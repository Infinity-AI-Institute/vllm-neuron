# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model.glm52_moe_dsa import model as model_module
from vllm_neuron.model.glm52_moe_dsa.artifact_preflight import (
    INDEX_FILENAME,
    preflight_checkpoint_artifact,
)
from vllm_neuron.model.glm52_moe_dsa.static_fp8 import (
    NEURON_LEGACY_E4M3FN_QMAX240,
    OCP_E4M3FN_QMAX448,
)

DIRECT = NEURON_LEGACY_E4M3FN_QMAX240
DIRECT_HYBRID_VERSION = (
    "glm52-trn2-static-fp8-direct-legacy-bf16-shared-v1"
)
OCP_HYBRID_VERSION = "glm52-trn2-static-fp8-bf16-shared-v1"
MANIFEST_FILENAME = "glm52-static-fp8-manifest.json"


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_direct_fixture(root: Path) -> None:
    root.mkdir()
    config = {
        "shared_expert_dtype": "bfloat16",
        "static_fp8_weight_format": DIRECT,
        "quantization_config": {
            "quant_method": "modelopt",
            "artifact_version": DIRECT_HYBRID_VERSION,
            "quantization": {
                "quant_algo": "FP8",
                "weight_format": DIRECT,
            },
        },
        "glm52_artifact": {
            "artifact_version": DIRECT_HYBRID_VERSION,
            "shared_expert_dtype": "bfloat16",
            "static_fp8_weight_format": DIRECT,
            "manifest_file": MANIFEST_FILENAME,
            "loader_ready": True,
            "index_closure_status": "passed",
        },
    }
    index = {
        "metadata": {
            "artifact_version": DIRECT_HYBRID_VERSION,
            "shared_expert_dtype": "bfloat16",
            "static_fp8_weight_format": DIRECT,
            "total_size": 1,
        },
        "weight_map": {"model.embed_tokens.weight": "model.safetensors"},
    }
    _write_json(root / "config.json", config)
    _write_json(root / INDEX_FILENAME, index)
    manifest = {
        "artifact_version": DIRECT_HYBRID_VERSION,
        "shared_expert_dtype": "bfloat16",
        "static_fp8_weight_format": DIRECT,
        "quantization": {"storage_format": DIRECT},
        "source": {"index_closure": {"status": "passed"}},
        "output": {
            "index_file": INDEX_FILENAME,
            "index_sha256": _sha256(root / INDEX_FILENAME),
        },
        "loader_validation": {
            "loader_ready": True,
            "required_artifact_version": DIRECT_HYBRID_VERSION,
            "required_static_fp8_weight_format": DIRECT,
        },
    }
    _write_json(root / MANIFEST_FILENAME, manifest)


def _preflight(root: Path) -> str:
    return preflight_checkpoint_artifact(
        root,
        expected_weight_format=DIRECT,
        expected_shared_expert_dtype="bfloat16",
    )


def test_valid_direct_artifact_closes_config_manifest_and_index(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _write_direct_fixture(root)

    assert _preflight(root) == DIRECT


def test_direct_artifact_rejects_missing_manifest(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_direct_fixture(root)
    (root / MANIFEST_FILENAME).unlink()

    with pytest.raises(ValueError, match="missing manifest"):
        _preflight(root)


@pytest.mark.parametrize("loader_name", ("load_weights", "load_weights_lite"))
def test_model_loaders_preflight_before_safetensors_checkpoint(
    tmp_path: Path,
    monkeypatch,
    loader_name: str,
) -> None:
    root = tmp_path / "artifact"
    _write_direct_fixture(root)
    (root / MANIFEST_FILENAME).unlink()
    checkpoint_opened = False

    def checkpoint_factory(*args, **kwargs):
        del args, kwargs
        nonlocal checkpoint_opened
        checkpoint_opened = True
        raise AssertionError("checkpoint opened before GLM artifact preflight")

    monkeypatch.setattr(
        model_module,
        "SafetensorsCheckpoint",
        checkpoint_factory,
    )
    fake_model = SimpleNamespace(
        config=SimpleNamespace(
            static_fp8_weight_format=DIRECT,
            shared_expert_dtype="bfloat16",
        )
    )

    with pytest.raises(ValueError, match="missing manifest"):
        getattr(model_module.Glm52MoeDsaForCausalLM, loader_name)(
            fake_model,
            str(root),
            torch.device("cpu"),
            None,
        )
    assert checkpoint_opened is False


def test_direct_artifact_rejects_stale_index_hash(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_direct_fixture(root)
    index = json.loads((root / INDEX_FILENAME).read_text(encoding="utf-8"))
    index["weight_map"]["model.norm.weight"] = "other.safetensors"
    _write_json(root / INDEX_FILENAME, index)

    with pytest.raises(ValueError, match="SHA-256 is stale or mismatched"):
        _preflight(root)


def test_direct_artifact_rejects_mixed_manifest_format(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_direct_fixture(root)
    manifest_path = root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["static_fp8_weight_format"] = OCP_E4M3FN_QMAX448
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="manifest static-FP8 format"):
        _preflight(root)


def test_direct_artifact_rejects_mixed_index_format_with_fresh_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _write_direct_fixture(root)
    index_path = root / INDEX_FILENAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["metadata"]["static_fp8_weight_format"] = OCP_E4M3FN_QMAX448
    _write_json(index_path, index)
    manifest_path = root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["index_sha256"] = _sha256(index_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="index static-FP8 format"):
        _preflight(root)


def test_direct_artifact_rejects_swapped_index_config_with_fresh_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _write_direct_fixture(root)
    index_path = root / INDEX_FILENAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["metadata"].update(
        artifact_version="glm52-trn2-static-fp8-direct-legacy-v1",
        shared_expert_dtype="fp8",
    )
    _write_json(index_path, index)
    manifest_path = root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["index_sha256"] = _sha256(index_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="index artifact_version"):
        _preflight(root)


def test_markerless_ocp_rejects_direct_index_metadata(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_direct_fixture(root)
    config = {
        "shared_expert_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "modelopt",
            "quantization": {"quant_algo": "FP8"},
        },
        "glm52_artifact": {
            "artifact_version": OCP_HYBRID_VERSION,
            "shared_expert_dtype": "bfloat16",
            "loader_ready": True,
            "index_closure_status": "passed",
        },
    }
    _write_json(root / "config.json", config)

    with pytest.raises(ValueError, match="cannot load direct qmax-240"):
        preflight_checkpoint_artifact(
            root,
            expected_weight_format=OCP_E4M3FN_QMAX448,
            expected_shared_expert_dtype="bfloat16",
        )


def test_valid_markerless_ocp_artifact_preserves_legacy_behavior(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    _write_json(
        root / "config.json",
        {
            "shared_expert_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "modelopt",
                "quantization": {"quant_algo": "FP8"},
            },
            "glm52_artifact": {
                "artifact_version": OCP_HYBRID_VERSION,
                "shared_expert_dtype": "bfloat16",
                "loader_ready": True,
                "index_closure_status": "passed",
            },
        },
    )
    _write_json(
        root / INDEX_FILENAME,
        {
            "metadata": {
                "artifact_version": OCP_HYBRID_VERSION,
                "shared_expert_dtype": "bfloat16",
            },
            "weight_map": {"model.embed_tokens.weight": "model.safetensors"},
        },
    )

    assert (
        preflight_checkpoint_artifact(
            root,
            expected_weight_format=OCP_E4M3FN_QMAX448,
            expected_shared_expert_dtype="bfloat16",
        )
        == OCP_E4M3FN_QMAX448
    )
