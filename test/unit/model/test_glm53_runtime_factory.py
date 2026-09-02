# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]
PACKAGE_PATH = ROOT / "vllm_neuron" / "model" / "glm53_flash"


def _load(name: str):
    module_name = f"vllm_neuron.model.glm53_flash.{name}"
    spec = importlib.util.spec_from_file_location(
        module_name, PACKAGE_PATH / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in (
    "vllm_neuron",
    "vllm_neuron.model",
    "vllm_neuron.model.glm53_flash",
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(PACKAGE_PATH)]
    sys.modules.setdefault(package_name, package)

CONVERTER = _load("checkpoint_converter")
_load("streaming_rank_writer")
_load("rank_plan")
RUNTIME_CONFIG = _load("runtime_config")
FACTORY = _load("runtime_factory")


def _profile() -> dict:
    return {
        "schema": RUNTIME_CONFIG.GLM53_RUNTIME_CONFIG_SCHEMA,
        "architecture": RUNTIME_CONFIG.GLM53_ARCHITECTURE,
        "checkpoint_revision": RUNTIME_CONFIG.GLM53_CHECKPOINT_REVISION,
        "source_commit": "f0fc84830b78b17bde7fb8e72439e6de3a91a28a",
        "source_tree": "524462b4127f186541675c94744b9deed342d658",
        "runtime_adapter": FACTORY.GLM53_RUNTIME_ADAPTER,
        "compiler_image_id": "sha256:" + "1" * 64,
        "compiler_image_digest": "example.invalid/neuron@sha256:" + "2" * 64,
        "compiler_version": "reviewed-but-not-yet-observed",
        "runtime_packages": {
            "neuronx-distributed-inference": "explicit-unresolved-identity",
            "torch-neuronx": "explicit-unresolved-identity",
            "vllm-neuron": "f0fc84830b78b17bde7fb8e72439e6de3a91a28a",
        },
        "compiler_flags": ["--explicit-unresolved-flag"],
        "tensor_parallel_degree": 32,
        "logical_neuron_cores": 2,
        "batch_size": 1,
        "max_sequence_length": 8192,
        "context_encoding_buckets": [128, 512, 8192],
        "token_generation_buckets": [128, 512, 8192],
        "weight_dtype": "bfloat16",
        "cache_dtype": "bfloat16",
        "runtime_quantization": "none",
        "sampling_mode": "greedy",
        "output_logits": True,
        "speculative_decode": False,
    }


def _policy() -> dict:
    return {
        "ownership_path": "/mnt/compile/OWNERSHIP.md",
        "active_compile_cap": 2,
        "systemd_unit": "glm53-compile-review-01",
        "systemd_nice": 15,
        "systemd_scope": False,
        "network_mode": "none",
        "atomic_staging_suffix": ".partial-<run-id>",
        "compile_permitted": False,
    }


def _materialize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint_dir = tmp_path / "checkpoint"
    rank_dir = tmp_path / "ranks"
    checkpoint_dir.mkdir()
    rank_dir.mkdir()
    plans = {}
    for rank in range(32):
        checkpoint = rank_dir / f"tp{rank}_sharded_checkpoint.safetensors"
        checkpoint.write_bytes(f"rank-{rank}".encode())
        tensor = SimpleNamespace(
            name=f"layers.{rank}.weight",
            dtype="torch.bfloat16",
            shape=(1,),
            nbytes=2,
        )
        inventory = SimpleNamespace(
            contract_sha256=f"{rank + 1:064x}",
            total_tensor_bytes=2,
            tensors=(tensor,),
        )
        plan = SimpleNamespace(
            inventory=inventory, contract_sha256=f"{rank + 101:064x}"
        )
        plans[rank] = plan
        payload = {
            "schema": "glm53-streaming-rank-v1",
            "source": {
                "model": "GLM-5.3-Flash",
                "revision": CONVERTER.GLM53_CHECKPOINT_REVISION,
                "config_sha256": CONVERTER.GLM53_CONFIG_SHA256,
                "index_sha256": CONVERTER.GLM53_INDEX_SHA256,
            },
            "rank": rank,
            "tp_degree": 32,
            "rank_inventory_sha256": inventory.contract_sha256,
            "rank_plan_sha256": plan.contract_sha256,
            "checkpoint": {
                "path": checkpoint.name,
                "bytes": checkpoint.stat().st_size,
                "sha256": FACTORY._sha256_file(checkpoint),
            },
            "resource_bound": {
                "configured_max_chunk_bytes": FACTORY.GLM53_MAX_CHUNK_BYTES,
                "observed_max_chunk_bytes": 2,
                "full_rank_tensor_bytes": 2,
            },
            "tensors": [
                {
                    "name": tensor.name,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "nbytes": tensor.nbytes,
                    "chunks": 1,
                }
            ],
        }
        checkpoint.with_suffix(checkpoint.suffix + ".manifest.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    monkeypatch.setattr(
        FACTORY, "build_glm53_rank_plan", lambda _root, *, rank, **_: plans[rank]
    )
    profile = RUNTIME_CONFIG.Glm53RuntimeConfig.from_mapping(_profile())
    requested = tmp_path / "requested.json"
    emitted = tmp_path / "emitted.json"
    requested.write_bytes(profile.canonical_bytes())
    emitted.write_bytes(profile.canonical_bytes())
    return checkpoint_dir, rank_dir, requested, emitted


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: dict | None = None):
    checkpoint, ranks, requested, emitted = _materialize(tmp_path, monkeypatch)
    return FACTORY.Glm53RuntimeFactory.from_paths(
        checkpoint_dir=checkpoint,
        rank_dir=ranks,
        requested_config=requested,
        emitted_config=emitted,
        compile_policy=policy or _policy(),
    )


def test_exact_bundle_verifies_all_32_ranks_without_authorizing_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _build(tmp_path, monkeypatch)
    payload = bundle.to_mapping()
    assert len(payload["ranks"]) == 32
    assert payload["runtime_config"]["equal"] is True
    assert payload["topology"]["output_logits"] is True
    assert payload["claims"] == {
        "rank_files_verified": True,
        "compile_permitted": False,
        "runtime_permitted": False,
        "correctness_40_of_40": False,
        "performance": False,
        "tokenomics": False,
    }
    assert len(bundle.sha256()) == 64
    assert FACTORY.get_runtime_factories() == [
        (RUNTIME_CONFIG.GLM53_ARCHITECTURE, FACTORY.Glm53RuntimeFactory)
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ownership_path", "/tmp/OWNERSHIP.md", "OWNERSHIP"),
        ("active_compile_cap", 3, "cap"),
        ("systemd_unit", "compile.service", "bounded named"),
        ("systemd_nice", 0, "nice=15"),
        ("systemd_scope", True, "scope"),
        ("network_mode", "host", "network"),
        ("atomic_staging_suffix", ".partial", "staging suffix"),
        ("compile_permitted", True, "cannot authorize"),
    ],
)
def test_launch_policy_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    policy = _policy()
    policy[field] = value
    with pytest.raises(FACTORY.Glm53RuntimeFactoryError, match=message):
        _build(tmp_path, monkeypatch, policy)


@pytest.mark.parametrize("mode", ["missing", "hash", "source", "tensor", "chunk"])
def test_rank_manifest_or_artifact_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    checkpoint, ranks, requested, emitted = _materialize(tmp_path, monkeypatch)
    artifact = ranks / "tp7_sharded_checkpoint.safetensors"
    manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mode == "missing":
        artifact.unlink()
    elif mode == "hash":
        manifest["checkpoint"]["sha256"] = "0" * 64
    elif mode == "source":
        manifest["source"]["revision"] = "0" * 40
    elif mode == "tensor":
        manifest["tensors"][0]["shape"] = [2]
    else:
        manifest["resource_bound"]["observed_max_chunk_bytes"] = 64 * 1024 * 1024 + 1
    if mode != "missing":
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FACTORY.Glm53RuntimeFactoryError):
        FACTORY.Glm53RuntimeFactory.from_paths(
            checkpoint_dir=checkpoint,
            rank_dir=ranks,
            requested_config=requested,
            emitted_config=emitted,
            compile_policy=_policy(),
        )


def test_requested_emitted_drift_and_duplicate_manifest_key_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, ranks, requested, emitted = _materialize(tmp_path, monkeypatch)
    changed = copy.deepcopy(_profile())
    changed["batch_size"] = 2
    emitted.write_bytes(
        RUNTIME_CONFIG.Glm53RuntimeConfig.from_mapping(changed).canonical_bytes()
    )
    with pytest.raises(Exception, match="drifted"):
        FACTORY.Glm53RuntimeFactory.from_paths(
            checkpoint_dir=checkpoint,
            rank_dir=ranks,
            requested_config=requested,
            emitted_config=emitted,
            compile_policy=_policy(),
        )

    emitted.write_bytes(
        RUNTIME_CONFIG.Glm53RuntimeConfig.from_mapping(_profile()).canonical_bytes()
    )
    manifest = ranks / "tp0_sharded_checkpoint.safetensors.manifest.json"
    payload = manifest.read_text(encoding="utf-8").replace(
        '"schema": "glm53-streaming-rank-v1",',
        '"schema": "glm53-streaming-rank-v1", "schema": "duplicate",',
        1,
    )
    manifest.write_text(payload, encoding="utf-8")
    with pytest.raises(FACTORY.Glm53RuntimeFactoryError, match="duplicate JSON key"):
        FACTORY.Glm53RuntimeFactory.from_paths(
            checkpoint_dir=checkpoint,
            rank_dir=ranks,
            requested_config=requested,
            emitted_config=emitted,
            compile_policy=_policy(),
        )


def test_mutually_equal_lnc1_configs_still_fail_exact_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, ranks, requested, emitted = _materialize(tmp_path, monkeypatch)
    changed = copy.deepcopy(_profile())
    changed["logical_neuron_cores"] = 1
    config = RUNTIME_CONFIG.Glm53RuntimeConfig.from_mapping(changed)
    requested.write_bytes(config.canonical_bytes())
    emitted.write_bytes(config.canonical_bytes())
    with pytest.raises(FACTORY.Glm53RuntimeFactoryError, match="requires LNC2"):
        FACTORY.Glm53RuntimeFactory.from_paths(
            checkpoint_dir=checkpoint,
            rank_dir=ranks,
            requested_config=requested,
            emitted_config=emitted,
            compile_policy=_policy(),
        )
