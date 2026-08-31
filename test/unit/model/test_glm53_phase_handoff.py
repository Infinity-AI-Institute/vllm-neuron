from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "vllm_neuron/model/glm53_flash/phase_handoff.py"
SPEC = importlib.util.spec_from_file_location("glm53_phase_handoff", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


SHARED = {
    "model": "GLM-5.3-Flash",
    "tp": 32,
    "lnc": 2,
    "batch": 1,
    "sequence": 128,
    "dtype": "bfloat16",
    "quantization": "none",
    "speculation": False,
    "source_commit": "a" * 40,
    "source_tree": "b" * 40,
    "nxdi_commit": "c" * 40,
    "checkpoint_revision": "d" * 40,
    "image": "public.ecr.aws/neuron/image@sha256:" + "e" * 64,
}


def _write_model(root: Path, phase: str, *, state_keys: tuple[str, ...]) -> None:
    model = root / "artifacts/model/model.pt"
    model.parent.mkdir(parents=True, exist_ok=True)
    name = "token_generation_model" if phase == "tkg" else "context_encoding_model"
    marker = "LayoutTransformation" if phase == "tkg" else "ops.neuron._parallel_load"
    code = f"{name} {marker} " + " ".join(state_keys)
    data = " ".join(state_keys).encode()
    with zipfile.ZipFile(model, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("model/code/__torch__/neuronx_distributed/trace/spmd.py", code)
        archive.writestr("model/data.pkl", data)


def _write_artifact(
    root: Path, phase: str, *, state_keys=("past_key_values.0",)
) -> None:
    _write_model(root, phase, state_keys=state_keys)
    artifacts = root / "artifacts"
    (artifacts / "effective-shape.json").write_text(
        json.dumps(
            {
                "model": "GLM-5.3-Flash",
                "tp": 32,
                "lnc": 2,
                "resident_batch": 1,
                "sequence": 128,
                "max_model_len": 128,
                "emit_phases": phase.upper(),
                "models_compiled": [
                    "token_generation_model"
                    if phase == "tkg"
                    else "context_encoding_model"
                ],
                "blockwise_use_shard_on_intermediate_dynamic_while": True,
                "kv_cache_quant_requested": False,
            }
        )
    )
    (artifacts / "model/neuron_config.json").write_text(
        json.dumps({"neuron_config": {"torch_dtype": "bfloat16"}})
    )
    (artifacts / "compile-result.json").write_text(
        json.dumps(
            {
                "neff_count": 1,
                "neuron_config_has_float8_e4m3fn": False,
                "neuron_config_has_bfloat16": True,
            }
        )
    )
    (root / "cache/compiler/model.hlo_module.pb").parent.mkdir(parents=True)
    (root / "cache/compiler/model.hlo_module.pb").write_bytes(b"hlo")
    (root / "cache/compiler/model.neff").write_bytes(b"neff")
    if phase == "tkg":
        (artifacts / "source-identity.json").write_text(json.dumps(SHARED))


def _compose(tkg: Path, cte: Path) -> Path:
    artifacts = {
        "tkg": {
            "compile_result_sha256": _sha(tkg / "artifacts/compile-result.json"),
            "effective_shape_sha256": _sha(tkg / "artifacts/effective-shape.json"),
            "emitted_config_sha256": _sha(tkg / "artifacts/model/neuron_config.json"),
            "hlo_sha256": _sha(tkg / "cache/compiler/model.hlo_module.pb"),
            "neff_sha256": _sha(tkg / "cache/compiler/model.neff"),
        },
        "cte": {
            "compile_result_sha256": _sha(cte / "artifacts/compile-result.json"),
            "effective_shape_sha256": _sha(cte / "artifacts/effective-shape.json"),
            "emitted_config_sha256": _sha(cte / "artifacts/model/neuron_config.json"),
            "hlo_sha256": _sha(cte / "cache/compiler/model.hlo_module.pb"),
            "neff_sha256": _sha(cte / "cache/compiler/model.neff"),
        },
    }
    path = cte / "artifacts/tkg-cte-compose-receipt.json"
    path.write_text(
        json.dumps(
            {
                "schema": "glm53-tkg-cte-compose-v1",
                "shared_contract": SHARED,
                "artifacts": artifacts,
                "reuse_policy": {
                    "tkg_neff_reused_unchanged": True,
                    "cte_neff_reused_unchanged": True,
                },
            }
        )
    )
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_handoff_binds_distinct_loaders_shared_state_and_model_bytes(tmp_path):
    tkg = tmp_path / "tkg"
    cte = tmp_path / "cte"
    _write_artifact(tkg, "tkg", state_keys=("past_key_values.0", "past_key_values.10"))
    _write_artifact(cte, "cte", state_keys=("past_key_values.0", "past_key_values.10"))
    receipt = MODULE.inspect_phase_handoff(
        tkg, cte, compose_receipt_path=_compose(tkg, cte)
    )
    assert receipt["handoff"]["weight_loader_difference_is_phase_local"] is True
    assert receipt["handoff"]["shared_state_schema"] is True
    assert receipt["handoff"]["model_pt_bound_for_both_phases"] is True
    assert receipt["tkg"]["serialized"]["loader"] == "LayoutTransformation"
    assert receipt["cte"]["serialized"]["loader"] == "_parallel_load"
    assert receipt["tkg"]["model_pt"]["bytes"] > 0


def test_phase_handoff_rejects_state_schema_drift(tmp_path):
    tkg = tmp_path / "tkg"
    cte = tmp_path / "cte"
    _write_artifact(tkg, "tkg", state_keys=("past_key_values.0", "past_key_values.10"))
    _write_artifact(cte, "cte", state_keys=("past_key_values.0", "past_key_values.11"))
    with pytest.raises(MODULE.Glm53PhaseHandoffError, match="state-key schema"):
        MODULE.inspect_phase_handoff(tkg, cte, compose_receipt_path=_compose(tkg, cte))


def test_phase_handoff_rejects_wrong_phase_loader(tmp_path):
    tkg = tmp_path / "tkg"
    cte = tmp_path / "cte"
    _write_artifact(tkg, "tkg")
    _write_artifact(cte, "cte")
    cte_model = cte / "artifacts/model/model.pt"
    with zipfile.ZipFile(cte_model, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "model/code/__torch__/neuronx_distributed/trace/spmd.py",
            "context_encoding_model LayoutTransformation past_key_values.0",
        )
        archive.writestr("model/data.pkl", b"past_key_values.0")
    with pytest.raises(MODULE.Glm53PhaseHandoffError, match="loader marker missing"):
        MODULE.inspect_phase_handoff(tkg, cte, compose_receipt_path=_compose(tkg, cte))


def test_phase_handoff_rejects_compose_hash_drift(tmp_path):
    tkg = tmp_path / "tkg"
    cte = tmp_path / "cte"
    _write_artifact(tkg, "tkg")
    _write_artifact(cte, "cte")
    compose = _compose(tkg, cte)
    payload = json.loads(compose.read_text(encoding="utf-8"))
    payload["artifacts"]["cte"]["neff_sha256"] = "0" * 64
    compose.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        MODULE.Glm53PhaseHandoffError, match="compose cte neff_sha256 drift"
    ):
        MODULE.inspect_phase_handoff(tkg, cte, compose_receipt_path=compose)


@pytest.mark.parametrize("missing_mode", ["null", "absent"])
def test_phase_handoff_rejects_absent_cte_launch_source_provenance(
    tmp_path, missing_mode
):
    tkg = tmp_path / "tkg"
    cte = tmp_path / "cte"
    _write_artifact(tkg, "tkg")
    _write_artifact(cte, "cte")
    compose = _compose(tkg, cte)
    launch = cte / "artifacts/launch-receipt.json"
    source = {
        "nxdi_commit": SHARED["nxdi_commit"],
        "checkpoint_revision": SHARED["checkpoint_revision"],
    }
    if missing_mode == "null":
        source.update({"source_commit": None, "source_tree": None})
    launch.write_text(json.dumps({"source": source}), encoding="utf-8")
    with pytest.raises(
        MODULE.Glm53PhaseHandoffError, match="CTE launch source disagrees"
    ):
        MODULE.inspect_phase_handoff(tkg, cte, compose_receipt_path=compose)


def test_phase_handoff_accepts_immutable_staged_artifacts_cache_layout(tmp_path):
    tkg = tmp_path / "tkg"
    cte = tmp_path / "cte"
    _write_artifact(tkg, "tkg")
    _write_artifact(cte, "cte")
    compose = _compose(tkg, cte)
    for root in (tkg, cte):
        source = root / "cache"
        staged = root / "artifacts" / "cache"
        staged.mkdir(parents=True)
        for path in source.rglob("*"):
            if path.is_file():
                target = staged / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
        for path in sorted(source.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        source.rmdir()
    receipt = MODULE.inspect_phase_handoff(tkg, cte, compose_receipt_path=compose)
    assert receipt["handoff"]["shared_state_schema"] is True
