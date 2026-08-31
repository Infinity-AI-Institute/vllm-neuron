from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]
PACKAGE_PATH = ROOT / "vllm_neuron" / "model" / "glm53_flash"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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

PREP = _load_module(
    "vllm_neuron.model.glm53_flash.card_prep", PACKAGE_PATH / "card_prep.py"
)
TOOL = _load_module("glm53_tkg_card_prep", ROOT / "tools" / "glm53_tkg_card_prep.py")


def _write_artifact(
    tmp_path: Path, *, shape_overrides: dict | None = None
) -> tuple[Path, Path]:
    root = tmp_path / "glm53-artifact"
    model = root / "artifacts" / "model"
    cache = root / "cache" / "compiler" / "MODULE_test"
    model.mkdir(parents=True)
    cache.mkdir(parents=True)
    emitted = {
        "tp_degree": 32,
        "world_size": 32,
        "logical_nc_config": 2,
        "batch_size": 1,
        "ctx_batch_size": 1,
        "tkg_batch_size": 1,
        "buckets": [128],
        "context_encoding_buckets": None,
        "token_generation_buckets": None,
        "seq_len": 128,
        "max_context_length": 128,
        "torch_dtype": "bfloat16",
        "kv_cache_quant": False,
        "quantized": False,
        "enable_eagle_speculation": False,
        "enable_fused_speculation": False,
    }
    emitted.update(shape_overrides or {})
    (model / "neuron_config.json").write_text(
        json.dumps({"neuron_config": emitted}), encoding="utf-8"
    )
    model_pt_path = model / "model.pt"
    model_pt_path.write_bytes(b"serialized-model")
    effective_shape_path = root / "artifacts" / "effective-shape.json"
    effective_shape_path.write_text(
        json.dumps(
            {
                "tp": 32,
                "lnc": 2,
                "resident_batch": 1,
                "sequence": 128,
                "emit_phases": "TKG",
                "models_compiled": ["token_generation_model"],
            }
        ),
        encoding="utf-8",
    )
    compile_result_path = root / "artifacts" / "compile-result.json"
    compile_result_path.write_text(
        json.dumps(
            {
                "neffs": [
                    {"path": "/runroot/cache/MODULE_test/model.neff", "bytes": 4}
                ],
                "neff_count": 1,
                "neuron_config_has_float8_e4m3fn": False,
                "neuron_config_has_bfloat16": True,
            }
        ),
        encoding="utf-8",
    )
    hlo_path = cache / "model.hlo_module.pb"
    neff_path = cache / "model.neff"
    hlo_path.write_bytes(b"HLO no sort")
    neff_path.write_bytes(b"NEFF")
    source_identity_path = root / "artifacts" / "source-identity.json"
    source_identity = {
        "source_commit": "source-commit",
        "source_tree": "source-tree",
        "nxdi_commit": "nxdi-commit",
        "checkpoint_revision": "checkpoint-revision",
    }
    source_identity_path.write_text(json.dumps(source_identity), encoding="utf-8")

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    packet = {
        "schema": "glm53-retained-tkg-artifact-v1",
        "artifact_role": "retained compiler evidence; not a fresh compile",
        "artifact_source": source_identity,
        "source_identity": {
            "relative_path": "artifacts/source-identity.json",
            "sha256": sha256(source_identity_path),
        },
        "model": {
            "relative_path": "artifacts/model/model.pt",
            "sha256": sha256(model_pt_path),
            "bytes": model_pt_path.stat().st_size,
        },
        "effective_shape": {
            "relative_path": "artifacts/effective-shape.json",
            "sha256": sha256(effective_shape_path),
        },
        "emitted_config": {
            "relative_path": "artifacts/model/neuron_config.json",
            "sha256": sha256(model / "neuron_config.json"),
        },
        "compile_result": {
            "relative_path": "artifacts/compile-result.json",
            "sha256": sha256(compile_result_path),
        },
        "compiler_evidence": {
            "hlo_relative_path": "cache/compiler/MODULE_test/model.hlo_module.pb",
            "hlo_sha256": sha256(hlo_path),
            "neff_relative_path": "cache/compiler/MODULE_test/model.neff",
            "neff_sha256": sha256(neff_path),
            "neff_bytes": neff_path.stat().st_size,
        },
        "topology": {
            "tp": 32,
            "lnc": 2,
            "batch": 1,
            "sequence": 128,
            "dtype": "bfloat16",
            "kv_cache_quant": False,
            "quantized": False,
            "speculation": False,
            "emit_phases": "TKG",
            "models_compiled": ["token_generation_model"],
        },
    }
    packet_path = tmp_path / "RETAINED-TKG-ARTIFACT-PACKET.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    return root, packet_path


def _fake_memory_inputs():
    audit = SimpleNamespace(
        shard_count=7,
        tensor_count=11,
        indexed_payload_bytes=1234,
        payload_bytes_loaded_during_audit=0,
    )
    reader = SimpleNamespace(audit_report=audit)

    def reader_factory(_):
        return reader

    def rank_plan_builder(_, *, rank, **__):
        specs = (
            SimpleNamespace(dtype="torch.bfloat16", nbytes=100 + rank),
            SimpleNamespace(dtype="torch.float32", nbytes=20),
        )
        return SimpleNamespace(inventory=SimpleNamespace(tensors=specs))

    return rank_plan_builder, reader_factory


def test_tkg_artifact_is_explicitly_not_fresh_prompt_ready(tmp_path: Path) -> None:
    root, packet = _write_artifact(tmp_path)
    receipt = PREP.inspect_tkg_artifact(root, retained_packet_path=packet)
    assert receipt.models_compiled == ("token_generation_model",)
    assert receipt.cte_artifact_present is False
    assert receipt.rank_bundle_present is False
    assert receipt.fresh_prompt_ready is False
    assert receipt.continuation_tkg_ready is False
    assert receipt.hlo_sha256 == hashlib.sha256(b"HLO no sort").hexdigest()
    assert receipt.neff_sha256 == hashlib.sha256(b"NEFF").hexdigest()
    assert "context_encoding_model" in receipt.blockers[0]
    assert "32 TP32 sharded" in receipt.blockers[1]
    assert receipt.to_mapping()["readiness"]["card_launch_authorized"] is False


def test_tkg_packet_rejects_serialized_model_hash_drift(tmp_path: Path) -> None:
    root, packet_path = _write_artifact(tmp_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["model"]["sha256"] = "0" * 64
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(
        PREP.Glm53CardPreparationError, match="serialized model hash drift"
    ):
        PREP.inspect_tkg_artifact(root, retained_packet_path=packet_path)


def test_memory_measurement_is_exact_and_audit_loads_no_payload() -> None:
    builder, reader_factory = _fake_memory_inputs()
    memory = PREP.measure_checkpoint_memory(
        "/pinned/checkpoint",
        rank_plan_builder=builder,
        reader_factory=reader_factory,
    )
    assert memory.source_indexed_payload_bytes == 1234
    assert memory.source_payload_bytes_loaded_during_audit == 0
    assert memory.rank_count == 32
    assert memory.rank_tensor_bytes == tuple(120 + rank for rank in range(32))
    assert memory.rank_bfloat16_bytes == tuple(100 + rank for rank in range(32))
    assert memory.rank_non_bfloat16_bytes == (20,) * 32


@pytest.mark.parametrize(
    "overrides",
    [
        {"tp_degree": 16},
        {"torch_dtype": "float16"},
        {"context_encoding_buckets": [128]},
        {"enable_fused_speculation": True},
    ],
)
def test_emitted_config_drift_fails_closed(tmp_path: Path, overrides: dict) -> None:
    root, packet = _write_artifact(tmp_path, shape_overrides=overrides)
    with pytest.raises(PREP.Glm53CardPreparationError, match="emitted config drift"):
        PREP.inspect_tkg_artifact(root, retained_packet_path=packet)


def test_shape_only_cte_cannot_publish_fresh_prompt_readiness(tmp_path: Path) -> None:
    root, packet = _write_artifact(tmp_path)
    cte_root = tmp_path / "shape-only-cte"
    (cte_root / "artifacts").mkdir(parents=True)
    (cte_root / "artifacts" / "effective-shape.json").write_text(
        json.dumps({"models_compiled": ["context_encoding_model"]}), encoding="utf-8"
    )
    receipt = PREP.inspect_tkg_artifact(
        root, cte_artifact_root=cte_root, retained_packet_path=packet
    )
    assert receipt.cte_artifact_present is False
    assert receipt.fresh_prompt_ready is False


def test_retained_hlo_sort_marker_is_rejected_even_when_packet_is_rehashed(
    tmp_path: Path,
) -> None:
    root, packet_path = _write_artifact(tmp_path)
    hlo_path = root / "cache" / "compiler" / "MODULE_test" / "model.hlo_module.pb"
    hlo_path.write_bytes(b"HLO %sort. op_type=aten__topk")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["compiler_evidence"]["hlo_sha256"] = hashlib.sha256(
        hlo_path.read_bytes()
    ).hexdigest()
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(PREP.Glm53CardPreparationError, match="unsupported"):
        PREP.inspect_tkg_artifact(root, retained_packet_path=packet_path)


def test_zero_byte_rank_bundle_cannot_publish_continuation_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, packet = _write_artifact(tmp_path)
    rank_dir = tmp_path / "ranks"
    rank_dir.mkdir()
    for rank in range(32):
        filename = f"tp{rank}_sharded_checkpoint.safetensors"
        (rank_dir / filename).write_bytes(b"")
        (rank_dir / f"{filename}.manifest.json").write_text("{}", encoding="utf-8")
    fake_memory = PREP.Glm53CheckpointMemory(
        source_shard_count=1,
        source_tensor_count=1,
        source_indexed_payload_bytes=1,
        source_payload_bytes_loaded_during_audit=0,
        rank_tensor_bytes=(1,) * 32,
        rank_bfloat16_bytes=(1,) * 32,
        rank_non_bfloat16_bytes=(0,) * 32,
        rank_inventory_sha256=("a" * 64,) * 32,
        rank_plan_sha256=("b" * 64,) * 32,
    )
    monkeypatch.setattr(PREP, "measure_checkpoint_memory", lambda _: fake_memory)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    receipt = PREP.inspect_tkg_artifact(
        root,
        checkpoint_dir=checkpoint_dir,
        rank_dir=rank_dir,
        retained_packet_path=packet,
    )
    assert receipt.rank_bundle_present is False
    assert receipt.continuation_tkg_ready is False


def test_cli_forwards_external_retained_packet_without_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    packet = tmp_path / "external" / "RETAINED-TKG-ARTIFACT-PACKET.json"
    output = tmp_path / "receipt.json"
    calls: dict[str, object] = {}

    def fake_inspect(*args, **kwargs):
        calls["args"] = args
        calls.update(kwargs)
        return SimpleNamespace(
            to_mapping=lambda: {"schema": "test"}, sha256=lambda: "a" * 64
        )

    monkeypatch.setattr(TOOL, "inspect_tkg_artifact", fake_inspect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "glm53_tkg_card_prep.py",
            "--artifact-root",
            str(artifact),
            "--retained-packet",
            str(packet),
            "--output",
            str(output),
        ],
    )

    assert TOOL.main() == 0
    assert calls["args"] == (artifact,)
    assert calls["retained_packet_path"] == packet
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "receipt_sha256": "a" * 64,
        "schema": "test",
    }


def test_cli_omitted_retained_packet_preserves_checked_in_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    def fake_inspect(*args, **kwargs):
        calls["args"] = args
        calls.update(kwargs)
        return SimpleNamespace(to_mapping=dict, sha256=lambda: "b" * 64)

    monkeypatch.setattr(TOOL, "inspect_tkg_artifact", fake_inspect)
    monkeypatch.setattr(
        sys,
        "argv",
        ["glm53_tkg_card_prep.py", "--artifact-root", str(tmp_path / "artifact")],
    )

    assert TOOL.main() == 0
    assert calls["retained_packet_path"] is None


def test_retained_artifact_gate_verifies_actual_hlo_neff_and_packet() -> None:
    artifact_value = os.environ.get("GLM53_RETAINED_ARTIFACT_ROOT")
    if not artifact_value:
        pytest.skip(
            "set GLM53_RETAINED_ARTIFACT_ROOT to run the retained artifact gate"
        )
    root = Path(artifact_value)
    packet = json.loads(
        (PACKAGE_PATH / "RETAINED-TKG-ARTIFACT-PACKET.json").read_text(encoding="utf-8")
    )
    receipt = PREP.inspect_tkg_artifact(root)
    assert receipt.effective_shape_sha256 == packet["effective_shape"]["sha256"]
    assert receipt.compile_result_sha256 == packet["compile_result"]["sha256"]
    assert receipt.hlo_sha256 == packet["compiler_evidence"]["hlo_sha256"]
    assert receipt.neff_sha256 == packet["compiler_evidence"]["neff_sha256"]
    assert (
        Path(receipt.hlo_path).relative_to(root).as_posix()
        == packet["compiler_evidence"]["hlo_relative_path"]
    )
    assert (
        Path(receipt.neff_path).relative_to(root).as_posix()
        == packet["compiler_evidence"]["neff_relative_path"]
    )
    source_identity_path = root / packet["source_identity"]["relative_path"]
    assert (
        hashlib.sha256(source_identity_path.read_bytes()).hexdigest()
        == packet["source_identity"]["sha256"]
    )
    source_identity = json.loads(source_identity_path.read_text(encoding="utf-8"))
    assert {
        key: source_identity[key]
        for key in (
            "source_commit",
            "source_tree",
            "nxdi_commit",
            "checkpoint_revision",
        )
    } == packet["artifact_source"]
    assert Path(receipt.hlo_path).read_bytes().find(b"%sort.") < 0
    assert Path(receipt.hlo_path).read_bytes().find(b"aten__topk") < 0
    compile_result = json.loads(
        (root / "artifacts" / "compile-result.json").read_text(encoding="utf-8")
    )
    assert compile_result["neff_count"] == 1
    assert (
        compile_result["neffs"][0]["bytes"] == packet["compiler_evidence"]["neff_bytes"]
    )
    assert compile_result["neuron_config_has_float8_e4m3fn"] is False
    assert compile_result["neuron_config_has_bfloat16"] is True
