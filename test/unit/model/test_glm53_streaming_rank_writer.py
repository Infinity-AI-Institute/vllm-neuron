from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

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

CONVERTER = _load_module(
    "vllm_neuron.model.glm53_flash.checkpoint_converter",
    PACKAGE_PATH / "checkpoint_converter.py",
)
WRITER = _load_module(
    "vllm_neuron.model.glm53_flash.streaming_rank_writer",
    PACKAGE_PATH / "streaming_rank_writer.py",
)
RANK_PLAN = _load_module(
    "vllm_neuron.model.glm53_flash.rank_plan",
    PACKAGE_PATH / "rank_plan.py",
)

Glm53ArchitectureMismatch = CONVERTER.Glm53ArchitectureMismatch
Glm53CheckpointReport = CONVERTER.Glm53CheckpointReport
Glm53StreamingError = WRITER.Glm53StreamingError
IndexedTensorReader = WRITER.IndexedTensorReader
RankInventory = WRITER.RankInventory
StreamingRankWriter = WRITER.StreamingRankWriter
TensorChunk = WRITER.TensorChunk
TensorSpec = WRITER.TensorSpec
stream_rank_checkpoint = WRITER.stream_rank_checkpoint
Glm53RankPlan = RANK_PLAN.Glm53RankPlan
TargetTensorPlan = RANK_PLAN.TargetTensorPlan
build_glm53_rank_plan = RANK_PLAN.build_glm53_rank_plan


def _source_report() -> Glm53CheckpointReport:
    return Glm53CheckpointReport(
        architecture="Glm5NextForConditionalGeneration",
        model_type="glm5_next",
        text_model_type="glm5_next_text",
        tensor_count=76_108,
        block_scale_count=37_338,
        vision_tensor_count=347,
        mtp_tensor_count=1_760,
        indexer_tensor_count=84,
        dsa_layers=tuple(range(3, 45, 4)),
        kda_layers=tuple(i for i in range(45) if i not in range(3, 45, 4)),
        weight_block_size=(128, 128),
    )


def _inventory() -> RankInventory:
    return RankInventory(
        rank=1,
        tp_degree=4,
        tensors=(
            TensorSpec("layers.0.weight", torch.bfloat16, (10_000,)),
            TensorSpec("layers.0.bias", torch.float32, (6,)),
        ),
    )


def test_streaming_writer_publishes_valid_safetensors_with_bounded_chunks(
    tmp_path: Path,
) -> None:
    output = tmp_path / "tp1.safetensors"
    inventory = _inventory()
    with StreamingRankWriter(
        output,
        inventory,
        source_report=_source_report(),
        max_chunk_bytes=200,
        plan_contract_sha256="a" * 64,
    ) as writer:
        expected_weight = torch.arange(10_000, dtype=torch.float32).to(torch.bfloat16)
        for start in range(0, 10_000, 100):
            writer.write_chunk(
                TensorChunk(
                    "layers.0.weight", start, expected_weight[start : start + 100]
                )
            )
        expected_bias = torch.arange(6, dtype=torch.float32)
        for start in range(0, 6, 5):
            writer.write_chunk(
                TensorChunk("layers.0.bias", start, expected_bias[start : start + 5])
            )
        assert not any(
            isinstance(value, torch.Tensor) for value in writer.__dict__.values()
        )
        manifest = writer.finalize()

    assert output.exists()
    assert manifest["rank"] == 1
    assert manifest["tp_degree"] == 4
    assert manifest["rank_inventory_sha256"] == inventory.contract_sha256
    assert manifest["rank_plan_sha256"] == "a" * 64
    assert manifest["resource_bound"]["observed_max_chunk_bytes"] == 200
    assert manifest["resource_bound"]["full_rank_tensor_bytes"] == 20_024
    assert manifest["resource_bound"]["observed_max_chunk_bytes"] * 100 < 20_024
    assert manifest["resource_bound"]["chunks_written"] == 102
    assert output.with_suffix(".safetensors.manifest.json").exists()
    with safe_open(output, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {"layers.0.weight", "layers.0.bias"}
        torch.testing.assert_close(
            handle.get_tensor("layers.0.weight"), expected_weight
        )
        torch.testing.assert_close(handle.get_tensor("layers.0.bias"), expected_bias)
        metadata = handle.metadata()
        assert metadata["revision"] == CONVERTER.GLM53_CHECKPOINT_REVISION
        assert metadata["rank"] == "1"
        assert metadata["plan_contract_sha256"] == "a" * 64


def test_writer_rejects_overlap_gap_oversize_and_never_publishes_partial(
    tmp_path: Path,
) -> None:
    output = tmp_path / "tp0.safetensors"
    inventory = RankInventory(
        rank=0,
        tp_degree=2,
        tensors=(TensorSpec("weight", torch.float32, (8,)),),
    )
    with StreamingRankWriter(
        output,
        inventory,
        source_report=_source_report(),
        max_chunk_bytes=16,
    ) as writer:
        writer.write_chunk(TensorChunk("weight", 0, torch.ones(4)))
        with pytest.raises(Glm53StreamingError, match="overlapping"):
            writer.write_chunk(TensorChunk("weight", 2, torch.ones(2)))
        with pytest.raises(Glm53StreamingError, match="bound"):
            writer.write_chunk(TensorChunk("weight", 4, torch.ones(5)))
        with pytest.raises(Glm53StreamingError, match="incomplete"):
            writer.finalize()
    assert not output.exists()
    assert not list(tmp_path.glob("*.partial-*"))


def test_rank_inventory_rejects_duplicates_and_invalid_rank() -> None:
    spec = TensorSpec("x", torch.float32, (1,))
    with pytest.raises(ValueError, match="duplicate"):
        RankInventory(rank=0, tp_degree=1, tensors=(spec, spec))
    with pytest.raises(ValueError, match="rank/TP"):
        RankInventory(rank=2, tp_degree=2, tensors=(spec,))
    with pytest.raises(ValueError, match="positive shape"):
        TensorSpec("bad", torch.float32, (0,))


def _small_checkpoint(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    root = tmp_path / CONVERTER.GLM53_CHECKPOINT_REVISION
    root.mkdir()
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, root / shard)
    index = {"weight_map": {key: shard for key in tensors}}
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    return root


def _patch_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(WRITER, "preflight_checkpoint_dir", lambda _: _source_report())


def test_indexed_reader_audits_exact_shards_and_reads_one_source_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preflight(monkeypatch)
    weight_key = "model.language_model.layers.3.self_attn.q_a_proj.weight"
    scale_key = f"{weight_key}_scale_inv"
    source = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.float8_e4m3fn)
    root = _small_checkpoint(
        tmp_path,
        {
            weight_key: source,
            scale_key: torch.full((1, 1), 0.5),
            "model.language_model.layers.0.holdout.weight": torch.ones(
                4, dtype=torch.bfloat16
            ),
        },
    )
    reader = IndexedTensorReader(root)
    assert reader.audit_report.shard_count == 1
    assert reader.audit_report.tensor_count == 3
    converted = reader.read_converted(weight_key, out_dtype=torch.float32)
    torch.testing.assert_close(converted, source.to(torch.float32) * 0.5)
    assert reader.source_groups_loaded == 1
    assert reader.max_source_group_bytes == source.nbytes + 4


@pytest.mark.parametrize(
    "mode",
    ["missing_shard", "missing_tensor", "extra_shard", "orphan_tensor", "orphan_fp8"],
)
def test_indexed_reader_fails_closed_on_incomplete_or_orphaned_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    _patch_preflight(monkeypatch)
    root = _small_checkpoint(
        tmp_path, {"model.language_model.layers.0.holdout.weight": torch.ones(2)}
    )
    shard = root / "model-00001-of-00001.safetensors"
    if mode == "missing_shard":
        shard.unlink()
        match = "missing"
    elif mode == "missing_tensor":
        index_path = root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["weight_map"]["model.language_model.layers.0.missing.weight"] = shard.name
        index_path.write_text(json.dumps(index), encoding="utf-8")
        match = "missing"
    elif mode == "extra_shard":
        save_file({"extra": torch.ones(1)}, root / "extra.safetensors")
        match = "extra"
    elif mode == "orphan_tensor":
        save_file(
            {
                "model.language_model.layers.0.holdout.weight": torch.ones(2),
                "orphan": torch.ones(1),
            },
            shard,
        )
        match = "orphan"
    else:
        save_file(
            {
                "model.language_model.layers.0.holdout.weight": torch.ones(
                    2, dtype=torch.float8_e4m3fn
                )
            },
            shard,
        )
        match = "orphan FP8"
    with pytest.raises(Glm53StreamingError, match=match):
        IndexedTensorReader(root)


def test_high_level_stream_audits_before_writing_and_records_source_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preflight(monkeypatch)
    source_key = "model.language_model.layers.0.holdout.weight"
    source = torch.arange(12, dtype=torch.float32).to(torch.bfloat16)
    root = _small_checkpoint(tmp_path, {source_key: source})
    output = tmp_path / "rank" / "tp0.safetensors"
    inventory = RankInventory(
        rank=0,
        tp_degree=2,
        tensors=(TensorSpec("layers.0.weight", torch.bfloat16, (6,)),),
    )

    def chunks(reader: IndexedTensorReader):
        converted = reader.read_converted(source_key)
        rank_slice = converted[:6]
        yield TensorChunk("layers.0.weight", 0, rank_slice[:3])
        yield TensorChunk("layers.0.weight", 3, rank_slice[3:])

    manifest = stream_rank_checkpoint(
        root, output, inventory, chunks, max_chunk_bytes=6
    )
    assert manifest["resource_bound"]["observed_max_chunk_bytes"] == 6
    assert manifest["resource_bound"]["source_max_group_bytes"] == source.nbytes
    assert manifest["source_audit"]["tensor_count"] == 1
    assert manifest["source_audit"]["payload_bytes_loaded_during_audit"] == 0
    with safe_open(output, framework="pt", device="cpu") as handle:
        torch.testing.assert_close(handle.get_tensor("layers.0.weight"), source[:6])


def test_high_level_stream_refuses_unproven_provenance_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wrong-revision"
    root.mkdir()

    def reject(_):
        raise Glm53ArchitectureMismatch("immutable provenance mismatch")

    monkeypatch.setattr(WRITER, "preflight_checkpoint_dir", reject)
    output = tmp_path / "tp0.safetensors"
    inventory = RankInventory(
        rank=0,
        tp_degree=1,
        tensors=(TensorSpec("x", torch.float32, (1,)),),
    )
    with pytest.raises(Glm53ArchitectureMismatch, match="provenance"):
        stream_rank_checkpoint(
            root,
            output,
            inventory,
            lambda _: [TensorChunk("x", 0, torch.ones(1))],
            max_chunk_bytes=4,
        )
    assert not output.exists()
    assert not list(tmp_path.glob("*.partial-*"))


def test_bounded_fp8_slice_uses_intersecting_reciprocal_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preflight(monkeypatch)
    key = "model.language_model.layers.0.mlp.gate_proj.weight"
    scale_key = f"{key}_scale_inv"
    source = torch.ones((260, 260), dtype=torch.float8_e4m3fn)
    scales = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=torch.float32,
    )
    root = _small_checkpoint(tmp_path, {key: source, scale_key: scales})
    reader = IndexedTensorReader(root)
    actual = reader.read_converted_slice(
        key, (slice(120, 140), slice(120, 140)), out_dtype=torch.float32
    )
    expected = torch.empty((20, 20), dtype=torch.float32)
    expected[:8, :8] = 1.0
    expected[:8, 8:] = 2.0
    expected[8:, :8] = 4.0
    expected[8:, 8:] = 5.0
    torch.testing.assert_close(actual, expected)
    assert reader.max_source_group_bytes < source.nbytes // 10


def test_rank_plan_is_deterministic_and_streams_shard_without_rank_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preflight(monkeypatch)
    key = "model.language_model.layers.0.holdout.weight"
    source = torch.arange(48, dtype=torch.float32).reshape(6, 8).to(torch.bfloat16)
    root = _small_checkpoint(tmp_path, {key: source})
    reader = IndexedTensorReader(root)
    operation = TargetTensorPlan(
        target=TensorSpec("layers.0.weight", torch.bfloat16, (6, 4)),
        kind="shard1",
        source_keys=(key,),
        source_shapes=((6, 8),),
    )
    inventory = RankInventory(rank=1, tp_degree=2, tensors=(operation.target,))
    plan = Glm53RankPlan(
        inventory=inventory, operations=(operation,), max_chunk_bytes=16
    )
    clone = Glm53RankPlan(
        inventory=inventory, operations=(operation,), max_chunk_bytes=16
    )
    assert plan.contract_sha256 == clone.contract_sha256
    chunks = list(plan.iter_chunks(reader))
    assert max(chunk.tensor.nbytes for chunk in chunks) <= 16
    actual = torch.cat([chunk.tensor.view(-1) for chunk in chunks]).reshape(6, 4)
    torch.testing.assert_close(actual, source[:, 4:])
    assert not any(isinstance(value, torch.Tensor) for value in plan.__dict__.values())


def test_rank_plan_rejects_source_shape_drift_before_yielding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preflight(monkeypatch)
    key = "model.language_model.layers.0.holdout.weight"
    root = _small_checkpoint(tmp_path, {key: torch.ones((2, 3))})
    reader = IndexedTensorReader(root)
    operation = TargetTensorPlan(
        target=TensorSpec("target", torch.float32, (2, 2)),
        kind="shard1",
        source_keys=(key,),
        source_shapes=((2, 4),),
    )
    plan = Glm53RankPlan(
        inventory=RankInventory(rank=0, tp_degree=2, tensors=(operation.target,)),
        operations=(operation,),
        max_chunk_bytes=16,
    )
    with pytest.raises(Glm53StreamingError, match="shape drift"):
        list(plan.iter_chunks(reader))


def test_specialized_lazy_plans_preserve_kda_and_expert_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preflight(monkeypatch)
    attn = "model.language_model.layers.0.self_attn."
    mlp = "model.language_model.layers.3.mlp."
    q = torch.arange(256, dtype=torch.float32).reshape(256, 1, 1).to(torch.bfloat16)
    k = (
        (torch.arange(256, dtype=torch.float32) + 1_000)
        .reshape(256, 1, 1)
        .to(torch.bfloat16)
    )
    v = (
        (torch.arange(256, dtype=torch.float32) + 2_000)
        .reshape(256, 1, 1)
        .to(torch.bfloat16)
    )
    tensors: dict[str, torch.Tensor] = {
        f"{attn}q_conv1d.weight": q,
        f"{attn}k_conv1d.weight": k,
        f"{attn}v_conv1d.weight": v,
    }
    for expert in range(2):
        base = torch.arange(12, dtype=torch.float32).reshape(4, 3) + expert * 100
        tensors[f"{mlp}experts.{expert}.gate_proj.weight"] = base.to(torch.bfloat16)
        tensors[f"{mlp}experts.{expert}.up_proj.weight"] = (base + 20).to(
            torch.bfloat16
        )
        tensors[f"{mlp}experts.{expert}.down_proj.weight"] = (
            torch.arange(12, dtype=torch.float32).reshape(3, 4) + expert * 200
        ).to(torch.bfloat16)
    root = _small_checkpoint(tmp_path, tensors)
    reader = IndexedTensorReader(root)
    conv = TargetTensorPlan(
        target=TensorSpec("conv", torch.bfloat16, (384, 1, 1)),
        kind="kda_conv",
        source_keys=tuple(
            f"{attn}{stream}_conv1d.weight" for stream in ("q", "k", "v")
        ),
        source_shapes=((256, 1, 1),) * 3,
    )
    gate_up_keys = tuple(
        f"{mlp}experts.{expert}.{projection}.weight"
        for expert in range(2)
        for projection in ("gate_proj", "up_proj")
    )
    gate_up = TargetTensorPlan(
        target=TensorSpec("gate_up", torch.bfloat16, (2, 3, 4)),
        kind="moe_gate_up",
        source_keys=gate_up_keys,
        source_shapes=((4, 3),) * 4,
    )
    down = TargetTensorPlan(
        target=TensorSpec("down", torch.bfloat16, (2, 2, 3)),
        kind="moe_down",
        source_keys=tuple(
            f"{mlp}experts.{expert}.down_proj.weight" for expert in range(2)
        ),
        source_shapes=((3, 4),) * 2,
    )
    plan = Glm53RankPlan(
        inventory=RankInventory(
            rank=1, tp_degree=2, tensors=(conv.target, gate_up.target, down.target)
        ),
        operations=(conv, gate_up, down),
        max_chunk_bytes=64,
    )
    emitted: dict[str, list[torch.Tensor]] = {}
    for chunk in plan.iter_chunks(reader):
        emitted.setdefault(chunk.tensor_name, []).append(chunk.tensor.view(-1))
        assert chunk.tensor.nbytes <= 64
    actual_conv = torch.cat(emitted["conv"]).reshape(conv.target.shape)
    expected_conv = torch.cat((q[128:], k[128:], v[128:]), dim=0).reshape(
        conv.target.shape
    )
    torch.testing.assert_close(actual_conv, expected_conv)
    actual_gate_up = torch.cat(emitted["gate_up"]).reshape(gate_up.target.shape)
    actual_down = torch.cat(emitted["down"]).reshape(down.target.shape)
    for expert in range(2):
        gate = tensors[f"{mlp}experts.{expert}.gate_proj.weight"]
        up = tensors[f"{mlp}experts.{expert}.up_proj.weight"]
        expected_gate_up = torch.cat((gate.t()[:, 2:4], up.t()[:, 2:4]), dim=1)
        torch.testing.assert_close(actual_gate_up[expert], expected_gate_up)
        expected_down = tensors[f"{mlp}experts.{expert}.down_proj.weight"].t()[2:4]
        torch.testing.assert_close(actual_down[expert], expected_down)


def test_actual_pinned_metadata_produces_stable_complete_rank_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_root = os.environ.get("GLM53_CHECKPOINT_METADATA_DIR")
    if not raw_root:
        pytest.skip("set GLM53_CHECKPOINT_METADATA_DIR for immutable metadata contract")
    root = Path(raw_root)
    plan = build_glm53_rank_plan(root, rank=0)
    repeat = build_glm53_rank_plan(root, rank=0)
    last_rank = build_glm53_rank_plan(root, rank=31)
    assert len(plan.operations) == 1_262
    assert len({op.target.name for op in plan.operations}) == 1_262
    assert plan.inventory.total_tensor_bytes == 19_859_704_056
    assert plan.inventory.contract_sha256 == (
        "0d7380ce03aeadb73d2b9dcb9a015c789a24a1a6220696717736c42cbe5d096a"
    )
    assert plan.contract_sha256 == (
        "b39c8e2e829048c06f80c8a5b223e4e670a83309da3cefe47aa5912854e274ea"
    )
    assert repeat.contract_sha256 == plan.contract_sha256
    assert last_rank.contract_sha256 != plan.contract_sha256

    adversarial = tmp_path / CONVERTER.GLM53_CHECKPOINT_REVISION
    adversarial.mkdir()
    shutil.copy2(root / "config.json", adversarial / "config.json")
    index_path = adversarial / "model.safetensors.index.json"
    index = json.loads((root / "model.safetensors.index.json").read_text())
    index["weight_map"]["model.language_model.unmapped.weight"] = (
        "model-00001-of-00063.safetensors"
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setattr(
        RANK_PLAN, "preflight_checkpoint_dir", lambda _: _source_report()
    )
    with pytest.raises(Glm53ArchitectureMismatch, match="unmapped"):
        build_glm53_rank_plan(adversarial, rank=0)
