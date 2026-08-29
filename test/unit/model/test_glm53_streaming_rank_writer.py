from __future__ import annotations

import importlib.util
import json
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

Glm53ArchitectureMismatch = CONVERTER.Glm53ArchitectureMismatch
Glm53CheckpointReport = CONVERTER.Glm53CheckpointReport
Glm53StreamingError = WRITER.Glm53StreamingError
IndexedTensorReader = WRITER.IndexedTensorReader
RankInventory = WRITER.RankInventory
StreamingRankWriter = WRITER.StreamingRankWriter
TensorChunk = WRITER.TensorChunk
TensorSpec = WRITER.TensorSpec
stream_rank_checkpoint = WRITER.stream_rank_checkpoint


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
