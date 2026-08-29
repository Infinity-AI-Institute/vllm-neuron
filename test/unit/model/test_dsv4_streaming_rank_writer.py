from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).parents[3]
MODEL_ROOT = ROOT / "vllm_neuron" / "model"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for package_name, package_path in (
    ("vllm_neuron", ROOT / "vllm_neuron"),
    ("vllm_neuron.model", MODEL_ROOT),
    ("vllm_neuron.model.glm53_flash", MODEL_ROOT / "glm53_flash"),
    ("vllm_neuron.model.dsv4_flash", MODEL_ROOT / "dsv4_flash"),
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules.setdefault(package_name, package)

_load_module(
    "vllm_neuron.model.glm53_flash.checkpoint_converter",
    MODEL_ROOT / "glm53_flash" / "checkpoint_converter.py",
)
_load_module(
    "vllm_neuron.model.glm53_flash.streaming_rank_writer",
    MODEL_ROOT / "glm53_flash" / "streaming_rank_writer.py",
)
_load_module(
    "vllm_neuron.model.dsv4_flash.config",
    MODEL_ROOT / "dsv4_flash" / "config.py",
)
CONVERTER = _load_module(
    "vllm_neuron.model.dsv4_flash.checkpoint_convert",
    MODEL_ROOT / "dsv4_flash" / "checkpoint_convert.py",
)
SHARDER = _load_module(
    "vllm_neuron.model.dsv4_flash.stream_shard",
    MODEL_ROOT / "dsv4_flash" / "stream_shard.py",
)


def _fixture(tmp_path: Path, *, orphan: bool = False) -> Path:
    root = tmp_path / "hf"
    root.mkdir()
    shard = root / "model-00001-of-00001.safetensors"
    tensors = {
        "embed.weight": torch.arange(16).reshape(4, 4),
        "norm.weight": torch.arange(4),
        "hc_head_fn": torch.arange(64, dtype=torch.float32).reshape(4, 16),
        "hc_head_base": torch.arange(4, dtype=torch.float32),
        "hc_head_scale": torch.ones(1, dtype=torch.float32),
        "layers.0.attn.wq_a.weight": torch.arange(16).reshape(4, 4),
    }
    if orphan:
        tensors["orphan.weight"] = torch.ones(1)
    save_file(tensors, shard)
    weight_map = {
        key: shard.name for key in tensors if not (orphan and key == "orphan.weight")
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    return root


def _source() -> SimpleNamespace:
    return SimpleNamespace(
        allow_reduced_shapes=True,
        num_hidden_layers=1,
        tie_word_embeddings=True,
        torch_dtype=torch.int64,
        hidden_size=4,
        hc_mult=4,
    )


def test_two_pass_rank_inventory_and_transactional_bounded_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture(tmp_path)

    def convert(state, layer_idx, src, **kwargs):
        del layer_idx, src, kwargs
        return {"layers.0.attn.mqa.wq_a.weight": state["layers.0.attn.wq_a.weight"]}

    monkeypatch.setattr(SHARDER, "_convert_one_layer", convert)
    out = tmp_path / "compiled"
    report = SHARDER.stream_shard_dsv4_checkpoint(
        str(root),
        str(out),
        _source(),
        tp_degree=2,
        ranks=[0, 1],
        max_chunk_bytes=16,
        _test_only_allow_unpinned_source=True,
    )
    assert report["ranks_written"] == [0, 1]
    assert report["source_audit"] == {
        "shard_count": 1,
        "tensor_count": 6,
        "payload_bytes_loaded_during_audit": 0,
    }
    assert set(report["rank_inventory_sha256"]) == {"0", "1"}
    assert all(
        item["observed_max_chunk_bytes"] <= 16
        for item in report["rank_manifest"].values()
    )
    rank0_path = out / "weights" / "tp0_sharded_checkpoint.safetensors"
    rank0 = load_file(rank0_path)
    assert torch.equal(rank0["embed_tokens.weight"], torch.arange(16).reshape(4, 4)[:2])
    assert rank0["hc_head_fn"].dtype == torch.float32
    assert torch.equal(
        rank0["hc_head_fn"], torch.arange(64, dtype=torch.float32).reshape(4, 16)
    )
    rank1 = load_file(out / "weights" / "tp1_sharded_checkpoint.safetensors")
    torch.testing.assert_close(rank1["hc_head_fn"], rank0["hc_head_fn"], rtol=0, atol=0)
    with safe_open(rank0_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    assert metadata["model"] == "DeepSeek-V4-Flash-0731"
    assert metadata["revision"] == SHARDER.DSV4_CHECKPOINT_REVISION
    manifest = json.loads(
        rank0_path.with_suffix(".safetensors.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "dsv4-streaming-rank-v1"
    assert manifest["rank_inventory_sha256"] == report["rank_inventory_sha256"]["0"]
    assert not list((out / "weights").glob("*.partial-*"))


def test_header_audit_rejects_orphan_before_publication(tmp_path: Path) -> None:
    root = _fixture(tmp_path, orphan=True)
    out = tmp_path / "compiled"
    with pytest.raises(ValueError, match="orphan"):
        SHARDER.stream_shard_dsv4_checkpoint(
            str(root),
            str(out),
            _source(),
            tp_degree=2,
            ranks=[0],
            _test_only_allow_unpinned_source=True,
        )
    assert not list(out.rglob("*.safetensors"))


def test_unpinned_source_bypass_is_reduced_fixture_only(tmp_path: Path) -> None:
    root = tmp_path / "hf"
    root.mkdir()
    source = _source()
    source.allow_reduced_shapes = False
    source.num_hidden_layers = 43
    with pytest.raises(ValueError, match="reduced test fixtures"):
        SHARDER.stream_shard_dsv4_checkpoint(
            str(root),
            str(tmp_path / "out"),
            source,
            tp_degree=32,
            ranks=[0],
            _test_only_allow_unpinned_source=True,
        )


@pytest.mark.parametrize(
    ("mlp_type", "converter_name"),
    [
        ("hash_moe", "_convert_hash_moe_block"),
        ("moe", "_convert_routed_moe_layer"),
    ],
)
def test_chunked_experts_emit_static_layer_state_once(
    monkeypatch: pytest.MonkeyPatch, mlp_type: str, converter_name: str
) -> None:
    calls: list[bool] = []
    expert_key = "layers.0.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
    down_key = "layers.0.mlp.expert_mlps.mlp_op.down_proj.weight"

    def convert(state, converted, layer_idx, src, *, expert_indices, include_static):
        del state, layer_idx, src
        calls.append(include_static)
        count = len(expert_indices)
        converted[expert_key] = torch.ones(count, 2, 2)
        converted[down_key] = torch.ones(count, 2, 2)
        if include_static:
            converted["layers.0.mlp.router.weight"] = torch.ones(2, 2)
            converted["_conversion_report"] = {"first_chunk": True}

    monkeypatch.setattr(SHARDER, converter_name, convert)
    monkeypatch.setattr(SHARDER, "_convert_csa_block", lambda *args: {})
    monkeypatch.setattr(SHARDER, "_convert_mhc_layer", lambda *args: {})
    source = SimpleNamespace(
        layer_types=["compressed_sparse_attention"],
        mlp_layer_types=[mlp_type],
        n_routed_experts=4,
        torch_dtype=torch.float32,
    )
    converted = SHARDER._convert_one_layer(
        {"layers.0.ffn_norm.weight": torch.ones(2)},
        0,
        source,
        expert_chunk_size=2,
    )
    assert calls == [True, False]
    assert converted[expert_key].shape == (4, 2, 2)
    assert converted[down_key].shape == (4, 2, 2)
    assert torch.equal(converted["layers.0.mlp.router.weight"], torch.ones(2, 2))
    assert all(not key.startswith("_") for key in converted)


def test_chunked_experts_still_reject_unexpected_later_chunk_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert_key = "layers.0.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
    down_key = "layers.0.mlp.expert_mlps.mlp_op.down_proj.weight"

    def buggy_convert(
        state, converted, layer_idx, src, *, expert_indices, include_static
    ):
        del state, layer_idx, src, include_static
        count = len(expert_indices)
        converted[expert_key] = torch.ones(count, 2, 2)
        converted[down_key] = torch.ones(count, 2, 2)
        converted["layers.0.mlp.router.weight"] = torch.ones(2, 2)

    monkeypatch.setattr(SHARDER, "_convert_hash_moe_block", buggy_convert)
    monkeypatch.setattr(SHARDER, "_convert_csa_block", lambda *args: {})
    monkeypatch.setattr(SHARDER, "_convert_mhc_layer", lambda *args: {})
    source = SimpleNamespace(
        layer_types=["compressed_sparse_attention"],
        mlp_layer_types=["hash_moe"],
        n_routed_experts=4,
        torch_dtype=torch.float32,
    )
    with pytest.raises(RuntimeError, match="unexpected duplicate keys"):
        SHARDER._convert_one_layer(
            {"layers.0.ffn_norm.weight": torch.ones(2)},
            0,
            source,
            expert_chunk_size=2,
        )


@pytest.mark.parametrize(
    ("mlp_type", "converter_name"),
    [
        ("hash_moe", "_convert_hash_moe_block"),
        ("moe", "_convert_routed_moe_layer"),
    ],
)
def test_later_chunk_converter_skips_all_static_outputs(
    monkeypatch: pytest.MonkeyPatch, mlp_type: str, converter_name: str
) -> None:
    def dequant(state, key, dtype):
        del state
        if ".w2." in key:
            return torch.ones(2, 1, dtype=dtype)
        return torch.ones(1, 2, dtype=dtype)

    monkeypatch.setattr(CONVERTER, "_dequant_expert_fp4_weight", dequant)
    source = SimpleNamespace(
        mlp_layer_types=[mlp_type],
        torch_dtype=torch.float32,
        hidden_size=2,
        moe_intermediate_size=1,
        n_routed_experts=2,
        num_experts_per_tok=1,
        vocab_size=4,
        quantization_config=SimpleNamespace(weight_block_size=(1, 1)),
    )
    converted = {}
    getattr(CONVERTER, converter_name)(
        {},
        converted,
        0,
        source,
        expert_indices=[0],
        include_static=False,
    )
    assert set(converted) == {
        "layers.0.mlp.expert_mlps.mlp_op.gate_up_proj.weight",
        "layers.0.mlp.expert_mlps.mlp_op.down_proj.weight",
    }
