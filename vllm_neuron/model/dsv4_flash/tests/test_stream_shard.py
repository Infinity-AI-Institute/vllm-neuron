# SPDX-License-Identifier: Apache-2.0
"""Offline invariants for the DSv4-Flash streaming sharder."""

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

ROOT = Path(__file__).parents[4]
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
_load_module(
    "vllm_neuron.model.dsv4_flash.checkpoint_convert",
    MODEL_ROOT / "dsv4_flash" / "checkpoint_convert.py",
)
stream_shard = _load_module(
    "vllm_neuron.model.dsv4_flash.stream_shard",
    MODEL_ROOT / "dsv4_flash" / "stream_shard.py",
)

_load_hf_index = stream_shard._load_hf_index
_row_shard = stream_shard._row_shard
_shard_expert_gate_up = stream_shard._shard_expert_gate_up
_wrapper_key = stream_shard._wrapper_key


def test_gate_up_shard_preserves_gate_then_up_order() -> None:
    tensor = torch.arange(2 * 3 * 8).reshape(2, 3, 8)
    rank0 = _shard_expert_gate_up(tensor, rank=0, tp_degree=2)
    rank1 = _shard_expert_gate_up(tensor, rank=1, tp_degree=2)
    expected0 = torch.cat((tensor[..., :2], tensor[..., 4:6]), dim=-1)
    expected1 = torch.cat((tensor[..., 2:4], tensor[..., 6:]), dim=-1)
    assert torch.equal(rank0, expected0)
    assert torch.equal(rank1, expected1)


def test_row_shard_rejects_ragged_axis() -> None:
    with pytest.raises(ValueError, match="cannot shard"):
        _row_shard(torch.zeros(5, 2), rank=0, tp_degree=2, dim=0)


def test_wrapper_key_inserts_mqa_module_only_for_direct_mqa_leaves() -> None:
    assert _wrapper_key("layers.0.attn.wq_a.weight") == "layers.0.attn.mqa.wq_a.weight"
    assert (
        _wrapper_key("layers.2.attn.compressor.wkv.weight")
        == "layers.2.attn.compressor.wkv.weight"
    )


def test_index_loader_requires_weight_map(tmp_path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty weight_map"):
        _load_hf_index(str(tmp_path))


def test_stream_shard_synthetic_33_shards_cpu_golden(tmp_path, monkeypatch) -> None:
    """Exercise the lazy index walk and TP slices over a 33-shard fixture.

    The converter is replaced with a deterministic CPU-golden function so the
    test validates sharder orchestration (all indexed shards opened, one layer
    fetched, rank slices written) without materialising a real DSv4 checkpoint.
    """
    model_path = tmp_path / "hf"
    model_path.mkdir()
    weight_map: dict[str, str] = {}
    for index in range(33):
        shard_name = f"model-{index + 1:05d}-of-00033.safetensors"
        if index == 0:
            key, tensor = "embed.weight", torch.arange(16).reshape(4, 4)
        elif index == 1:
            key, tensor = "norm.weight", torch.arange(4)
        elif index == 2:
            key, tensor = (
                "layers.0.attn.wq_a.weight",
                torch.arange(16).reshape(4, 4),
            )
        elif index == 3:
            key, tensor = (
                "hc_head_fn",
                torch.arange(64, dtype=torch.float32).reshape(4, 16),
            )
        elif index == 4:
            key, tensor = "hc_head_base", torch.arange(4, dtype=torch.float32)
        elif index == 5:
            key, tensor = "hc_head_scale", torch.ones(1, dtype=torch.float32)
        else:
            key, tensor = f"filler.{index}", torch.tensor([index])
        save_file({key: tensor}, str(model_path / shard_name))
        weight_map[key] = shard_name
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )

    def cpu_golden_parts(state, layer_idx, src, **kwargs):
        del layer_idx, src, kwargs
        # Return the exact CPU tensor; stream_shard applies the frozen TP slice.
        yield stream_shard._ConvertedTensorPart(
            "layers.0.attn.mqa.wq_a.weight",
            state["layers.0.attn.wq_a.weight"],
        )

    monkeypatch.setattr(stream_shard, "_iter_converted_layer_parts", cpu_golden_parts)
    src = SimpleNamespace(
        allow_reduced_shapes=True,
        num_hidden_layers=1,
        tie_word_embeddings=True,
        torch_dtype=torch.float32,
        hidden_size=4,
        hc_mult=4,
    )
    out = tmp_path / "compiled"
    report = stream_shard.stream_shard_dsv4_checkpoint(
        str(model_path),
        str(out),
        src,
        tp_degree=2,
        ranks=[0, 1],
        max_chunk_bytes=16,
        _test_only_allow_unpinned_source=True,
    )
    assert len(weight_map) == 33
    assert report["ranks_written"] == [0, 1]
    assert report["source_audit"] == {
        "shard_count": 33,
        "tensor_count": 33,
        "payload_bytes_loaded_during_audit": 0,
    }
    assert set(report["rank_inventory_sha256"]) == {"0", "1"}
    assert all(
        item["observed_max_chunk_bytes"] <= 16
        for item in report["rank_manifest"].values()
    )
    rank0 = load_file(str(out / "weights/tp0_sharded_checkpoint.safetensors"))
    rank1 = load_file(str(out / "weights/tp1_sharded_checkpoint.safetensors"))
    assert torch.equal(rank0["embed_tokens.weight"], torch.arange(16).reshape(4, 4)[:2])
    assert torch.equal(rank1["embed_tokens.weight"], torch.arange(16).reshape(4, 4)[2:])
    assert torch.equal(
        rank0["layers.0.attn.mqa.wq_a.weight"], torch.arange(16).reshape(4, 4)[:2]
    )
    assert torch.equal(
        rank1["layers.0.attn.mqa.wq_a.weight"], torch.arange(16).reshape(4, 4)[2:]
    )
    with safe_open(
        out / "weights/tp0_sharded_checkpoint.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        metadata = handle.metadata()
    assert metadata["model"] == "DeepSeek-V4-Flash-0731"
    assert metadata["revision"] == stream_shard.DSV4_CHECKPOINT_REVISION
    manifest = json.loads(
        (out / "weights/tp0_sharded_checkpoint.safetensors.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema"] == "dsv4-streaming-rank-v1"
    assert manifest["rank_inventory_sha256"] == report["rank_inventory_sha256"]["0"]


def test_stream_shard_rejects_orphan_tensor_before_output(
    tmp_path, monkeypatch
) -> None:
    model_path = tmp_path / "hf"
    model_path.mkdir()
    shard = model_path / "model-00001-of-00001.safetensors"
    save_file(
        {
            "embed.weight": torch.ones(4, 2),
            "norm.weight": torch.ones(2),
            "orphan.weight": torch.ones(1),
        },
        shard,
    )
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "embed.weight": shard.name,
                    "norm.weight": shard.name,
                }
            }
        ),
        encoding="utf-8",
    )
    src = SimpleNamespace(
        allow_reduced_shapes=True,
        num_hidden_layers=0,
        tie_word_embeddings=True,
        torch_dtype=torch.float32,
    )
    out = tmp_path / "compiled"
    with pytest.raises(ValueError, match="orphan"):
        stream_shard.stream_shard_dsv4_checkpoint(
            str(model_path),
            str(out),
            src,
            tp_degree=2,
            ranks=[0],
            _test_only_allow_unpinned_source=True,
        )
    assert not list(out.rglob("*.safetensors"))


def test_unpinned_source_bypass_is_test_fixture_only(tmp_path) -> None:
    model_path = tmp_path / "hf"
    model_path.mkdir()
    src = SimpleNamespace(
        allow_reduced_shapes=False,
        num_hidden_layers=43,
        tie_word_embeddings=True,
        torch_dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="reduced test fixtures"):
        stream_shard.stream_shard_dsv4_checkpoint(
            str(model_path),
            str(tmp_path / "out"),
            src,
            tp_degree=32,
            ranks=[0],
            _test_only_allow_unpinned_source=True,
        )
