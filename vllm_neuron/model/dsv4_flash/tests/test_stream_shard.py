# SPDX-License-Identifier: Apache-2.0
"""Offline invariants for the DSv4-Flash streaming sharder."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from vllm_neuron.model.dsv4_flash import stream_shard
from vllm_neuron.model.dsv4_flash.stream_shard import (
    _load_hf_index,
    _row_shard,
    _shard_expert_gate_up,
    _wrapper_key,
)


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
        else:
            key, tensor = f"filler.{index}", torch.tensor([index])
        save_file({key: tensor}, str(model_path / shard_name))
        weight_map[key] = shard_name
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )

    def cpu_golden(state, layer_idx, src, **kwargs):
        del layer_idx, src, kwargs
        # Return the exact CPU tensor; stream_shard applies the frozen TP slice.
        return {"layers.0.attn.mqa.wq_a.weight": state["layers.0.attn.wq_a.weight"]}

    monkeypatch.setattr(stream_shard, "_convert_one_layer", cpu_golden)
    src = SimpleNamespace(
        allow_reduced_shapes=True,
        num_hidden_layers=1,
        tie_word_embeddings=True,
        torch_dtype=torch.float32,
    )
    out = tmp_path / "compiled"
    report = stream_shard.stream_shard_dsv4_checkpoint(
        str(model_path), str(out), src, tp_degree=2, ranks=[0, 1]
    )
    assert len(weight_map) == 33
    assert report["ranks_written"] == [0, 1]
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
