# SPDX-License-Identifier: Apache-2.0
"""Offline invariants for the DSv4-Flash streaming sharder."""

from __future__ import annotations

import json

import pytest
import torch

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
    assert (
        _wrapper_key("layers.0.attn.wq_a.weight")
        == "layers.0.attn.mqa.wq_a.weight"
    )
    assert (
        _wrapper_key("layers.2.attn.compressor.wkv.weight")
        == "layers.2.attn.compressor.wkv.weight"
    )


def test_index_loader_requires_weight_map(tmp_path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty weight_map"):
        _load_hf_index(str(tmp_path))
