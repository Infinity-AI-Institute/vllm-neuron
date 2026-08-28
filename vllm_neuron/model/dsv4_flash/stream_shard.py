# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash streaming TP sharder.

Safetensors files are mmap-opened; each layer is fetched lazily, converted,
TP-sliced, and released before the next layer.  This avoids materialising the
167 GB source or a 334 GB BF16 copy on rank 0.
"""

from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Mapping
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .checkpoint_convert import (
    _convert_csa_block,
    _convert_hash_moe_block,
    _convert_hca_block,
    _convert_routed_moe_layer,
    _convert_sliding_only_block,
)
from .config import DeepseekV4FlashInferenceConfig


class _MmapState(Mapping[str, torch.Tensor]):
    """Read-through mapping for the currently selected layer."""

    def __init__(self, weight_map: dict[str, str], handles: dict[str, Any], keys: set[str]) -> None:
        self._weight_map = weight_map
        self._handles = handles
        self._keys = keys
        self._cache: dict[str, torch.Tensor] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._keys:
            return default
        if key not in self._cache:
            shard = self._weight_map[key]
            self._cache[key] = self._handles[shard].get_tensor(key)
        return self._cache[key]


def _load_hf_index(model_path: str) -> dict[str, str]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"missing safetensors index at {index_path!r}")
    with open(index_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    weight_map = raw.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path!r} has no non-empty weight_map")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()):
        raise ValueError(f"{index_path!r} contains malformed weight_map entries")
    return weight_map


def _open_shards(model_path: str, shards: set[str]) -> dict[str, Any]:
    handles: dict[str, Any] = {}
    try:
        for shard in sorted(shards):
            path = os.path.join(model_path, shard)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"index points at missing shard {path!r}")
            handles[shard] = safe_open(path, framework="pt", device="cpu")
    except Exception:
        _close_shards(handles)
        raise
    return handles


def _close_shards(handles: dict[str, Any]) -> None:
    for handle in handles.values():
        try:
            handle.__exit__(None, None, None)
        except Exception:  # pragma: no cover
            pass


def _row_shard(tensor: torch.Tensor, rank: int, tp_degree: int, dim: int) -> torch.Tensor:
    size = int(tensor.shape[dim])
    if size % tp_degree:
        raise ValueError(
            f"cannot shard dimension {dim} of shape {tuple(tensor.shape)} across TP={tp_degree}"
        )
    step = size // tp_degree
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(rank * step, (rank + 1) * step)
    return tensor[tuple(slices)].contiguous()


def _shard_expert_gate_up(tensor: torch.Tensor, rank: int, tp_degree: int) -> torch.Tensor:
    """Shard NxDI ``[expert, hidden, gate|up]`` on the fused axis."""
    if tensor.ndim != 3 or tensor.shape[-1] % (2 * tp_degree):
        raise ValueError(
            "gate_up_proj must be [experts, hidden, 2*intermediate] with "
            f"an intermediate axis divisible by TP={tp_degree}; got {tuple(tensor.shape)}"
        )
    intermediate = tensor.shape[-1] // 2
    step = intermediate // tp_degree
    gate = tensor[..., rank * step : (rank + 1) * step]
    up = tensor[..., intermediate + rank * step : intermediate + (rank + 1) * step]
    return torch.cat((gate, up), dim=-1).contiguous()


def _target_shard(key: str, tensor: torch.Tensor, rank: int, tp_degree: int) -> torch.Tensor:
    """Apply the frozen DSv4 first-fire TP layout to one converted tensor."""
    if key in {"embed_tokens.weight", "lm_head.weight"}:
        return _row_shard(tensor, rank, tp_degree, 0)
    if key.endswith("router.weight") or key.endswith("tid2eid") or key.endswith("e_score_correction_bias"):
        return tensor.contiguous()
    if key.endswith("shared_expert.gate_proj.weight") or key.endswith("shared_expert.up_proj.weight"):
        return _row_shard(tensor, rank, tp_degree, 0)
    if key.endswith("shared_expert.down_proj.weight"):
        return _row_shard(tensor, rank, tp_degree, 1)
    if key.endswith("expert_mlps.mlp_op.gate_up_proj.weight"):
        return _shard_expert_gate_up(tensor, rank, tp_degree)
    if key.endswith("expert_mlps.mlp_op.down_proj.weight"):
        return _row_shard(tensor, rank, tp_degree, 1)
    # MQA projections are TP-sharded; compressor/indexer leaves are replicated
    # for the first fire (their local head axes are deliberately not sliced).
    if key.endswith("attn.wq_a.weight") or key.endswith("attn.wq_b.weight"):
        return _row_shard(tensor, rank, tp_degree, 0)
    if key.endswith("attn.wo_a.weight"):
        return _row_shard(tensor, rank, tp_degree, 0)
    if key.endswith("attn.wo_b.weight"):
        return _row_shard(tensor, rank, tp_degree, 1)
    return tensor.contiguous()


def _convert_one_layer(state: Mapping[str, torch.Tensor], layer_idx: int, src: DeepseekV4FlashInferenceConfig) -> dict[str, torch.Tensor]:
    layer_state = dict(state)
    layer_type = src.layer_types[layer_idx]
    if layer_type == "sliding_attention":
        converted = _convert_sliding_only_block(layer_state, layer_idx, src)
    elif layer_type == "compressed_sparse_attention":
        converted = _convert_csa_block(layer_state, layer_idx, src)
    elif layer_type == "heavily_compressed_attention":
        converted = _convert_hca_block(layer_state, layer_idx, src)
    else:  # pragma: no cover
        raise ValueError(f"unsupported attention layer type {layer_type!r} at {layer_idx}")

    ffn_norm_key = f"layers.{layer_idx}.ffn_norm.weight"
    ffn_norm = layer_state.get(ffn_norm_key)
    if ffn_norm is None:
        raise KeyError(f"missing {ffn_norm_key!r}")
    converted[ffn_norm_key] = ffn_norm.to(src.torch_dtype)
    mlp_type = src.mlp_layer_types[layer_idx]
    if mlp_type == "hash_moe":
        _convert_hash_moe_block(layer_state, converted, layer_idx, src)
    elif mlp_type == "moe":
        _convert_routed_moe_layer(layer_state, converted, layer_idx, src)
    else:  # pragma: no cover
        raise ValueError(f"unsupported MLP layer type {mlp_type!r} at {layer_idx}")
    return {key: value for key, value in converted.items() if not key.startswith("_")}


def stream_shard_dsv4_checkpoint(
    hf_model_path: str,
    compiled_model_path: str,
    src: DeepseekV4FlashInferenceConfig,
    *,
    tp_degree: int,
    ep_degree: int | None = None,
    ranks: list[int] | None = None,
) -> dict[str, Any]:
    """Stream-write per-rank safetensors from a DSv4-Flash HF snapshot."""
    if tp_degree <= 0:
        raise ValueError(f"tp_degree must be positive; got {tp_degree}")
    if ep_degree is None:
        ep_degree = tp_degree
    if ep_degree <= 0 or ep_degree > tp_degree:
        raise ValueError(f"ep_degree must be in [1, tp_degree]; got {ep_degree}")
    if not os.path.isdir(hf_model_path):
        raise FileNotFoundError(f"HF model path {hf_model_path!r} is not a directory")
    if not src.allow_reduced_shapes and src.num_hidden_layers != 43:
        raise ValueError("full DSv4-Flash sharding requires the frozen 43-layer config")
    ranks_iter = list(range(tp_degree)) if ranks is None else list(ranks)
    if not ranks_iter or any(rank < 0 or rank >= tp_degree for rank in ranks_iter):
        raise ValueError(f"ranks must be a non-empty subset of [0, {tp_degree})")
    if len(set(ranks_iter)) != len(ranks_iter):
        raise ValueError("ranks contains duplicates")

    os.makedirs(compiled_model_path, exist_ok=True)
    weights_dir = os.path.join(compiled_model_path, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weight_map = _load_hf_index(hf_model_path)
    required = {"embed_tokens.weight", "norm.weight"}
    if not src.tie_word_embeddings:
        required.add("lm_head.weight")
    missing = sorted(required - weight_map.keys())
    if missing:
        raise KeyError(f"HF index is missing required top-level keys: {missing}")

    handles = _open_shards(hf_model_path, set(weight_map.values()))
    report: dict[str, Any] = {
        "model_path": hf_model_path,
        "compiled_model_path": compiled_model_path,
        "tp_degree": tp_degree,
        "ep_degree": ep_degree,
        "ranks_requested": ranks_iter,
        "ranks_written": [],
        "rank_bytes": {},
        "rank_wall_s": {},
        "layers": src.num_hidden_layers,
        "peak_layer_key_count": 0,
        "start_unix": int(time.time()),
    }
    try:
        top_state = _MmapState(weight_map, handles, {key for key in weight_map if not key.startswith("layers.")})
        embed = top_state.get("embed_tokens.weight")
        final_norm = top_state.get("norm.weight")
        lm_head = top_state.get("lm_head.weight")
        if embed is None or final_norm is None:
            raise KeyError("embed_tokens.weight and norm.weight are required")
        if lm_head is None:
            if not src.tie_word_embeddings:
                raise KeyError("lm_head.weight is required when tie_word_embeddings=False")
            lm_head = embed

        for rank in ranks_iter:
            started = time.time()
            rank_dict: dict[str, torch.Tensor] = {
                "embed_tokens.weight": _row_shard(embed.to(src.torch_dtype), rank, tp_degree, 0),
                "final_norm_weight": final_norm.to(src.torch_dtype).contiguous(),
                "lm_head.weight": _row_shard(lm_head.to(src.torch_dtype), rank, tp_degree, 0),
            }
            for layer_idx in range(src.num_hidden_layers):
                layer_keys = {key for key in weight_map if key.startswith(f"layers.{layer_idx}.")}
                state = _MmapState(weight_map, handles, layer_keys)
                converted = _convert_one_layer(state, layer_idx, src)
                report["peak_layer_key_count"] = max(report["peak_layer_key_count"], len(converted))
                rank_dict.update({key: _target_shard(key, value, rank, tp_degree) for key, value in converted.items()})
                del converted, state
                gc.collect()
            output_path = os.path.join(weights_dir, f"tp{rank}_sharded_checkpoint.safetensors")
            save_file(rank_dict, output_path)
            report["ranks_written"].append(rank)
            report["rank_bytes"][str(rank)] = os.path.getsize(output_path)
            report["rank_wall_s"][str(rank)] = round(time.time() - started, 2)
            del rank_dict
            gc.collect()
    finally:
        _close_shards(handles)
    report["end_unix"] = int(time.time())
    report["total_wall_s"] = report["end_unix"] - report["start_unix"]
    return report


__all__ = ["stream_shard_dsv4_checkpoint"]
