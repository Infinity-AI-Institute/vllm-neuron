# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash streaming TP sharder.

Safetensors files are mmap-opened; each layer is fetched lazily, converted,
TP-sliced, and released before the next layer.  This avoids materialising the
167 GB source or a 334 GB BF16 copy on rank 0.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from vllm_neuron.model.glm53_flash.streaming_rank_writer import (
    RankInventory,
    StreamingRankWriter,
    TensorChunk,
    TensorSpec,
)

from .checkpoint_convert import (
    _convert_csa_block,
    _convert_hash_moe_block,
    _convert_hca_block,
    _convert_mhc_head,
    _convert_mhc_layer,
    _convert_routed_moe_layer,
    _convert_sliding_only_block,
)
from .config import DeepseekV4FlashInferenceConfig

DSV4_CHECKPOINT_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DSV4_CONFIG_SHA256 = "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
DSV4_INDEX_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
DSV4_SHARD_COUNT = 48
DEFAULT_MAX_CHUNK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class Dsv4SourceReport:
    shard_count: int
    tensor_count: int
    payload_bytes_loaded_during_audit: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "shard_count": self.shard_count,
            "tensor_count": self.tensor_count,
            "payload_bytes_loaded_during_audit": self.payload_bytes_loaded_during_audit,
        }


@dataclass(frozen=True)
class _ConvertedTensorPart:
    """One bounded piece of a converted target tensor.

    Expert tensors are emitted a source expert-chunk at a time. ``expert_start``
    locates that chunk in the final tensor; non-expert tensors are complete.
    """

    key: str
    tensor: torch.Tensor
    expert_start: int | None = None
    full_experts: int | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _MmapState(Mapping[str, torch.Tensor]):
    """Read-through mapping for the currently selected layer."""

    def __init__(
        self, weight_map: dict[str, str], handles: dict[str, Any], keys: set[str]
    ) -> None:
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
    if not all(
        isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()
    ):
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
        except Exception:  # pragma: no cover  # noqa: BLE001,S110
            pass


def _preflight_and_audit_source(
    model_path: str,
    weight_map: Mapping[str, str],
    handles: Mapping[str, Any],
    *,
    test_only_allow_unpinned_source: bool,
) -> Dsv4SourceReport:
    """Bind provenance and compare every index route with actual shard headers."""
    root = Path(model_path).resolve(strict=True)
    if test_only_allow_unpinned_source:
        if root.name == DSV4_CHECKPOINT_REVISION:
            raise ValueError(
                "test-only provenance bypass is forbidden for the pinned snapshot"
            )
    else:
        if root.name != DSV4_CHECKPOINT_REVISION:
            raise ValueError(
                "checkpoint directory must be the exact pinned revision "
                f"{DSV4_CHECKPOINT_REVISION}; got {root.name!r}"
            )
        for filename, expected in (
            ("config.json", DSV4_CONFIG_SHA256),
            ("model.safetensors.index.json", DSV4_INDEX_SHA256),
        ):
            actual = _sha256_file(root / filename)
            if actual != expected:
                raise ValueError(
                    f"immutable {filename} SHA-256 mismatch: expected {expected}, got {actual}"
                )

    expected_by_shard: dict[str, set[str]] = {}
    for key, shard_name in weight_map.items():
        shard_path = Path(shard_name)
        if shard_path.is_absolute() or shard_path.name != shard_name:
            raise ValueError(f"unsafe shard path in index: {shard_name!r}")
        expected_by_shard.setdefault(shard_name, set()).add(key)
    actual_files = {path.name for path in root.glob("*.safetensors")}
    referenced = set(expected_by_shard)
    if actual_files != referenced:
        raise ValueError(
            "source shard inventory mismatch: "
            f"missing={sorted(referenced - actual_files)[:4]} "
            f"extra={sorted(actual_files - referenced)[:4]}"
        )
    if not test_only_allow_unpinned_source and len(referenced) != DSV4_SHARD_COUNT:
        raise ValueError(
            f"pinned DSv4 snapshot requires {DSV4_SHARD_COUNT} shards; got {len(referenced)}"
        )

    seen: set[str] = set()
    for shard_name in sorted(referenced):
        actual_keys = set(handles[shard_name].keys())
        expected_keys = expected_by_shard[shard_name]
        if actual_keys != expected_keys:
            raise ValueError(
                f"tensor routing mismatch in {shard_name}: "
                f"missing={sorted(expected_keys - actual_keys)[:4]} "
                f"orphan={sorted(actual_keys - expected_keys)[:4]}"
            )
        duplicate = seen & actual_keys
        if duplicate:
            raise ValueError(
                f"tensor appears in multiple shards: {sorted(duplicate)[:4]}"
            )
        seen.update(actual_keys)
    if seen != set(weight_map):
        raise ValueError(
            "audited shard headers do not exactly equal the immutable index"
        )
    return Dsv4SourceReport(
        shard_count=len(referenced),
        tensor_count=len(seen),
    )


def _row_shard(
    tensor: torch.Tensor, rank: int, tp_degree: int, dim: int
) -> torch.Tensor:
    size = int(tensor.shape[dim])
    if size % tp_degree:
        raise ValueError(
            f"cannot shard dimension {dim} of shape {tuple(tensor.shape)} across TP={tp_degree}"
        )
    step = size // tp_degree
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(rank * step, (rank + 1) * step)
    return tensor[tuple(slices)].contiguous()


def _shard_expert_gate_up(
    tensor: torch.Tensor, rank: int, tp_degree: int
) -> torch.Tensor:
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


def _target_shard(
    key: str, tensor: torch.Tensor, rank: int, tp_degree: int
) -> torch.Tensor:
    """Apply the frozen DSv4 first-fire TP layout to one converted tensor."""
    if key in {"embed_tokens.weight", "lm_head.weight"}:
        return _row_shard(tensor, rank, tp_degree, 0)
    if key.endswith(("router.weight", "tid2eid", "e_score_correction_bias")):
        return tensor.contiguous()
    if key.endswith(("shared_expert.gate_proj.weight", "shared_expert.up_proj.weight")):
        return _row_shard(tensor, rank, tp_degree, 0)
    if key.endswith("shared_expert.down_proj.weight"):
        return _row_shard(tensor, rank, tp_degree, 1)
    if key.endswith("expert_mlps.mlp_op.gate_up_proj.weight"):
        return _shard_expert_gate_up(tensor, rank, tp_degree)
    if key.endswith("expert_mlps.mlp_op.down_proj.weight"):
        return _row_shard(tensor, rank, tp_degree, 1)
    # MQA projections are TP-sharded; compressor/indexer leaves are replicated
    # for the first fire (their local head axes are deliberately not sliced).
    if key.endswith(
        (
            "attn.wq_a.weight",
            "attn.wq_b.weight",
            "attn.mqa.wq_a.weight",
            "attn.mqa.wq_b.weight",
        )
    ):
        return _row_shard(tensor, rank, tp_degree, 0)
    if key.endswith(("attn.wo_a.weight", "attn.mqa.wo_a.weight")):
        return _row_shard(tensor, rank, tp_degree, 0)
    if key.endswith(("attn.wo_b.weight", "attn.mqa.wo_b.weight")):
        return _row_shard(tensor, rank, tp_degree, 1)
    return tensor.contiguous()


def _wrapper_key(key: str) -> str:
    """Normalize converter MQA leaves to the Round-8 module tree.

    The per-block converter intentionally retains the HF-like spelling
    ``layers.i.attn.wq_a.weight`` for its standalone tests.  The NxDI wrapper
    composes that block under ``attn.mqa``; emitted checkpoint keys therefore
    need the one-level insertion below.  Compressor and indexer leaves are
    already nested and must remain untouched.
    """
    mqa_names = {
        "wq_a.weight",
        "wq_b.weight",
        "wkv.weight",
        "wo_a.weight",
        "wo_b.weight",
        "q_norm.weight",
        "kv_norm.weight",
        "attn_sink",
    }
    marker = ".attn."
    if marker not in key:
        return key
    prefix, leaf = key.split(marker, 1)
    if leaf in mqa_names:
        return f"{prefix}.attn.mqa.{leaf}"
    return key


def _iter_converted_layer_parts(
    state: Mapping[str, torch.Tensor],
    layer_idx: int,
    src: DeepseekV4FlashInferenceConfig,
    *,
    expert_chunk_size: int = 8,
) -> Iterator[_ConvertedTensorPart]:
    """Yield one converted layer without concatenating its expert chunks."""
    layer_state = dict(state)
    layer_type = src.layer_types[layer_idx]
    if layer_type == "sliding_attention":
        converted = _convert_sliding_only_block(layer_state, layer_idx, src)
    elif layer_type == "compressed_sparse_attention":
        converted = _convert_csa_block(layer_state, layer_idx, src)
    elif layer_type == "heavily_compressed_attention":
        converted = _convert_hca_block(layer_state, layer_idx, src)
    else:  # pragma: no cover
        raise ValueError(
            f"unsupported attention layer type {layer_type!r} at {layer_idx}"
        )

    ffn_norm_key = f"layers.{layer_idx}.ffn_norm.weight"
    ffn_norm = layer_state.get(ffn_norm_key)
    if ffn_norm is None:
        raise KeyError(f"missing {ffn_norm_key!r}")
    converted[ffn_norm_key] = ffn_norm.to(src.torch_dtype)
    converted.update(_convert_mhc_layer(layer_state, layer_idx, src))
    mlp_type = src.mlp_layer_types[layer_idx]
    if mlp_type not in {"hash_moe", "moe"}:
        raise ValueError(f"unsupported MLP layer type {mlp_type!r} at {layer_idx}")
    if expert_chunk_size <= 0:
        raise ValueError(f"expert_chunk_size must be positive; got {expert_chunk_size}")

    for key, value in converted.items():
        if not key.startswith("_"):
            yield _ConvertedTensorPart(_wrapper_key(key), value)

    # Dequantize only a small expert group at a time, and expose each piece to
    # the layer-major fan-out before converting the next piece. This bounds
    # peak host memory independently of TP degree and requested rank count.
    expert_key = f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
    down_key = f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.weight"
    for start in range(0, src.n_routed_experts, expert_chunk_size):
        indices = list(
            range(start, min(start + expert_chunk_size, src.n_routed_experts))
        )
        partial: dict[str, Any] = {}
        if mlp_type == "hash_moe":
            _convert_hash_moe_block(
                layer_state,
                partial,
                layer_idx,
                src,
                expert_indices=indices,
                include_static=start == 0,
            )
        else:
            _convert_routed_moe_layer(
                layer_state,
                partial,
                layer_idx,
                src,
                expert_indices=indices,
                include_static=start == 0,
            )
        gate_chunk = partial.pop(expert_key)
        down_chunk = partial.pop(down_key)
        if start == 0:
            for key, value in partial.items():
                if not key.startswith("_"):
                    yield _ConvertedTensorPart(_wrapper_key(key), value)
        elif partial:
            raise RuntimeError(
                "chunked DSv4 expert conversion emitted unexpected duplicate "
                f"keys at layer {layer_idx}: {sorted(partial)}"
            )
        yield _ConvertedTensorPart(
            _wrapper_key(expert_key),
            gate_chunk,
            expert_start=start,
            full_experts=src.n_routed_experts,
        )
        yield _ConvertedTensorPart(
            _wrapper_key(down_key),
            down_chunk,
            expert_start=start,
            full_experts=src.n_routed_experts,
        )
        del partial
        gc.collect()


def _convert_one_layer(
    state: Mapping[str, torch.Tensor],
    layer_idx: int,
    src: DeepseekV4FlashInferenceConfig,
    *,
    expert_chunk_size: int = 8,
    rank: int | None = None,
    tp_degree: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compatibility collector for tests and the legacy rank iterator."""
    if (rank is None) != (tp_degree is None):
        raise ValueError("rank and tp_degree must be provided together")
    converted: dict[str, torch.Tensor] = {}
    expert_chunks: dict[str, list[torch.Tensor]] = {}
    for part in _iter_converted_layer_parts(
        state, layer_idx, src, expert_chunk_size=expert_chunk_size
    ):
        tensor = part.tensor
        # Preserve the legacy contract: expert chunks are TP-sliced before
        # concatenation to bound memory; the caller shards all other tensors.
        if part.expert_start is not None and rank is not None and tp_degree is not None:
            tensor = _target_shard(part.key, tensor, rank, tp_degree)
        if part.expert_start is None:
            if part.key in converted:
                raise RuntimeError(f"duplicate converted layer key: {part.key}")
            converted[part.key] = tensor
        else:
            expert_chunks.setdefault(part.key, []).append(tensor)
    for key, chunks in expert_chunks.items():
        converted[key] = torch.cat(chunks, dim=0).contiguous()
    return converted


def _iter_rank_tensors(
    weight_map: dict[str, str],
    handles: dict[str, Any],
    src: DeepseekV4FlashInferenceConfig,
    *,
    rank: int,
    tp_degree: int,
):
    top_state = _MmapState(
        weight_map,
        handles,
        {key for key in weight_map if not key.startswith("layers.")},
    )
    embed = top_state.get("embed.weight")
    final_norm = top_state.get("norm.weight")
    lm_head = top_state.get("head.weight")
    if embed is None or final_norm is None:
        raise KeyError("embed.weight and norm.weight are required")
    if lm_head is None:
        if not src.tie_word_embeddings:
            raise KeyError("head.weight is required when tie_word_embeddings=False")
        lm_head = embed
    yield (
        "embed_tokens.weight",
        _row_shard(embed.to(src.torch_dtype), rank, tp_degree, 0),
    )
    yield "final_norm_weight", final_norm.to(src.torch_dtype).contiguous()
    yield "lm_head.weight", _row_shard(lm_head.to(src.torch_dtype), rank, tp_degree, 0)
    for key, value in _convert_mhc_head(top_state, src).items():
        yield key, value

    for layer_idx in range(src.num_hidden_layers):
        layer_keys = {
            key for key in weight_map if key.startswith(f"layers.{layer_idx}.")
        }
        state = _MmapState(weight_map, handles, layer_keys)
        converted = _convert_one_layer(
            state,
            layer_idx,
            src,
            rank=rank,
            tp_degree=tp_degree,
        )
        for key, value in converted.items():
            if not key.endswith(
                (
                    "expert_mlps.mlp_op.gate_up_proj.weight",
                    "expert_mlps.mlp_op.down_proj.weight",
                )
            ):
                value = _target_shard(key, value, rank, tp_degree)
            yield key, value
        del converted, state
        gc.collect()


def _rank_inventory(
    weight_map: dict[str, str],
    handles: dict[str, Any],
    src: DeepseekV4FlashInferenceConfig,
    *,
    rank: int,
    tp_degree: int,
) -> RankInventory:
    specs: list[TensorSpec] = []
    for key, tensor in _iter_rank_tensors(
        weight_map, handles, src, rank=rank, tp_degree=tp_degree
    ):
        specs.append(TensorSpec(key, tensor.dtype, tuple(tensor.shape)))
        del tensor
        gc.collect()
    return RankInventory(rank=rank, tp_degree=tp_degree, tensors=tuple(specs))


def _walk_layer_major_rank_tensors(
    weight_map: dict[str, str],
    handles: dict[str, Any],
    src: DeepseekV4FlashInferenceConfig,
    *,
    ranks: list[int],
    tp_degree: int,
    consume: Callable[[int, str, torch.Tensor, int, tuple[int, ...]], None],
) -> int:
    """Convert each source layer once, then fan each bounded piece to ranks."""
    top_state = _MmapState(
        weight_map,
        handles,
        {key for key in weight_map if not key.startswith("layers.")},
    )
    embed = top_state.get("embed.weight")
    final_norm = top_state.get("norm.weight")
    lm_head = top_state.get("head.weight")
    if embed is None or final_norm is None:
        raise KeyError("embed.weight and norm.weight are required")
    if lm_head is None:
        if not src.tie_word_embeddings:
            raise KeyError("head.weight is required when tie_word_embeddings=False")
        lm_head = embed
    top_tensors = {
        "embed_tokens.weight": embed.to(src.torch_dtype),
        "final_norm_weight": final_norm.to(src.torch_dtype).contiguous(),
        "lm_head.weight": lm_head.to(src.torch_dtype),
        **_convert_mhc_head(top_state, src),
    }
    for key, tensor in top_tensors.items():
        for rank in ranks:
            rank_tensor = _target_shard(key, tensor, rank, tp_degree)
            consume(rank, key, rank_tensor, 0, tuple(rank_tensor.shape))
            del rank_tensor
        del tensor
    del top_tensors, top_state
    gc.collect()

    layer_visits = 0
    for layer_idx in range(src.num_hidden_layers):
        layer_keys = {
            key for key in weight_map if key.startswith(f"layers.{layer_idx}.")
        }
        state = _MmapState(weight_map, handles, layer_keys)
        layer_visits += 1
        for part in _iter_converted_layer_parts(state, layer_idx, src):
            for rank in ranks:
                rank_tensor = _target_shard(part.key, part.tensor, rank, tp_degree)
                if part.expert_start is None:
                    start_element = 0
                    full_shape = tuple(rank_tensor.shape)
                else:
                    if part.full_experts is None:
                        raise RuntimeError(
                            "expert tensor part has no full expert count"
                        )
                    trailing_elements = math.prod(rank_tensor.shape[1:])
                    start_element = part.expert_start * trailing_elements
                    full_shape = (part.full_experts, *rank_tensor.shape[1:])
                consume(rank, part.key, rank_tensor, start_element, full_shape)
                del rank_tensor
            del part
            gc.collect()
        del state
        gc.collect()
    return layer_visits


def _layer_major_rank_inventories(
    weight_map: dict[str, str],
    handles: dict[str, Any],
    src: DeepseekV4FlashInferenceConfig,
    *,
    ranks: list[int],
    tp_degree: int,
) -> tuple[dict[int, RankInventory], int]:
    specs: dict[int, dict[str, TensorSpec]] = {rank: {} for rank in ranks}
    next_elements: dict[tuple[int, str], int] = {}

    def collect(
        rank: int,
        key: str,
        tensor: torch.Tensor,
        start_element: int,
        full_shape: tuple[int, ...],
    ) -> None:
        candidate = TensorSpec(key, tensor.dtype, full_shape)
        existing = specs[rank].get(key)
        if existing is None:
            if start_element != 0:
                raise RuntimeError(
                    f"first inventory part for rank {rank} {key} starts at "
                    f"{start_element}, not zero"
                )
            specs[rank][key] = candidate
        elif existing != candidate:
            raise RuntimeError(
                f"inventory drift across pieces for rank {rank} {key}: "
                f"{existing} != {candidate}"
            )
        expected = next_elements.get((rank, key), 0)
        if start_element != expected:
            raise RuntimeError(
                f"non-contiguous inventory parts for rank {rank} {key}: "
                f"expected {expected}, got {start_element}"
            )
        next_elements[(rank, key)] = start_element + tensor.numel()

    layer_visits = _walk_layer_major_rank_tensors(
        weight_map,
        handles,
        src,
        ranks=ranks,
        tp_degree=tp_degree,
        consume=collect,
    )
    inventories: dict[int, RankInventory] = {}
    for rank in ranks:
        for key, spec in specs[rank].items():
            if next_elements[(rank, key)] != spec.numel:
                raise RuntimeError(
                    f"inventory for rank {rank} {key} covers "
                    f"{next_elements[(rank, key)]}/{spec.numel} elements"
                )
        inventories[rank] = RankInventory(
            rank=rank,
            tp_degree=tp_degree,
            tensors=tuple(specs[rank].values()),
        )
    return inventories, layer_visits


def _write_bounded_tensor(
    writer: StreamingRankWriter,
    key: str,
    tensor: torch.Tensor,
    *,
    start_element: int = 0,
) -> None:
    flat = tensor.contiguous().view(-1)
    elements_per_chunk = max(1, writer.max_chunk_bytes // flat.element_size())
    for start in range(0, flat.numel(), elements_per_chunk):
        writer.write_chunk(
            TensorChunk(
                key,
                start_element + start,
                flat[start : start + elements_per_chunk],
            )
        )


def _write_layer_major_tensor(
    writers: Mapping[int, StreamingRankWriter],
    rank: int,
    key: str,
    tensor: torch.Tensor,
    start_element: int,
    full_shape: tuple[int, ...],
) -> None:
    del full_shape
    _write_bounded_tensor(writers[rank], key, tensor, start_element=start_element)


def stream_shard_dsv4_checkpoint(
    hf_model_path: str,
    compiled_model_path: str,
    src: DeepseekV4FlashInferenceConfig,
    *,
    tp_degree: int,
    ep_degree: int | None = None,
    ranks: list[int] | None = None,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    _test_only_allow_unpinned_source: bool = False,
) -> dict[str, Any]:
    """Transactionally stream complete per-rank SafeTensors checkpoints."""
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
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")
    if _test_only_allow_unpinned_source and not src.allow_reduced_shapes:
        raise ValueError(
            "unpinned-source bypass is restricted to reduced test fixtures"
        )

    os.makedirs(compiled_model_path, exist_ok=True)
    weights_dir = os.path.join(compiled_model_path, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weight_map = _load_hf_index(hf_model_path)
    # The pinned DeepSeek-V4 snapshot spells the top-level leaves `embed`
    # and `head`; the converter emits the wrapper spellings below.
    required = {"embed.weight", "norm.weight"}
    if not src.tie_word_embeddings:
        required.add("head.weight")
    missing = sorted(required - weight_map.keys())
    if missing:
        raise KeyError(f"HF index is missing required top-level keys: {missing}")

    handles = _open_shards(hf_model_path, set(weight_map.values()))
    source_report = _preflight_and_audit_source(
        hf_model_path,
        weight_map,
        handles,
        test_only_allow_unpinned_source=_test_only_allow_unpinned_source,
    )
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
        "source_audit": source_report.to_dict(),
        "rank_inventory_sha256": {},
        "rank_manifest": {},
        "max_chunk_bytes": max_chunk_bytes,
        "layer_conversion_strategy": "layer-major-multi-rank-two-pass",
        "layer_conversion_passes": 2,
        "layer_conversion_visits": 0,
        "start_unix": int(time.time()),
    }
    try:
        inventories, inventory_visits = _layer_major_rank_inventories(
            weight_map,
            handles,
            src,
            ranks=ranks_iter,
            tp_degree=tp_degree,
        )
        write_started = time.time()
        writers: dict[int, StreamingRankWriter] = {}
        output_paths: dict[int, str] = {}
        with ExitStack() as stack:
            for rank in ranks_iter:
                output_path = os.path.join(
                    weights_dir, f"tp{rank}_sharded_checkpoint.safetensors"
                )
                output_paths[rank] = output_path
                writers[rank] = stack.enter_context(
                    StreamingRankWriter(
                        output_path,
                        inventories[rank],
                        source_report=source_report,
                        max_chunk_bytes=max_chunk_bytes,
                        source_metadata={
                            "model": "DeepSeek-V4-Flash-0731",
                            "revision": DSV4_CHECKPOINT_REVISION,
                            "config_sha256": DSV4_CONFIG_SHA256,
                            "index_sha256": DSV4_INDEX_SHA256,
                        },
                        manifest_schema="dsv4-streaming-rank-v1",
                    )
                )

            write_visits = _walk_layer_major_rank_tensors(
                weight_map,
                handles,
                src,
                ranks=ranks_iter,
                tp_degree=tp_degree,
                consume=partial(_write_layer_major_tensor, writers),
            )
            report["layer_conversion_visits"] = inventory_visits + write_visits
            expected_visits = 2 * src.num_hidden_layers
            if report["layer_conversion_visits"] != expected_visits:
                raise RuntimeError(
                    "layer-major conversion count drift: expected "
                    f"{expected_visits}, got {report['layer_conversion_visits']}"
                )

            for rank in ranks_iter:
                writer = writers[rank]
                manifest = writer.finalize()
                report["ranks_written"].append(rank)
                report["rank_bytes"][str(rank)] = os.path.getsize(output_paths[rank])
                report["rank_wall_s"][str(rank)] = round(time.time() - write_started, 2)
                inventory = inventories[rank]
                report["rank_inventory_sha256"][str(rank)] = inventory.contract_sha256
                report["rank_manifest"][str(rank)] = {
                    "path": os.path.basename(writer.manifest_path),
                    "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                    "chunks_written": manifest["resource_bound"]["chunks_written"],
                    "observed_max_chunk_bytes": manifest["resource_bound"][
                        "observed_max_chunk_bytes"
                    ],
                }
                del manifest
        del inventories, writers
        gc.collect()
    finally:
        _close_shards(handles)
    report["end_unix"] = int(time.time())
    report["total_wall_s"] = report["end_unix"] - report["start_unix"]
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_model_path")
    parser.add_argument("compiled_model_path")
    parser.add_argument("--tp-degree", type=int, default=32)
    parser.add_argument("--ep-degree", type=int)
    parser.add_argument("--ranks", type=int, nargs="+")
    parser.add_argument("--max-chunk-bytes", type=int, default=DEFAULT_MAX_CHUNK_BYTES)
    args = parser.parse_args()
    config = DeepseekV4FlashInferenceConfig.from_pretrained(args.hf_model_path)
    report = stream_shard_dsv4_checkpoint(
        args.hf_model_path,
        args.compiled_model_path,
        config,
        tp_degree=args.tp_degree,
        ep_degree=args.ep_degree,
        ranks=args.ranks,
        max_chunk_bytes=args.max_chunk_bytes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - exercised on compile host
    _main()


__all__ = [
    "DEFAULT_MAX_CHUNK_BYTES",
    "DSV4_CHECKPOINT_REVISION",
    "DSV4_CONFIG_SHA256",
    "DSV4_INDEX_SHA256",
    "stream_shard_dsv4_checkpoint",
]
