# SPDX-License-Identifier: Apache-2.0
"""Executable GLM-5.2 DSA indexer and IndexShare state semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (
    apply_glm52_interleaved_rope,
    glm52_index_scores,
    glm52_mask_index_scores,
)
from .cache_layout import IndexerCacheBinding
from .cache_ops import gather_paged_cache, write_paged_cache
from .config import Glm52MoeDsaConfig


@dataclass(frozen=True)
class Glm52IndexShareState:
    """Top-k selection propagated from a full indexer into shared layers."""

    topk_indices: torch.Tensor
    source_layer_idx: int


def advance_index_share_state(
    config: Glm52MoeDsaConfig,
    *,
    layer_idx: int,
    previous: Glm52IndexShareState | None,
    computed_topk: torch.Tensor | None,
) -> Glm52IndexShareState:
    """Apply the frozen full/shared indexer schedule for one decoder layer."""
    if not 0 <= layer_idx < config.num_hidden_layers:
        raise IndexError(f"layer index {layer_idx} is outside the backbone")

    indexer_type = config.indexer_types[layer_idx]
    if indexer_type == "full":
        if computed_topk is None:
            raise ValueError(f"full indexer layer {layer_idx} must compute top-k")
        return Glm52IndexShareState(
            topk_indices=computed_topk,
            source_layer_idx=layer_idx,
        )

    if computed_topk is not None:
        raise ValueError(f"shared indexer layer {layer_idx} cannot replace top-k")
    if previous is None:
        raise ValueError(
            f"shared indexer layer {layer_idx} requires a previous full indexer"
        )
    return previous


@dataclass(frozen=True)
class Glm52IndexerProjection:
    query: torch.Tensor
    key: torch.Tensor
    head_weights: torch.Tensor


class Glm52FullIndexer(nn.Module):
    """Full GLM indexer for layers that own a DSA top-k computation.

    The module is replicated over TP ranks, matching the checkpoint. It consumes
    MLA's normalized ``q_resid`` for query projection and normalized hidden
    states for the index key and FP32 head weights.
    """

    def __init__(
        self,
        config: Glm52MoeDsaConfig,
        *,
        layer_idx: int,
        cache_binding: IndexerCacheBinding,
        dtype: torch.dtype | None = None,
        topk_backend: str = "torch",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if config.indexer_types[layer_idx] != "full":
            raise ValueError(f"layer {layer_idx} does not own a full indexer")
        if cache_binding.layer_idx != layer_idx:
            raise ValueError("cache binding belongs to a different indexer layer")
        if topk_backend not in ("torch", "neuron"):
            raise ValueError("topk_backend must be 'torch' or 'neuron'")

        self.config = config
        self.layer_idx = layer_idx
        self.cache_binding = cache_binding
        self.dtype = dtype or config.torch_dtype
        self.topk_backend = topk_backend
        self.wq_b = nn.Linear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
            dtype=self.dtype,
            device=device,
        )
        self.wk = nn.Linear(
            config.hidden_size,
            config.index_head_dim,
            bias=False,
            dtype=self.dtype,
            device=device,
        )
        self.k_norm = nn.LayerNorm(
            config.index_head_dim,
            eps=1e-6,
            dtype=self.dtype,
            device=device,
        )
        self.weights_proj = nn.Linear(
            config.hidden_size,
            config.index_n_heads,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        self.requires_grad_(False)
        self.key_cache: torch.Tensor | None = None
        self.register_buffer(
            "cache_quant_multiplier",
            torch.ones(1, dtype=torch.float32),
            persistent=False,
        )

    def bind_key_cache(self, cache: torch.Tensor) -> None:
        if cache.ndim != 4:
            raise ValueError("indexer cache must be a four-dimensional tensor")
        if cache.shape[1] != 1:
            raise ValueError("indexer cache must contain one MQA key head")
        if cache.shape[-1] != self.config.index_head_dim:
            raise ValueError("indexer cache head dimension is incorrect")
        self.key_cache = cache

    def set_cache_quant_multiplier(self, multiplier: float) -> None:
        if not multiplier > 0:
            raise ValueError("cache quantization multiplier must be positive")
        self.cache_quant_multiplier.fill_(multiplier)

    def project(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Glm52IndexerProjection:
        if hidden_states.shape[:-1] != q_resid.shape[:-1]:
            raise ValueError("hidden_states and q_resid token axes must match")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden_states has an incorrect hidden dimension")
        if q_resid.shape[-1] != self.config.q_lora_rank:
            raise ValueError("q_resid has an incorrect LoRA dimension")

        query = F.linear(
            q_resid.to(self.wq_b.weight.dtype),
            self.wq_b.weight,
        )
        query = query.view(
            *hidden_states.shape[:-1],
            self.config.index_n_heads,
            self.config.index_head_dim,
        )
        # PR #13 fix: Tensor.split([list], dim=-1) returns wrong data by up to 6.0
        # on trn2 (see device_split_repro on deepseek-v4-flash-base). Use slicing.
        _rope = self.config.qk_rope_head_dim
        q_rot = query[..., :_rope]
        q_pass = query[..., _rope:]

        key = F.linear(
            hidden_states.to(self.wk.weight.dtype),
            self.wk.weight,
        )
        key = self.k_norm(key)
        # PR #13 fix: same split-list-overload trn2 miscompile — use slicing.
        k_rot = key[..., :_rope]
        k_pass = key[..., _rope:]

        q_rot, k_rot = apply_glm52_interleaved_rope(
            q_rot,
            k_rot.unsqueeze(-2),
            cos,
            sin,
            unsqueeze_dim=-2,
        )
        query = torch.cat((q_rot, q_pass), dim=-1)
        key = torch.cat((k_rot.squeeze(-2), k_pass), dim=-1)
        head_weights = F.linear(
            hidden_states.to(torch.float32),
            self.weights_proj.weight,
        )
        return Glm52IndexerProjection(
            query=query,
            key=key,
            head_weights=head_weights,
        )

    def _select_topk(
        self,
        scores: torch.Tensor,
        *,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selected = min(self.config.index_topk, scores.shape[-1])
        if selected == scores.shape[-1]:
            # At the 2,048-token baseline, GLM's index_topk equals the full
            # context. Selecting every position does not require ranking them.
            # Avoid torch.topk(k == context), whose full-sort HLO is unsupported
            # on Trn2, preserve logical order, and keep the signed-int64 index
            # invariant required by the PR #13 sentinel-wrap fix.
            indices = torch.arange(
                selected,
                dtype=torch.int64,
                device=scores.device,
            )
            leading_ones = (1,) * (scores.ndim - 1)
            return indices.view(*leading_ones, selected).expand(
                *scores.shape[:-1],
                selected,
            )

        masked_scores = glm52_mask_index_scores(
            scores,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        if self.topk_backend == "neuron":
            # Lazy import keeps CPU-only model/unit-test imports independent of
            # NKI while making the compiled path use rotational_topk instead of
            # torch.topk's unsupported full-sort lowering on Trn2.
            from vllm_neuron.functional.topk import topk as neuron_topk

            _, indices = neuron_topk(
                masked_scores,
                k=selected,
                dim=-1,
                gather_dim=-1,
            )
        else:
            _, indices = torch.topk(masked_scores, selected, dim=-1)
        # PR #13 fix: torch.topk index override returns unsigned int32 on trn2;
        # any -1 sentinel wraps to 4294967295 sending scatter_ oob (nrta status=1006).
        # Cast to int64 before any sentinel comparison downstream.
        return indices.to(torch.int64)

    def forward_dense(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        key_cache: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reference top-k over an already assembled dense index-key cache."""
        projection = self.project(hidden_states, q_resid, cos, sin)
        scores = glm52_index_scores(
            projection.query,
            key_cache,
            projection.head_weights,
        )
        return self._select_topk(
            scores,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

    def forward_paged(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attn_metadata: dict[str, dict[str, torch.Tensor | int]],
        key_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write current keys, gather the paged history, and compute top-k.

        Inputs use the flattened, request-major token layout of vLLM-Neuron.
        The returned tensor is flattened to ``[tokens, selected_keys]`` so it
        can be passed unchanged through subsequent shared decoder layers.
        """
        active_cache = key_cache if key_cache is not None else self.key_cache
        if active_cache is None:
            raise RuntimeError("indexer key cache has not been bound")
        if hidden_states.ndim != 2 or q_resid.ndim != 2:
            raise ValueError("paged indexer inputs must be flattened token matrices")

        metadata = attn_metadata[self.cache_binding.cache_name]
        slot_mapping = metadata["slot_mapping"]
        block_size = metadata["block_size"]
        block_table = metadata["block_table_tensor"]
        if not isinstance(slot_mapping, torch.Tensor):
            raise TypeError("slot_mapping metadata must be a tensor")
        if not isinstance(block_table, torch.Tensor):
            raise TypeError("block_table_tensor metadata must be a tensor")
        if not isinstance(block_size, int):
            raise TypeError("block_size metadata must be an integer")

        projection = self.project(hidden_states, q_resid, cos, sin)
        quant_multiplier = (
            self.cache_quant_multiplier
            if active_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            else None
        )
        write_paged_cache(
            active_cache,
            projection.key.unsqueeze(1),
            slot_mapping,
            block_size=block_size,
            quant_multiplier=quant_multiplier,
        )
        dense_cache = gather_paged_cache(
            active_cache,
            block_table,
            block_size=block_size,
            output_dtype=self.dtype,
            quant_multiplier=quant_multiplier,
        ).squeeze(2)

        batch_size = block_table.shape[0]
        tokens = hidden_states.shape[0]
        if tokens % batch_size:
            raise ValueError("flattened token count must divide over requests")
        query_tokens = tokens // batch_size
        query = projection.query.reshape(
            batch_size,
            query_tokens,
            self.config.index_n_heads,
            self.config.index_head_dim,
        )
        head_weights = projection.head_weights.reshape(
            batch_size,
            query_tokens,
            self.config.index_n_heads,
        )
        positions = position_ids.reshape(batch_size, query_tokens)
        scores = glm52_index_scores(query, dense_cache, head_weights)
        topk_indices = self._select_topk(
            scores,
            position_ids=positions,
        )
        return topk_indices.reshape(tokens, topk_indices.shape[-1])
