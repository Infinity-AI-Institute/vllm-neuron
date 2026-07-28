# SPDX-License-Identifier: Apache-2.0
"""CPU-reference attention/indexer semantics for GLM-5.2."""

import torch
import torch.nn.functional as F

from .cache_ops import gather_selected_paged_cache


def apply_glm52_interleaved_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply GLM's interleaved-pair RoPE and emit its split-pair layout."""
    if query.shape[-1] % 2 or key.shape[-1] % 2:
        raise ValueError("rotary dimensions must be even")
    pair_dim = cos.shape[-1] // 2
    cos_pairs = cos[..., :pair_dim].unsqueeze(unsqueeze_dim)
    sin_pairs = sin[..., :pair_dim].unsqueeze(unsqueeze_dim)

    query_even, query_odd = query[..., 0::2], query[..., 1::2]
    key_even, key_odd = key[..., 0::2], key[..., 1::2]
    if query_even.shape[-1] != pair_dim or key_even.shape[-1] != pair_dim:
        raise ValueError("cos/sin pair count must match the rotary dimensions")

    query_rotated = torch.cat(
        (
            query_even * cos_pairs - query_odd * sin_pairs,
            query_odd * cos_pairs + query_even * sin_pairs,
        ),
        dim=-1,
    )
    key_rotated = torch.cat(
        (
            key_even * cos_pairs - key_odd * sin_pairs,
            key_odd * cos_pairs + key_even * sin_pairs,
        ),
        dim=-1,
    )
    return query_rotated, key_rotated


def glm52_index_scores(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    head_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute DSA scores with ReLU before the weighted head reduction.

    Args:
        query: ``[batch, query_tokens, index_heads, index_head_dim]``.
        key_cache: ``[batch, key_tokens, index_head_dim]``.
        head_weights: ``[batch, query_tokens, index_heads]``.
    """
    if query.ndim != 4 or key_cache.ndim != 3 or head_weights.ndim != 3:
        raise ValueError("unexpected GLM indexer tensor rank")
    if query.shape[0] != key_cache.shape[0]:
        raise ValueError("query and key cache batch sizes differ")
    if query.shape[:3] != head_weights.shape:
        raise ValueError("head_weights must match query batch/token/head axes")
    if query.shape[-1] != key_cache.shape[-1]:
        raise ValueError("query and key dimensions differ")

    head_dim_scale = query.shape[-1] ** -0.5
    scores = (
        torch.matmul(
            query.to(torch.float32),
            key_cache.transpose(-1, -2).to(torch.float32).unsqueeze(1),
        )
        * head_dim_scale
    )
    scores = F.relu(scores)
    scaled_head_weights = head_weights.to(torch.float32) * (
        query.shape[-2] ** -0.5
    )
    return torch.matmul(
        scaled_head_weights.unsqueeze(-2),
        scores,
    ).squeeze(-2)


def glm52_index_topk(
    index_scores: torch.Tensor,
    *,
    top_k: int,
    position_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply causal masking and return DSA token indices."""
    masked_scores = glm52_mask_index_scores(
        index_scores,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    selected = min(top_k, index_scores.shape[-1])
    return masked_scores.topk(selected, dim=-1).indices.to(torch.int32)


def glm52_mask_index_scores(
    index_scores: torch.Tensor,
    *,
    position_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the causal/padding mask shared by CPU and Neuron top-k paths."""
    if attention_mask is not None:
        return index_scores + attention_mask
    if position_ids is None:
        raise ValueError("position_ids are required without an attention mask")
    key_positions = torch.arange(
        index_scores.shape[-1],
        device=index_scores.device,
    )
    causal = key_positions[None, None, :] > position_ids[:, :, None]
    return index_scores.masked_fill(causal, float("-inf"))


def glm52_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    scaling: float,
    key_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference DSA attention over the indexer's selected token positions.

    Args:
        query: ``[batch, query_tokens, heads, qk_head_dim]``.
        key_cache: ``[batch, key_tokens, heads, qk_head_dim]``.
        value_cache: ``[batch, key_tokens, heads, value_head_dim]``.
        topk_indices: ``[batch, query_tokens, selected_keys]``.
        position_ids: Absolute position of every query token.
        scaling: MLA attention softmax scale.
        key_lengths: Optional valid cached-key count per request.
    """
    if query.ndim != 4 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("query and cache tensors must be four-dimensional")
    if topk_indices.ndim != 3 or position_ids.ndim != 2:
        raise ValueError("topk_indices and position_ids have unexpected rank")
    if query.shape[:2] != topk_indices.shape[:2]:
        raise ValueError("topk token axes must match the query")
    if query.shape[:2] != position_ids.shape:
        raise ValueError("position_ids must match the query token axes")
    if key_cache.shape[:3] != value_cache.shape[:3]:
        raise ValueError("key and value cache axes must match")
    if query.shape[0] != key_cache.shape[0]:
        raise ValueError("query and cache batch sizes differ")
    if query.shape[2] != key_cache.shape[2]:
        raise ValueError("query and cache head counts differ")
    if query.shape[-1] != key_cache.shape[-1]:
        raise ValueError("query and key head dimensions differ")

    selected = topk_indices.to(torch.long)
    if not torch.compiler.is_compiling() and selected.numel() and (
        selected.min().item() < 0 or selected.max().item() >= key_cache.shape[1]
    ):
        raise ValueError("top-k index is outside the key cache")

    batch_indices = torch.arange(
        query.shape[0],
        device=query.device,
    )[:, None, None]
    selected_keys = key_cache[batch_indices, selected]
    selected_values = value_cache[batch_indices, selected]
    return _glm52_selected_attention(
        query,
        selected_keys,
        selected_values,
        selected,
        position_ids=position_ids,
        scaling=scaling,
        key_lengths=key_lengths,
    )


def glm52_paged_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_size: int,
    position_ids: torch.Tensor,
    scaling: float,
    key_lengths: torch.Tensor | None = None,
    key_quant_multiplier: torch.Tensor | None = None,
    value_quant_multiplier: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run DSA attention with direct top-k gathers from paged K/V caches."""
    selected = topk_indices.to(torch.long)
    selected_keys = gather_selected_paged_cache(
        key_cache,
        block_table,
        selected,
        block_size=block_size,
        output_dtype=query.dtype,
        quant_multiplier=key_quant_multiplier,
    )
    selected_values = gather_selected_paged_cache(
        value_cache,
        block_table,
        selected,
        block_size=block_size,
        output_dtype=query.dtype,
        quant_multiplier=value_quant_multiplier,
    )
    return _glm52_selected_attention(
        query,
        selected_keys,
        selected_values,
        selected,
        position_ids=position_ids,
        scaling=scaling,
        key_lengths=key_lengths,
    )


def _glm52_selected_attention(
    query: torch.Tensor,
    selected_keys: torch.Tensor,
    selected_values: torch.Tensor,
    selected_positions: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    scaling: float,
    key_lengths: torch.Tensor | None,
) -> torch.Tensor:
    """Compute attention over pre-gathered ``[B,Q,K,H,D]`` tensors."""
    scores = (
        query.to(torch.float32).unsqueeze(2)
        * selected_keys.to(torch.float32)
    ).sum(dim=-1)
    scores = scores * scaling

    invalid = selected_positions > position_ids[:, :, None]
    if key_lengths is not None:
        if key_lengths.shape != (query.shape[0],):
            raise ValueError("key_lengths must contain one value per request")
        invalid = invalid | (
            selected_positions >= key_lengths[:, None, None]
        )
    scores = scores.masked_fill(invalid.unsqueeze(-1), float("-inf"))
    probabilities = torch.softmax(scores, dim=2)
    probabilities = torch.nan_to_num(probabilities)
    output = (
        probabilities.unsqueeze(-1) * selected_values.to(torch.float32)
    ).sum(dim=2)
    return output.to(query.dtype)
