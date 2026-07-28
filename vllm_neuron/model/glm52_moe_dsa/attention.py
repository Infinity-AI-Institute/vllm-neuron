# SPDX-License-Identifier: Apache-2.0
"""CPU-reference attention/indexer semantics for GLM-5.2."""

import torch
import torch.nn.functional as F


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
    if attention_mask is not None:
        masked_scores = index_scores + attention_mask
    else:
        if position_ids is None:
            raise ValueError("position_ids are required without an attention mask")
        key_positions = torch.arange(
            index_scores.shape[-1],
            device=index_scores.device,
        )
        causal = key_positions[None, None, :] > position_ids[:, :, None]
        masked_scores = index_scores.masked_fill(causal, float("-inf"))

    selected = min(top_k, index_scores.shape[-1])
    return masked_scores.topk(selected, dim=-1).indices.to(torch.int32)
