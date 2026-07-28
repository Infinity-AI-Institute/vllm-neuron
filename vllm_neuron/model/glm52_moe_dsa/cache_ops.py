# SPDX-License-Identifier: Apache-2.0
"""Paged-cache operations shared by the GLM-5.2 indexer and MLA path."""

from __future__ import annotations

import torch


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_TRN2_E4M3_MAX = 240.0


def _indexable_cache(cache: torch.Tensor) -> torch.Tensor:
    """Return a CPU-indexable view without changing the Neuron graph path.

    PyTorch CPU does not implement advanced indexing or index_select for FP8
    tensors. CPU is only the numerical reference/test path, so cast there
    before selecting rows. XLA/Neuron keeps indexing the original FP8 cache.
    """

    if cache.device.type == "cpu" and cache.dtype in _FP8_DTYPES:
        return cache.to(torch.float32)
    return cache


def _validate_cache_shape(cache: torch.Tensor, block_size: int) -> None:
    if cache.ndim != 4:
        raise ValueError(
            "paged cache must have shape [num_blocks, num_heads, block_size, head_dim]"
        )
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if cache.shape[2] != block_size:
        raise ValueError(
            f"cache block size {cache.shape[2]} does not match metadata "
            f"block size {block_size}"
        )


def _validate_quant_multiplier(
    cache: torch.Tensor,
    quant_multiplier: torch.Tensor | None,
) -> None:
    if cache.dtype in _FP8_DTYPES:
        if quant_multiplier is None or quant_multiplier.numel() != 1:
            raise ValueError(
                "FP8 paged caches require one scalar quantization multiplier"
            )
    elif quant_multiplier is not None:
        raise ValueError("a quantization multiplier is only valid for FP8 caches")


def write_paged_cache(
    cache: torch.Tensor,
    values: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    block_size: int,
    quant_multiplier: torch.Tensor | None = None,
) -> None:
    """Write token-major values to a standard Neuron paged cache.

    Args:
        cache: ``[num_blocks, num_heads, block_size, head_dim]``.
        values: ``[tokens, num_heads, head_dim]``.
        slot_mapping: One physical slot per token.
        block_size: Cache block size from attention metadata.
        quant_multiplier: For FP8, a scalar ``q = value * multiplier``.

    Invalid padded slots are redirected to an identical valid-token write.
    When every slot is invalid, they rewrite slot zero with its existing
    value. This preserves static scatter shapes without allowing padding to
    overwrite a live cache entry.
    """
    _validate_cache_shape(cache, block_size)
    _validate_quant_multiplier(cache, quant_multiplier)
    if values.ndim != 3:
        raise ValueError("values must have shape [tokens, num_heads, head_dim]")
    if values.shape[0] != slot_mapping.numel():
        raise ValueError("slot_mapping must contain one entry per token")
    if values.shape[1] != cache.shape[1] or values.shape[2] != cache.shape[3]:
        raise ValueError("value head axes do not match the paged cache")

    num_blocks, num_heads, _, head_dim = cache.shape
    max_slot = num_blocks * block_size
    flat_slots = slot_mapping.reshape(-1)
    valid_slots = (flat_slots >= 0) & (flat_slots < max_slot)
    has_valid_slot = valid_slots.any()
    first_valid = valid_slots & (valid_slots.to(torch.int64).cumsum(0) == 1)
    reference_slot = torch.where(
        has_valid_slot,
        torch.where(first_valid, flat_slots, torch.zeros_like(flat_slots)).sum(),
        torch.zeros((), dtype=flat_slots.dtype, device=flat_slots.device),
    )
    safe_slots = torch.where(valid_slots, flat_slots, reference_slot).to(torch.long)

    stored = values
    if cache.dtype in _FP8_DTYPES:
        multiplier = quant_multiplier.to(torch.float32).reshape(1, 1, 1)
        clamp_max = (
            _TRN2_E4M3_MAX
            if cache.dtype == torch.float8_e4m3fn
            else torch.finfo(cache.dtype).max
        )
        stored = (
            (values.to(torch.float32) * multiplier)
            .clamp(-clamp_max, clamp_max)
            .to(cache.dtype)
        )
    else:
        stored = values.to(cache.dtype)

    reference_from_values = (
        stored.to(torch.float32) * first_valid.reshape(-1, 1, 1).to(torch.float32)
    ).sum(dim=0)
    reference_value = torch.where(
        has_valid_slot,
        reference_from_values,
        cache[0, :, 0, :].to(torch.float32),
    ).to(cache.dtype)
    stored = torch.where(
        valid_slots.reshape(-1, 1, 1),
        stored,
        reference_value.unsqueeze(0),
    )

    block_indices = safe_slots // block_size
    block_offsets = safe_slots % block_size
    head_indices = torch.arange(
        num_heads,
        device=values.device,
        dtype=torch.long,
    ).repeat(values.shape[0])
    cache.index_put_(
        (
            block_indices.repeat_interleave(num_heads),
            head_indices,
            block_offsets.repeat_interleave(num_heads),
        ),
        stored.reshape(-1, head_dim),
    )


def gather_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_size: int,
    output_dtype: torch.dtype,
    quant_multiplier: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather a request-major dense view from a standard paged cache.

    Returns ``[batch, max_blocks * block_size, num_heads, head_dim]``.
    Causal/sequence-length masking remains the caller's responsibility.
    """
    _validate_cache_shape(cache, block_size)
    _validate_quant_multiplier(cache, quant_multiplier)
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [batch, max_blocks]")

    safe_blocks = block_table.to(torch.long).clamp(0, cache.shape[0] - 1)
    gathered = _indexable_cache(cache)[safe_blocks]
    gathered = gathered.permute(0, 1, 3, 2, 4).contiguous()
    gathered = gathered.reshape(
        block_table.shape[0],
        block_table.shape[1] * block_size,
        cache.shape[1],
        cache.shape[3],
    )

    if cache.dtype in _FP8_DTYPES:
        multiplier = quant_multiplier.to(torch.float32).reshape(1, 1, 1, 1)
        gathered = gathered.to(torch.float32) / multiplier
    return gathered.to(output_dtype)


def gather_selected_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    selected_positions: torch.Tensor,
    *,
    block_size: int,
    output_dtype: torch.dtype,
    quant_multiplier: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather only selected logical token positions from a paged cache.

    Args:
        cache: ``[num_blocks, num_heads, block_size, head_dim]``.
        block_table: Physical block IDs, ``[batch, max_blocks]``.
        selected_positions: Logical positions, ``[batch, query_tokens, topk]``.

    Returns:
        ``[batch, query_tokens, topk, num_heads, head_dim]``.

    Resolving logical positions before the row gather avoids materializing a
    dense context-sized cache view for DSA's fixed top-k attention.
    """
    _validate_cache_shape(cache, block_size)
    _validate_quant_multiplier(cache, quant_multiplier)
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [batch, max_blocks]")
    if selected_positions.ndim != 3:
        raise ValueError(
            "selected_positions must have shape [batch, query_tokens, topk]"
        )
    if selected_positions.shape[0] != block_table.shape[0]:
        raise ValueError("selected positions and block table batch sizes differ")

    selected = selected_positions.to(torch.long)
    logical_limit = block_table.shape[1] * block_size
    if (
        not torch.compiler.is_compiling()
        and selected.numel()
        and (selected.min().item() < 0 or selected.max().item() >= logical_limit)
    ):
        raise ValueError("selected logical position is outside the block table")

    safe_selected = selected.clamp(0, logical_limit - 1)
    logical_blocks = safe_selected // block_size
    block_offsets = safe_selected % block_size
    physical_blocks = torch.gather(
        block_table.to(torch.long),
        1,
        logical_blocks.reshape(block_table.shape[0], -1),
    ).reshape_as(logical_blocks)
    physical_blocks = physical_blocks.clamp(0, cache.shape[0] - 1)
    physical_slots = physical_blocks * block_size + block_offsets

    flat_cache = _indexable_cache(cache).permute(0, 2, 1, 3).reshape(
        cache.shape[0] * block_size,
        cache.shape[1],
        cache.shape[3],
    )
    selected_values = torch.index_select(
        flat_cache,
        0,
        physical_slots.reshape(-1),
    ).contiguous()
    selected_values = selected_values.reshape(
        *selected.shape,
        cache.shape[1],
        cache.shape[3],
    )

    if cache.dtype in _FP8_DTYPES:
        multiplier = quant_multiplier.to(torch.float32).reshape(1, 1, 1, 1, 1)
        selected_values = selected_values.to(torch.float32) / multiplier
    return selected_values.to(output_dtype)
