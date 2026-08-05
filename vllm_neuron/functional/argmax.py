# SPDX-License-Identifier: Apache-2.0
"""
Distributed argmax kernel for tensor-parallel inference.

This module provides a distributed argmax operation that works across
sharded tensors in a tensor-parallel setting.
"""

import logging
from typing import Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed._functional_collectives import all_gather_tensor
import nki
from nkilib.core.max.cascaded_max import cascaded_max
from vllm_neuron import envs
from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

cascaded_max_jit = nki.jit()(cascaded_max)

logger = logging.getLogger(__name__)

# Greedy sampling must preserve the true BF16 winner. Only exact equal logits
# are canonicalized by token ID; treating merely close values as ties changes
# model semantics and can select a lower, non-maximal token.
_BF16_NEAR_TIE_RTOL = 0.0


def argmax(
    tensor: Tensor,
    dim: int,
    gather_dim: int,
    keepdim: bool = False,
    process_group: Optional[ProcessGroup] = None,
    deterministic_ties: bool = False,
    rank: Optional[Tensor] = None,
) -> Tensor:
    """
    Performs distributed argmax on sharded tensors using 2-step algorithm.

    This function implements a distributed argmax operation for tensor-parallel
    inference where tensors are sharded across multiple devices. It uses a
    2-step approach:

    1. **Local argmax**: Each rank computes argmax on its local shard using
       NKI cascaded_max kernel (when conditions are met) or torch.max fallback
    2. **Global argmax**: Results are gathered and global argmax is computed

    **Sharding Layout**:
    The input tensor is assumed to be uniformly sharded along `gather_dim` across
    all ranks in the process group. For example, with TP=2 and vocab_size=256:
    - Rank 0: logits[:, 0:128]   (first 128 vocab tokens)
    - Rank 1: logits[:, 128:256] (last 128 vocab tokens)

    **When to Use**:

    This function is intended for distributed argmax. It will fall back to torch.argmax when process_group is not provided.

    **Kernel**:
    In step 1 we compute the local maxs using either a nki kernel or torch.max.
    NKI cascaded_max kernel is used when: dim=-1, 2D/3D tensor, size>=128, and
    not in CPU mode. Otherwise falls back to torch.max.

    Args:
        tensor: Input tensor to perform argmax on. Must be uniformly sharded
            along `gather_dim` across all ranks in `process_group`.
        dim: Dimension along which to find argmax.
        gather_dim: Dimension the tensor is sharded on. When `dim == gather_dim`,
            indices are corrected for sharding offset.
        keepdim: Whether to keep the reduced dimension. Defaults to False.
        process_group: Process group for distributed operations.
            Must be provided for distributed execution.
        deterministic_ties: Select the lowest global token index when BF16
            logits are exactly equal.
        rank: Rank in ``process_group`` as a tensor. Required when reducing the
            sharded dimension so local indices can be converted to global IDs
            before the collective without baking rank into the graph.

    Returns:
        Tensor with global argmax indices across all shards. All ranks return
        the same result.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from vllm_neuron.functional.argmax import argmax
        >>>
        >>> # Tensor-parallel inference with TP=2, vocab_size=256
        >>> # Each rank has logits of shape (batch=1, vocab_per_rank=128)
        >>> logits = torch.randn(1, 128)  # Local shard on this rank
        >>> pg = dist.new_group([0, 1])
        >>>
        >>> # Compute global argmax across both ranks
        >>> rank = torch.tensor(dist.get_rank(pg), dtype=torch.int32)
        >>> token_id = argmax(
        ...     logits, dim=1, gather_dim=1, process_group=pg, rank=rank
        ... )
        >>> # Returns shape (1,) with global token index in range [0, 256)
        >>> # Rank 0 indices are in [0, 128), Rank 1 indices are in [128, 256)
    """
    if process_group is None:
        raise ValueError("process_group must be provided for distributed argmax")

    tp_degree = torch.distributed.get_world_size(group=process_group)

    # Fast path for single device
    if tp_degree == 1:
        return torch.argmax(tensor, dim=dim, keepdim=keepdim)

    # TODO: Remove this upcast - NKI kernel and functional tests work with bfloat16, but E2E models show accuracy regression without it
    tensor = tensor.to(torch.float32)
    if deterministic_ties:
        # Use compiler-native reductions so the deterministic path defines its
        # candidate set independently of the fused value/index fast path.
        local_value = torch.amax(tensor, dim=dim, keepdim=True)
        local_index = _lowest_tied_index(
            tensor,
            local_value,
            dim,
            relative_tolerance=_BF16_NEAR_TIE_RTOL,
        )
    else:
        local_value, local_index = _compute_local_max(tensor, dim)

    # Convert local shard indices to global IDs before gathering. Collective
    # output order is a transport detail and must not be used to infer rank;
    # every rank must feed the same sampled token into the next decode step.
    if gather_dim == dim:
        if rank is None:
            raise ValueError(
                "rank must be provided as a tensor when gather_dim == dim"
            )
        local_index = _globalize_local_index(
            local_index,
            rank,
            tensor.shape[gather_dim],
        )

    normalized_dim = dim if dim >= 0 else tensor.ndim + dim
    normalized_gather_dim = (
        gather_dim if gather_dim >= 0 else tensor.ndim + gather_dim
    )
    can_fuse_gathers = (
        deterministic_ties
        and normalized_dim == normalized_gather_dim == tensor.ndim - 1
        # Global token IDs are transported as FP32 in the tagged pair. Keep the
        # conversion exact rather than silently rounding an unusually wide ID.
        and tensor.shape[normalized_gather_dim] * tp_degree < 2**24
    )

    # The deterministic greedy path transports each rank's maximum and tagged
    # global token ID in one gather. Besides saving a collective, tagging makes
    # the result independent of collective output ordering.
    if can_fuse_gathers:
        local_pair = _pack_local_max(local_value, local_index, normalized_dim)
        global_pair = all_gather_tensor(
            local_pair,
            normalized_gather_dim,
            group=process_group,
        )
        global_values, global_indices = _unpack_gathered_max(
            global_pair,
            normalized_dim,
            local_index.dtype,
        )
    else:
        global_values = all_gather_tensor(
            local_value,
            gather_dim,
            group=process_group,
        )
        global_indices = all_gather_tensor(
            local_index,
            gather_dim,
            group=process_group,
        )

    # all_gather preserves process-group rank order. Each rank already chose
    # its lowest local exact tie, and contiguous vocab shards are rank ordered,
    # so first-max semantics also select the lowest global token on exact
    # cross-rank ties. Avoid a second amax/where/amin reduction over candidates.
    global_argmax = torch.argmax(global_values, dim=dim, keepdim=True)
    final_indices = torch.gather(global_indices, dim, global_argmax)

    if not keepdim:
        return final_indices.squeeze(dim)
    return final_indices


def _lowest_tied_index(
    tensor: Tensor,
    maximum: Tensor,
    dim: int,
    *,
    relative_tolerance: float = 0.0,
) -> Tensor:
    """Return the lowest index within ``relative_tolerance`` of a maximum."""
    normalized_dim = dim if dim >= 0 else tensor.ndim + dim
    index_shape = [1] * tensor.ndim
    index_shape[normalized_dim] = tensor.shape[normalized_dim]
    indices = torch.arange(
        tensor.shape[normalized_dim],
        dtype=torch.int32,
        device=tensor.device,
    ).reshape(index_shape)
    threshold = maximum - (torch.abs(maximum) * relative_tolerance)
    candidates = torch.where(
        tensor >= threshold,
        indices,
        tensor.shape[normalized_dim],
    )
    return torch.amin(candidates, dim=normalized_dim, keepdim=True)


def _globalize_local_index(
    local_index: Tensor,
    rank: Tensor,
    shard_size: int,
) -> Tensor:
    """Convert a local shard index to its global index using runtime rank."""
    return local_index + rank.to(local_index.dtype) * shard_size


def _pack_local_max(
    local_value: Tensor,
    global_index: Tensor,
    dim: int,
) -> Tensor:
    """Pack one value/index pair for a single tagged all-gather."""
    return torch.cat((local_value, global_index.to(local_value.dtype)), dim=dim)


def _unpack_gathered_max(
    gathered_pair: Tensor,
    dim: int,
    index_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Unpack rank-major ``[value, global_index]`` pairs after all-gather."""
    normalized_dim = dim if dim >= 0 else gathered_pair.ndim + dim
    if normalized_dim != gathered_pair.ndim - 1:
        raise ValueError("tagged max pairs can only be unpacked along the last dim")
    pair_shape = (*gathered_pair.shape[:-1], gathered_pair.shape[-1] // 2, 2)
    pairs = gathered_pair.reshape(pair_shape)
    return pairs[..., 0], pairs[..., 1].to(index_dtype)


def _compute_local_max(tensor: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """Compute local max using NKI kernel when possible, otherwise torch.max."""
    if _can_use_nki_max(tensor, dim):
        is_3d = len(tensor.shape) == 3
        input_tensor = tensor.squeeze(0) if is_3d else tensor

        cascaded_max_nki = wrap_nki(cascaded_max_jit)
        value, index = cascaded_max_nki[2](input_tensor)
        # NKI kernel returns uint32 indices; gloo all_gather doesn't support
        # uint32, so cast to int64 (matching torch.max) in CPU mode.
        if envs.VLLM_NEURON_CPU_MODE:
            index = index.to(torch.int64)
        # Restore dimension if squeezed
        if is_3d:
            value = value.unsqueeze(0)
            index = index.unsqueeze(0)
        return value, index
    else:
        return torch.max(tensor, dim=dim, keepdim=True)


def _can_use_nki_max(tensor: Tensor, dim: int) -> bool:
    """
    Check if we can use the NKI max kernel.

    Requirements:
    - NKI kernels can run (hardware or CPU simulator)
    - dim must be the last dimension (NKI kernel only supports this)
    - Tensor must be 2D or 3D with shape[0] == 1
    - Reduction dimension must have at least 128 elements
    """
    if not can_run_kernel(tensor):
        return False

    shape = tensor.shape
    num_dims = len(shape)

    # NKI kernel only works on last dimension
    if dim != num_dims - 1 and dim != -1:
        return False

    # Check tensor dimensionality
    if not (num_dims == 2 or (num_dims == 3 and shape[0] == 1)):
        return False

    # Check minimum reduction size
    return not shape[dim] < 128
