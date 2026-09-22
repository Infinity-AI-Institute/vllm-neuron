# SPDX-License-Identifier: Apache-2.0
"""
One Gated DeltaNet layer step over vLLM's flat token layout
===========================================================

The single function the model's layer calls and the CPU tests exercise, so
the tested path and the served path are the same code:

    flat tokens -> per-request padded -> read state by slot -> mixer
                -> write state by slot -> flat tokens

Pure torch; no vLLM imports.
"""

import torch

from . import gated_deltanet as gdn
from .runner_state import GdnStepInputs
from .state import RecurrentStateBuffers


# Chunk width for the prefill delta rule. 128 is the only size that
# neuronx-cc 2.32 compiles: 64 and 32 trip an internal vectorizer assertion
# ("MacroGeneration ... Trying to vectorize non loop axis", NCC_IMGN901) when
# a tile narrower than an SBUF partition feeds the chunk loop's matmuls, and
# 256 fails ISA validation (NCC_IINAR001). It only sets summation order, so
# the numerics are unchanged; docs/qwen3_5/RUNBOOK.md has the bisect.
CHUNK_SIZE = 128


def gdn_layer_step(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    buffers: RecurrentStateBuffers,
    layer: int,
    step: GdnStepInputs,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float,
    decode: bool,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    """Run layer ``layer``'s token mixer for one step; update its state in place.

    Args:
        hidden: ``[num_tokens, hidden]`` flat tokens, as vLLM lays them out.
        weights: this rank's sharded Gated DeltaNet weights (transformers
            names, ``[out, in]`` matrices, conv as ``[C, K]``).
        buffers: the model's recurrent state buffers.
        step: this step's :class:`GdnStepInputs`.
        num_k_heads, num_v_heads: this rank's head counts after TP sharding.

    Returns:
        ``[num_tokens, hidden]``; a row-parallel partial sum under TP, which
        the caller reduces.
    """
    # counts/offsets come from the runner: computing them here with
    # index_add_ gives wrong values on Trainium2 (see runner_state).
    counts, offsets = step.counts, step.offsets
    padded = gdn.tokens_to_padded(hidden, offsets, counts, step.max_len)

    conv, recurrent = buffers.read(layer, step.state_slots, step.state_reset)
    out, conv, recurrent = gdn.gated_deltanet_mixer(
        padded,
        weights,
        conv,
        recurrent,
        step.lengths,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        eps=eps,
        decode=decode,
        chunk_size=chunk_size,
    )
    buffers.write(layer, step.state_slots, conv, recurrent)
    return gdn.padded_to_tokens(out, step.request_of_token, offsets)
