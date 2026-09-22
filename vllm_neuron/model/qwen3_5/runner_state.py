# SPDX-License-Identifier: Apache-2.0
"""
Per-step Gated DeltaNet inputs from the runner's own bookkeeping
================================================================

Pure numpy/torch and CPU-tested. The runner calls it at three points:

- a request is first scheduled     -> :meth:`GdnRunnerState.acquire`
- a request finishes or is aborted -> :meth:`GdnRunnerState.release`
- a step's model inputs are built  -> :meth:`GdnRunnerState.step_inputs`

What the runner's layout forces
-------------------------------
``NeuronModelRunner._prepare_model_input_impl`` pads each request separately,
giving a flat layout ``[req0_real, req0_pad, req1_real, req1_pad, ...]`` with
``padded_req_indices = np.repeat(arange(num_reqs), num_scheduled_tokens_padded)``.
Padding tokens are therefore attributed to their own request. Attention makes
that harmless by redirecting padding KV writes to a reserved null block; the
recurrence has no such block, so segmenting on ``padded_req_indices`` alone
would run padding through each request's state.

Within a request's segment the real tokens come first, so it is right
padding: segment on the padded counts, and mask with the real
``actual_num_tokens`` as ``lengths``. ``gated_deltanet`` turns masked
positions into exact no-ops.

Two more rules:

- **Prefix caching must be off.** A restored prefix makes a request start
  with ``num_computed_tokens > 0``, but recurrent state is not part of the
  prefix cache, and a reused slot may still hold a finished request's state.
  :meth:`step_inputs` raises instead of continuing on stale state.
- **One trailing sink row, always.** Tokens past the per-request segments
  (bucket padding of the whole flattened batch) go to it, which keeps the
  batch dimension fixed per compile bucket: ``num_reqs_padded + 1``.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .state import SlotAllocator


@dataclass
class GdnStepInputs:
    """Everything the Gated DeltaNet layers need for one step.

    ``rows`` is ``num_reqs_padded + 1``; the last row is the sink.
    """

    request_of_token: torch.Tensor  # [num_tokens] int64: segment row per flat token
    lengths: torch.Tensor  # [rows] int64: real tokens per row
    state_slots: torch.Tensor  # [rows] int64: stable state slot per row
    state_reset: torch.Tensor  # [rows] bool: start this row from zero state
    max_len: int  # static padded segment length for this bucket

    @property
    def rows(self) -> int:
        return int(self.lengths.shape[0])


class GdnRunnerState:
    """Owns the ``request_id -> slot`` map across steps."""

    def __init__(self, num_slots: int):
        self.allocator = SlotAllocator(num_slots)
        self.sink = num_slots
        self._fresh: set[str] = set()

    def acquire(self, request_id: str) -> int:
        new = request_id not in self.allocator._slot
        slot = self.allocator.acquire(request_id)
        if new:
            self._fresh.add(request_id)
        return slot

    def release(self, request_id: str) -> None:
        self.allocator.release(request_id)
        self._fresh.discard(request_id)

    def step_inputs(
        self,
        req_ids: list[str],
        num_computed_tokens: np.ndarray,
        actual_num_tokens: np.ndarray,
        num_scheduled_tokens_padded: np.ndarray,
        num_reqs_padded: int,
        num_tokens: int,
        device: torch.device | str = "cpu",
    ) -> GdnStepInputs:
        """Build this step's tensors.

        Args:
            req_ids: live requests in current batch-row order (post-condense).
            num_computed_tokens: per request, at the *start* of this step.
            actual_num_tokens: real tokens scheduled per request.
            num_scheduled_tokens_padded: per-request segment sizes including
                that request's own padding.
            num_reqs_padded: batch-row bucket (``>= len(req_ids)``).
            num_tokens: length of the flat ``input_ids`` for this step.
        """
        num_reqs = len(req_ids)
        if num_reqs_padded < num_reqs:
            raise ValueError("num_reqs_padded is smaller than the live request count")
        actual = np.asarray(actual_num_tokens, dtype=np.int64)
        padded = np.asarray(num_scheduled_tokens_padded, dtype=np.int64)
        if np.any(actual > padded):
            raise ValueError("a request has more real tokens than its padded segment")

        computed = np.asarray(num_computed_tokens, dtype=np.int64)
        reset = np.ones(num_reqs_padded + 1, dtype=bool)
        for row, rid in enumerate(req_ids):
            if rid in self._fresh:
                if computed[row] != 0:
                    raise RuntimeError(
                        f"request {rid} starts with {computed[row]} computed tokens: "
                        "prefix caching restored KV blocks, but Gated DeltaNet state "
                        "is not cached. Serve Qwen3.5 with --no-enable-prefix-caching."
                    )
                self._fresh.discard(rid)
            reset[row] = computed[row] == 0

        segments = np.repeat(np.arange(num_reqs, dtype=np.int64), padded)
        if segments.shape[0] > num_tokens:
            raise ValueError("per-request segments exceed the flat token count")
        sink_row = num_reqs_padded
        tail = np.full(num_tokens - segments.shape[0], sink_row, dtype=np.int64)
        request_of_token = np.concatenate([segments, tail])

        lengths = np.zeros(num_reqs_padded + 1, dtype=np.int64)
        lengths[:num_reqs] = actual

        slots = np.full(num_reqs_padded + 1, self.sink, dtype=np.int64)
        slots[:num_reqs] = self.allocator.slots_for_rows(list(req_ids), self.sink)

        # The sink row's segment holds any tail tokens; its length stays 0.
        counts = np.bincount(request_of_token, minlength=num_reqs_padded + 1)
        max_len = int(max(1, counts.max()))

        as_tensor = lambda a, dtype: torch.as_tensor(a, dtype=dtype, device=device)
        return GdnStepInputs(
            request_of_token=as_tensor(request_of_token, torch.int64),
            lengths=as_tensor(lengths, torch.int64),
            state_slots=as_tensor(slots, torch.int64),
            state_reset=as_tensor(reset, torch.bool),
            max_len=max_len,
        )

    def warmup_inputs(
        self,
        num_reqs_padded: int,
        num_tokens: int,
        max_len: int,
        device: torch.device | str = "cpu",
    ) -> GdnStepInputs:
        """Dummy inputs of the right shapes for compile warmup.

        The runner requires every model kwarg to be present during warmup so
        ``torch.compile`` traces one graph per shape. Every row points at the
        sink with length 0, so warmup never touches a live request's state.
        """
        rows = num_reqs_padded + 1
        return GdnStepInputs(
            request_of_token=torch.full(
                (num_tokens,), num_reqs_padded, dtype=torch.int64, device=device
            ),
            lengths=torch.zeros(rows, dtype=torch.int64, device=device),
            state_slots=torch.full(
                (rows,), self.sink, dtype=torch.int64, device=device
            ),
            state_reset=torch.ones(rows, dtype=torch.bool, device=device),
            max_len=max_len,
        )
