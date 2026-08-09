# Kimi K3 validated async scheduling prototype

Status: local, off by default, no device qualification.

## Decision

Unrestricted vLLM-Neuron async scheduling is rejected for Kimi K3 exact
greedy. A narrower prototype is safe enough for local review: async scheduling
may remain enabled, but the prior sampled-token future must be copied to the
host and validated before the runner applies the next `SchedulerOutput`.

The prototype can overlap scheduler work with the preceding forward. It does
not permit an unvalidated token future to drive the next forward, so it does
not remove the token readback dependency. No Trainium launcher or model
default is changed here.

vLLM 0.21 normally resolves an unspecified `async_scheduling` value to enabled
when the executor supports it. That is not a safe experiment gate. A model
with a host sampling validator therefore refuses async model load unless
`VLLM_NEURON_EXPERIMENTAL_HOST_VALIDATED_ASYNC_SCHEDULING=1` is set explicitly.
Without that opt-in the operator must use `--no-async-scheduling`.

## Exact source candidates

- NDI `be0a584d1b19dbd17810236de59e8d21939d0070` adds the TP64 HLO
  qualification on top of `c42923f26b482e81755a23cb91a228b08a1a9397`, the
  hardened distributed exact-greedy implementation.
- vLLM-Neuron `85f8092698ae49d9ed9e1536b79627c5430d8e1d` is the local K3 runner
  source inspected for this prototype.
- vLLM `ad7125a431e176d4161099480a66f0169609a690` (`v0.21.0`) supplies
  `AsyncScheduler` and the two-entry engine batch queue.

## Why unrestricted async is unsafe

K3's distributed argmax uses the first padded-vocabulary ID as a failure
sentinel for a malformed collective payload, invalid runtime rank, or
non-finite participating logits. `validate_exact_greedy_token_ids` can reject
that value only after a CPU transfer. Its ABI explicitly requires rejection
before the token enters request state or becomes the next step's `input_ids`.

The ordinary async runner violates that contract in steady state:

1. `AsyncScheduler` advances requests with output placeholders and can enqueue
   the next batch before the prior output is consumed.
2. `_maybe_swap_async_input_ids` feeds the prior device tensor directly to the
   next forward.
3. `AsyncNeuronModelRunnerOutput.get_output` performs CPU validation later, on
   the output path.

An argmax sentinel can therefore reach embedding lookup and mutate MLA KV and
KDA recurrent state before the host notices it. Validating only before client
emission is not fail-closed.

## Prototype invariant

`_update_states_after_async_sampling_validation` creates one ordering point:

```text
previous forward completion
  -> CPU token-ID validation
  -> apply next SchedulerOutput
  -> condense InputBatch / sync KDA slots
  -> submit next forward
```

The barrier is selected only when four conditions hold: async scheduling,
on-device sampling, a callable model sampling-output validator, and the
explicit vLLM-Neuron experiment flag. Other models retain the existing
unmaterialized future path.

`get_async_scheduling_stats()` reports `host_validation_barrier_steps` in
addition to the existing async/fallback counters. A K3 receipt must include it:
otherwise an `async_steps` count cannot distinguish unrestricted future
chaining from this host-validated mode.

If validation raises, the raw tensor remains installed in the pending async
output and `_update_states` is not called. Retrying reaches the same validator
again. Consequently a failed request cannot be removed, cannot release or
reuse its KDA slot, and cannot submit another state-mutating forward through
this runner.

The existing composition-change barrier remains complementary. It handles
valid EOS overscheduling, aborts, preemption, condensation, and slot reuse by
waiting for the prior output before membership changes. The new barrier covers
the same-request steady-state case that composition comparison cannot detect.

## What is and is not proven locally

The unit tests prove the host ordering and fail-closed mutation boundary:

- a valid exact-greedy token is materialized before scheduler state update;
- an invalid token prevents scheduler state update and leaves request/output
  state untouched;
- a model without a validator keeps the existing async path.

The proof relies on the runner's established tensor-future contract: `.cpu()`
waits for the producing forward, whose KDA scatter is part of the same graph.
No device, cache, HLO, container, or remote host was used.

This does not qualify a performance win or unrestricted device chaining.
Device qualification would still be needed before enabling the option in a
launcher.

## Requirements for a future barrier-free design

Removing the host barrier requires a graph-visible poison protocol, not merely
another output check:

1. Return a valid, safe token ID plus a separate per-row status bit; never feed
   the out-of-range sentinel to embedding lookup.
2. Carry status into every queued successor and suppress both MLA KV writes and
   KDA gather/reset/scatter for poisoned rows.
3. Propagate poison to subsequent outputs until the host acknowledges the
   failure; do not allow slot release/reuse to erase it silently.
4. Tag every submission with request ID, scheduler sequence, slot number, and
   slot generation. Assert the same tuple at output materialization.
5. Instrument `schedule`, runner state update, slot acquire/reset/commit/release,
   forward submit/complete, token validation, EOS discard, abort, and
   preemption. Exercise condensation, same-step EOS overshoot, abort with two
   queued batches, preemption/resume, and sentinel injection.

Those graph changes would require a new graph/cache identity. They must remain
separate from the B1 T256 baseline; this prototype changes only host ordering
and does not alter B1 T256 compile identity or claim its measurements.
