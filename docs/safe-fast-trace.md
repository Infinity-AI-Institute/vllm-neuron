# Safe fast-trace A/B

`VLLM_NEURON_FAST_TRACE=1` is an opt-in cold-start experiment that suppresses
only successful FX text dumps. It does not skip graph mutation, any
`gm.recompile()`, either direct-compile cache-hash call, normalized-copy
recompilation, FX or HLO passes, compiler work, or final HLO, NEFF, and cache
metadata.

Fast mode still writes `example_inputs.txt`. An ordinary pipeline failure
writes exactly one `fxgraph_failure.txt` containing the current graph and
generated code. `MemoryError`, `KeyboardInterrupt`, and `SystemExit` (including
a wrapped `MemoryError`) do not render the graph because doing so can worsen a
low-memory or shutdown condition.

## Metrics ownership

Set `VLLM_NEURON_TRACE_METRICS=1` to collect a baseline receipt without changing
dump behavior. Fast mode enables the same receipt automatically. Each
FX-to-HLO invocation creates its own metrics object; caller compilation options
are never mutated or used to retain counters. The receipt is written as
`trace_metrics.json` in that graph's compile directory.

The receipt reports trace wall time, per-pass wall time, graph/code renders,
written and suppressed dump counts, failure diagnostics, and process RSS
high-water marks. RSS is the trace child's `VmHWM` (or platform equivalent),
not aggregate memory across TP ranks; use host telemetry for the latter.

## Qualification

Use fresh shape-compatible cache keys and hold model revision, source pins,
weights, dtype, TP/EP, batch size, buckets, compiler flags, prompts, and trace
concurrency constant.

```bash
# Baseline: diagnostics unchanged, metrics enabled.
VLLM_NEURON_TRACE_METRICS=1 \
VLLM_NEURON_FAST_TRACE=0 \
STACK_CACHE_KEY=k3-safe-trace-baseline-<source-sha> \
make run MODEL=kimi-k3 STACK_MODE=source

# Candidate: successful FX text dumps suppressed.
VLLM_NEURON_FAST_TRACE=1 \
STACK_CACHE_KEY=k3-safe-trace-candidate-<source-sha> \
make run MODEL=kimi-k3 STACK_MODE=source
```

Accept the candidate only if final cache keys, normalized HLO hashes, NEFF
hashes, cache metadata, serving output fingerprints, and correctness results
match. Compare host-level peak memory and cold-start wall time externally;
per-child receipts cannot establish a TP64 aggregate peak.
