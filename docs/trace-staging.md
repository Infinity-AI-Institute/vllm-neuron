# Representative-rank trace staging

Large tensor-parallel models can spend a long time in Dynamo and FakeTensor
model tracing before the graph-capture backend receives an FX graph. HLO count
is therefore not a liveness signal during that interval. If the failure is
model-wide, allowing every rank to enter the same trace only multiplies CPU and
memory use before they all report the same exception.

This experiment is off by default. It nominates one global rank to validate the
Python/FakeTensor portion first. Other ranks wait in a CPU object broadcast.
The representative uses the existing fork pool and meta swap, but the capture
backend returns the existing `CaptureComplete` sentinel as soon as Dynamo hands
it an FX graph. The child is then reaped.

By default the preflight intentionally performs no FX passes, FX-to-HLO
lowering, HLO write, compiler invocation, or cache write. After it succeeds,
every rank, including the representative, executes the unmodified normal
extraction. No graph or Dynamo state is transferred between ranks, and
leader-only graph reuse semantics are not enabled or changed.

For the default-off same-process lowering experiment only, set
`VLLM_NEURON_TRACE_PREFLIGHT_LOWER_HLO=1` and provide a dedicated local path in
`VLLM_NEURON_TRACE_PREFLIGHT_HLO_RECEIPT_DIR`. The representative child then
continues its live FX `GraphModule` through FX-to-HLO and writes `graph.hlo`,
FX/input ABI hashes, lowering metadata, and an explicit diagnostic receipt.
It performs no cache lookup/publication, writes no NEFF, enables no runtime
bypass, and still exits before normal all-rank extraction.

## Integration

Use a run-specific milestone directory. For the K3 T256 retry, whose target job
list is one prefill shape followed by one decode shape:

```bash
export VLLM_NEURON_TRACE_PREFLIGHT_RANK=0
export VLLM_NEURON_TRACE_PREFLIGHT_JOBS=2
export VLLM_NEURON_TRACE_PREFLIGHT_TIMEOUT_SECONDS=14400
export VLLM_NEURON_TRACE_PREFLIGHT_HEARTBEAT_SECONDS=300
export VLLM_NEURON_TRACE_MILESTONE_DIR=/scratch/trainium-logs/kimi-k3/$RUN_ID-trace
```

The staging rendezvous uses a dedicated all-rank process group created before
the representative starts tracing. Its four-hour default deadline applies only
to this control-plane broadcast; the default process group and model
collectives retain their existing timeout. Parked ranks emit
`preflight_wait_heartbeat` every five minutes without resetting that deadline.
The heartbeat must be shorter than the deadline, and both values are validated
before tracing begins.

Keep the existing `VLLM_NEURON_TRACE_RANK_CONCURRENCY` setting. The preflight
child uses the same host slot protocol, so it cannot overlap the waiting ranks'
normal trace work. Remove `VLLM_NEURON_TRACE_PREFLIGHT_JOBS` to stage every job,
or set it to `1` when the first shape is known to cover a model-wide failure and
successful-start latency is more important than shape-specific preflight
coverage.

The files `rank-0.jsonl` through `rank-(world_size-1).jsonl` contain schema-v1
records. Important events are:

- `preflight_selected`, `preflight_waiting`,
  `preflight_control_group_ready`, `preflight_wait_heartbeat`,
  `preflight_rendezvous_failed`, `preflight_failed`, and `preflight_released`;
- `pool_started`, `host_slot_waiting`, `host_slot_acquired`, and
  `pool_completed` or `pool_failed`;
- `meta_swap_started`, `meta_swap_completed`, `job_started`, `job_completed`,
  and `job_failed`;
- `capture_backend_reached`, which proves that Python/FakeTensor tracing
  finished for that preflight job.

If a stream ends at `job_started` without `capture_backend_reached`, the worker
is still in or failed inside Dynamo/FakeTensor model tracing; zero HLOs is then
expected, but elapsed time between milestones makes the lack of progress
explicit. If `capture_backend_reached` exists, the preflight job passed its
intended boundary. Normal-stage events then locate later FX/HLO failures by job.

## Failure propagation and limitations

The representative catches ordinary trace exceptions and broadcasts a bounded
error payload. Waiting ranks raise before forking any normal trace children. A
hard process death is bounded by the preflight-only process group's deadline.
Changing that deadline does not weaken ordinary collective timeout handling.

Preflight success does not prove rank-specific graph correctness, FX-pass
correctness, HLO equivalence, or compilation correctness. Those remain covered
by the subsequent normal all-rank extraction. The feature makes no
rank-invariant-HLO assumption.

On a failing cold start, CPU and memory exposure is reduced from up to the
configured host-wide trace concurrency to one representative rank and the
failure is reported once. On a successful cold start, wall time increases by
one representative Python/FakeTensor trace for each staged job. Milestone I/O
is one short append per boundary and is disabled when its directory is unset.
Use preflight while qualifying a new model or risky graph change, then disable
it once the path is stable.
