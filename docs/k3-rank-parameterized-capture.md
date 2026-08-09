# Kimi K3 one-capture rank-parameterized lowering qualification

Status: CPU-only prototype, default off, not integrated with capture or runtime.
Source bases reviewed: vLLM-Neuron `2c2197e` plus integrated cache hardening
`342a388`; NDI `c9f5da2`.

## Decision

One Dynamo/FX capture followed by per-rank FX-to-HLO lowering is technically
plausible, but the current K3 model is not yet safe to use that way. The safe
reuse boundary is narrower than “all ranks have the same tensor shapes.” It
requires every rank-dependent computation to be tensor dataflow, identical FX
operator/control-flow semantics, identical input shape/dtype/stride/alias ABI,
identical collective replica groups, and process-valid NKI registry state.

The prototype in `vllm_neuron/compile/rank_parameterized_capture.py` exercises
that boundary without changing production behavior. It requires an exact env
opt-in, one-element integer rank tensor, a source-audit digest, K3 rank metadata,
resolved replica groups, and ABI equality. It rejects semantic graph, input,
alias, collective, source-audit, or K3 ownership changes. It deliberately
rejects cross-process reuse because FX/NKI registry transport is not qualified.

### Critical-path go/no-go

**NO-GO: serialize the current full K3 `GraphModule` from the disposable
preflight child and lower it in another rank/process.** `GraphModule` itself is
pickleable, including an NKI HOP call target, but that is not the complete
lowering program:

- The NKI HOP stores only `kernel_idx` and `constant_args_key`; FX-to-HLO calls
  the process-global `NKIRegistry` to recover the kernel function, argument
  names/defaults, and trace-created constant arguments. A CPU experiment
  pickled/restored the HOP graph successfully, then reset the registry; lookup
  failed with `KeyError: 0` and the registry remained empty. The pickle did not
  carry it. Both identifiers are process-derived: kernel indices depend on
  registration order and constant keys use Python `hash(...)`. The `342a388`
  cache normalization removes their volatility from the cache identity; it does
  not make them usable for HLO lowering in a fresh process.
- Collective FX nodes carry process-local group-name strings. The replica-group
  pass resolves those strings through the consumer process's c10d registry.
  vLLM 0.21 creates names with a per-process counter (`name:N`), so equal names
  require identical construction order and still require a live local group.
  Neither the group handle nor its registration is in the FX pickle.
  Additionally, all-gather/reduce-scatter channel IDs use
  `hash(group_name) & 0x7fffffff`, so independently lowered HLO also requires a
  controlled Python hash seed or canonical channel-ID replacement.
- Rank-local tensor payloads are not the transport blocker: Dynamo lifts
  ordinary parameters into `example_inputs`, and HLO placeholder creation reads
  shape and dtype. The blockers are non-input lowering state plus today's
  Python rank constants.

**GO, conditional: keep the captured FX graph in the preflight child and lower
there before the child exits.** This avoids transporting NKI and collective
state. It becomes sound after embedding and EP ownership arithmetic use the
existing rank tensor and a 64-rank input-ABI manifest proves that weights,
buffers, KDA/MLA state, and aliases differ only in payload. First lower ranks
0/47/48/63 and compare HLO semantics before deduplicating.

`FxGraphHandoffArtifact` is the smallest fail-closed serialization prototype.
It stores the FX pickle plus semantic digest, input/alias ABI, rank slot, source
audit, K3 ownership contract, NKI signatures, and replica groups. Portable CPU
graphs round-trip and bind distinct weight payloads. A cross-process artifact
with any NKI or collective dependency is rejected before lowering; tampered
graph bytes are rejected by digest. This makes the current K3 no-go executable
rather than documentary.

## Rank-dependence map

| Value | Current representation | Consequence | Required one-capture form |
|---|---|---|---|
| TP/EP rank | Python `groups.rank` / `groups.ep_rank`; runner also supplies a rank tensor | Model construction and downstream Python constants can specialize FX | Use the existing runner rank tensor for all device arithmetic; retain Python rank only for load-time checks |
| Expert ownership | `start = config.ep_rank * 14` in `ep64.py` | `14 * rank` is an FX literal today | Compute start from rank tensor; metadata must prove `[14r, 14(r+1))` |
| Source-rank routing metadata | `torch.full_like(..., config.ep_rank)` | Rank literal is embedded in the dispatch graph | Broadcast/expand the rank tensor |
| Embedding ownership | `PackedQ80Embedding.vocab_start = rank * 2560` and subtract in forward | Per-rank FX literal today | Compute `2560 * rank_tensor` in forward |
| Expert bank `expert_start` | Python attribute | Mostly construction/load validation; unsafe if read by device forward | Keep load-time only or replace any forward read with rank tensor |
| LM-head/embedding/expert weights | Dynamo lifts parameters as example-input placeholders (verified by CPU capture) | Payloads may differ if full tensor ABI is identical | Bind rank-local tensors per rank; never leave rank-local tensors as `get_attr` constants |
| Heads 0..95 padded to 128 | Loader emits fixed-size shards; ranks 0..47 have two real heads, ranks 48..63 have two zero heads | Boundary 47/48 changes payload/source bytes, not declared output size | Accept only with identical shape/dtype/stride/alias ABI and metadata recording 2/0 real heads |
| Collectives | FX nodes carry process-group names; FX pass resolves names to rank lists | A process-local name or changed list can change HLO replica groups/channel behavior | Validate every name resolves to the same ordered `[0..63]` group before each lowering |
| NKI kernels | FX HOP nodes carry registry index, grid, constants, backend config; lowering consults a process-global registry | A deserialized FX graph alone is insufficient in another process | Lower in capture process initially; qualify explicit registry-safe transport separately |

The 47/48 classification is documented in NDI architecture and rank-view tests:
rank 47 consumes real bytes, rank 48 emits the same output size entirely from
zero fill. Dense padded tensors can also have partial/all-zero tail-rank
payloads. Therefore “padding only changes payload” is valid only after checking
every runtime input ABI, not from architecture arithmetic alone.

## Lowering/cache/parallel-compile implications

Documented in current source:

1. Dynamo calls the capture backend with an FX graph and flattened example
   inputs.
2. The backend hashes normalized FX semantics, resolved collective replica
   groups, input shape/dtype/stride, stack versions, NKI identity, and compiler
   arguments.
3. FX passes mutate the graph, then `convert_fx_to_hlo` executes it on XLA
   placeholders. XLA values not corresponding to example inputs are inlined as
   constants.
4. Capture writes `graph.hlo` under a rank subdirectory. Parallel compile only
   starts after HLO exists; for a shared hash it chooses one rank directory,
   removes sibling rank directories, compiles one NEFF, and moves it to the hash
   root.

The cache key excludes input payload values but includes input
shape/dtype/stride, normalized FX semantics, resolved replica groups, compiler
arguments, stack versions, and NKI identity. Distinct rank-local weights can
share a key only when they are FX inputs with identical ABI. A rank-specific
Python integer changes FX/hash semantics. A rank-specific tensor `get_attr` or
other non-input XLA value is inlined as an HLO constant and cannot safely share
HLO. The prototype rejects every tensor `get_attr`; model tensors must be lifted
inputs.

That collective guarantee is conditional on successful local resolution. The
current cache helper catches process-group resolution failure and returns
`None`, omitting replica groups from the key. Consequently a deserialized K3 FX
graph in a process without the exact c10d registry can produce an incomplete
cache identity instead of a hard failure. The handoff prototype is stricter and
rejects this state before hashing/lowering.

Inference from those facts: once rank arithmetic is an input and every other
semantic/ABI field matches, all ranks should hash to one graph and may need only
one HLO/NEFF, not 64 rank-specific HLOs. That stronger result is not yet proven.
The conservative first integration should still lower all 64 rank bindings,
compare canonical HLO semantics, and only deduplicate compilation after the
HLO census agrees.

## What the CPU prototype proves and does not prove

It proves:

- Dynamo can lift model parameters as graph inputs while leaving a rank tensor
  as dataflow, enabling distinct rank-local payloads under one FX structure.
- One captured FX graph can be cloned and passed to 64 sequential lowerer calls.
- Wrong `14 * rank`, the 47/48 metadata boundary, shape/dtype/stride/alias
  changes, Python-scalar rank, source-audit changes, replica-group changes, and
  supplied candidate-graph changes can fail closed before lowering.
- A plain lifted-parameter FX graph can be serialized, restored, and rebound to
  a different rank's weight payload under the same ABI.
- An NKI HOP target surviving pickle does not restore its kernel/constant
  registry; the handoff rejects that incomplete cross-process state.

It does not prove:

- Current K3 has removed all Python rank literals (it has not).
- A full K3 FX GraphModule can be serialized between processes with NKI HOP and
  collective state intact.
- Full K3 rank input ABI is identical across all 64 loaded models.
- Rank-parameterized K3 produces equivalent HLO/NEFF or correct device output.
- Runtime workers can bypass their own Dynamo compilation using a leader
  artifact; current cache lookup still happens after Dynamo reaches the backend.

## Next smallest safe experiment

1. The isolated NDI prototype `codex/k3-rank-tensor-dataflow` now replaces
   embedding `vocab_start`, EP dispatch `14 * ep_rank`, and `source_ranks` with
   arithmetic from the existing rank tensor behind the exact default-off flag.
   CPU execution proves integer and tensor EP dispatch are exact at ranks
   0/47/48/63; Dynamo captures ranks 47 and 48 as identical 98-node FX graphs.
   Finish a packed-embedding capture test before integration.
2. Produce a 64-rank input manifest without Dynamo: tensor name, role,
   shape/dtype/stride/layout/device/alias class plus K3 ownership metadata.
   Reject on any mismatch. This specifically verifies the padded-head and dense
   tail-padding cases rather than assuming them.
3. On Trainium only after those pass, change the representative preflight child
   to continue from captured rank-0 FX directly into lowering. Do not send an FX
   pickle to another process. Lower bindings 0, 47, 48, and 63 in that same
   process. Compare canonical HLO instructions, constants, custom-call configs,
   aliases, channel IDs, and replica groups.
4. If the four-rank boundary census matches, expand lowering to 64 ranks. Keep
   NEFF compilation deduplication disabled until all HLO hashes agree and the
   existing correctness gates pass.
5. Separately design a runtime artifact loader. Leader-only extraction by itself
   does not remove Dynamo work from normal rank workers.

## Wall-time and memory model

Observed measurement supplied by the current K3 trace investigation: one rank
reaches the capture backend in about 94 minutes with about 52.99 GiB peak RSS.

- Current wave-4 all-rank extraction: `ceil(64 / 4) * 94 min = 1,504 min`, about
  25.1 hours of capture wall time per graph shape, before FX-to-HLO and NEFF
  compilation. Approximate concurrent child RSS is `4 * 52.99 = 211.96 GiB`,
  plus parent/model/runtime overhead.
- One capture: about 94 minutes of observed capture work. This is a 16x reduction
  in the capture-dominated wave-4 wall component, not a promise for end-to-end
  compilation.
- Added work: up to 64 sequential FX-to-HLO lowerings and graph clones. Full-K3
  lowering time and clone/RSS cost have not been measured, so an end-to-end
  speedup cannot yet be stated. Lowerings can be bounded to one at a time to
  avoid recreating wave-4 memory pressure.
- If all 64 HLOs are proven identical, one HLO/NEFF path could remove most of
  that added work. This is an inference and must not be enabled from FX equality
  alone.

The highest-leverage result is therefore conditional but large: parameterizing
three known Python rank uses could remove roughly 24 hours from each cold
wave-4 capture cycle. The gating engineering is modest; proving full-K3 ABI,
HLO, NKI, collective, and runtime-loader equivalence is the material work.
