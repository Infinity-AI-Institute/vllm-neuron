# GLM-5.2 DSA PR #13 defect audit — 2026-08-28

Provenance: Infinity-AI-Institute/trainium PR #13 (deepseek-v4-flash-base)
Callsign: codex-bravo

## Scope and method

The handoff named 15 files. The live directory contains those 15 plus five
support modules (`__init__.py`, `artifact_preflight.py`, `cache_layout.py`,
`cache_ops.py`, and `weight_manifest.py`), so this audit covered all 20 Python
files in `vllm_neuron/model/glm52_moe_dsa/`.

Two independent scans were used:

1. Text search for `split(`, `.topk(`, `torch.topk(`, `torch.int32`, sentinel
   comparisons, and scatter calls.
2. Python AST inspection of every executable call. This distinguishes comments
   mentioning a forbidden pattern from live code and distinguishes module-level
   `torch.topk` from bound-method `Tensor.topk`.

## Defect disposition

### 1. List-of-sizes split on the last dimension

PASS: zero executable `torch.split` or bound-method `.split` calls remain in
all 20 files. `indexer.py` and `mla.py` use explicit slices for their logical
last-dimension fields. The only `split` text hits are explanatory comments
describing why slices are required.

### 2. Unsupported TopK lowering

PASS for the qualified Trn2 path: zero bound-method `.topk()` calls remain.
There are exactly three module-level `torch.topk` calls:

- `attention.py::glm52_index_topk`: CPU/reference helper, returned indices are
  normalized immediately to signed int64. It has no production model call site.
- `indexer.py::Glm52FullIndexer._select_topk`: the qualified static-FP8 Trn2
  path selects `vllm_neuron.functional.topk.topk`; the module-level
  `torch.topk` branch is the CPU/BF16 reference fallback. Both join before the
  signed-int64 conversion.
- `moe.py::select_glm52_experts`: the qualified static-FP8 Trn2 path selects
  the NKI rotational top-k; the module-level `torch.topk` branch is the CPU/BF16
  reference fallback. Both now join before the signed-int64 conversion.

No `AwsNeuronCustomLoweringType.apply_overridings("TopK")` initializer exists
in this repository. That override is not required by the qualified GLM-5.2
model graph because `factory.py` constructs the static-FP8 implementation and
`mla.py`/`sparse_mlp.py` select the NKI top-k backend for both DSA and expert
routing. The remaining `torch.topk` calls are module-level as required if an
external CPU/reference harness installs the PR #13 override.

### 3. Unsigned top-k index and sentinel wrap

PASS after this audit's residual fix:

- `attention.py` returns signed int64 immediately after `torch.topk`.
- `indexer.py` converts both NKI and torch top-k returns to signed int64; its
  full-context no-sort fast path now also returns signed int64.
- `moe.py` now converts both NKI and torch expert-selection returns to signed
  int64 before `torch.gather` or kernel dispatch.

Intentional int32 uses were reviewed and are safe:

- `expert_kernels.py` narrows already-validated, non-negative expert IDs only
  at the `moe_tkg` int32 ABI. The blockwise mapper's signed token IDs are not a
  TopK result; narrowing preserves its `-1` padding sentinel and
  `skip_token=True` consumes it.
- `model.py` narrows position tensors for Neuron position/DGE ABI requirements.
  Neither tensor is a top-k result or a sentinel-bearing sparse index.

## File-by-file result for the 15 handoff files

| File | Result |
|---|---|
| `attention.py` | module-level TopK + immediate int64; no split |
| `checkpoint_converter.py` | no audited patterns |
| `checkpoint_mapping.py` | no audited patterns |
| `config.py` | no audited patterns |
| `dense_mlp.py` | no audited patterns |
| `expert_kernels.py` | intentional int32 kernel ABI uses justified above |
| `factory.py` | no audited patterns; validates qualified static-FP8 path |
| `indexer.py` | slices only; NKI/module TopK join at int64; fast path int64 |
| `mla.py` | explicit slicing; full/shared indexer wiring verified |
| `model.py` | end-to-end wiring verified; intentional position int32 only |
| `moe.py` | residual top-k int64 normalization fixed |
| `parallelism.py` | no audited patterns |
| `shared_expert.py` | no audited patterns |
| `sparse_mlp.py` | NKI router selected for qualified static-FP8 path |
| `static_fp8.py` | no audited patterns; no normalization rewrite made |

The five additional support modules also have no executable split, top-k, or
suspect int32 conversion matching the audit patterns.

## `model.py` wire-in cross-check

`Glm52Model` builds every decoder layer through `Glm52DecoderLayer`, which
constructs `Glm52MlaAttention`. `Glm52MlaAttention` constructs
`Glm52FullIndexer` exactly when `config.indexer_types[layer_idx] == "full"`;
shared layers cannot instantiate an alternate indexer and instead propagate
`Glm52IndexShareState`. No wrapper or tensor-slicing macro re-emits split.

Sparse MLP routing constructs `Glm52ExpertRouter` with the NKI backend whenever
`static_fp8=True`, and `Glm52RoutedExperts` is the only route into the qualified
TKG/CTE expert kernels. There is no hidden fallback that calls a bound-method
TopK.
