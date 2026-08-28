# GLM-5.2 ten-token gate still open — 2026-08-28

Provenance: Infinity-AI-Institute/trainium PR #13 (deepseek-v4-flash-base)
Callsign: codex-bravo

## Verdict

The PR #13 static defect audit is clean, but the ten-token correctness gate is
not closed. A fresh post-audit cosine cannot be measured from the supplied
CPU-only test because its live case requires a pre-generated logits artifact,
and no such artifact exists in the referenced staging tree or the local GLM
worktrees.

This receipt deliberately does not turn the synthetic test pass into a model
correctness claim.

## Baseline versus post-audit

| Measurement | Token-0 centred cosine | Status |
|---|---:|---|
| Recorded pre-audit baseline | 0.9695 | FAIL (`< 0.99`) |
| Fresh post-audit measurement | unavailable | live artifact missing |
| Last comparable number carried forward after audit | 0.9695 | still the only measured value; not a fresh run |

The two already-landed DSA fixes (`de91cb4`, `320f828`) and this audit's MoE
top-k signed-int64 fix (`1bb758a`) change source semantics, but none generates
the required 10-by-vocabulary reference/candidate logits. With device compile
explicitly out of scope and no captured candidate supplied, there is no honest
way to claim that any of the three defects moved 0.9695.

## Gate execution receipt

The handoff-prescribed invocation failed before collection because the actual
test does not define `--model-branch` or `--vllm-neuron`:

```text
error: unrecognized arguments: --model-branch ... --vllm-neuron ...
```

The supported invocation was then run without those options:

```text
.....s
5 passed, 1 skipped in 0.46s
SKIPPED: GLM_TEN_TOKEN_ARTIFACT not set
```

The five passing cases validate only the gate evaluator against synthetic
vectors. The skipped case is the only live-artifact comparison. The test reads
`GLM_TEN_TOKEN_ARTIFACT`; it does not load the model branch or execute a model.
It also checks only token-0 argmax equality, not the required token-0 centred
cosine threshold. Consequently, even supplying the JSON would not make this
particular staging test sufficient evidence for the user-specified bar.

## PR #13 audit result

- Defect 1: zero executable `split` calls remain across all 20 live module
  files; the Q/K projections use explicit slices.
- Defect 2: zero bound-method `.topk()` calls remain. Qualified Trn2 DSA and
  MoE routing use the NKI top-k backend; all remaining torch calls are
  module-level CPU/reference fallbacks.
- Defect 3: every DSA and MoE top-k result is normalized to signed int64
  immediately. Intentional int32 kernel ABI uses are justified in
  `PR13-DEFECT-AUDIT-2026-08-28.md`.

Focused unit verification: 17 tests passed for attention, indexer, MoE routing,
and expert-kernel contracts. A broader sparse-MLP collection attempt is blocked
on this Windows host by the unavailable Neuron `nki` package.

## Top two remaining hypotheses

1. **NxDI container MoE blockwise-matmul configuration drift.** The GLM-5.2
   launcher/config tree has no occurrence of
   `use_shard_on_intermediate_dynamic_while=True`, despite the campaign rule
   that this flag must be set before `InferenceConfig` construction. If the
   effective runtime config falls back to the alternate blockwise path, expert
   dispatch/layout behavior can diverge independently of the now-clean DSA
   top-k plumbing. First next check: capture the effective `NeuronConfig` from
   the exact candidate artifact and prove the flag before any future compile.

2. **`normalize_static_fp8_weight_format(None)` silently selects OCP-448.** In
   `static_fp8.py`, `None` returns `ocp_e4m3fn_qmax448`. The qualified served
   artifact is expected to declare `neuron_legacy_e4m3fn_qmax240`, and the
   factory/benchmark contain downstream marker checks, but an omitted marker
   during an earlier config or conversion step can still select OCP scaling and
   introduce the documented double-rounding/scale drift. First next check: run
   the existing live checkpoint audit with the operator-provided converted
   checkpoint pointer and bind its config, manifest, and logits artifact hashes.
   Do not rewrite the normalizer in this audit branch; that remains a Fleet A
   core change unless the live evidence implicates it.

## Required input to close the gate

Provide either:

- `GLM_TEN_TOKEN_ARTIFACT` pointing to a JSON object with 10 reference and 10
  candidate logit rows plus their greedy token IDs; or
- the original `.pt` reference/candidate pair used by
  `models/glm5-2/tools/validate-glm52-ten-token-logits.py`, from which the JSON
  can be produced without teacher forcing.

The live converted-checkpoint path remains an operator ask as well; it is
needed to test the FP8 hypothesis rather than infer it.
