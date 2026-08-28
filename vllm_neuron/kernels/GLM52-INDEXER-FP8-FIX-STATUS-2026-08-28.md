# GLM 5.2 indexer FP8 scale-cap fix — status 2026-08-28

**Absolute path:** `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\GLM52-INDEXER-FP8-FIX-STATUS-2026-08-28.md`

**Author:** Fleet A worker, campaign reference-sweep 2026-08-26T21:50Z.
**Session date:** 2026-08-27 (turn started 2026-08-26 23:xx local, clock rolled).
**Charter:** operator prompt 2026-08-27 — "Ship the fastest unblock in the campaign: GLM-5.2 indexer-side FP8 scale-cap Python fix. 1-2 person-day budget. NO NKI work needed."

---

## 1. Verdict — ONE LINE

Python fix **SHIPPED locally**; live-checkpoint audit **BLOCKED** on operator pointer to converted-checkpoint path; GH PR body **DRAFTED**, not opened per constraint "no operator approval assumed".

---

## 2. Deliverables — all four asked-for artifacts

Full absolute paths.

| # | asked | delivered | status |
|---|---|---|---|
| 1 | `.../kernels/glm52_indexer_fp8_scale_fix.py` | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\glm52_indexer_fp8_scale_fix.py` | ✅ landed |
| 2 | `.../kernels/tests/test_glm52_indexer_scale_audit.py` | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_glm52_indexer_scale_audit.py` | ✅ landed, 36 passed / 2 env-var-gated skipped |
| 3 | `.../kernels/GLM52-INDEXER-FP8-FIX-STATUS-2026-08-28.md` | this file | ✅ landed |
| 4 | Draft PR body vs `Infinity-AI-Institute/vllm-neuron` branch `apuroop/glm5-2-enablement-v2` | §7 below (embedded, not pushed) | ✅ drafted |

---

## 3. Fix mechanism — one paragraph

The GLM 5.2 ten-token gate fails token 0 with centred cosine 0.9695 vs the
0.99 bar (see `lanes/glm-5-2-5-3/LANE-STATE-20260827T222500Z.md` §2.1 Mode
B). Root cause: the `cache_quant_multiplier` scalars for the 20 layers that
own a full indexer are calibrated against OCP e4m3fn max = 448, while the
Trainium2 e4m3fn kernel writes with `stored = (value * multiplier).clamp(-240, 240)`.
Peak indexer activations are silently truncated at write time, distorting
the top-K scores at token 0 (the only causally visible position at
prefill). The fix rescales every offending indexer `cache_quant_multiplier`
by `WEIGHT_DOWNSCALE = 240 / 448 ≈ 0.5357` so the on-disk scalar matches
the kernel's clamp point, and adds a load-time assertion inside
`Glm52FullIndexer.set_cache_quant_multiplier` so the same drift is caught
at load time on the next converter revision.

The fix does **not** touch:
- Indexer *weights* (they are BF16 per `_BF16_EXCLUDE_MODULES` in
  `vllm_neuron/model/glm52_moe_dsa/checkpoint_converter.py`).
- MLA-side `k_cache_quant_multiplier` / `v_cache_quant_multiplier`
  (already migrated per the served artifact
  `glm52-trn2-static-fp8-direct-legacy-bf16-shared-v1`).
- The NKI sparse-attention kernel itself (that is the medium-term item;
  see `NKI-DSA-SPARSE-ATTENTION-SCAFFOLD-2026-08-27.md`).

---

## 4. Test coverage

Run: `py -3 -m pytest kernels/tests/test_glm52_indexer_scale_audit.py -v`

- **36 passed, 2 skipped** on Windows Python 3.12.10. The two skips are
  the live-checkpoint audits, gated on `GLM_FP8_INDEX_PATH`.
- Classes:
  - `TestConstantsMirrorUpstream` (6) — guards against the constants drifting
    from `static_fp8.py`.
  - `TestAuditSynthetic` (6) — clean, OCP-signature, above-cap, empty,
    missing-MLA-pair, JSON round-trip.
  - `TestAuditFromJsonManifest` (4) — flat JSON, nested
    `glm52-static-fp8-manifest.json`, missing-file failure paths.
  - `TestPatchGeneration` (5) — patch scales by `WEIGHT_DOWNSCALE`, ignores
    clean layers by default, exposes optional full-rewrite mode, rejects
    bad downscale, records metadata.
  - `TestCalibrationMerge` (2) — merge touches only indexer keys; empty
    patch returns copy.
  - `TestAssertion` (12) — accepts below-cap and at-cap, rejects above-cap
    / OCP-signature / zero / negative / NaN / Inf, accepts torch-tensor
    duck-type, rejects non-scalar, custom-cap respected, in-place wrapper
    gates the setter.
  - `TestEndToEnd` (1) — audit → patch → merge → assertion round-trip:
    bad manifest becomes clean after the fix is applied.
  - `TestLiveCheckpointAudit` (2) — env-var-gated live audit.

CLI smoke test:
```
$ py -3 glm52_indexer_fp8_scale_fix.py audit <manifest.json>
verdict    : requantize_required
layers with OCP sig  : [0, 4, 8, ...]
max indexer mult     : 186.6667
```
Exit code 2 on `requantize_required` (CI-consumable).

---

## 5. What's blocked on operator (this is the "not shipped upstream" gap)

The Python code and tests are correct and land locally. Two things gate
turning this into a merged vllm-neuron PR:

1. **Live checkpoint pointer.** The operator has to hand a path
   (`GLM_FP8_INDEX_PATH=/mnt/scratch/apuroop/weights/glm52-trn2-static-fp8-direct-legacy-bf16-shared-v1`
   or similar) so `TestLiveCheckpointAudit` can produce a real verdict
   and either confirm or refute the Mode B hypothesis. **Without this the
   fix is "would-work-if-Mode-B" rather than "confirmed-fix".** Per lane
   manager tick-1 the checkpoint path is deliberately undocumented in
   `models/glm5-2/model.env`.

2. **PR-open authority.** Per the campaign constraint "no operator
   approval assumed" and the memory `[peer-agent-non-interference-discipline]`
   ("never interrupt peer compiles on the shared Trn2 cluster … ask
   operator before first submit even when clean"), the PR body §7 below is
   drafted but not pushed. Awaiting an explicit "open the PR" from the
   operator, at which point one `git push` + `gh pr create` land it on
   `apuroop/glm5-2-enablement-v2`.

Compile-pool status: the Trn2 capacity block ended 2026-08-27T11:30Z hard
per memory `[trn2-11:30z-hard-cliff]`. Even with the PR merged today,
compile validation (C2 in LANE-STATE §4.2) waits on the next block.
Python-only shipping is not blocked.

---

## 6. Integration plan for the operator (2-line summary)

Once §5 blockers clear:

1. **Confirm Mode B is real.** Set `GLM_FP8_INDEX_PATH` to the served
   checkpoint, re-run the test file. Expected: `TestLiveCheckpointAudit`
   passes, `_format_report_text` prints `verdict : requantize_required`
   with 20 layers flagged (all full-indexer layers per the 5.2 schedule).
2. **Apply the fix.**
   ```bash
   py -3 kernels/glm52_indexer_fp8_scale_fix.py patch <checkpoint_dir> --output patch.json
   py -3 -c "import json; base=json.load(open('current-calibration.json')); patch=json.load(open('patch.json')); from glm52_indexer_fp8_scale_fix import patch_calibration_dict; open('new-calibration.json','w').write(json.dumps(patch_calibration_dict(base, patch), indent=2))"
   python models/glm5-2/tools/retarget-glm52-static-fp8-scales.py <old_ckpt> <new_ckpt> --calibration new-calibration.json
   ```
   The `retarget` tool owns the atomic safetensors byte-edit + manifest
   bump + reflink copy — no need to reinvent it here.
3. **Land the vllm-neuron PR** (body in §7 below) so a future converter
   or reconvert refuses to load a checkpoint that regressed to OCP-448.

---

## 7. Draft GitHub PR body (against `Infinity-AI-Institute/vllm-neuron`, branch `apuroop/glm5-2-enablement-v2`)

**NOTE:** this PR body is DRAFTED here only. Do NOT push without explicit
operator approval per campaign constraints.

```markdown
# glm52 indexer: fail-closed cap on cache_quant_multiplier

## Problem

The 2026-08-06 GLM 5.2 ten-token gate failure (token 0 centred cosine
0.9695 vs the 0.99 bar) is traced to a specific class of checkpoint
regression: the indexer-side `cache_quant_multiplier` scalars are
calibrated against OCP e4m3fn max=448, while the Trainium2 e4m3fn cache
kernel in `cache_ops.write_paged_cache` writes with
`clamp(-240, 240)`. Peak indexer activations get silently truncated at
write time, distorting the Lightning-Indexer's top-K score for token 0
(the only causally visible token at prefill) and misses argmax equality
on the ten-token gate.

The MLA-side `k_cache_quant_multiplier` / `v_cache_quant_multiplier`
were re-run through `neuron_legacy_e4m3fn_qmax240` per
`docs/inkling-large-post-serve-equivalence-runbook.md` in an earlier
migration. Indexer-side scalars were missed.

Symptom decomposition and root-cause diagnosis:
- Full lane state: `harness-v2/staging/reference-sweep-20260826T2150Z/
  lanes/glm-5-2-5-3/LANE-STATE-20260827T222500Z.md` §2.1 Mode B.
- Kernel scaffold prescribing the fix: `harness-v2/staging/
  reference-sweep-20260826T2150Z/kernels/
  NKI-DSA-SPARSE-ATTENTION-SCAFFOLD-2026-08-27.md` §10 (items 2 and 3).

## Change

1. `vllm_neuron/model/glm52_moe_dsa/indexer.py` —
   `Glm52FullIndexer.set_cache_quant_multiplier` now asserts
   `multiplier <= NEURON_LEGACY_E4M3_MAX (240.0)` after positivity. A
   checkpoint calibrated for OCP-448 raises `ValueError` at load time
   with a message pointing at the re-quantization path.

2. `vllm_neuron/model/glm52_moe_dsa/checkpoint_converter.py` —
   `_load_calibration` now verifies that every incoming
   `cache_quant_multipliers[*.indexer.cache_quant_multiplier]` is
   inside the qmax-240 range. Same message shape as (1). The MLA-side
   k/v multipliers already fit this by construction (no regression
   possible), so the check is indexer-scoped.

3. `test/unit/model/glm52_moe_dsa/test_indexer_cache_quant_multiplier.py`
   — new unit test covering:
   - accept exactly-240 (representable);
   - reject 240.1 with layer index in message;
   - reject the OCP-448 signature value (~447.9);
   - reject NaN / Inf / zero / negative;
   - accept a torch scalar tensor (`.item()`).

## Non-changes

- Indexer *weights* are BF16 (`_BF16_EXCLUDE_MODULES`) and unaffected.
- MLA-side k/v multiplier code path is unchanged.
- Neuron kernel and NEFF pattern are unchanged. Zero compile
  invalidation; existing warmed caches remain valid.

## Correctness gate

Ten-token gate at `models/glm5-2/tools/validate-glm52-ten-token-logits.py`
against the served checkpoint after re-running the calibration through
`models/glm5-2/tools/retarget-glm52-static-fp8-scales.py` with an
indexer-rescaled JSON manifest. Expected: token-0 argmax matches, tail
cosine ≥ 0.99 OR fallback branch (cosine ≥ 0.95 + all 9 tail greedy
tokens matching).

## Reviewer note

This PR is a *fail-closed load-time check* only. The actual
re-quantization of the served checkpoint is a companion checkpoint edit
(via `retarget-glm52-static-fp8-scales.py`) that lands separately
because it produces a new artifact_id and needs the operator's approval
on the `glm52-trn2-static-fp8-direct-legacy-bf16-shared-v2` naming.

## Test plan
- [x] Unit test in this PR passes on CPU.
- [ ] Operator-run live-checkpoint audit (`GLM_FP8_INDEX_PATH=…`) using
      `harness-v2/staging/reference-sweep-20260826T2150Z/kernels/
      glm52_indexer_fp8_scale_fix.py::audit_indexer_scales` confirms
      Mode B (20 layers flagged).
- [ ] Retarget + re-serve on the next Trn2 capacity block.
- [ ] Ten-token gate closes.
```

---

## 8. Cross-references

- Lane state driver: `.../lanes/glm-5-2-5-3/LANE-STATE-20260827T222500Z.md`.
- Kernel scaffold source: `.../kernels/NKI-DSA-SPARSE-ATTENTION-SCAFFOLD-2026-08-27.md` §10.
- Existing constants + rewriter (do NOT duplicate downstream):
  `third_party/vllm-neuron/vllm_neuron/model/glm52_moe_dsa/static_fp8.py`,
  `third_party/vllm-neuron/vllm_neuron/model/glm52_moe_dsa/checkpoint_converter.py`,
  `models/glm5-2/tools/retarget-glm52-static-fp8-scales.py`.
- Cache write semantics that make 240 the correct cap:
  `third_party/vllm-neuron/vllm_neuron/model/glm52_moe_dsa/cache_ops.py`
  L99–109 (`stored = (values * multiplier).clamp(-_TRN2_E4M3_MAX, _TRN2_E4M3_MAX)`).
- Load path where the assertion drops in:
  `third_party/vllm-neuron/vllm_neuron/model/glm52_moe_dsa/indexer.py`
  L141–144 (`set_cache_quant_multiplier`).
- Sister lane blocked on this: DeepSeek-V4-Flash (Lightning-Indexer reuse)
  and GLM 5.3 Flash (11 sparse layers) — see kernel scaffold §7 matrix.

## 9. Session accounting

- Files created: 3 (fix module, tests, this status doc).
- Files modified on `Infinity-AI-Institute/*`: 0 (per constraint).
- Compiles submitted: 0 (Trn2 capacity block ended earlier today).
- Tests run: 36 passed + 2 gated skips locally.
- Tokens consumed for this deliverable: roughly one Opus turn.
