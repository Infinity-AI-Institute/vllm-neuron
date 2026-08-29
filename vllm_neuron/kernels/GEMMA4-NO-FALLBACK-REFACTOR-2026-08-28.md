# Gemma-4 no-fallback refactor — 2026-08-28

**Author:** Claude Code worker agent (Fleet A, `gemma4-no-fallback-agent`), spawned to refactor `gemma4_cpu_fallback_replacement.py` per operator concern about the misleading file name + shrink scope now that AWS PR #172 has superseded triggers #1 and #2.

**Scope:** Rename + refactor the Gemma-4 mitigation module, author a universal no-CPU-fallback assertion test suite, update the README + MoE-dispatch status doc, land the follow-up commit on PR #4 (`worker/fleet-a-nki-kernels-2026-08-28`).

**Discipline anchor:** operator hard rule — "we never plan for CPU fallback."  The old file name (`gemma4_cpu_fallback_replacement.py`) read as if we ship a replacement CPU path.  The intent is the opposite: prevent fallback, and fail loudly when we can't.

---

## 1. Why the rename

The old name conflated two very different things:

* **What it does:** wire the mitigations that prevent Gemma-4-26B-A4B CPU fallback.
* **What the name suggests:** a Python-side implementation that we intentionally use *as* a CPU-side execution path.

The confusion was flagged by the operator.  A future maintainer glancing at the file tree would reasonably assume the second interpretation and, worst case, write code that treats CPU fallback as an acceptable outcome — the exact discipline the campaign forbids (Gemma-4 hit MFU 0.06% for precisely this reason).

**Rename:** `gemma4_cpu_fallback_replacement.py` → `gemma4_no_fallback_mitigations.py`.  Docstring now leads with:

> **Intent:** this module PREVENTS CPU fallback by wiring the AWS PR #172 upstream kernels for triggers #1 / #2 and shipping stopgaps for triggers #3 / #4.  It NEVER plans for a fallback, never authors a "CPU replacement path," and never accepts a silent CPU emission.

---

## 2. What PR #172 supersedes

AWS PR #172 is the NxDI contrib port of `google/gemma-4-26B-A4B-it` (text-only MoE sibling of Gemma-4-31B-IT / PR #106).  Snapshot at `C:\Users\apumu\AppData\Local\Temp\claude\C--Users-apumu-research-InfinityAI-gemma4-trn2-handoff\fe22bc9a-b8c7-4107-ac55-1d55ba4bd33a\scratchpad\aws-pr172\` (populated 2026-08-28 with the PR review copy).

### Trigger #1 — `head_dim > _MAX_D_HEAD=128`

**Superseded.**  PR #172 ships `nki_flash_attn_d256_swa.py` (SWA layers, `head_dim=256`, 25/30 layers) and `nki_flash_attn_large_d.py` (global layers, `head_dim=512`, 5/30 layers) — both reused verbatim from PR #106 with Stage 5 canonical-chat validation matching HF CPU bf16 at 100% token agreement for 11/12 greedy/sample combos.  Configuration: TP=8, bfloat16, seq_len=256, LNC=2, trn2.48xlarge.

**Effect on this module:**

* `HeadDimSignOffRequired` exception and `SIGN_OFF_REQUIRED_HEAD_DIM` sentinel removed.
* `Gemma4AttentionGeometry` dataclass and `GEMMA4_26B_A4B_ATTENTION_GEOMETRY` constant removed — geometry is pinned by PR #172 at HF revision `24548b62aa021d562695c04aaf7758a1ea47990b`.
* `set_gemma4_head_dim()` helper removed — no longer needed.
* `make_flash_attention_hybrid_sliding_global_kernel()` removed — replaced by the PR #172 adapter shim `import_pr172_flash_attention(variant="d256_swa" | "large_d")`.

### Trigger #2 — hybrid sliding + global KV manager off

**Superseded.**  PR #172 ships a subclassed `Gemma4KVCacheManager` in `modeling_gemma4_neuron.py` that handles the per-layer heterogeneous KV shapes (8×`head_dim=256` SWA vs 2×`head_dim=512` global, post-TP sharding).

**Effect on this module:**

* `enable_hybrid_kv_cache_manager()` helper removed — replaced by `import_pr172_kv_cache_manager()` adapter shim.
* `build_gemma4_layer_to_cache_size_mapping()` helper removed — PR #172 owns the mapping.

### Trigger #3 — vocab=262K > nc_find_index8 cap

**KEPT.**  PR #172 was validated at batch_size=1 only.  The `nc_find_index8` 16,384-partition cap on Gemma-4-26B-A4B (vocab 262,144) still bites at TP<=8 for B>128 decode, requiring `disable_argmax_kernel=True`.  `should_disable_argmax_kernel(vocab_size, tp_degree, batch_size)` remains the mitigation until Part B #2 `argmax_kernel_partitioned` (§B.9) lands.

The function is universal — Qwen3-30B-A3B TP=8 with vocab=151k also hits this at high B.  The file name is gemma4-scoped for lineage reasons; the mitigation is model-agnostic.

### Trigger #4 — `(GLU, GELU_TANH_APPROX)` activation combo

**KEPT + verify against PR #172 on first fire.**  PR #172 uses NxDI `moe_v2` / `initialize_moe_module` directly and does not route through the fused-dispatch NEFF this campaign compiles.  The scaffold §B5 hazard — NKI kernel v16 silently fell back to `torch_blockwise_matmul_inference` on `(GLU, GELU_TANH_APPROX)` — is still in force whenever the fused-dispatch NEFF is the one on device.

`verify_activation_branch_coverage()` continues as the Tier-1 CPU-battery guard, greping `moe_dispatch.MoEActivation` for the `GELU_TANH_APPROX` branch and raising if it's missing.

---

## 3. Adapter shim contract

Both `import_pr172_flash_attention()` and `import_pr172_kv_cache_manager()` resolve their imports through a single env-var switch:

```
GEMMA4_USE_UPSTREAM_PR172=1  →  neuronx_distributed_inference.contrib.gemma_4_26b_a4b_it
(default)                    →  local vendored snapshot in scratchpad/aws-pr172/
```

**Failure mode:** if the target module cannot be resolved (neither upstream nor local), the shim raises `ImportError`.  It **never** returns a CPU stub.  A missing kernel is a HARD FAIL — the whole point of the rename is to make silent fallback impossible.

**Post-merge migration:** when PR #172 lands on the compile host, set `GEMMA4_USE_UPSTREAM_PR172=1` in the compile env.  No code change required — the shim is stable across the merge.

---

## 4. Universal no-CPU-fallback assertion tests

New file: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_no_cpu_fallback.py`

Three coverage layers, all universal (not Gemma-4-specific despite living in this kernels tree):

### 4a. Compile-log grep

Pure text search over the compile log.  Runs on any host, requires no Trn2.  Grep pattern `CPU_FALLBACK_GREP_PATTERNS` is exported from the module for downstream reuse:

```
falling back to cpu
torch_blockwise_matmul_inference
op fallback
cpu-side
emitting host code
nki\b.*\bnot (?:found|available)
partition cap exceeded
unsupported operation.*fallback
fell back to (?:cpu|host|torch)
host code emitted
```

Each alternative maps to a canonical Neuron-compiler or NxDI fallback marker.  Rationale documented in the module docstring.

Invocation:
```
pytest kernels/tests/test_no_cpu_fallback.py --compile-log /path/to/compile.log
cat compile.log | pytest kernels/tests/test_no_cpu_fallback.py --compile-log -
```

### 4b. NEFF-content assertion

For a landed compile artifact directory (`<slug>/logs/` and/or `<slug>/model.pt`):
* every `*.log` and `*.txt` in the tree is scanned with the same grep pattern;
* if `neuron-mlir-tool` or `neuron_cc.disasm` is on PATH, the NEFF is disassembled and grepped for unresolved `stablehlo` ops that lower to host.

Invocation:
```
pytest kernels/tests/test_no_cpu_fallback.py --artifact-dir /path/to/compiled/model_dir
```

### 4c. Runtime probe

Best-effort.  Runs `neuron-top --sample 10` during a served inference and verifies no host-CPU activity beyond driver polling (>20% CPU across samples fails).  Skips cleanly on any host where `neuron-top` is absent or Trn2 is not accessible.

Invocation:
```
pytest kernels/tests/test_no_cpu_fallback.py --runtime-probe
```

### 4d. Universal harness

The four smoke tests in `TestUniversalHarnessSmoke` run in the default `pytest -q` sweep with no options.  They exercise the grep pattern with known positive + negative examples, verify `find_cpu_fallback_matches` handles arbitrary `Iterable[str]` sources (including `io.StringIO` for stdin-piped logs), and confirm `gemma4_no_fallback_mitigations` imports cleanly with all four public helpers exposed.

### 4e. Discipline

Every future compile lane on any model must invoke this test module.  Add it to the compile driver's post-compile assertion chain alongside the existing Tier-1 CPU battery:

```
python compile_driver.py --model <slug> --tp <n> 2>&1 | tee compile.log
pytest kernels/tests/test_no_cpu_fallback.py --compile-log compile.log --artifact-dir <artifact_dir>
```

Documented as a first-class campaign rule in `vllm_neuron/kernels/README.md` under **Rules the kernels enforce**.

---

## 5. Test count post-refactor

Baseline (pre-refactor, `worker/fleet-a-nki-kernels-2026-08-28` at `611b838`): **307 tests · 300 passed · 7 skipped · 0 failed**.

Post-refactor (this change): **317 tests · 306 passed · 11 skipped · 0 failed**.

Delta: **+10 tests, +6 pass, +4 environmental skip** (compile-log / artifact-dir / runtime-probe / neuron-top).  Zero regressions.

Reproducer (from `vllm_neuron/kernels/`):
```
py -3 -m pytest
```

Wall time: 165 s (Windows Python 3.12.10, pytest 8.4.2).

---

## 6. Files changed on PR #4

| Path (absolute local) | Change |
|---|---|
| `C:\Users\apumu\research\InfinityAI\vllm-neuron-fleet-a-refactor\vllm_neuron\kernels\gemma4_no_fallback_mitigations.py` | NEW — renamed + refactored from `gemma4_cpu_fallback_replacement.py` |
| `C:\Users\apumu\research\InfinityAI\vllm-neuron-fleet-a-refactor\vllm_neuron\kernels\gemma4_cpu_fallback_replacement.py` | DELETED — replaced by rename above |
| `C:\Users\apumu\research\InfinityAI\vllm-neuron-fleet-a-refactor\vllm_neuron\kernels\tests\test_no_cpu_fallback.py` | NEW — universal no-CPU-fallback assertion tests |
| `C:\Users\apumu\research\InfinityAI\vllm-neuron-fleet-a-refactor\vllm_neuron\kernels\tests\conftest.py` | NEW — pytest CLI options + fixtures (`--compile-log`, `--artifact-dir`, `--runtime-probe`) |
| `C:\Users\apumu\research\InfinityAI\vllm-neuron-fleet-a-refactor\vllm_neuron\kernels\moe_dispatch.py` | AMEND — docstring cross-reference now points at `gemma4_no_fallback_mitigations.import_pr172_flash_attention` |
| `C:\Users\apumu\research\InfinityAI\vllm-neuron-fleet-a-refactor\vllm_neuron\kernels\README.md` | AMEND — Gemma-4 row rewritten; new "No CPU fallback" rule added under "Rules the kernels enforce"; test-count line updated; status-doc index adds this file |
| `C:\Users\apumu\research\InfinityAI\vllm-neuron-fleet-a-refactor\vllm_neuron\kernels\MOE-DISPATCH-STATUS-2026-08-28.md` | AMEND — trigger-status table amended for PR #172 supersession; head_dim sign-off section marked RESOLVED |
| `C:\Users\apumu\research\InfinityAI\vllm-neuron-fleet-a-refactor\vllm_neuron\kernels\GEMMA4-NO-FALLBACK-REFACTOR-2026-08-28.md` | NEW — this doc |

Staging tree at `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\` mirrors every change.

---

## 7. Not changed (deliberately)

* **No test coverage removed.**  Every pre-existing test continues to pass.  Only additions.
* **`moe_dispatch.py` MoE branch coverage** — unchanged; PR #172 uses NxDI `moe_v2` directly, so our fused-dispatch NEFF still needs `MoEActivation.GELU_TANH_APPROX`.
* **`should_disable_argmax_kernel()` conservative default** — PR #172 was validated at B=1 only, so the mitigation is not orthogonal; the function is model-agnostic and remains relevant for Qwen3-30B-A3B and any other decoder with a large vocab.
* **In-flight compiles** — no preemption of GPT-OSS-120B h3072 or Llama-70B corrected sweep authoring.  This refactor touches only the kernels/ tree, not any lane state or compile driver.

---

## 8. Receipt map

* Renamed module (staging): `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\gemma4_no_fallback_mitigations.py`
* Renamed module (push repo): `C:\Users\apumu\research\InfinityAI\vllm-neuron-fleet-a-refactor\vllm_neuron\kernels\gemma4_no_fallback_mitigations.py`
* Test module (staging): `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_no_cpu_fallback.py`
* Test module (push repo): `C:\Users\apumu\research\InfinityAI\vllm-neuron-fleet-a-refactor\vllm_neuron\kernels\tests\test_no_cpu_fallback.py`
* Test conftest (staging): `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\conftest.py`
* Test conftest (push repo): `C:\Users\apumu\research\InfinityAI\vllm-neuron-fleet-a-refactor\vllm_neuron\kernels\tests\conftest.py`
* AWS PR #172 snapshot: `C:\Users\apumu\AppData\Local\Temp\claude\C--Users-apumu-research-InfinityAI-gemma4-trn2-handoff\fe22bc9a-b8c7-4107-ac55-1d55ba4bd33a\scratchpad\aws-pr172\`
* PR #4 URL: `https://github.com/Infinity-AI-Institute/vllm-neuron/pull/4`
* Base branch head at task start: `611b838` (README + status docs cleanup by `readme-cleanup-agent`)

MEMORY receipts:
* `[Gemma-4 deferred (CPU fallback)]` — Gemma-4-26B-A4B is still a Fleet-B follow-on; this refactor prepares the mitigation stack for a re-open, does NOT re-open the lane.
* `[Peer-agent non-interference discipline]` — refactor executed in a fresh worktree (`vllm-neuron-fleet-a-refactor`); peer worktree at `vllm-neuron-readme-fix` and any in-flight compiles untouched.
* `[Push-on-every-win policy]` — commit lands on `worker/fleet-a-nki-kernels-2026-08-28`, not main.
* `[No spec-decode methodology 2026-08-27]` — the universal test module extends the existing kernel-source scan discipline; no spec-decode fields introduced.

---

**End of refactor doc.**
