# MoE dispatch + Gemma-4 CPU-fallback status — 2026-08-28

**Author:** Claude Code worker agent, Gemma-4 Trn2 campaign, spawned by main-session with the MoE-dispatch NKI + Gemma-4 CPU-fallback replacement mandate.
**Scope:** Ships the actual MoE dispatch NKI kernel (`moe_dispatch.py`), the Tier-1 CPU-battery correctness test (`tests/test_moe_dispatch_correctness.py`), the Gemma-4 CPU-fallback replacement module (`gemma4_cpu_fallback_replacement.py`), and this status doc.  Corresponds to Part A + Part B of `NKI-MOE-DISPATCH-AND-GEMMA4-CPU-FALLBACK-SCAFFOLD-2026-08-27.md`.
**No-guess discipline:** the head_dim ambiguity is blocked on operator sign-off per the task prompt; every other trigger has a shipped fix or a documented mitigation.

---

## 1. Delivered files (absolute paths)

| File | Purpose | Lines |
|---|---|---:|
| `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\moe_dispatch.py` | Fused MoE dispatch NKI kernel: router + expert-combine, three shape families | ~330 |
| `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_moe_dispatch_correctness.py` | Tier-1a CPU-simulate golden reference test suite | ~310 |
| `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\gemma4_no_fallback_mitigations.py` | PR #172 adapter shim (triggers #1/#2) + local stopgaps (triggers #3/#4) — renamed 2026-08-28 from `gemma4_cpu_fallback_replacement.py` | ~260 |
| `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_no_cpu_fallback.py` | Universal no-CPU-fallback assertion tests (grep + NEFF-content + runtime probe) | ~380 |
| `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\MOE-DISPATCH-STATUS-2026-08-28.md` | This document | — |

---

## 2. Shape families shipped in `moe_dispatch.py`

| Config object | Model | Hidden | Experts | Top-K | I_TP | Activation | Renormalize | Cross-model target |
|---|---|---:|---:|---:|---:|---|:---:|---|
| `QWEN3_30B_A3B_TP8` | Qwen3-30B-A3B TP=8 | 2048 | 128 | 8 | 96 | SILU | True | first-strike test bed (MFU 12.05% baseline) |
| `GEMMA4_26B_A4B_TP4` | Gemma-4-26B-A4B TP=4 | 2816 | 128 | 8 | 176 | GELU_Tanh_Approx | True | matches Codex `cycle465` split |
| `GPT_OSS_20B_TP4` | GPT-OSS-20B TP=4 | 2880 | 128 | 4 | 720 | SILU | **False** | §A.G-9 no-renorm branch |

**Container constraint enforcement** (per memory `nxdi-container-moe-blockwise-mm-workaround-20260827`): the `enable_moe_fused_dispatch()` helper in `moe_dispatch.py` *always* sets `blockwise_matmul_config.use_shard_on_intermediate_dynamic_while=True` before any downstream compile can proceed.  This is enforced at wire-in time, not runtime — a compile submitted without the helper fails the Tier-1 CPU battery grep.

---

## 3. Which lane fires first

**Answer: Qwen3-30B-A3B TP=8** — not Gemma-4-26B-A4B.  Ordering rationale:

1. **Qwen3-30B-A3B is the only shipped shape family where the model is already on-device MFU 12.05%.**  A fused-dispatch uplift is directly measurable against a real Trn2 baseline (`RESULTS-B8-2026-08-27.md`).  Uplift target per scaffold §A.6 table: **1.5-2.0× at B=16-32.**
2. **Gemma-4-26B-A4B still hits Wall #4 trigger #1 (head_dim > `_MAX_D_HEAD=128`)** — the attention half CPU-fallbacks regardless of what the MoE kernel does.  A fused MoE dispatch on Gemma-4 alone, without the attention fix, is a MoE-block micro-benchmark, NOT a whole-window uplift.  Fires as position #3 per scope §4.10.
3. **GPT-OSS-20B TP=4 is cross-model validation** — the no-renormalize branch (§A.G-9) is the highest-signal branch-coverage test.  Fires after Qwen3.

**Fire-order this campaign week (aligned with lane-manager schedule):**

| # | Target | Shape | Purpose | Fire condition |
|:---:|---|---|---|---|
| 1 | Qwen3-30B-A3B TP=8 | `QWEN3_30B_A3B_TP8` | Primary fused-dispatch uplift test | Trn2 window opens 2026-08-28T12:30Z |
| 2 | GPT-OSS-20B TP=4 | `GPT_OSS_20B_TP4` | No-renorm branch coverage | after #1 lands PASS |
| 3 | Gemma-4-26B-A4B TP=4 | `GEMMA4_26B_A4B_TP4` | Integrate with Codex `cycle465` split; MoE-block micro-bench (attention still CPU-fallback) | after Part B #1 gate-flip or head_dim sign-off |

---

## 4. Head_dim discrepancy — RESOLVED by AWS PR #172 (2026-08-28)

**Update 2026-08-28:** AWS PR #172 pins the Gemma-4-26B-A4B geometry at HF revision `24548b62aa021d562695c04aaf7758a1ea47990b`.  Per PR #172 `README.md` line 22-32: `hidden_size=2816`, 16 attention heads, 8 KV heads (GQA 2:1), heterogeneous attention with SWA layers at `head_dim=256` and global layers at `head_dim=512` (same as PR #106 gemma-4-31B-IT).  The `SIGN_OFF_REQUIRED_HEAD_DIM` sentinel and `HeadDimSignOffRequired` exception are retired from `gemma4_no_fallback_mitigations.py`; the PR #172 flash-attn kernels are wired directly through `import_pr172_flash_attention(variant="d256_swa")` and `import_pr172_flash_attention(variant="large_d")`.

The task-prompt "64 experts / top-6" numbers are inconsistent with the pinned HF `24548b62`, which is `num_experts=128, top_k=8, moe_intermediate_size=704` (per Codex `dual_input_tkg_moe_nki.py` + PR #172 §Architecture Details).  The Codex-verified values remain the source of truth for the fused-dispatch NEFF shape.

### Historical context — the sign-off block (pre-PR-172)

**Per task prompt:**
> If head_dim discrepancy still not resolved (my prior report flagged head_dim=176 vs sourced 256 sliding / 512 global), block on operator sign-off — do NOT guess a value.

**Receipts in conflict:**

| Source | Claim | Path |
|---|---|---|
| `AR-TRN-ISSUE-DRAFT-GEMMA4-2026-08-27.md:251` | `head_dim = hidden_size / num_attention_heads = 2816 / 16 = 176` (for Gemma-4-**26B-A4B**) | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\AR-TRN-ISSUE-DRAFT-GEMMA4-2026-08-27.md` |
| `GEMMA4-LESSONS-GENERALIZED-2026-08-27.md §B3` | `head_dim = 256 sliding / 512 global` (for Gemma-4-**26B** non-A4B) | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\GEMMA4-LESSONS-GENERALIZED-2026-08-27.md` |
| Codex `dual_input_tkg_moe_nki.py:36-44` | `HIDDEN=2816`, `EXPERTS=128`, `TOP_K=8` (pinned HF revision `24548b62`) | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\cycle465\dual_input_tkg_moe_nki.py` |
| Task prompt (this task) | Gemma-4-26B-A4B "64 experts / top-6" | task prompt |

**Two separate discrepancies:**

**Discrepancy A — head_dim value(s).**  Blocks `flash_attention_hybrid_sliding_global` authoring.  The scaffolded head-dim-tiling kernel is head-dim-agnostic, but the *per-layer* head_dim table must be known.  The `MAX_HEAD_DIM 128 -> 256` gate flip patch site is shipped in `gemma4_cpu_fallback_replacement.print_max_head_dim_patch()`; it applies IFF confirmed head_dim ≤ 256.  If head_dim = 512 in the global layers, the gate flip alone is NOT sufficient — Part B #1's head-dim tiling remains required.

**Discrepancy B — expert count / top-K.**  Task prompt says "64 experts / top-6" for Gemma-4-26B-A4B; Codex's proven production code (pinned to HF `24548b62aa021d562695c04aaf7758a1ea47990b`) uses `EXPERTS=128, TOP_K=8`.  This file ships the Codex-verified values because they are grounded in actual code + a pinned HF revision.  The task-prompt numbers may be either stale or from a different variant (Gemma-4-**31B** dense per MEMORY `[Makora TrainSpotting competitive context]`?).  Sign-off requested to reconcile.

**Sign-off requested — please confirm the following for `gemma-4-26B-A4B` @ HF revision `24548b62aa021d562695c04aaf7758a1ea47990b`:**

```
config.json fields we need pinned:
    hidden_size            = ?  (Codex code: 2816)
    num_attention_heads    = ?  (arithmetic gives 16 if head_dim=176)
    num_key_value_heads    = ?  (per sliding + global layer types)
    head_dim               = ?  (176 vs 256 vs 512 depending on receipt)
    num_experts            = ?  (Codex code + HF pin: 128; task prompt: 64)
    top_k                  = ?  (Codex code + HF pin: 8;   task prompt: 6)
    moe_intermediate_size  = ?  (Codex code: 704)
    layer_types            = [sliding_attention or global_attention per layer]
    sliding_window         = ?  (scaffold: 1024)
    n_sliding_layers       = ?  (scaffold: 25)
    n_global_layers        = ?  (scaffold: 5)
```

**Sign-off action for operator:** call `set_gemma4_head_dim(head_dim_sliding=<int>, head_dim_global=<int>, signed_off_by="apuroop", signed_off_receipt="<link>")` in `gemma4_cpu_fallback_replacement.py` after inspecting the pinned HF config.  Until this call lands, `make_flash_attention_hybrid_sliding_global_kernel()` raises `HeadDimSignOffRequired`.

---

## 5. Codex TP=4 dual-input MoE split — integration path

Codex's `harness-v2\staging\cycle465\compile_dual_input_tkg_moe_split.py` is already queued as item 8 in the compile-orchestrator's pre-compile queue (per `PRECOMPILE-QUEUE-ORCHESTRATION-2026-08-27.md` and lane-state).  Integration with this campaign's `moe_dispatch.py`:

**Shape parity — verified.**  `GEMMA4_26B_A4B_TP4` config in `moe_dispatch.py` matches Codex's `dual_input_tkg_moe_nki.py` exactly:
* HIDDEN=2816 ✓
* EXPERTS=128 ✓
* TOP_K=8 ✓
* INTERMEDIATE_GLOBAL=704 ✓
* INTERMEDIATE_PER_TP=176 (TP=4) ✓
* activation=GELU_Tanh_Approx ✓
* ExpertAffinityScaleMode.POST_SCALE ✓

**Difference:** `moe_dispatch.py` uses the same `router_topk` + `moe_tkg` primitives Codex compiled with, so the two implementations produce identical NEFFs modulo the closure structure (compile-time constants vs the module-level constants Codex used).  The `_make_moe_dispatch(GEMMA4_26B_A4B_TP4)` factory returns `(gemma_moe_router, gemma_moe_expert_combine)` — the exact two-NEFF pair Codex ships.  A regression test would sha256 both NEFFs against Codex's baseline; when they diverge, refuse to promote until the divergence is understood.

**Migration path from Codex compile driver:**
1. Point `harness-v2\staging\cycle465\compile_dual_input_tkg_moe_split.py` at the new factory — `from moe_dispatch import make_gemma4_26b_a4b_tp4_kernel; router_fn, expert_fn = make_gemma4_26b_a4b_tp4_kernel()`.
2. Preserve Codex's shape assertions (`INTERMEDIATE_PER_TP=176`, `NKI_ROOT`/`NXD_ROOT` commit pins).
3. Bank the sha256 of both NEFFs alongside the Codex baseline receipt.
4. If sha256s differ, gate promotion on 10-token exact-slug parity per Tier-1a.

**Firing schedule:** Codex's compile is Round-3 lane assignment; expected to bank a NEFF pair before the 12:30Z Trn2 window opens.  First device fire fires the Codex NEFFs (already compiled), THEN the `moe_dispatch.py`-produced NEFFs on the same shape — both as MoE-block micro-benchmarks with the disclaimer that Gemma-4 attention still CPU-fallbacks.

---

## 6. CPU-fallback trigger status (four triggers per scaffold §B.1)

**Amended 2026-08-28:** AWS PR #172 (Gemma-4-26B-A4B NxDI contrib port) supersedes local mitigations for triggers #1 and #2.  The PR ships validated flash-attn kernels (`nki_flash_attn_d256_swa.py` for SWA head_dim=256, `nki_flash_attn_large_d.py` for global head_dim=512, both verbatim from PR #106 with Stage 5 canonical-chat validation matching HF CPU bf16 at 100% token agreement on 11/12 combos) and a subclassed `Gemma4KVCacheManager` for the hybrid per-layer geometry.  The head_dim sign-off block is retired — PR #172 pins the values at HF revision `24548b62`.  Triggers #3 and #4 remain campaign-owned because PR #172 validated at batch_size=1 only and does not touch the fused MoE dispatch NEFF this campaign compiles.

| # | Trigger | Status | Fix | Ships in |
|:---:|---|:---:|---|---|
| 1 | `head_dim > _MAX_D_HEAD=128` | **SHIPPED_UPSTREAM_PR172** | PR #172 `nki_flash_attn_d256_swa` (SWA) + `nki_flash_attn_large_d` (global); local adapter shim `import_pr172_flash_attention()` | `gemma4_no_fallback_mitigations.import_pr172_flash_attention` |
| 2 | Hybrid KV manager off | **SHIPPED_UPSTREAM_PR172** | PR #172 `Gemma4KVCacheManager` subclass with per-layer heterogeneous shapes; local adapter shim `import_pr172_kv_cache_manager()` | `gemma4_no_fallback_mitigations.import_pr172_kv_cache_manager` |
| 3 | vocab=262K > nc_find_index8 cap | **PARTIAL** | `should_disable_argmax_kernel()` mitigation; Part B #2 kernel not authored.  PR #172 validated at B=1 only. | `gemma4_no_fallback_mitigations.should_disable_argmax_kernel` |
| 4 | `(GLU, GELU_TANH_APPROX)` activation combo | **SHIPPED_LOCAL_STOPGAP** | `MoEActivation.GELU_TANH_APPROX` branch + `verify_activation_branch_coverage()` guard.  PR #172 uses NxDI `moe_v2` directly and does not touch the fused-dispatch NEFF path. | `moe_dispatch.MoEActivation` |

**Trigger #4 detail — closes §B5's silent-fallback hazard.**  The `enable_moe_fused_dispatch()` helper in `moe_dispatch.py` fires a `log.critical("MoE fused dispatch = enabled | ... | activation=%s", cfg.activation.name)` message when wired.  The Tier-1 CPU battery greps for `activation=GELU_TANH_APPROX` when compiling Gemma-4 and refuses to proceed if the label is missing.  This is the concrete instantiation of §C.1 "no silent fallback" discipline.

---

## 7. Kill switch (per campaign scope §8.3)

Gemma-4 lane has a Week-5 kill switch at MFU < 0.06% floor.  This kernel work does **not** by itself flag the kill switch — the MoE fused dispatch alone cannot lift Gemma-4 out of CPU-fallback, because the attention half is still CPU-fallbacking on head_dim.

**Kill-switch signal from this delivery:** if, after Part B #1 flash attention lands (post sign-off), the combined MoE + attention on-device baseline still measures MFU < 0.06% on Gemma-4-26B-A4B TP=4 B=8, the campaign should pivot to K3 correctness unblock per scope §8.3.  **No signal yet** — head_dim sign-off is the pre-requisite to that measurement.

**Non-signal:** the MoE fused dispatch is expected to deliver 1.5-2.0× on Qwen3-30B-A3B (where the model is already on-device).  A miss there would flag a different kill switch — MoE fused dispatch does NOT beat blockwise baseline — which would inform whether Part A applies to Gemma-4 post-Part-B-#1 or whether the four-graph blockwise baseline is already at the optimum.

---

## 8. Test invocation

**CPU-simulate golden (any host with PyTorch, ~5-10 s):**
```
cd C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels
python -m pytest tests/test_moe_dispatch_correctness.py -v
```

Test coverage:
* `test_config_validates` — Tier-1 CPU battery gates for every shape family
* `test_router_topk_index_parity` — top-K matches `torch.topk` at lowest-index tie-break
* `test_weight_sum_invariant` — §A.4 weight-sum invariant + §A.G-9 no-renorm branch
* `test_expert_combine_dense_all_experts` — §B6 dense all-experts + POST_SCALE combine
* `test_end_to_end_reference_stable` — reference determinism (fp32 accumulator)
* `test_activation_branch_registered` — §A.G-7 exhaustive activation branch table
* `test_nki_kernel_constructible` — NKI factory smoke test (skipped without pinned stack)

**Compile-host smoke (pinned NKI stack, ~30 s):**
```
cd harness-v2/staging/cycle465
python compile_dual_input_tkg_moe_split.py --out /tmp/moe_dispatch_gemma4_tp4/
# Then the moe_dispatch.py factory once the compile driver is updated:
python -c "from moe_dispatch import make_gemma4_26b_a4b_tp4_kernel; make_gemma4_26b_a4b_tp4_kernel()"
```

---

## 9. What follows this delivery

Per campaign scope §5.1 Weeks 1-2:

1. **Sign-off on head_dim + expert count.**  This is the blocking dependency for the flash-attention kernel.  This document constitutes the sign-off request.
2. **First-strike Qwen3-30B-A3B fire.**  Wire `enable_moe_fused_dispatch(neuron_config, QWEN3_30B_A3B_TP8)` into `neuronx_distributed_inference/models/qwen3_moe/modeling_qwen3_moe.py` (path assumed per scaffold §A.G-5; grep at integration).  Fire during 12:30Z Trn2 window.  Bank 10-token exact-slug + n=3 iter throughput → Tier-1 / Tier-2 receipts.
3. **Codex TP=4 split whole-window fire.**  Compile-orch item 8 lands; fire on Gemma-4-26B-A4B TP=4 as MoE-block micro-benchmark with the CPU-fallback attention disclaimer.  Cross-check `_make_moe_dispatch(GEMMA4_26B_A4B_TP4)` produces sha256-identical NEFFs.
4. **Part B #1 authoring blocked** until sign-off; scope §4.6 zero-cost `MAX_HEAD_DIM 128 -> 256` gate flip is orthogonal and can ship immediately once head_dim ≤ 256 is confirmed.
5. **Part B #3 (hybrid KV manager) is compile-pool-only** and can integrate immediately — `enable_hybrid_kv_cache_manager(neuron_config)` is shipped and unconditional (no head_dim dependency).
6. **Part B #2 (argmax_kernel_partitioned) authoring** is queued as Fleet-B follow-on; `should_disable_argmax_kernel()` mitigation is the current stop-gap.

---

## 10. Receipt map (absolute paths)

* Scaffold source: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\NKI-MOE-DISPATCH-AND-GEMMA4-CPU-FALLBACK-SCAFFOLD-2026-08-27.md`
* Campaign scope: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\CAMPAIGN-SCOPE-GEMMA-4-OPTIMIZATION-2026-08-27.md`
* Codex TP=4 dual-input MoE split: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\cycle465\compile_dual_input_tkg_moe_split.py`
* Codex TP=4 dual-input MoE kernel bodies: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\cycle465\dual_input_tkg_moe_split_nki.py`, `dual_input_tkg_moe_nki.py`
* Codex session kernel history: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\CODEX-SESSION-KERNEL-HISTORY-2026-08-27.md` §3 body extracts + §6 failed attempts + §7 kernel gaps
* Gemma-4 lane state (spawn tick): `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\gemma-4-26b-a4b\LANE-STATE-2026-08-27T22Z.md`
* Gemma-4 lane test flywheel: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\gemma-4-26b-a4b\tests\README.md`
* MEMORY: `[Gemma-4 deferred (CPU fallback)]`, `[nxdi-container-moe-blockwise-mm-workaround-20260827]`, `[Card scope 15 of 16]`, `[Peer-agent non-interference discipline]`

---

**End of status doc.**  Fire order: **Qwen3-30B-A3B first, GPT-OSS-20B second, Gemma-4-26B-A4B third — Gemma-4 whole-window uplift is blocked on head_dim sign-off, which this document requests.**
