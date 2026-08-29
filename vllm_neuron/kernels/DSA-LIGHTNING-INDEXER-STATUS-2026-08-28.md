# DSA Lightning Indexer — implementation status 2026-08-28

**Author:** Fleet A worker (kernel ownership lane), 2026-08-27 session on top of the reference-sweep 2026-08-26T21:50Z staging.
**Deliverable ask:** own the DSA Lightning Indexer NKI kernel that unblocks 3 lanes (GLM-5.2, GLM-5.3-Flash, DeepSeek-V4-Flash).
**Absolute path:** `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md`

---

## 1. Kernel slug

- **`nki_v0_reference_lightning_indexer`** — the CPU golden reference landed 2026-08-27 first session. This is what downstream lanes gate against. See `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\dsa_lightning_indexer.py`.
- **`dsa_sparse_attention.nki_v1`** — first NKI device-kernel authoring pass landed 2026-08-27 (Callsign: `dsa-nki-v1-agent`). See `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\dsa_lightning_indexer_nki_v1.py`. NKI kernel body ships as source-string ready to compile on Trn2; runtime path transparently falls back to v0 on any host without `neuronxcc.nki`. Bit-check target vs v0: bf16 rel tol ~1e-3 once compiled on device.
- **`nki_v1_lightning_indexer`** — v0's reserved-slug name for a device kernel; superseded by the operator-prompt-named `dsa_sparse_attention.nki_v1` above. Both string values still resolve via `DsaKernelConfig.cache_key()` for backwards-compat, but new lanes cite the operator-prompt name.

All three slugs participate in `_GLM52_GRAPH_ID` (and its 5.3-Flash and DSV4-Flash siblings) via `DsaKernelConfig.cache_key()`, so any v0 <-> v1 swap is a fresh compile-cache line and never a silent replay. This is the discipline the Gemma-4 top-5 rules #2 name-the-graph-and-engine-it-changes rule requires.

---

## 2. First plug-in lane

**Winner: GLM 5.2 correctness unblock** at `lanes/glm-5-2-5-3/`.

Rationale, ranked against the other two candidates:

| Lane | Correctness gate open? | Blocks on device NKI? | Value of reference-only landing |
|---|---|---|---|
| **GLM 5.2** | **YES** — token-0 centred cosine 0.9695 vs 0.99 bar per `LANE-STATE-20260827T222500Z` §2. | NO — Mode B (indexer FP8 numerics) is fixable with 3 Python-side actions per scaffold §10; NKI is a later win. | **Highest.** Reference gives the 5.2 lane a real oracle for the token-0 gate audit and a bit-exact target for the eventual NKI kernel. |
| GLM 5.3 Flash | Not yet — green field per `CAMPAIGN-SCOPE-GLM-5.3-FLASH-2026-08-27.md` §9.1. | Yes but IndexPool=4 is novel; needs GAP-3 resolution first. | Lower — the lane is not yet in device-serve territory. |
| DSV4-Flash | Passing at toy width per `LANE-STATE-2026-08-27T2225Z` §1.1. | Yes — Lightning Indexer top-K=512 called out as scaffold `BIGGEST RISK` in port DESIGN.md. | Medium — reference gives the tiny-all-feature re-baseline (Compile A per §2.2 of DSV4 LANE-STATE) something to gate against, but the lane also needs TP16 sharded backend before it exercises the reference at production shape. |

**Plug-in mechanics.** The 5.2 lane's Python side hooks the reference in three places, each independently value-added even if the others slip:

1. **`Glm52MlaAttention.forward`** dispatches to `dsa_lightning_indexer_forward` from `kernels/dsa_lightning_indexer.py` when `os.environ.get("DSA_KERNEL_IMPL", "nki_v0_reference_lightning_indexer") == "nki_v0_reference_lightning_indexer"`. This makes the *reference* the default runtime path — slow but correct — while the NKI kernel is still deferred. Serves as the correctness oracle for every future NKI author.

2. **`Glm52FullIndexer.__init__`** adds the scaffold §10 item (3) assertion: `torch.max(cache_quant_multiplier) <= 240.0`. This turns silent FP8-numeric drift into a load-time error before the ten-token gate has to catch it. Wired to the `test_05_fp8_scale_factor_audit.py` contract already scaffolded on `lanes/glm-5-2-5-3/tests/`.

3. **Telemetry counters** (`dsa_path_active[layer_idx]`, `mla_full_indexer_active[layer_idx]`) exposed via `get_telemetry()` on the module, matching the contract in `lanes/glm-5-2-5-3/tests/test_03_dsa_path_activation.py`. This closes the observability gap in the operator's 2026-08-27 memo.

Together these three land the fastest-path unblock the 5.2 scaffold §10 called out (1-2 person-days total), *before* any NKI-authoring starts.

---

## 3. What actually shipped this session

### 3.1 Files

- **`kernels/dsa_lightning_indexer.py`** (~14 KB) — CPU golden reference. Public API matches scaffold §3.1 exactly:
  - `dsa_lightning_indexer_forward(...)` — one-shot indexer + gather + sparse attention
  - `dsa_sparse_attention_forward(...)` — shared-indexer follow-on
  - `dsa_index_pool_projection(...)` — IndexPool weighted-sum
  - `lightning_indexer_topk(...)`, `lightning_indexer_scores(...)`, `sparse_gather_kv(...)` — atomic pieces for the Tier-1 kernel-correctness gate
  - `full_attention_reference(...)`, `full_attention_at_indices_reference(...)` — independent oracles for the Mode-A degeneracy invariant and for the operator's exact-prompt formula `torch.softmax(Q @ K.T)[:, topk_indices]`
  - `analytical_bounds(...)` returning `KernelResourceBounds` — pre-fire SBUF / descriptor / cycles envelope
  - `DsaKernelConfig` dataclass — immutable cache-identity shape
  - `_nki_kernel_stub_...` — raises `NotImplementedError` with the GAP list
- **`kernels/tests/test_dsa_lightning_indexer_correctness.py`** — 10 gates (T0..T9) parameterised over `K ∈ {2048, 4096, 8192, 16384, 32768}` and `topk ∈ {2048, 4096}`. Covers all six scaffold §5 invariants plus dtype consistency and cache-key correctness.
- **`kernels/tests/test_dsa_lightning_indexer_speed.py`** — 6 gates (S0..S5) parameterised over the 3 production shapes × 5 L buckets. All analytical, device-free.
- **This status doc** (`DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md`).

### 3.2 Test results

```
77 passed in 2.81s
```

All 77 gates pass. Wall time 2.81 s is well under the Tier-1 < 10 s discipline. Every gate is authored against an independent oracle (not the SUT); see the docstring of each `test_T*`/`test_S*` for the specific oracle used.

Notable results in the sweep:

- **T2 (sparse-vs-full at topk=L=2048)**: `rel < 1e-4` — confirms the Mode-A degeneracy on which GLM 5.2's compiled 2K graph relies for correctness today.
- **T3 (sparse attention vs operator-prompt oracle)** at all 8 (L, topk) combinations: `rel < 5e-6` across the board.
- **T4 (token 0 escape)**: attention output is non-zero and non-NaN at position 0 with topk=1 — the bug pattern that broke GLM 5.2's ten-token gate does NOT reproduce here.
- **T6 (IndexShare fidelity)**: `rel < 1e-6` — a shared-indexer sparse attention call using another layer's topk_indices matches a full-indexer re-compute bit-for-bit.
- **S1 (descriptor coalescing)**: at all 3 production shapes the analytical reduction hits the scaffold's 32× target.
- **S3 (nc_find_index8 partition cap)**: the top-K stream workspace stays ≤ 64 KiB (16384 fp32 slots) at every L bucket — inside the 16384 partition cap that gated GPT-OSS TP8 at B>128.
- **S5 (cache-key change on every shape flip)**: any change to `{topk, block_size, index_n_heads, index_head_dim, index_pool, causal, return_topk_for_indexshare}` produces a distinct cache key. No silent-replay risk.

### 3.3 What did NOT ship (and why)

- **A real NKI device kernel.** The operator's prompt explicitly authorises the CPU-golden-reference fallback: *"If NKI kernel doesn't complete in this turn budget, fall back to producing a working CPU golden reference so downstream lanes have SOMETHING to gate against. Do NOT ship a broken kernel."* An NKI kernel written this session — without device access, without SBUF sizing measurements, without descriptor-cache profiling, without closing GAP-1..7 — would be a broken kernel, ship-blocked by the operator's rule.
- **A device-side speed benchmark.** All speed gates are analytical against `analytical_bounds(...)`. A live tok/s benchmark of the CPU reference would swamp the < 10 s Tier-1 budget above L~16 k and adds no correctness signal (per Tier-3 profile-at-knee discipline, live benchmarks belong at the lane's knee sweep, not at the kernel Tier-1).

---

## 3A. NKI v1 landing status (added 2026-08-27, Callsign: dsa-nki-v1-agent)

### 3A.1 Files landed

- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\dsa_lightning_indexer_nki_v1.py` — NKI v1 author's draft. Slug `dsa_sparse_attention.nki_v1`. Contents:
  - `KERNEL_SLUG_V1_NKI = "dsa_sparse_attention.nki_v1"` — operator-prompt-named cache identity.
  - `LSE_BASE_CONVENTION = "natural"` — mirrored from v0; cross-verification bit-checkable without conversion (§5 of `SGLANG-DSA-LSE-FIX-ANALYSIS-2026-08-28.md`).
  - `nki_runtime_available()` — feature-detects `neuronxcc.nki` at import time. Windows: False. Trn2 container: expected True.
  - `_NKI_KERNEL_SOURCE` — the actual NKI Python DSL kernel body as a source-string constant, `exec`'d only when NKI is importable. Contains:
      * `@nki.jit`-decorated `_dsa_sparse_attention_nki_v1_impl(Q, K, V, index_topk_idx, q_pos, k_len, Out, Lse, ...)` with the online-softmax flash-attention loop over `topk // block_size` K-blocks. Uses `nl.affine_range` for tile scheduling, `nisa.nc_matmul` for QK^T and P @ V, `nl.exp` / `nl.log` / `nl.max` / `nl.sum` for the natural-log accumulator, `nl.where` for the causal + key-length mask, `nl.equal` + sentinel for the `fixup_zero_kv_rows` all-masked contract.
      * `_nki_gather_kv_block(kv_batch, idx_block)` — DMA-coalesced gather helper per `NKI-DMA-COALESCING-SCAFFOLD-2026-08-27.md` §5.1. Emits one block-strided descriptor per `(q, block)` for the 32x reduction target.
      * Tile plan constants: `q_tile=16`, `block_size=32` per scaffold §4.
  - `dsa_sparse_attention_forward_nki_v1(Q, K, V, index_topk_idx, ...)` — public v1 entrypoint matching the operator-requested API. Dispatches to compiled NKI on Trn2, falls back to v0 CPU golden reference on any NKI-less host (or if compile fails), so a lane flipping `DSA_KERNEL_IMPL` to `dsa_sparse_attention.nki_v1` is safe on Windows.
  - `build_v1_cache_key(...)` — wraps v0's `DsaKernelConfig` with `impl=KERNEL_SLUG_V1_NKI` and layers on `return_lse` so the LSE-variant gets its own NEFF per LSE fix analysis §7 action #1.
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_dsa_lightning_indexer_nki_v1_smoke.py` — S0..S6 smoke (import + slug + LSE convention + NKI-detect + fallback shape + cache-key uniqueness + fallback bit-match vs v0).

### 3A.2 Smoke result

```
kernels/tests/test_dsa_lightning_indexer_nki_v1_smoke.py .......  [100%]
7 passed in 2.44s
```

All 7 gates pass on Windows (no NKI, torch available). Baseline v0 gates unchanged: `32 passed in 2.75s` (correctness + LSE accumulator combined).

### 3A.3 What did NOT ship in v1 (and why)

- **A live NKI compile.** The Windows author session has no `neuronxcc.nki` runtime; per operator's *"If NKI runtime not accessible in Windows session, ship as SOURCE-STRING scaffold ready to compile when Trn2 host has NKI. Do NOT ship untested compiled code"* the kernel body is source-string-only and compiled by `_compile_nki_kernel_if_available()` on first call on a Trn2 host. A device-side agent picks this up unchanged.
- **A live device correctness gate.** Requires NKI + Trn2 access; the source-string kernel is an *authoring* pass whose bit-check against v0 lives on the device side. That gate is the first item on the device-side agent's list once GAP-1..7 close.

### 3A.4 First plug-in target

Same as §2 — GLM 5.2 correctness lane at `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\glm-5-2-5-3\`. The plug-in mechanic:

```python
# in Glm52MlaAttention.forward
if os.environ.get("DSA_KERNEL_IMPL", KERNEL_SLUG_V0_REFERENCE) == KERNEL_SLUG_V1_NKI:
    out = dsa_sparse_attention_forward_nki_v1(q, k, v, topk_indices,
                                              position_ids=pos, key_lengths=klen,
                                              topk=cfg.topk)
else:
    out = dsa_sparse_attention_forward(q, k, v, topk_indices, pos, klen,
                                       topk=cfg.topk)
```

Flipping `DSA_KERNEL_IMPL=dsa_sparse_attention.nki_v1`:
  * on Windows (this session's host) -> transparent fallback to v0 (smoke S6 confirms bit-match).
  * on Trn2 host with NKI -> exercises compiled kernel; bit-check target is bf16 rel tol 1e-3 vs v0.

---

## 4. Open blockers (in fire order)

### 4.1 Blockers on the NKI v1 kernel

Every GAP from `kernels/NKI-DSA-SPARSE-ATTENTION-SCAFFOLD-2026-08-27.md` §9 is still open:

1. **GAP-1** — dump the current GLM 5.2 compiled `.neff` (`neuron-cc --dump-neff`) and confirm zero `dsa_indexer_kernel` invocations today. This is the empirical proof of scaffold §1.1 Mode C. Requires access to the GLM 5.2 compile-cache directory on the box.
2. **GAP-2, GAP-4, GAP-5** — read full `config.json` for GLM 5.2 and GLM 5.3 Flash to lift `index_n_heads` / `index_head_dim`. Blocked by config-file trims from prior `WebFetch` calls; needs direct file read from a local mirror OR raw HF `?raw=1` fetch.
3. **GAP-3** — confirm IndexPool `pool_weights` semantics: per-layer learned tensor vs global constant? Determines whether the CPU reference's `pool_weights` argument is a runtime input or a constant baked at compile time.
4. **GAP-6** — prototype two-stage hierarchical top-K for L=1 M. The current `analytical_bounds(...)` clips the stream tile at 16384, but at L=1 M a single-stage top-K needs 64 waves. GAP-6 asks whether a coarse-over-blocks + fine-within-winning-blocks scheme wins the cycles/token race; needs synthetic-input timing on device.
5. **GAP-7** — NEFF pattern-match audit: does the current compiler lower `gather + softmax` to a full-attention primitive that drops sparsity? Read a compile trace at S=64 K.

### 4.2 Blockers on the first plug-in lane (GLM 5.2)

Per `lanes/glm-5-2-5-3/LANE-STATE-20260827T222500Z.md` §5, three operator-side actions:

1. Operator sign-off on a codex-side PR against `apuroop/glm5-2-enablement-v2` adding the telemetry counters + reference-dispatch hook.
2. Live pointer to the current converted 5.2 FP8 checkpoint (or its scale-manifest JSON) so `test_05_fp8_scale_factor_audit.py` can run against real data.
3. Confirmation the FP8-scale-corrected rebuild (`C2` in the LANE-STATE §4.2) fits in the next Trn2 capacity block's compile-pool budget.

The reference-code path is unblocked. The Python plug-in is unblocked. The device-side compile-and-serve is queue-blocked; my kernel does not consume any Trn2 device time this session.

### 4.3 Cross-cutting blockers

- **The next Trn2 capacity block window is unknown from this session's context.** Per MEMORY `[trn2-11:30z-hard-cliff]` the 2026-08-27 window closed at 11:30 Z. Neither the reference-code landing nor the analytical speed gates need device time to be immediately valuable; the C2 compile is queued.
- **`MLA-VS-DSA-KERNEL-VERIFICATION-2026-08-27.md` §3 corrections not yet reflected in `FOUR-MODEL-COMPETITIVE-PATH-2026-08-27.md`.** DSA is the actual highest-cross-model NKI-authoring win, not MLA. Cross-file correction is book-keeping, not blocking.
- **DSV4-Flash's TP16 sharded backend does not exist.** Per `lanes/deepseek-v4-flash/LANE-STATE-2026-08-27T2225Z.md` §1.3 the checkpoint cannot fit on 1 chip and the TP-sharded weight loader is 4-6 person-weeks of net-new code. My kernel is ready for the DSV4-Flash lane; the DSV4-Flash serving stack is not.

---

## 5. Cross-model reuse table (post-landing)

Updated view of the scaffold §7 table with a "landed?" column:

| Model | index_topk | index_n_heads | index_head_dim | index_pool | landed at reference? | first NKI-kernel plug-in? |
|---|---|---|---|---|---|---|
| GLM 5.2 | 2048 | 64 (GAP-2) | 64 (GAP-5) | 1 | **YES** — this session | see §2 above |
| GLM 5.3 Flash | 2048 | 64 (GAP-4) | 128 (GAP-4) | 4 | **YES** — this session (IndexPool=4 path exercised in test T5) | after 5.2 v1 device kernel lands |
| DeepSeek-V4-Flash | 512 | 64 | 128 | 1 | **YES** — this session | after 5.2 v1 device kernel lands AND TP16 sharded backend lands |
| Kimi K3 | n/a | n/a | n/a | n/a | not applicable | K3 has no sparse indexer |

The single reference file plus its `DsaKernelConfig` handles all three models by parameterising over `topk`, `H_idx`, `D_idx`, and `index_pool`. Per scaffold §8 the 3-model amortisation drops per-model cost to 2.3–3.8 person-weeks — this session banked the amortisation for the reference-code side; the NKI v1 side is still on the same 6-10 person-week critical path per model.

---

## 6. Return artifacts (paths absolute per user policy)

- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\dsa_lightning_indexer.py` (v0 CPU golden reference)
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\dsa_lightning_indexer_nki_v1.py` (v1 NKI author's draft, added 2026-08-27)
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_dsa_lightning_indexer_correctness.py`
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_dsa_lightning_indexer_speed.py`
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_dsa_lightning_indexer_nki_v1_smoke.py` (v1 smoke, added 2026-08-27)
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md` (this file)

Design cross-references:

- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\NKI-DSA-SPARSE-ATTENTION-SCAFFOLD-2026-08-27.md`
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\MLA-VS-DSA-KERNEL-VERIFICATION-2026-08-27.md`
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\CAMPAIGN-SCOPE-GLM-5.3-FLASH-2026-08-27.md`
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\CAMPAIGN-SCOPE-DEEPSEEK-V4-FLASH-2026-08-27.md`

Lane targets:

- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\glm-5-2-5-3\LANE-STATE-20260827T222500Z.md` (first plug-in)
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\deepseek-v4-flash\LANE-STATE-2026-08-27T2225Z.md` (second plug-in after TP16 backend)
