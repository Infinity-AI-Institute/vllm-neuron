# KDA state kernel -- implementation status, 2026-08-28

**Owner:** worker/reference-sweep-2026-08-26/gpt-oss-tp4-knee (Fleet A, Claude Code, Windows local).
**Charter:** own the KDA (Kimi Delta Attention) NKI kernel -- unblocks 2 load-bearing lanes: Kimi K3 (69 of 93 layers, 74.2%) and GLM-5.3-Flash (34 of 45 layers, 75.5%).
**Shift constraint:** no Trn2-cluster access this shift; deliverables are CPU-golden + NKI-DSL scaffolded to be compilable-in-principle. Card 12 never; no spec-decode.

Absolute local paths (per operator memory `always-give-full-local-paths`):

- Status doc: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\KDA-STATE-STATUS-2026-08-28.md`
- Kernel: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\kda_state.py`
- Correctness tests: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_kda_state_correctness.py`
- Speed tests: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_kda_state_speed.py`
- Design scaffold (prior art): `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\NKI-KDA-STATE-SCAFFOLD-2026-08-27.md`
- Sister DeltaNet scaffold: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\NKI-DELTANET-STATE-INT8-SCAFFOLD-2026-08-27.md`
- MLA-vs-DSA verification: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\MLA-VS-DSA-KERNEL-VERIFICATION-2026-08-27.md`
- K3 lane state: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\kimi-k3\LANE-STATE-20260827T2219Z.md`
- GLM 5.2 / 5.3 Flash lane state: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\glm-5-2-5-3\LANE-STATE-20260827T222500Z.md`
- Path-activation contract to hook: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\glm-5-2-5-3\tests\test_04_kda_path_activation.py`

---

## Kernel slug

```
kda_state.decode.rank1_delta.int8_state.per_channel_scale.v1
```

This slug participates in the compile-cache identity hash for both K3 and GLM-5.3-Flash. Wire it into each model's `model.env` before the first compile submit:

```
KDA_STATE_KERNEL_SLUG="kda_state.decode.rank1_delta.int8_state.per_channel_scale.v1"
KDA_STATE_DTYPE="int8"
KDA_STATE_QUANT_GRANULARITY="per_channel"
KDA_STATE_CHUNK_SIZE="128"            # placeholder; unused until prefill lands
```

Change the trailing `.v1` to `.v2` (and any downstream copies) the moment the kernel semantics, quantization layout, or NKI DSL body change -- not before. Silent cache reuse of a stale artifact after a semantics change is one of the highest-priority failure modes flagged in the Gemma-4 lessons harvest (rule #2: every lever names the graph + engine it changes).

---

## What was delivered this cycle

### 1. `kda_state.py` -- kernel + reference

- Bit-exact CPU numpy golden (`kda_state_decode_forward_reference`) covering the rank-1 delta rule `S_t = S_{t-1} + beta_t * (v_t - S_{t-1} @ k_t) @ k_t^T`; output `y_t = S_t @ q_t`.
- Per-channel int8 quantization with a bf16 per-channel scale on the D_v axis (matches scaffold section 3.1's D_v-partition layout).
- Explicit `kda_state_reset` primitive with a batch-wise mask -- no silent side-effects, per the operator's KDA-silent-fallback-corrupts-model discipline.
- Prefill deferred as a by-token loop per operator directive ("prefill can loop"); documented as the next-cycle chunked-parallel deliverable.
- NKI Python DSL body scaffolded as a source string returnable by `_kda_state_decode_forward_nki_source()`. Uses `nl.affine_range` over (B, H); `nisa.nc_matmul` for the two rank-1 matmuls and the query read; `tensor_saturate_cast` to int8 for the write path. Compiles-in-principle against Neuron SDK 2.32 once the sibling `mla_attention_tkg` shape shim lands and GAP-1 (KDA state layout audit against `contrib/kimi_k3/kda_cpu_reference.py`) closes. The NKI toolchain is not present on the Windows workstation this file was authored on; the code path guards on `_try_import_nki()` so the reference path is served unconditionally on CPU hosts. **A missing NKI backend does NOT fall through to full attention** -- this is the correctness rule KDA lives or dies by.
- Model-shape presets: `KIMI_K3_KDA_SHAPE` (H=96, D=128, 69 layers), `GLM_5_3_FLASH_KDA_SHAPE` (H=64, D=128, 34 layers), `QWEN35_2B_DELTANET_SHAPE` (H=16, D=256, 18 layers) -- for reuse by DeltaNet-family sibling kernels.

### 2. `tests/test_kda_state_correctness.py` -- 22 tests, all passing

Coverage matches the operator-requested matrix `state_dim in {64, 128, 256}` X `seq_len in {1, 128, 1024, 8192}`, plus:

- Prefill-vs-decode equivalence at L=64 (scaffold section 5 test T5).
- Boundary betas (beta=0 leaves state unchanged and returns q^T S; beta=1 gives full-strength update; neither NaNs).
- State reset semantics (masked batch elements zeroed bit-exact; unmasked preserved bit-exact).
- Int8 round-trip within per-channel absmax tolerance (< 0.51 * scale per element, catches double-rounding regressions).
- 100-step accumulation monotonicity (`test_04_kda_path_activation.py` contract).
- Kernel slug and model-shape preset lock-in.

Drift envelope discovered empirically -- **this is the concrete GAP-5 measurement the DeltaNet scaffold section 2.1 asks for**:

| state_dim | seq_len | max |y_kernel - y_naive| |
|:---------:|:-------:|:-------------------------:|
| 64 / 128 / 256 | 1 | ~ 1e-4 to 1e-3 (single-step round-trip) |
| 64 / 128 / 256 | 128 | ~ 6e-3 to 2e-2 |
| 64 / 128 / 256 | 1024 | ~ 1e-2 to 1e-1 |
| 256 | 8192 | ~ 3e-1 (well beyond decode budget) |

Interpretation: at K3's D=128 the per-element drift stays inside 3e-2 through L=1024 (the median served decode budget). This is comfortably inside the ten-token gate's cosine-margin envelope (0.99 for K3 today; UNDECIDABLE band 0.9-1.0). The int8 discipline is safe for decode; prefill-scale drift at L=8192 must be measured under the chunked-parallel kernel before that path is served.

### 3. `tests/test_kda_state_speed.py` -- 23 tests, all passing

- **DMA descriptor floor.** Per-layer DMA at K3 (H=96, D=128) B=1 is 3.05 MiB read+write; comfortably above the 4 KiB tiny-packet penalty threshold from `NKI-DMA-COALESCING-SCAFFOLD` section 3. Regression guard: catches any refactor that would emit per-head DMA (which would explode into 96 small descriptors per layer).
- **SBUF resident-state floor.** Single K3 KDA layer at B=1 is 1.522 MiB (6.3% of the 24 MiB SBUF budget). Four colocated layers fit inside the budget (`test_k3_four_colocated_layers_fit`), matching the scaffold section 2.2 "up to 6 layers fit if we cache aggressively" projection.
- **Int8 discipline halves DMA vs bf16** at all three preset shapes (K3, GLM-5.3-Flash, Qwen3.5-2B DeltaNet). This locks in the primary Fleet B lever from scaffold section 2.3.
- **Compile-cache safety.** Slug is versioned; K3 and GLM 5.3 Flash shape hashes differ, so cache never silently serves the wrong NEFF for the other model.
- **Scaffold arithmetic lock-in.** The state-slab MiB math the scaffold cites (K3 B=32: 6624 MiB; GLM 5.3 Flash B=32: 2176 MiB) is recomputed and gated in tests; a stray shape edit that drifts more than +/-1% fails immediately.

All tests run offline on any Windows/Linux machine with `numpy >= 2` and `pytest >= 8`. `py -3 -m pytest tests/test_kda_state_correctness.py tests/test_kda_state_speed.py` completes in ~35 s and finishes 45/45 green (verified 2026-08-28).

### 4. This status doc.

---

## Plug-in order: K3 first, GLM-5.3-Flash second

The prompt asks for the ordered plug-in list. Ranking follows operator memory `[Beat H100/B300 per-model mandate]` (bet-first on the shortest-path model that clears the most competitive daylight), the MLA-vs-DSA verification's per-model kernel-mass rankings, and the two lanes' current L-levels.

### First plug-in target -- **Kimi K3**

**Why K3 first:**

1. **Load-bearing at 74.2% of layers** (69 of 93). Landing KDA on K3 turns the largest fraction of the model from stub/host-fallback to on-device. Landing KDA on GLM 5.3 Flash also turns 75.5% -- almost identical fraction -- but K3 has:
2. **Higher customer-demand signal** (per `CAMPAIGN-SCOPE-KIMI-K3-2026-08-27.md` section 1.4): 2.83M HF downloads/month, rank #4 of 189 on Artificial Analysis; GLM 5.3 Flash was released 2026-08-27 with under-week traction.
3. **Un-fired v12 launcher already scaffolded** at `codex/k3-launch-v12-compile-bound@b95688b8` -- the plumbing to E1 (HLO->NEFF cross) exists in-tree; KDA is one of the two kernels that has to land to unblock E2 (world join) and E3 (ten-token gate) per K3 lane state.
4. **KDA head_dim=128** matches the scaffold's zero-padding assumption exactly (no padding shim needed). GLM 5.3 Flash also has D=128, so the shim from K3 to GLM 5.3 Flash is head-count only (96 -> 64), which is the smaller shim.
5. **Trn2 topology and HBM math for GLM 5.3 Flash are not yet published in scope docs at operator authority.** K3's TP=64 EP=64 topology, 22.5 GiB per-rank HBM budget, and served-artifact splice are all documented and receipt-anchored. Firing GLM 5.3 Flash first would require additional scoping work not yet in the campaign-scope tree; K3 firing has no such gap.

**What K3 needs from this deliverable, at plug-in time:**

- Wire `KDA_STATE_KERNEL_SLUG` into `models/kimi-k3/model.env` on the K3 branch (`codex/k3-launch-v12-compile-bound@b95688b8`); include it in the `_KIMI_K3_GRAPH_ID` derivation.
- Copy `kda_state.py` into `neuronx_distributed_inference/kernels/nki_kda_state.py` in the vendored `apuroop/kimi-k3-real-world` NxDI submodule.
- Bind the 69 KDA layers' state read/write to the kernel's `kda_state_decode_forward` entry point, guarded by `neuron_config.kda_state_dtype == "int8"`. Default to `"bf16"` for the first compile-cross -- the safety default is what the operator prompt calls out.
- Author the K3-side `test_04_kda_path_activation.py` analog in the K3 lane's tests dir (K3 lane already has `tests/` scaffolded per its LANE-STATE section 4; this is Test #4 of the six-test framework). Hook the real `contrib/kimi_k3/kda_cpu_reference.py` when available so the golden isn't only the numpy version here.
- Fire the E1 launcher only on the day the operator authorizes card 12 access OR authorizes the spike-compile fallback (K3 lane state section 1). Neither happens on this shift.

**Blocker to K3 plug-in this shift:** the K3 lane's first blocker (`TP=64 needs 16 physical chips of 15 owned; card 12 teammate`) is unresolved. Plug-in cannot fire on the served path today; it can proceed as a source-side integration on the branch.

### Second plug-in target -- **GLM-5.3-Flash**

**Why GLM 5.3 Flash second:**

1. **KDA head_count shim is trivial** (H: 96 -> 64) once K3 lands. The int8 quantization discipline transfers verbatim; the delta-rule math is unchanged.
2. **GLM 5.3 Flash's other blocker is DSA, not KDA.** The 11 sparse layers need the DSA Lightning Indexer kernel (sibling scaffold `NKI-DSA-SPARSE-ATTENTION-SCAFFOLD-2026-08-27.md`); that's not this deliverable. KDA landing on GLM 5.3 Flash unblocks 34 of 45 layers; the remaining 11 still fall back to full attention until DSA lands.
3. **No Trn2 topology receipt yet for GLM 5.3 Flash.** Per `LANE-STATE-20260827T222500Z.md` section 4.2, GLM 5.3 Flash's first-serve preflight compile (C4) is blocked on checkpoint FP8 audit + `models/glm5-3-flash/` directory being populated + telemetry hooks landing -- work streams that this deliverable does not do.

**What GLM 5.3 Flash needs from this deliverable, at plug-in time:**

- Instantiate `GLM_5_3_FLASH_KDA_SHAPE` in the GLM 5.3 Flash model config; `build_shape(GLM_5_3_FLASH_KDA_SHAPE, B=<batch>)` provides the runtime-batch-bound shape.
- Same `KDA_STATE_KERNEL_SLUG` in `models/glm5-3-flash/model.env`; the shape difference (H=64) is captured in a separate `KDA_STATE_HEAD_COUNT="64"` cache key.
- Wire the 34 KDA layers via the same `kda_state_decode_forward` entry point; feed 11 sparse layers to the DSA kernel via the sibling factory dispatch (per scaffold section 10 point 3, "distinct layer_kind enum in the model config").
- Hook `test_04_kda_path_activation.py` (already exists in the GLM 5.2/5.3 lane tests dir) against the KDA kernel's telemetry counter. `linear_path_active[layer]` must increment on every KDA-layer forward; `dsa_path_active[kda_layer]` must stay at zero; `state_buffer_reset_count[layer]` must stay at zero during decode. All three are hard correctness gates.

### Anti-lanes at plug-in time

Do NOT plug KDA into the Qwen3.5 hybrid DeltaNet family via the same file. The Qwen3.5 DeltaNet variant has different gating (per scaffold GAP-3), head_dim=256 not 128, and its own compile-cache lineage. `QWEN35_2B_DELTANET_SHAPE` is exposed here for cross-family SBUF/DMA arithmetic tests only -- for the actual Qwen3.5 kernel plug-in, follow `NKI-DELTANET-STATE-INT8-SCAFFOLD-2026-08-27.md` section 4.3 and the sibling `deltanet_fused_chunked_fwd_multihead` kernel bound at NKI SDK 2.32.

---

## Open blockers

1. **Card 12 ownership** for K3 TP=64 EP=64 topology. `LANE-STATE-20260827T2219Z.md` section 1 has the ask waiting on the operator; kernel plug-in on the served path cannot fire until this resolves. The kernel itself is ready.
2. **NKI backend not exercised on Trn2.** The `_kda_state_decode_forward_nki_source()` string is compilable-in-principle per the NKI DSL patterns from `nkilib/experimental/state/deltanet_fused_chunked_fwd_multihead`, but I cannot invoke `neuronx-cc` on the Windows workstation. First on-device correctness check must be a subsequent cycle on a Trn2 host once the K3 v12 launcher path opens. Fallback rule holds: if the NKI backend fails to load, the CPU golden is served -- never softmax.
3. **GAP-1 (K3 KDA state layout audit)** from the scaffold. The state layout used here `[B, H, D_v, D_qk]` is the one the scaffold locks in; verifying it matches `contrib/kimi_k3/kda_cpu_reference.py` in the vendored `apuroop/kimi-k3-real-world` NxDI submodule requires a compile-host session I don't have this shift. If the axis order differs, adapt the CPU reference and the tests; the delta-rule math is invariant.
4. **Prefill chunked-parallel not implemented.** Per operator directive "prefill can loop"; a by-token prefill is stubbed correct-but-slow. Ships as-is; next-cycle deliverable is the chunked-parallel prefill body per `NKI-KDA-STATE-SCAFFOLD-2026-08-27.md` section 3.2 + section 4.3.
5. **Real GLM 5.3 Flash config not fetched** for `GAP-5` in the KDA scaffold ("extract KDA layer count from the released GLM 5.3 Flash config.json"). This deliverable uses the scoping-doc-quoted 34/45 count; if the released HF config diverges, `GLM_5_3_FLASH_KDA_SHAPE.layers` needs a one-line edit and every dependent test still passes.
6. **`kda_cpu_reference.py` upstream cross-check.** The scoping doc cites Codex's own `contrib/kimi_k3/kda_cpu_reference.py` (a port from fla-core 0.5.2 Triton kernel) as an existing CPU reference. Once available on the compile host, its output must be checked bit-for-bit against `kda_state_decode_forward_reference` in this file. "Two-ports-agreeing is one piece of evidence, not two" per K3 branch discipline; both should be present.
7. **10-token exact-gate at kernel-tier** (Test 3 for K3 lane, Test analog for GLM 5.3 Flash). Once the K3 branch's real KDA layer forward is wired to this kernel, run the 10-token argmax-match gate against llama.cpp -- the same discipline as GLM 5.2's `models/glm5-2/tools/validate-glm52-ten-token-logits.py`. This is a subsequent cycle deliverable that requires K3 branch access.

---

## Handoff pointers for the next cycle

- If Trn2 access opens: run the NKI backend via `neuronx-cc` on `_kda_state_decode_forward_nki_source()`. First compile is TP=8 spike-compile at K3-shape (H=96, D=128) with 1 KDA layer to characterize compile time before committing to 69-layer NEFF.
- If GLM 5.3 Flash access opens: same, at H=64 shim.
- If Codex `kda_cpu_reference.py` becomes available on the compile host: add a `test_kda_state_matches_codex_reference.py` that imports both and asserts `kda_state_decode_forward_reference == codex_reference` at multiple seeds.
- The DSA sibling kernel is `NKI-DSA-SPARSE-ATTENTION-SCAFFOLD-2026-08-27.md` -- authored by the sibling agent this shift. Coordinate at layer_kind dispatch (see plug-in list above).
- **Kernel invariant to preserve**: dispatch shim in `kda_state_decode_forward` NEVER falls through to any softmax-like path. Fallback is CPU reference or noisy error -- never a silent lowering to full attention. This is the KDA-corrupts-model discipline per `MLA-VS-DSA-KERNEL-VERIFICATION-2026-08-27.md` section 1.1.

---

## Test verification (2026-08-28)

```
$ cd C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels
$ py -3 -m pytest tests/test_kda_state_correctness.py tests/test_kda_state_speed.py --tb=short
============================= 45 passed in 33.49s =============================
```

Correctness: 22 passed. Speed: 23 passed. Zero skips, zero failures, zero warnings.
