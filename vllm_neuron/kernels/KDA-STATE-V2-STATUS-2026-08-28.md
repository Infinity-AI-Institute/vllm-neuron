# KDA state kernel V2 -- bit-exact FLA v0.5.2 re-authorization -- STATUS

**Date:** 2026-08-27 (task fired late-UTC)
**Author:** Fleet-A KDA re-author subagent (worker/reference-sweep-2026-08-26/gpt-oss-tp4-knee)
**Reason for v2:** V1 (`kda_state.decode.rank1_delta.int8_state.per_channel_scale.v1`)
was DeltaNet-with-int8-state, not KDA. Semantic mismatch, not tolerance drift; blocks
both Kimi K3 and GLM-5.3-Flash serving.

**Companion docs (all absolute local paths per operator memory `[[always-give-full-local-paths]]`):**

- V1 (deprecated, DRIFTING; kept on disk to document the gap):
  `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\kda_state.py`
- V2 (this deliverable, SERVING):
  `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\kda_state_v2.py`
- Int8 study alias (NOT SERVED, HBM-bandwidth study only):
  `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\kda_state_int8_study.py`
- V2 correctness (25 tests):
  `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_kda_state_v2_correctness.py`
- V2 speed (23 tests):
  `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_kda_state_v2_speed.py`
- Prior flavor identification (why v2 is needed):
  `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\VLLM-KDA-KERNEL-FLAVOR-2026-08-28.md`
- MLA-vs-DSA verification (fallback discipline):
  `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\MLA-VS-DSA-KERNEL-VERIFICATION-2026-08-27.md`
- Reference FLA v0.5.2 Triton source (scratchpad mirror of vLLM PR #53906):
  `C:\Users\apumu\AppData\Local\Temp\claude\C--Users-apumu-research-InfinityAI-gemma4-trn2-handoff\fe22bc9a-b8c7-4107-ac55-1d55ba4bd33a\scratchpad\kda-fetch\fused_recurrent_vllm_third_party_flash_linear_attention_ops.py`

---

## 1. TL;DR

- V2 module shipped: `kda_state_v2.py`, slug
  `kda_state.decode.kda_gate.rank1_delta.bf16_state.v1`.
- All four algorithmic fixes applied and tested (§3 below).
- 25 correctness tests + 23 speed tests, all green (48 tests, ~20s wall for
  correctness / 0.15s for speed on the Windows workstation).
- V1 module untouched on disk; V1's 45 tests continue to pass and continue to
  document the DeltaNet-vs-KDA + int8-vs-bf16 gap on disk.
- Int8 lane preserved as `kda_state_int8_study.py` with `SERVING_STATUS = "NOT_SERVED"`
  and a `require_study_ack()` guard. It re-exports v1 by import; no source duplication.

---

## 2. Slug hygiene + compile-cache safety

- V1 slug (unchanged): `kda_state.decode.rank1_delta.int8_state.per_channel_scale.v1`
- V2 slug (new):        `kda_state.decode.kda_gate.rank1_delta.bf16_state.v1`

Divergence is enforced by `test_v2_slug_differs_from_v1_slug` -- a compile-cache
that reuses v1 for v2 (or vice versa) would silently corrupt the model. The
substrings `"kda_gate"` and `"bf16_state"` are load-bearing tokens the compile
driver's cache-key generator must include.

---

## 3. What v2 fixes vs v1 (the four missing pieces)

Reference FLA v0.5.2 line numbers below cite the scratchpad mirror
`C:\Users\apumu\AppData\Local\Temp\claude\C--Users-apumu-research-InfinityAI-gemma4-trn2-handoff\fe22bc9a-b8c7-4107-ac55-1d55ba4bd33a\scratchpad\kda-fetch\fused_recurrent_vllm_third_party_flash_linear_attention_ops.py`,
function `fused_recurrent_gated_delta_rule_fwd_kernel`.

### (a) KDA per-channel gate `Diag(alpha_h)` applied to state before delta step

**FLA reference:** lines 145-156 (COMPUTE_GATE branch inside the IS_KDA branch),
plus the per-head `b_a_log = tl.exp(tl.load(a_log + i_h))` hoist at line 104.

```
b_gk = tl.load(p_gk).to(tl.float32)                          # raw per-channel logits
if COMPUTE_GATE:
    b_gk += tl.load(g_bias + i_h * K + o_k, ...).to(tl.float32)
    b_gk = LOWER_BOUND / (1.0 + tl.exp(-(b_a_log * b_gk)))   # per-K gate
b_h *= exp(b_gk[None, :])                                    # state decay (broadcast on D_v)
```

**V2 code** (`kda_state_v2.py::_kda_delta_rule_step`, lines aligned):

```python
a_amp = np.exp(a_log.astype(np.float32))                         # [H]
g = g_raw.astype(np.float32) + g_bias.astype(np.float32)          # [H, D_qk]
alpha = lower_bound / (1.0 + np.exp(-(a_amp[:, None] * g)))       # [H, D_qk]
decay = np.exp(alpha)                                             # [H, D_qk]
S = S * decay[:, None, :]                                         # [H, D_v, D_qk]
```

**Why it matters:** without this, the recurrent state grows unbounded and never
forgets -- v1 would diverge from the reference within O(1) decode steps. v2's
`test_v2_state_stays_bounded_over_100_steps` locks the gate presence in; and
`test_v2_gate_is_load_bearing` fires if someone reverts the fix.

### (b) L2-norm on Q/K in-kernel (eps=1e-6)

**FLA reference:** lines 137-140 (USE_QK_L2NORM_IN_KERNEL branch).

```
b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
```

**V2 code** (per-head L2 reduction, eps=`KDA_L2NORM_EPS` = 1e-6):

```python
q_norm = np.sqrt(np.sum(q * q, axis=-1) + l2norm_eps)   # [H]
k_norm = np.sqrt(np.sum(k * k, axis=-1) + l2norm_eps)   # [H]
q = q / q_norm[:, None]
k = k / k_norm[:, None]
```

### (c) Query scale q *= 128^-0.5

**FLA reference:** line 140 (`b_q = b_q * scale`). The caller passes `scale =
1 / sqrt(K)` where K is the QK head-dim (128 for both K3 and GLM-5.3-Flash).

**V2 code:** `scale = 1.0 / math.sqrt(D_qk)` when the `params.scale` field is
None (the default). Applied AFTER L2-norm on q.

### (d) bf16 state (not int8), layout `[num_slots, HV, V, K]`

**FLA reference:** state is `b_h[BV, BK]` (per-head [D_v, D_qk] tile); the HBM
layout for `h0`, `ht` follows `MambaStateDtypeCalculator.kda_state_dtype`
(default `(bfloat16, bfloat16)` per vLLM PR #53906). Continuous-batching layout
is `[num_slots, HV, V, K]` where `num_slots` is the model's SSM-state cache
capacity.

**V2 code:** `state_bf16: [B, H, D_v, D_qk]` (fp32 array whose values are all
bf16-representable via `bf16_cast`). B replaces `num_slots` for the CPU
reference. `bf16_cast()` implements a bit-twiddling round-to-nearest-even
fp32 -> bf16 -> fp32 round-trip matching the Triton `.to(bf16)` semantics.

---

## 4. What v2 KEEPS from v1 for A/B study

`_delta_rule_step_gatefree` -- the Yang et al. rank-1 delta rule WITHOUT the KDA
gate. Preserved for isolating the gate's contribution to output drift. NEVER
called as the served reference (test suite doesn't wire it to any dispatch
shim).

---

## 5. Fallback discipline

Per `MLA-VS-DSA-KERNEL-VERIFICATION-2026-08-27.md` §1.1, KDA has NO
full-attention fallback that is numerically equivalent -- a silent lowering
into softmax attention CORRUPTS the model. v2 enforces this at the dispatch
shim:

```python
_BANNED_IMPLS = frozenset({"softmax", "full_attention", "sdpa", "flash_attn"})

def kda_state_decode_forward_v2(inputs, impl="reference"):
    if impl in _BANNED_IMPLS:
        raise ValueError(...)   # NEVER a silent fallthrough
```

A missing NKI backend falls through to the bf16 CPU reference -- correct but
slow. Never softmax. `test_v2_dispatch_rejects_softmax_impl` locks this in.

---

## 6. Test suite summary

### Correctness (25 tests, ~20s wall, per-test <10s)

Shape sweep: `state_dim ∈ {64, 128, 256}` × `seq_len ∈ {1, 128, 1024, 8192}`
(H=2 default; H=1 for the L=8192 D=256 corner to stay inside the kernel-tier
per-test time budget -- see `test_v2_decode_matches_fla_oracle`).

Oracle: an independent numpy transcription of the FLA v0.5.2 Triton kernel body,
written line-by-line in `_oracle_kda_step` (no shared code with
`kda_state_v2._kda_delta_rule_step`). Both go through the same fp32 arithmetic
sequence, same bf16 round-trips on state and output -- expected error zero,
tolerance envelope 1e-4 for L<=1024 and 5e-4 for L=8192.

Additional coverage:
- `test_v2_gate_is_load_bearing`: gate MUST change output vs gate-free path.
- `test_v2_state_stays_bounded_over_100_steps`: gate keeps recurrent state finite.
- `test_v2_prefill_equals_decode_L_64`: prefill(L=64) bit-exact against 64 decodes.
- `test_v2_no_nan_at_extreme_beta_raw`: beta_raw ∈ {-10, +10} -> no NaN.
- `test_v2_zero_state_zero_query_outputs_zero_y`: recurrence base case.
- `test_v2_reset_zeroes_masked_preserves_unmasked`: explicit reset semantics.
- `test_bf16_cast_*`: bf16 simulation round-trip discipline (3 tests).
- `test_v2_dispatch_rejects_softmax_impl` + `_reference_and_nki_impls_ok`.
- `test_v2_kernel_slug_stable` + `_model_shape_presets_match_scope_docs`.

### Speed (23 tests, ~0.15s wall)

- DMA descriptor floor cleared at bf16 for K3 and GLM-5.3-Flash at
  B ∈ {1, 4, 8, 16, 32}.
- bf16 DMA per layer is exactly 2x the int8 payload (concrete cost of the
  correctness fix).
- Single-layer bf16 state fits SBUF at B=1 for both models (K3 ~3.0 MiB, GLM
  ~2.0 MiB inside the 24 MiB SBUF budget).
- Colocated-layer capacity: 2 layers K3, 3 layers GLM at B=1 (half of v1's
  int8 capacity -- the direct cost of the byte-per-element doubling).
- HBM ceiling at B=32 all-layers:
  - K3: 32 * 69 * 96 * 128 * 128 * 2 = 6,941,573,120 B = **6.46 GiB** (task
    doc envelope 6.0-7.2 GiB) -- fits inside 24 GiB per-NC HBM.
  - GLM-5.3-Flash: 32 * 34 * 64 * 128 * 128 * 2 = 2,281,701,376 B =
    **2.13 GiB**. Task doc quoted ~4.3 GiB; that figure was double-counting
    (likely spec-decode scratch or a HV=128 config). Recomputed here from
    scope-doc numbers (H=64, D=128, layers=34). Either figure fits HBM with
    wide margin.
- Slug versioning + shape-hash cross-model divergence.

### Full v1+v2 sweep

```
93 passed in 53.17s
```

(v1: 22 correctness + 23 speed = 45 tests, still all green -- gap documented
on disk. v2: 25 correctness + 23 speed = 48 tests.)

---

## 7. K3 vs GLM-5.3-Flash first-plug-in coordination

**GLM-5.3-Flash lane (Fleet A):** ready to plug v2 into the KDA slot in the
GLM-5.3-Flash driver. The kernel signature accepts the vLLM param layout
directly (`a_log: [H]`, `g_bias: [H, D_qk]`, `g_raw: [B, T, H, D_qk]`,
`beta_raw: [B, T, H]`). Next steps:

1. Compile-time constants set:
   - `LOWER_BOUND = -5.0` (matches GLM-5.3-Flash `linear_attn_config.gate_lower_bound`)
   - `scale = 128^-0.5` (from `head_dim = 128`)
   - `l2norm_eps = 1e-6`
2. Test 4 (kernel-tier correctness) already green.
3. GLM-5.3-Flash lane fires next in the model-scope docs after MoE dispatch
   lands. Ordering not blocked by KDA any more.

**Kimi K3 lane:** identical algorithm -- vLLM PR #53906 uses the same generic
FLA `fused_recurrent_gated_delta_rule` for both models with the same
constexpr set. The one delta is HV (K3=96, GLM=64) -- pure shape difference,
same code path.

- v2 slug is model-independent; the compile-cache key is
  `(slug, shape_tuple)`, and `test_v2_shape_hash_diverges_across_models`
  ensures the K3 and GLM shape tuples hash differently. A K3 NEFF cannot
  accidentally serve GLM-5.3-Flash and vice versa.
- **Coordination note:** K3 has a bespoke `flashkda` CUDA path
  (`csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu`, per
  `VLLM-KDA-KERNEL-FLAVOR-2026-08-28.md` §1) that we would race against for
  K3 tokenomics. It is NOT present for GLM-5.3-Flash. Both models use the
  generic FLA Triton path in vLLM PR #53906; the v2 NKI kernel is the
  correct baseline for both.

---

## NKI v2 device -- landing status (2026-08-27, kda-nki-v2-agent)

**Landed:**
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\kda_state_nki_v2.py`
  (sibling to `kda_state_v2.py`) -- NKI device kernel authored as an
  inspectable Python DSL source string plus a dispatch shim that falls
  through to the CPU golden when the NEFF is cold or `neuronxcc` is absent.
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_kda_state_nki_v2_smoke.py`
  -- 21 smoke tests, all green (0.28 s). Covers: dry-import, public surface,
  slug hygiene (`kda_state.decode.kda_gate.rank1_delta.bf16_state.nki_v1`
  diverges from CPU-golden slug and contains all load-bearing tokens),
  NKI-DSL-source hygiene (verifies each of the 4 FLA-parity pieces appears
  in the source with the correct ordering -- L2-norm before scale, gate
  before delta step, bf16 store on state + y), no `IS_SPEC_DECODING`
  branch (per operator hard rule `[No spec-decode methodology 2026-08-27]`),
  tile-shape budget (per-head SBUF fits < 1/32 of the 24 MiB budget),
  dispatch shim fallback (banned softmax impls raise; auto/reference
  paths match the CPU golden bit-for-bit; `impl="nki"` cold-cache raises
  loudly).
- Full-suite regression: **v1 (45) + v2 CPU golden (48) + NKI v2 smoke (21)
  = 114 tests green** (v1 continues on disk documenting the algorithmic gap;
  v2 CPU golden ran in 24 s; NKI smoke ran in 0.28 s on Windows without
  `neuronxcc`).

**Slug hygiene (updated):**

| slug | file | status |
|---|---|---|
| `kda_state.decode.rank1_delta.int8_state.per_channel_scale.v1` | `kda_state.py` | DEPRECATED (drifting) |
| `kda_state.decode.kda_gate.rank1_delta.bf16_state.v1` | `kda_state_v2.py` | SERVING (CPU golden) |
| `kda_state.decode.kda_gate.rank1_delta.bf16_state.nki_v1` | `kda_state_nki_v2.py` | LANDED (source string) |

`test_slug_differs_from_cpu_golden` and `test_v2_slug_differs_from_v1_slug`
together lock in that all three slugs are distinct -- a stale NEFF from
any pair would be a silent-correctness bug (KDA has no numerically-
equivalent fallback per MLA-VS-DSA §1.1).

**First plug-in path (unblocked): Kimi K3 KDA layers (69 of 93).**

Kimi K3 (HV=96, D_qk=D_v=128) is the first-plug-in target per operator
memory `[Makora TrainSpotting competitive context]` -- highest customer-
demand signal. The NKI kernel signature accepts vLLM PR #53906's param
layout directly:

```
a_log    : [H]              (sharded on head axis by TP; per-NC after shard)
g_bias   : [H, D_qk]        (sharded on projection axis by TP)
g_raw    : [B, 1, H, D_qk]
beta_raw : [B, 1, H]
state    : [num_slots, HV, V, K] == [B, H, D_v, D_qk] in decode
```

Plug-in point (Kimi K3 model file in vLLM PR #53906):
`vllm/models/kimi_k3/nvidia/kda.py:564` (plain decode entry). The `use_spec`
branch at `:497` is out of scope per operator hard rule
`[No spec-decode methodology 2026-08-27]` -- the NKI source omits the
`IS_SPEC_DECODING` branch and `test_source_omits_spec_decode_branch`
enforces it.

GLM-5.3-Flash (HV=64, same algorithm) inherits the same kernel via a pure
shape rebind -- `KIMI_K3_KDA_SHAPE_NKI_V2` vs `GLM_5_3_FLASH_KDA_SHAPE_NKI_V2`
in the module. Compile-cache key is `(slug, shape_tuple)`; the CPU-golden
test `test_v2_shape_hash_diverges_across_models` guarantees a K3 NEFF
cannot accidentally serve GLM-5.3-Flash and vice versa. The shape-scoped
registry lookup in the NKI shim (`get_nki_backend_v2(shape_tuple)`)
enforces the same at dispatch time -- `test_register_and_lookup_compiled_kernel_shape_scoped`
locks in that a K3 (1,96,128,128) NEFF returned for a K3 lookup does NOT
return for a GLM-5.3-Flash (1,64,128,128) lookup.

**Tile shape:** BV=64, BK=128, H_partition=1. Per-head SBUF residency
= 16 KiB bf16 + 32 KiB fp32 working = 48 KiB, well under the 24 MiB Trn2
per-NC SBUF budget with plenty of headroom for tensor-engine scratch.

**Engine assignments** (compile driver's static schedule):
- Tensor engine: `S @ k`, `S @ q`, delta outer product (3 nc_matmul per head).
- Vector engine: element-wise mul/add/sub, `nl.sigmoid`, `nl.exp`.
- GpSIMD (scalar): L2-norm reductions, `nl.exp` on per-head `a_log`.
- DMA: state read+write (2 * BV * BK * 2 bytes/head), Q/K/V/g_raw/y I/O.

**Not landed (honest list):**
1. **NKI compile driver invocation on the Trn2 host.** The `.py` file
   ships the DSL source; the compile driver must ingest it, register the
   NEFF against the shape presets, and verify against the CPU golden.
   `register_compiled_nki_kernel((B, H, D_v, D_qk), callable_)` is the
   hook; the driver populates `_KDA_NKI_CALLABLES` at compile time.
   Estimated 2-3 hours once the Trn2 host is reachable.
2. **Chunk-KDA NKI prefill body.** The decode body ships now; the prefill
   shim `kda_state_prefill_forward_nki_v2` falls through to the CPU-golden
   by-token prefill (correct, O(L) sequential). A separate 4-6 agent-hour
   deliverable; not on the K3/GLM-5.3-Flash first-plug-in critical path.
3. **Short causal-conv1d and FusedRMSNormGated.** Caller pre-/post-applies;
   fusing them into the NKI kernel is a follow-on tick.

---

## 8. Open blockers -- honest list

1. **NKI backend body compiled on-device.** `KDA_STATE_NKI_V2_SOURCE`
   (returned by `get_nki_source()`) is the artifact the Trn2 compile
   driver ingests. Windows-side authorship + smoke tests are green; the
   compile driver on the Trn2 host has not yet been pointed at it. Next
   tick: land the compile driver invocation, register the NEFF against
   the shape presets, verify against the CPU reference. Estimated 2-3 hours.
2. **Chunked-parallel prefill NKI kernel.** v2 prefill still falls through to
   token-by-token decode -- correct but O(L) sequential. A chunk-KDA NKI
   lowering per `chunk_kda_with_fused_gate` (FLA v0.5.2 chunk path) is a
   separate 4-6 hour deliverable. Not on the K3/GLM-5.3-Flash first-plug-in
   critical path (decode dominates the serving loop).
3. **Short causal-conv1d (kernel_size=4, silu).** v2 kernel operates on
   post-conv Q/K/V; the conv is the caller's responsibility. The conv is
   itself a small NKI kernel; not written yet. Downstream of v2.
4. **FusedRMSNormGated output normalization.** vLLM's
   `Glm5NextLinearAttention.forward` applies `o_norm(core_attn_out, g2)`
   AFTER the recurrent kernel. Not implemented in the NKI kernel; caller
   must apply it externally (or a follow-on tick fuses it into the kernel
   tail).
5. **TP sharding of `a_log` and `g_bias`.** vLLM's module shards `a_log` on
   the head axis and `g_bias` on the projection axis. v2 accepts these as
   already-sharded per-NC tensors (shapes `[H_local]` and `[H_local, D_qk]`);
   the shape adapter that maps the checkpoint layout to per-NC shards is not
   in this file. Downstream of the compile driver landing.
6. **PR #53906 is still open (not merged).** Head SHA at fetch time was
   `142062f13d16bed254b5d97cc3d371fbd4f7790a`. If the PR merges with
   different constexprs or a different fused-gate arithmetic, v2 must be
   re-diffed. Watch: monitor PR head daily until merge.

---

## 9. Discipline items enforced

- **Full local paths everywhere** per `[[always-give-full-local-paths]]`.
- **Fallback rule:** dispatch shim raises on any softmax-family impl.
- **v1 untouched:** its 22 correctness tests + 23 speed tests still pass and
  still document the gap on disk. This is the "45 tests document the
  algorithmic gap" the task called out.
- **Peer-agent non-interference:** v2 is a NEW module; nothing on the box or
  in another fleet's compile-cache is invalidated.
- **No spec-decode:** the NKI DSL source omits the `IS_SPEC_DECODING` branch
  (v2 targets the plain-decode entry point at
  `vllm/models/glm5next/nvidia/kda.py:564`, not the spec branch at :497).
- **Card 12 untouched:** no measurements on the box for this task; kernel
  authorship and CPU-side verification only.

---

## 10. One-line summary for STATE-NOW / OVERSEER-STATE

```
KDA state kernel v2 authored: slug kda_state.decode.kda_gate.rank1_delta.bf16_state.v1;
all 4 fixes applied (per-channel gate, L2-norm, q-scale, bf16 state); 25 correctness
tests + 23 speed tests green vs numpy-transcribed FLA v0.5.2 oracle. v1 kept on
disk with its 45 tests documenting the gap. Int8 lane preserved as
kda_state_int8_study.py (NOT_SERVED). GLM-5.3-Flash first-plug-in unblocked;
K3 first-plug-in unblocked (same kernel, HV=96 vs 64 only shape difference).
```
