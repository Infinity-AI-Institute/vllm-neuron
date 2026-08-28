# vLLM's KDA kernel flavor for GLM-5.3-Flash — identified, and the budget impact on our NKI kernel

**Date:** 2026-08-27 (early-UTC)
**Author:** Fleet-A KDA-flavor subagent (Trn2 campaign, worker/reference-sweep-2026-08-26/gpt-oss-tp4-knee)
**Companion docs (all absolute):**

- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\GLM-5-3-FLASH-ARCHITECTURE-2026-08-28.md` (architecture deepdive; this closes §15 unknown #3)
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\kda_state.py` (our shipped NKI-KDA state kernel + reference)
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\NKI-KDA-STATE-SCAFFOLD-2026-08-27.md` (kernel scaffold)
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\MLA-VS-DSA-KERNEL-VERIFICATION-2026-08-27.md` (why silent softmax fallback is banned)

**Fetched at PR head:** vLLM PR #53906 `[Model] add GLM-5.3-Flash support`, branch `glm-release` on `ZJY0516/vllm`, head commit `142062f13d16bed254b5d97cc3d371fbd4f7790a` (committed 2026-08-27T08:01:05Z, open).

**Scratchpad copy of all fetched files (local):**
- `C:\Users\apumu\AppData\Local\Temp\claude\C--Users-apumu-research-InfinityAI-gemma4-trn2-handoff\fe22bc9a-b8c7-4107-ac55-1d55ba4bd33a\scratchpad\kda-fetch\kda_vllm_models_glm5next_nvidia.py` (604 lines)
- `.../scratchpad/kda-fetch/model_vllm_models_glm5next_nvidia.py` (1206 lines)
- `.../scratchpad/kda-fetch/kda_vllm_third_party_flash_linear_attention_ops.py` (1723 lines)
- `.../scratchpad/kda-fetch/fused_recurrent_vllm_third_party_flash_linear_attention_ops.py` (656 lines)
- `.../scratchpad/kda-fetch/kda_vllm_models_kimi_k3_nvidia.py` (1017 lines) — for K3 comparison
- `.../scratchpad/kda-fetch/flashkda.cmake` (K3-only, DOES NOT apply to GLM-5.3-Flash)
- `.../scratchpad/kda-fetch/fused_kda_decode_kernel.cu` (K3-only)
- `.../scratchpad/kda-fetch/benchmark_kimi_k3_kda_decode.py`

---

## 1. TL;DR

**Kernel flavor:** Vendored `flash-linear-attention` (FLA) library, generic `fused_recurrent_gated_delta_rule` Triton kernel with the KDA constexpr branch enabled (`IS_KDA=True`, `SIGMOID_BETA=True`, `COMPUTE_GATE=True`, `SAFE_GATE=True`, `LOWER_BOUND=-5.0`, `USE_QK_L2NORM_IN_KERNEL=True`, `INPLACE_FINAL_STATE=True`). The FLA source pins to upstream `fla-org/flash-linear-attention` **v0.5.2** (2026-07-27) — the same version Codex's `contrib/kimi_k3/kda_cpu_reference.py` was ported from.

**File paths (in-repo of `vllm-project/vllm`, PR #53906):**
- Decode + spec-verify: `vllm/models/glm5next/nvidia/kda.py` → calls `fused_recurrent_kda(...)` from `vllm/third_party/flash_linear_attention/ops/kda.py:138`.
- Prefill: same file → calls `chunk_kda_with_fused_gate(...)` from same module at `:1550`.
- Kernel body (generic + KDA branch, Triton): `vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:27`, function `fused_recurrent_gated_delta_rule_fwd_kernel`.
- Layer wrapper base class: `GatedDeltaNetAttention` from `vllm/model_executor/layers/mamba/gdn/base.py` — GLM-5.3-Flash sub-classes and adds the bounded-gate override (`self.kda_safe_gate = True`, `self.kda_lower_bound = -5.0`).

**Not FlashInfer. Not TileLang. Not the bespoke `flashkda` CUDA project.**

- FlashInfer: applies only to the DSA (sparse-MLA) layers, per the vLLM recipe.
- TileLang: SGLang-only; not in the vLLM path.
- FlashKDA (`csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu` + `cmake/external_projects/flashkda.cmake`, pinned commit `ee0be888cd0e972f9409bf53756f8c38c6652173` of `github.com/vllm-project/FlashKDA`): **Kimi K3 exclusive.** GLM-5.3-Flash's `glm5next/nvidia/kda.py` does not import `flashkda`, does not call `torch.ops._flashkda_C.fwd`, does not register `is_fused_kda_decode_supported`. Grep of the GLM-5.3-Flash sources for `flashkda` and `fused_kda_decode` returns zero hits.

**Bit-exact vs numerically-similar characterization:** vLLM's in-kernel fused-gate path is bit-for-bit equivalent to FLA v0.5.2's separate `kda_gate_fwd_kernel` + `fused_recurrent_kda_fwd_kernel` composition — the vLLM source comment (`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:148-151`) documents this explicitly: "Replicates kda_gate_fwd_kernel's SAFE_GATE branch bit-for-bit (same `tl.exp`, same fp32 math; the intermediate gate value this replaces was stored/reloaded as fp32, which is lossless)." Codex's CPU reference, per operator, is a "fla-core 0.5.2 Triton port". Chain of equivalence therefore holds: vLLM Triton kernel ≡ upstream FLA v0.5.2 SAFE_GATE branch ≡ Codex CPU reference.

**Impact on our NKI KDA kernel budget: we are DRIFTING, not golden-matched.**

Our shipped `kda_state.decode.rank1_delta.int8_state.per_channel_scale.v1` (see `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\kda_state.py`) implements a plain **Yang et al. 2024 gated-delta-rule** step; it is missing three algorithmic elements that vLLM/FLA v0.5.2 KDA has, and it stores state in per-channel int8 whereas the reference stores in bf16. This is a semantic mismatch, not a tolerance-drift; the model output will diverge from the reference within a few decode steps and monotonically thereafter. Full delta below in §4.

---

## 2. The actual algorithm vLLM runs, step by step

**Per-token fp32 body of `fused_recurrent_gated_delta_rule_fwd_kernel` with the GLM-5.3-Flash constexpr set:**

```python
# vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:132-173, IS_KDA=True branch
b_q = load(q_t).float()
b_k = load(k_t).float()
b_v = load(v_t).float()
if USE_QK_L2NORM_IN_KERNEL:                                # TRUE for GLM-5.3-Flash
    b_q = b_q / sqrt(sum(b_q * b_q) + 1e-6)                # L2 norm, eps=1e-6
    b_k = b_k / sqrt(sum(b_k * b_k) + 1e-6)
b_q = b_q * scale                                          # scale = K^-0.5 = 128^-0.5
# Per-channel gate — the KDA-specific piece:
b_gk = load(g_raw_t).float() + load(g_bias[i_h]).float()   # add per-head bias
b_gk = LOWER_BOUND / (1.0 + exp(-exp(a_log[i_h]) * b_gk))  # = -5.0 * sigmoid(exp(A_log_h)*(g+bias))
b_h *= exp(b_gk[None, :])                                  # per-channel decay: state *= exp(gk)
# Delta rule:
b_v -= sum(b_h * b_k[None, :], axis=1)                     # b_v = v - S @ k
if SIGMOID_BETA:                                           # TRUE for GLM-5.3-Flash
    b_beta = tl.sigmoid(load(beta_t).float())              # beta raw logits → sigmoided in-kernel
b_v *= b_beta                                              # (v - S@k) * beta
b_h += b_v[:, None] * b_k[None, :]                         # S += beta * (v - S@k) ⊗ k
b_o = sum(b_h * b_q[None, :], axis=1)                      # y = S @ q
store(o_t, b_o.to(bf16))
# state (b_h) advances in-place in fp32 registers between tokens; written to HBM in bf16 on the last step
```

**State layout in HBM (h0, ht):** `[num_state_tokens, HV, V, K]` = `[num_slots, 64, 128, 128]` per KDA layer for GLM-5.3-Flash. Stored in **bf16** by default (`MambaStateDtypeCalculator.kda_state_dtype` returns `(bf16, bf16)` for the standard config; can be overridden by `--mamba-cache-dtype`).

**Fixed constants for GLM-5.3-Flash:**
- `LOWER_BOUND = -5.0` (from `linear_attn_config.gate_lower_bound = -5.0`)
- `scale = 128^-0.5 ≈ 0.08838834765` (K = head_dim = 128)
- L2-norm eps = `1e-6`
- Beta arrives as raw scalar logit per-head (shape `[1, T, H]`), sigmoided in-kernel.
- `g` arrives as raw per-channel logits (shape `[1, T, H, K]`), gate computed in-kernel (avoids one extra kernel launch + fp32 intermediate `[T, H, K]` in HBM).
- `a_log` is a per-head learned scalar (shape `[H]`), `g_bias` is a per-head per-channel learned vector (shape `[H, K]`, called `dt_bias` in the module — sharded 0 across TP).

**Two entry points** in `vllm/models/glm5next/nvidia/kda.py`:

1. **Prefill / chunked-parallel** (`num_prefills > 0` branch) → `chunk_kda_with_fused_gate(...)` at line 532. Uses `chunk_kda_with_fused_gate_fwd` at `flash_linear_attention/ops/kda.py:1470`. Same algorithm, chunked-parallel form (BT=32/64/128 autotuned), takes pre-sigmoided beta (chunk kernel does not sigmoid internally — see line 539 `_cast_sigmoid(beta_ns.squeeze(0))`).
2. **Decode / spec-verify** (`num_decodes > 0` branch, and the mirrored `use_spec` branch) → `fused_recurrent_kda(...)` at line 497 (spec) and 564 (plain decode). Sigmoids beta in-kernel (`sigmoid_beta=True`), computes gate in-kernel (`compute_gate=True`), writes state in-place (`inplace_final_state=True`).

**Numerical equivalence between the two paths** is asserted by the vLLM authors in the fused-gate comment cited above and by the chunk vs recurrent hand-off logic in `_forward` (both write to the same `recurrent_state` HBM tensor). The chunk kernel is the throughput path for prefill; the recurrent kernel is what our NKI decode kernel is being written against.

---

## 3. FLA library version identification

**vLLM vendors the source, does not link to `fla-core` as a package.** The vendored path is `vllm/third_party/flash_linear_attention/ops/*.py`. Each file bears the header:

```
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
```

The vendored code is a **structural refactor of upstream `fla-org/flash-linear-attention` v0.5.2** (released 2026-07-27). Two evidence points:

1. **Constexpr set matches v0.5.2 exactly.** Upstream v0.5.2 `fla/ops/kda/fused_recurrent.py` defines `fused_recurrent_kda_fwd_kernel` with `USE_LOWER_BOUND` (`lower_bound is not None`), `LOWER_BOUND` scalar, SAFE_GATE handling, `USE_QK_L2NORM_IN_KERNEL`, sigmoid-beta branch. vLLM's vendored source folds the same set into the general `fused_recurrent_gated_delta_rule_fwd_kernel` via `IS_KDA` / `COMPUTE_GATE` / `SAFE_GATE` / `SIGMOID_BETA` constexpr branches, but the arithmetic in each branch is identical.
2. **The vLLM refactor rationale (in-kernel fused gate)** matches an optimization landed upstream between v0.5.1 and v0.5.2. In v0.5.2 the gate is still a separate `kda_gate_fwd_kernel`; the fused-in-kernel form is vLLM-local. The vLLM source comment cited above notes bit-for-bit equivalence to `kda_gate_fwd_kernel`.

**Latest upstream FLA releases (for the record):** v0.5.2 (2026-07-27), v0.5.1 (2026-06-18), v0.5.0 (2026-04-21). Upstream `fla/ops/kda/` has files `chunk.py`, `chunk_bwd.py`, `chunk_fwd.py`, `chunk_intra.py`, `chunk_intra_token_parallel.py`, `fused_recurrent.py`, `gate.py`, `naive.py`, `wy_fast.py`, `backends/`.

**Codex's `contrib/kimi_k3/kda_cpu_reference.py`** was specified by the operator as a "fla-core 0.5.2 Triton port". Given (1) v0.5.2 was the latest upstream at time of writing and (2) vLLM's vendored source is a v0.5.2 refactor with bit-for-bit equivalence documented in-source, our expected numerical parity chain is:

```
vLLM decode kernel (Triton, in-kernel fused gate)
    ≡ upstream FLA v0.5.2 fused_recurrent_kda + kda_gate composition
    ≡ Codex contrib/kimi_k3/kda_cpu_reference.py
```

Codex's CPU reference is therefore the authoritative bit-exact target our NKI kernel must match. This closes GAP-1 from `NKI-KDA-STATE-SCAFFOLD-2026-08-27.md`.

---

## 4. Delta from our shipped `kda_state.decode.rank1_delta.int8_state.per_channel_scale.v1`

Reading `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\kda_state.py` `_delta_rule_step` line 70-107 as ground truth:

| element | vLLM/FLA reference | Our shipped `kda_state.py` reference | delta |
|---|---|---|---|
| **KDA per-channel gate** `b_h *= exp(-5.0 * sigmoid(exp(A_log_h) * (g_t + g_bias_h)))` (per K channel, applied to state **before** delta step) | **PRESENT** — the defining "KDA" step; consumes `a_log`, `g_bias` (called `dt_bias`), and raw `g` inputs | **MISSING** — reference doesn't take `A_log`, `g_bias`, or `g`; the delta step runs against undecayed prior state | **algorithmic**: our shipped kernel is DeltaNet, not KDA. Without this decay, the recurrent state grows unbounded and forgets nothing — the model output diverges within O(1) decode steps. |
| **L2-norm on Q and K** (in-kernel, eps=1e-6) | **PRESENT** — `USE_QK_L2NORM_IN_KERNEL=True` | **MISSING** — assumes Q/K pre-normalized by caller | **contract**: silent divergence if caller doesn't pre-normalize; can be fixed by promoting to an in-kernel step or by asserting a normalized-input precondition at the reference boundary. |
| **Query scale** `q *= 1/sqrt(K)` = `q *= 128^-0.5 ≈ 0.0884` | **PRESENT** — applied inside kernel after L2-norm, before state @ q | **MISSING** — reference does raw `S @ q` | **numerical**: output magnitude off by `sqrt(K) ≈ 11.31×` per layer, compounding as `~11.31^34 ≈ 10^36×` if not corrected. |
| **Sigmoid on beta** | **PRESENT** — `SIGMOID_BETA=True`, beta arrives as raw logit and is sigmoided in-kernel | **AMBIGUOUS** — reference multiplies by beta directly, expects sigmoided input; caller contract not enforced | **contract**: fix by adding an assertion at the CPU reference boundary that beta ∈ [0, 1], or by adding a `sigmoid_beta` toggle to the shim (matching vLLM's flag). |
| **State dtype in HBM** | bf16 (per `MambaStateDtypeCalculator.kda_state_dtype`, default; overridable via `--mamba-cache-dtype`) | **int8 per-channel absmax with fp32 per-D_v scale** (our design choice for Trn2 SBUF-residency) | **numerical**: our int8 quant adds ~1/127 ≈ 0.79% quantization noise per step per channel. Independent noise; over 34 KDA layers × T decode steps in a request, expected accumulated RMS error is ~sqrt(34·T)·0.79%. At T=1024 this is ~148%, i.e. state is destroyed. **This is untenable unless (a) matched by an int8 reference for the accuracy gate, or (b) revised to bf16 for correctness parity and int8 kept only as a study lane.** |
| **State layout `[H, D_v, D_qk]`** with D_v as slow axis, D_qk as fast axis | vLLM kernel indexes `b_h[BV, BK]` with V outer, K inner — matches our layout convention | **MATCHES** | none |
| **Beta shape** — per-head scalar (`H`) | supported (`IS_BETA_HEADWISE=False` for GLM-5.3-Flash: beta stored as `[bos*HV + i_hv]`) | **MATCHES** — `[H]` per token | none |
| **Short-conv1d on Q/K/V** (kernel_size=4, silu activation, per-channel independent) | **PRESENT** — `causal_conv1d_fn` (prefill) / `causal_conv1d_update` (decode) upstream of the recurrent step | **NOT MODELED** — reference expects post-conv q/k/v inputs | **contract**: needs a companion "short-conv" reference kernel; the recurrent-step reference alone is correct if the caller pre-applies the short conv. Document at the API boundary; not a numerics-parity blocker on the recurrent step itself. |
| **Output gate: FusedRMSNormGated** (RMSNorm on head-dim with sigmoid gate `x * sigmoid(g)`, eps=1e-5, elementwise-affine weight) | **PRESENT** — applied by the enclosing `Glm5NextLinearAttention.forward` at line 322 (`o_norm(core_attn_out, g2)`) | **NOT MODELED** — reference outputs pre-o-norm y | **contract**: same as short-conv — this is a companion post-recurrence step. The recurrent-step reference is correct with the pre-norm output; the o-norm should be a separate reference tested against `FusedRMSNormGated.forward_native` at `flash_linear_attention/ops/kda.py:505`. |

**None of the missing elements is a "close-enough" numerical drift.** The KDA gate is the algorithmic identity of the KDA operator; the L2-norm and query-scale are load-bearing for numerical stability; the int8 state quantization multiplies against a bf16-reference expectation.

---

## 5. Recommended budget + slug hygiene

**Recommendation** (do not implement in this task — deliverable is this document; the operator's `.claude/agents/*.md` fleet will pick up the fixes):

1. **Bump slug to `kda_state.decode.kda_gate.rank1_delta.bf16_state.v1`** as the primary correctness-parity target. Match FLA v0.5.2 bit-close (within Trn2 bf16 fp accumulation ULP for the `tl.exp` and `tl.sigmoid` intrinsics).
2. **Keep `kda_state.decode.kda_gate.rank1_delta.int8_state.per_channel_scale.v2` as a study lane**, with an explicit note in its slug docstring that it's a Trn2-specific int8 quantization intended for HBM-bandwidth studies, not a numerical-parity kernel. Gate its unit tests on a tolerance envelope that matches its steady-state quantization RMS, and forbid its use in the model's serving loop until an accuracy gate proves it's within the reference-model quality bar.
3. **Refactor the reference `_delta_rule_step`** in `kda_state.py` to add the KDA gate, L2-norm, and query-scale steps. Its full signature becomes:
   ```
   _kda_delta_rule_step(
       S_prev, q_raw, k_raw, v, g_raw,      # per-token inputs
       beta_raw,                             # will be sigmoided
       A_log_h, g_bias_h,                    # per-head learned params
       lower_bound=-5.0, l2norm_eps=1e-6,    # config knobs
       scale=None,                           # default K^-0.5
   ) -> (y, S_t)
   ```
   Preserve `_delta_rule_step` as a private `_delta_rule_step_gatefree` for A/B numerical-comparison tests, but never as the model reference.
4. **Add the `chunk_kda_with_fused_gate` reference** as a companion module — it's a different algorithm (chunked-parallel, uses `chunk_local_cumsum`, `chunk_kda_scaled_dot_kkt_fwd`, `solve_tril`) and needs its own bit-exact CPU golden before we can validate an NKI prefill lowering.
5. **Add a `kda_test_vector` fixture** derived from Codex's `contrib/kimi_k3/kda_cpu_reference.py` — feed random Q/K/V/g/beta through the CPU reference, save inputs + outputs, and use as ten-token golden vectors for the NKI kernel-load test (mirrors what MLA-VS-DSA-KERNEL-VERIFICATION §1.1 requires).
6. **Do NOT worry about `flashkda`** for GLM-5.3-Flash. If the campaign expands to Kimi K3 later, the `csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu` bespoke CUDA path is what we'd race against for K3 tokenomics — but it's not in the GLM-5.3-Flash reference path and is unlikely to be soon.

---

## 6. Status doc: what's answered, what's still open

**Answered:**
- vLLM's KDA kernel flavor for GLM-5.3-Flash: **vendored FLA v0.5.2, Triton, generic `fused_recurrent_gated_delta_rule` kernel with the `IS_KDA`/`SAFE_GATE`/`COMPUTE_GATE`/`SIGMOID_BETA` branch, LOWER_BOUND=-5.0, L2-norm in-kernel eps=1e-6, bf16 state.**
- File paths and PR SHA (all above, absolute where local).
- Bit-exact vs numerically-similar: **bit-exact within FLA v0.5.2 arithmetic**, per source comment cited above.
- Codex CPU reference status: matches vLLM path via the v0.5.2 chain of equivalence.
- Our NKI kernel budget: **drifting, not golden-matched** — the shipped slug is DeltaNet-with-int8-state, not KDA.
- Delta enumerated in §4.

**Still open (out of scope for this 3-4 agent-hour task; flagged for the fleet):**
- **Chunk-KDA prefill kernel comparison.** We have not read the full `chunk_kda_with_fused_gate_fwd` body yet — the recurrent kernel is the priority for decode NKI parity, but a prefill NKI kernel (see `NKI-KDA-STATE-SCAFFOLD §3.2 / §4.3`) will need its own analysis and its own reference. Estimated 2-3 agent-hours.
- **FusedRMSNormGated on Trn2.** vLLM's `forward_cuda` calls into `rms_norm_gated` (a separate Triton kernel from `mamba_ssm.ops.triton.layer_norm_gated` in real installations, or Inductor-compiled `forward_native` in the vendored path). We should decide whether to write an NKI RMSNormGated or fuse it into the state kernel's tail. 1-2 agent-hours.
- **A_log and g_bias sharding across TP.** In vLLM's `Glm5NextLinearAttention.__init__` (line 236-268), `A_log` is sharded on the head axis (`sharded_weight_loader(2)`, per `local_num_heads`), and `dt_bias` is sharded on the projection axis (`sharded_weight_loader(0)`, per `local_projection_size`). Our NKI kernel needs to consume the same shard shapes; a shape/stride adapter has not been written. 1 agent-hour.
- **PR #53906 is open, not merged.** Head SHA at fetch time was `142062f13d16bed254b5d97cc3d371fbd4f7790a`. If the PR merges with different constexprs or a different fused-gate arithmetic, our NKI kernel must be re-verified. Watch: monitor the PR head daily until merge; re-diff on any change to the KDA path. Estimated ongoing 15-min/day.

---

## 7. One-line summary for STATE-NOW / OVERSEER-STATE

```
KDA kernel flavor for GLM-5.3-Flash identified: vendored FLA v0.5.2 Triton recurrent (fused_recurrent_kda) + Triton chunk (chunk_kda_with_fused_gate); NOT flashkda (K3 only), NOT FlashInfer (DSA only), NOT TileLang (SGLang only). Our shipped kda_state.decode.rank1_delta.int8_state.per_channel_scale.v1 slug is DRIFTING (missing per-channel gate, L2-norm, query-scale; int8 state where reference is bf16) — needs bump to kda_state.decode.kda_gate.rank1_delta.bf16_state.v1 for correctness parity before serving.
```

---

*End of KDA kernel flavor identification. Return: kernel flavor + budget impact + status doc, as requested.*
