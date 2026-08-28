# SPDX-License-Identifier: Apache-2.0
# NKI kernel body -- file-import version -- slug: kda_state.decode.kda_gate.rank1_delta.bf16_state.nki_v1
#
# This file exists as a PHYSICAL file (not an `exec()`-populated namespace)
# because @nki.jit's KernelRewriter.reparse_function calls
# `inspect.getsource(<decorated_fn>)` at compile time, and that call requires
# a real file on disk to walk. `exec(source_str, ns)` leaves the function's
# `__module__` as "<string>" and `inspect.getsource` raises `OSError: could
# not get source code`. See EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md for the
# full bug diagnosis.
#
# Bit-exact transcription of FLA v0.5.2 fused_recurrent_gated_delta_rule_fwd_kernel
# with KDA constexpr set (IS_KDA=True, COMPUTE_GATE=True, SAFE_GATE=True,
# USE_QK_L2NORM_IN_KERNEL=True, SIGMOID_BETA=True, LOWER_BOUND=-5.0).
# Reference source (scratchpad mirror of vLLM PR #53906):
#   C:/Users/apumu/AppData/Local/Temp/claude/
#     C--Users-apumu-research-InfinityAI-gemma4-trn2-handoff/
#     fe22bc9a-b8c7-4107-ac55-1d55ba4bd33a/scratchpad/kda-fetch/
#     fused_recurrent_vllm_third_party_flash_linear_attention_ops.py
# Function `fused_recurrent_gated_delta_rule_fwd_kernel`, lines 132-173.

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

# Constexpr constants -- baked in at file-authoring time (defaults from
# kda_state_v2 CPU golden). For non-default constants, the compile driver
# must materialize a per-constant-set body file; the shim's
# `_kda_state_nki_v2_source(...)` generator still emits the interpolated
# text for that path.
KDA_LOWER_BOUND = -5.0
KDA_L2NORM_EPS = 1e-06
BV = 64
BK = 128


@nki.jit(mode="baremetal")
def kda_state_decode_forward_nki_v2_body(
    query,        # HBM [B, 1, H, D_qk] bf16     (post-conv, pre-L2norm)
    key,          # HBM [B, 1, H, D_qk] bf16     (post-conv, pre-L2norm)
    value,        # HBM [B, 1, H, D_v]  bf16     (post-conv)
    g_raw,        # HBM [B, 1, H, D_qk] bf16     (per-channel gate logits)
    beta_raw,     # HBM [B, 1, H]       bf16     (per-head beta logit)
    a_log,        # HBM [H]             bf16     (learned per-head scalar)
    g_bias,       # HBM [H, D_qk]       bf16     (learned per-head-per-channel)
    state_in,     # HBM [B, H, D_v, D_qk] bf16   (prior recurrent state)
    scale_value,  # python float (constexpr): 1 / sqrt(D_qk)
):
    """Decode-step NKI body -- rank-1 delta rule with KDA per-channel gate,
    bf16 state.

    Engine assignments (per per-head iteration, chosen for the Trn2 static
    schedule):
      - Tensor engine  : nc_matmul for S@k, S@q, and the delta outer product.
      - Vector engine  : mul/add/sub, sigmoid, exp element-wise.
      - GpSIMD/scalar  : nl.sum for L2-norm reductions, exp on a_log.
      - DMA            : bf16 load state (BV*BK*2 B/head), bf16 store state
                         and y at the end.

    Numerical contract: fp32 body, bf16 store on state and y. Matches
    FLA v0.5.2 `.to(p_ht.dtype.element_ty)` semantics (lines 185, 189 of
    fused_recurrent.py).
    """
    B, _, H, D_qk = query.shape
    _, _, _, D_v = value.shape
    # BV, BK are compile-time constants -- the compile driver picks the
    # SBUF residency based on them (see docstring at top of file).

    y_out = nl.ndarray((B, 1, H, D_v), dtype=nl.bfloat16, buffer=nl.hbm)
    state_out = nl.ndarray(
        (B, H, D_v, D_qk), dtype=nl.bfloat16, buffer=nl.hbm
    )

    for b in nl.affine_range(B):
        for h in nl.affine_range(H):
            # ---- (1) HBM -> SBUF, promote to fp32 ------------------------
            s_bf16 = nl.load(state_in[b, h])                # [D_v, D_qk] bf16
            q_bf16 = nl.load(query[b, 0, h])                # [D_qk] bf16
            k_bf16 = nl.load(key[b, 0, h])                  # [D_qk] bf16
            v_bf16 = nl.load(value[b, 0, h])                # [D_v]  bf16
            g_bf16 = nl.load(g_raw[b, 0, h])                # [D_qk] bf16
            br_bf16 = nl.load(beta_raw[b, 0, h])            # scalar bf16
            al_bf16 = nl.load(a_log[h])                     # scalar bf16
            gb_bf16 = nl.load(g_bias[h])                    # [D_qk] bf16

            s = s_bf16.astype(nl.float32)
            q = q_bf16.astype(nl.float32)
            k = k_bf16.astype(nl.float32)
            v = v_bf16.astype(nl.float32)
            g = g_bf16.astype(nl.float32)
            br = br_bf16.astype(nl.float32)
            al = al_bf16.astype(nl.float32)
            gb = gb_bf16.astype(nl.float32)

            # ---- (2) L2-norm on q, k (eps=1e-06) -------------------------
            # FLA reference: fused_recurrent.py:137-140
            q_sq_sum = nl.sum(q * q)
            k_sq_sum = nl.sum(k * k)
            q_denom = nl.sqrt(q_sq_sum + KDA_L2NORM_EPS)
            k_denom = nl.sqrt(k_sq_sum + KDA_L2NORM_EPS)
            q = q / q_denom
            k = k / k_denom

            # ---- (3) Query scale q *= 1/sqrt(D_qk) -----------------------
            # FLA reference: fused_recurrent.py:140 (`b_q = b_q * scale`)
            q = q * scale_value

            # ---- (4) KDA per-channel gate --------------------------------
            # FLA reference: fused_recurrent.py:148-155 (COMPUTE_GATE +
            # SAFE_GATE branch inside IS_KDA branch)
            #   b_a_log = exp(a_log[i_h])                    (per-head scalar)
            #   b_gk    = g_raw + g_bias                     (per-channel)
            #   b_gk    = LOWER_BOUND / (1 + exp(-b_a_log * b_gk))
            #   b_h    *= exp(b_gk[None, :])                 (per-channel decay)
            a_amp = nl.exp(al)                              # scalar
            g_plus = g + gb                                 # [D_qk]
            alpha = KDA_LOWER_BOUND / (
                1.0 + nl.exp(-(a_amp * g_plus))
            )                                               # [D_qk]
            decay = nl.exp(alpha)                           # [D_qk]

            # ---- (5) State decay: S *= decay[None, :] --------------------
            # Broadcast on D_v axis -- FLA reference: fused_recurrent.py:157
            s = s * decay[None, :]

            # ---- (6) delta = v - S @ k -----------------------------------
            # FLA reference: fused_recurrent.py:159-162
            # Tensor-engine matmul [BV, BK] @ [BK, 1] -> [BV, 1]
            Sk = nisa.nc_matmul(s, k[:, None])              # [D_v, 1]
            Sk = Sk[:, 0]                                   # [D_v]
            delta = v - Sk

            # ---- (7) beta = sigmoid(beta_raw); delta *= beta -------------
            # FLA reference: fused_recurrent.py:164-166 (SIGMOID_BETA branch)
            beta = nl.sigmoid(br)                           # scalar
            delta = delta * beta                            # [D_v]

            # ---- (8) S += delta[:, None] * k[None, :] --------------------
            # Outer product rank-1 update -- FLA reference:
            # fused_recurrent.py:168-170
            # Tensor-engine matmul [BV, 1] @ [1, BK] -> [BV, BK]
            outer = nisa.nc_matmul(delta[:, None], k[None, :])  # [D_v, D_qk]
            s = s + outer

            # ---- (9) y = S @ q -------------------------------------------
            # FLA reference: fused_recurrent.py:172-173
            # y is POST-update state @ q (vLLM's Triton stores post-update).
            y = nisa.nc_matmul(s, q[:, None])               # [D_v, 1]
            y = y[:, 0]                                     # [D_v]

            # ---- (10) Cast back to bf16, HBM store -----------------------
            # Matches FLA `.to(p_ht.dtype.element_ty)` at fused_recurrent.py:185, 189
            nl.store(state_out[b, h], s.astype(nl.bfloat16))
            nl.store(y_out[b, 0, h], y.astype(nl.bfloat16))

    return y_out, state_out


# ---- Prefill body: chunk-KDA -- TODO(nki-chunk-kda tick) -----------------
# Chunked-parallel prefill (chunk_kda_with_fused_gate FLA equivalent) is a
# separate 4-6 agent-hour deliverable. Until it lands, the dispatch shim
# falls through to the CPU golden by-token prefill (correct, O(L)).
#
# Reference for the future body:
#   flash_linear_attention/ops/kda.py:1470  chunk_kda_with_fused_gate_fwd
#   vLLM entry point at glm5next/nvidia/kda.py:532 (num_prefills > 0)
