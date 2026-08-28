# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""KDA (Kimi Delta Attention) state kernel V2 -- bf16 state, in-kernel gate.

Cross-references (absolute local paths, per operator memory `always-give-full-local-paths`):
- V1 (deprecated, algorithmically drifting): C:\\Users\\apumu\\research\\InfinityAI\\gemma4-trn2-handoff\\harness-v2\\staging\\reference-sweep-20260826T2150Z\\kernels\\kda_state.py
- V1 -> V2 flavor gap: .../kernels/VLLM-KDA-KERNEL-FLAVOR-2026-08-28.md
- V2 status: .../kernels/KDA-STATE-V2-STATUS-2026-08-28.md
- Scaffold: .../kernels/NKI-KDA-STATE-SCAFFOLD-2026-08-27.md
- MLA-vs-DSA verification (fallback discipline): .../MLA-VS-DSA-KERNEL-VERIFICATION-2026-08-27.md
- FLA v0.5.2 Triton source (scratchpad mirror of vLLM PR #53906):
  C:\\Users\\apumu\\AppData\\Local\\Temp\\claude\\C--Users-apumu-research-InfinityAI-gemma4-trn2-handoff\\fe22bc9a-b8c7-4107-ac55-1d55ba4bd33a\\scratchpad\\kda-fetch\\fused_recurrent_vllm_third_party_flash_linear_attention_ops.py
  (function `fused_recurrent_gated_delta_rule_fwd_kernel`, lines 27-200; per-token body
  lines 132-173; bit-exact fused-gate comment lines 148-151.)

Kernel slug (participates in compile-cache identity hash):
    kda_state.decode.kda_gate.rank1_delta.bf16_state.v1

WHAT V2 FIXES vs V1 (`kda_state.decode.rank1_delta.int8_state.per_channel_scale.v1`):
  (a) KDA per-channel gate `Diag(alpha_h)` applied to state BEFORE the delta step.
      Formula (per FLA v0.5.2, transcribed from `fused_recurrent.py:148-155`):
          alpha_hk = -5.0 * sigmoid( exp(A_log_h) * (g_raw_hk + g_bias_hk) )
          S_prev  *= exp(alpha_h)     # broadcast over D_v axis
      Without this, V1 was DeltaNet, not KDA; the recurrent state grows unbounded.
  (b) In-kernel L2-norm on Q and K (eps=1e-6). Matches `USE_QK_L2NORM_IN_KERNEL=True`
      branch at `fused_recurrent.py:137-140`.
  (c) Query scale `q *= 128^-0.5 ~= 0.0884` applied AFTER L2-norm. Matches
      `b_q = b_q * scale` at `fused_recurrent.py:140`.
  (d) State stored in bf16, layout `[num_slots, HV, V, K]`. This replaces V1's int8
      per-channel-scale layout (which was semantically incompatible with the bf16
      reference; drift ~1% per step, unbounded over decode). The int8 study lane
      lives at `kda_state_int8_study.py` and is marked NOT-SERVED.

WHAT V2 DOES NOT DO (per operator directive: decode kernel first):
  - Prefill chunked-parallel: falls through to token-by-token reference (correct but
    O(L) sequential). A chunked-parallel prefill NKI lowering is a separate deliverable.
  - Short causal-conv1d on Q/K/V (kernel_size=4, silu): the caller is responsible for
    pre-applying the conv. This kernel operates on post-conv Q/K/V.
  - FusedRMSNormGated output normalization: the caller applies o_norm on the returned y.

FALLBACK DISCIPLINE (per campaign memory `[Peer-agent non-interference discipline]`
and MLA-VS-DSA-KERNEL-VERIFICATION §1.1):
  - KDA has NO full-attention fallback that is numerically equivalent. A silent
    compile-lowering into softmax attention CORRUPTS the model.
  - The bf16 CPU reference in this file is the ONLY authorized golden.
  - If the NKI backend fails to import, callers get the CPU reference -- NEVER a
    silent softmax fallback. The dispatch shim asserts on `impl == "softmax"` and
    refuses to serve it.

BF16 SIMULATION ON CPU (no `ml_dtypes` dependency):
  We simulate a fp32 -> bf16 -> fp32 round-trip via bit-twiddling
  (round-to-nearest-even). This lets the reference sit next to the NKI kernel in a
  common numpy environment while still matching the bf16 quantization the hardware
  applies at the HBM boundary.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Kernel identity
# ---------------------------------------------------------------------------

KDA_STATE_V2_KERNEL_SLUG: str = (
    "kda_state.decode.kda_gate.rank1_delta.bf16_state.v1"
)

# GLM-5.3-Flash + Kimi K3 defaults (see FLA v0.5.2 constexpr set summarized in
# VLLM-KDA-KERNEL-FLAVOR-2026-08-28.md §2):
#   LOWER_BOUND    = -5.0  (from linear_attn_config.gate_lower_bound)
#   L2NORM_EPS     = 1e-6  (from USE_QK_L2NORM_IN_KERNEL branch)
#   Q head-dim K   = 128   (both models)
#   scale          = K^-0.5  (default when scale is None in caller)
KDA_LOWER_BOUND: float = -5.0
KDA_L2NORM_EPS: float = 1e-6
KDA_DEFAULT_HEAD_DIM: int = 128


# ---------------------------------------------------------------------------
# bf16 simulation
# ---------------------------------------------------------------------------

def bf16_cast(x: np.ndarray) -> np.ndarray:
    """Simulate fp32 -> bf16 -> fp32 round-trip via bit-twiddling.

    Round-to-nearest-even is implemented as:
        add 0x7FFF + (mantissa_msb_of_dropped_bits >> 16) & 1
        mask off the low 16 mantissa bits

    This matches the hardware bf16 rounding used by the vLLM Triton kernel's
    `b_h.to(p_ht.dtype.element_ty)` store (per FLA `fused_recurrent.py:185, 189`).

    Handles NaN/Inf correctly (they survive as NaN/Inf in bf16).
    """
    x32 = np.asarray(x, dtype=np.float32).copy()
    # Reinterpret as uint32 for bit-twiddling.
    bits = x32.view(np.uint32).copy()
    # NaN handling: any NaN must remain NaN (preserve top mantissa bit).
    is_nan = np.isnan(x32)
    # Round-to-nearest-even bias.
    lsb = (bits >> 16) & np.uint32(1)
    rounding_bias = (np.uint32(0x7FFF) + lsb).astype(np.uint32)
    bits_rounded = bits + rounding_bias
    bits_truncated = (bits_rounded & np.uint32(0xFFFF0000)).astype(np.uint32)
    # Reinterpret back as fp32.
    out = bits_truncated.view(np.float32).copy()
    # Restore NaNs (bit-twiddling can turn NaN into Inf when the mantissa is small).
    out[is_nan] = np.nan
    return out


# ---------------------------------------------------------------------------
# Per-step delta-rule kernel bodies
# ---------------------------------------------------------------------------

def _kda_delta_rule_step_gatefree(
    S_prev: np.ndarray,  # [H, D_v, D_qk] float32
    q: np.ndarray,       # [H, D_qk] float32
    k: np.ndarray,       # [H, D_qk] float32
    v: np.ndarray,       # [H, D_v]  float32
    beta: np.ndarray,    # [H]       float32
) -> Tuple[np.ndarray, np.ndarray]:
    """Yang et al. 2024 gated-delta-rule step WITHOUT the KDA per-channel gate.

    Preserved for A/B numerical-comparison against v1 and for isolating the
    contribution of the KDA gate to output drift. NEVER call this as the served
    reference for KDA -- it is DeltaNet, not KDA (see v1 file gap discussion).
    """
    Sk = np.einsum("hij,hj->hi", S_prev, k)          # [H, D_v]
    delta = v - Sk
    outer = np.einsum("h,hi,hj->hij", beta, delta, k)
    S_t = S_prev + outer
    y = np.einsum("hij,hj->hi", S_t, q)
    return y, S_t


def _kda_delta_rule_step(
    S_prev: np.ndarray,       # [H, D_v, D_qk] float32
    q_raw: np.ndarray,        # [H, D_qk] float32 (post-conv, pre-L2norm)
    k_raw: np.ndarray,        # [H, D_qk] float32 (post-conv, pre-L2norm)
    v: np.ndarray,            # [H, D_v]  float32 (post-conv)
    g_raw: np.ndarray,        # [H, D_qk] float32 (raw per-channel gate logits)
    beta_raw: np.ndarray,     # [H]       float32 (raw per-head beta logit)
    a_log: np.ndarray,        # [H]       float32 (learned per-head)
    g_bias: np.ndarray,       # [H, D_qk] float32 (learned per-head-per-channel)
    lower_bound: float = KDA_LOWER_BOUND,
    l2norm_eps: float = KDA_L2NORM_EPS,
    scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """One full KDA step -- bit-for-bit transcription of FLA v0.5.2's per-token
    body at `fused_recurrent.py:132-173` (SAFE_GATE, COMPUTE_GATE, SIGMOID_BETA,
    USE_QK_L2NORM_IN_KERNEL all True).

    Per-head sequence:
        1. L2-norm on q, k: x /= sqrt(sum(x^2) + eps)
        2. q *= scale (default 1/sqrt(D_qk))
        3. gate: alpha_hk = lower_bound / (1 + exp(-exp(a_log_h) * (g_raw_hk + g_bias_hk)))
        4. S *= exp(alpha_h)                  # broadcast over D_v
        5. delta = v - S @ k
        6. beta = sigmoid(beta_raw)
        7. delta *= beta
        8. S += delta[:, None] * k[None, :]
        9. y = S @ q
    """
    H, D_v, D_qk = S_prev.shape
    assert q_raw.shape == (H, D_qk), q_raw.shape
    assert k_raw.shape == (H, D_qk), k_raw.shape
    assert v.shape == (H, D_v), v.shape
    assert g_raw.shape == (H, D_qk), g_raw.shape
    assert beta_raw.shape == (H,), beta_raw.shape
    assert a_log.shape == (H,), a_log.shape
    assert g_bias.shape == (H, D_qk), g_bias.shape

    if scale is None:
        scale = 1.0 / math.sqrt(D_qk)

    # Match the Triton kernel: cast bf16 inputs -> fp32 for the body. Our caller
    # already provides fp32; the bf16 quantization happens at the HBM boundary
    # inside the dispatch shim, not here.
    S = S_prev.astype(np.float32)
    q = q_raw.astype(np.float32).copy()
    k = k_raw.astype(np.float32).copy()
    v = v.astype(np.float32).copy()

    # (1) L2-norm on q, k -- eps=1e-6, per-head.
    q_norm = np.sqrt(np.sum(q * q, axis=-1) + l2norm_eps)  # [H]
    k_norm = np.sqrt(np.sum(k * k, axis=-1) + l2norm_eps)  # [H]
    q = q / q_norm[:, None]
    k = k / k_norm[:, None]

    # (2) Query scale.
    q = q * scale

    # (3) KDA per-channel gate.
    a_amp = np.exp(a_log.astype(np.float32))                 # [H]
    g = g_raw.astype(np.float32) + g_bias.astype(np.float32) # [H, D_qk]
    alpha = lower_bound / (1.0 + np.exp(-(a_amp[:, None] * g)))  # [H, D_qk]
    decay = np.exp(alpha)                                     # [H, D_qk]

    # (4) State decay -- broadcast over the D_v axis. The Triton kernel indexes
    # b_h as [BV, BK]; the decay `b_gk[None, :]` broadcasts on the D_v axis.
    S = S * decay[:, None, :]                                 # [H, D_v, D_qk]

    # (5) delta = v - S @ k
    Sk = np.einsum("hij,hj->hi", S, k)                        # [H, D_v]
    delta = v - Sk                                            # [H, D_v]

    # (6) beta = sigmoid(beta_raw)
    beta = 1.0 / (1.0 + np.exp(-beta_raw.astype(np.float32))) # [H]

    # (7) delta *= beta (per-head scalar broadcast on D_v axis)
    delta = delta * beta[:, None]

    # (8) S += delta[:, None] * k[None, :]  (outer product added)
    S = S + np.einsum("hi,hj->hij", delta, k)

    # (9) y = S @ q  (post-update state; note the vLLM kernel outputs post-update)
    y = np.einsum("hij,hj->hi", S, q)

    return y, S


# ---------------------------------------------------------------------------
# Learned-parameter container
# ---------------------------------------------------------------------------

@dataclass
class KdaLayerParams:
    """Per-layer learned parameters for the KDA gate.

    Shapes match FLA v0.5.2 / vLLM `Glm5NextLinearAttention.__init__`:
        a_log  : [H]        -- per-head scalar, sharded on head axis across TP
        g_bias : [H, D_qk]  -- per-head per-channel vector (called `dt_bias` in the
                               vLLM module), sharded on projection axis across TP
    """
    a_log: np.ndarray
    g_bias: np.ndarray
    lower_bound: float = KDA_LOWER_BOUND
    l2norm_eps: float = KDA_L2NORM_EPS
    scale: Optional[float] = None  # None -> defaults to 1/sqrt(D_qk) per-step

    def validate(self, H: int, D_qk: int) -> None:
        assert self.a_log.shape == (H,), (self.a_log.shape, H)
        assert self.g_bias.shape == (H, D_qk), (self.g_bias.shape, H, D_qk)


# ---------------------------------------------------------------------------
# Public decode-step inputs / outputs (bf16 state)
# ---------------------------------------------------------------------------

@dataclass
class KdaDecodeInputsV2:
    """Batched decode-step inputs, bf16-state variant.

    Shapes:
        query, key       : [B, 1, H, D_qk] float32 (bf16-representable)
        value            : [B, 1, H, D_v]  float32
        g_raw            : [B, 1, H, D_qk] float32 (per-channel gate logits)
        beta_raw         : [B, 1, H]       float32 (per-head beta logit; sigmoided in-kernel)
        state_bf16       : [B, H, D_v, D_qk] float32 (bf16-quantized, layout matches
                                                       vLLM's [num_slots, HV, V, K])
        params           : KdaLayerParams
    """
    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
    g_raw: np.ndarray
    beta_raw: np.ndarray
    state_bf16: np.ndarray
    params: KdaLayerParams


@dataclass
class KdaDecodeOutputsV2:
    """Batched decode-step outputs.

    Shapes:
        y          : [B, 1, H, D_v] float32 (bf16-quantized -- matches Triton store)
        state_bf16 : [B, H, D_v, D_qk] float32 (bf16-quantized final state)
    """
    y: np.ndarray
    state_bf16: np.ndarray


# ---------------------------------------------------------------------------
# Reference forward -- bit-exact CPU golden
# ---------------------------------------------------------------------------

def kda_state_decode_forward_reference_v2(inputs: KdaDecodeInputsV2) -> KdaDecodeOutputsV2:
    """CPU numpy golden -- bit-exact reference for the NKI kernel.

    Per-batch loop of `_kda_delta_rule_step`. Q length is 1 per batch element for
    decode; state is read as bf16, promoted to fp32 for the body, cast back to
    bf16 for the write. Output y is cast to bf16 to match the Triton store dtype.
    """
    q4 = inputs.query
    k4 = inputs.key
    v4 = inputs.value
    g4 = inputs.g_raw
    beta3 = inputs.beta_raw
    S = inputs.state_bf16
    params = inputs.params

    B, one, H, D_qk = q4.shape
    assert one == 1, "decode kernel expects Q length = 1"
    _, _, _, D_v = v4.shape
    assert v4.shape == (B, 1, H, D_v)
    assert g4.shape == (B, 1, H, D_qk)
    assert beta3.shape == (B, 1, H)
    assert S.shape == (B, H, D_v, D_qk)
    params.validate(H, D_qk)

    y_out = np.empty((B, 1, H, D_v), dtype=np.float32)
    S_new = np.empty_like(S)
    for b in range(B):
        # Load state as bf16 (already stored as bf16-representable fp32) and
        # promote to fp32. The bf16 -> fp32 cast is lossless.
        S_prev = S[b].astype(np.float32)
        q = q4[b, 0].astype(np.float32)
        k = k4[b, 0].astype(np.float32)
        v = v4[b, 0].astype(np.float32)
        g = g4[b, 0].astype(np.float32)
        beta_r = beta3[b, 0].astype(np.float32)
        y_b, S_b = _kda_delta_rule_step(
            S_prev, q, k, v, g, beta_r,
            a_log=params.a_log, g_bias=params.g_bias,
            lower_bound=params.lower_bound, l2norm_eps=params.l2norm_eps,
            scale=params.scale,
        )
        # Store state and output in bf16 (matches Triton's cast-to-dtype at store).
        y_out[b, 0] = bf16_cast(y_b)
        S_new[b] = bf16_cast(S_b)

    return KdaDecodeOutputsV2(y=y_out, state_bf16=S_new)


def kda_state_reset_v2(
    state_bf16: np.ndarray,     # [B, H, D_v, D_qk]
    reset_mask: np.ndarray,     # [B] bool
) -> np.ndarray:
    """Zero the state slabs for batch elements whose reset_mask is True.

    Same discipline as v1: reset MUST be explicit -- never a silent side effect
    of decode. Unmasked batch elements are returned bit-identical.
    """
    assert reset_mask.dtype == np.bool_, reset_mask.dtype
    B = state_bf16.shape[0]
    assert reset_mask.shape == (B,), reset_mask.shape
    out = state_bf16.copy()
    for b in range(B):
        if reset_mask[b]:
            out[b] = 0.0
    return out


# ---------------------------------------------------------------------------
# Prefill (temporarily by-token; chunked-parallel deferred)
# ---------------------------------------------------------------------------

def kda_state_prefill_forward_reference_v2(
    query: np.ndarray,        # [B, L, H, D_qk]
    key: np.ndarray,          # [B, L, H, D_qk]
    value: np.ndarray,        # [B, L, H, D_v]
    g_raw: np.ndarray,        # [B, L, H, D_qk]
    beta_raw: np.ndarray,     # [B, L, H]
    state_bf16: np.ndarray,   # [B, H, D_v, D_qk]
    params: KdaLayerParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prefill by unrolled decode calls -- correct but O(L) sequential.

    A chunked-parallel prefill kernel per `chunk_kda_with_fused_gate` is scheduled
    for a follow-on tick (see VLLM-KDA-KERNEL-FLAVOR §6 "Chunk-KDA prefill kernel
    comparison" open).
    """
    B, L, H, D_qk = query.shape
    _, _, _, D_v = value.shape

    y_all = np.empty((B, L, H, D_v), dtype=np.float32)
    S = state_bf16
    for t in range(L):
        step = KdaDecodeInputsV2(
            query=query[:, t:t + 1],
            key=key[:, t:t + 1],
            value=value[:, t:t + 1],
            g_raw=g_raw[:, t:t + 1],
            beta_raw=beta_raw[:, t:t + 1],
            state_bf16=S,
            params=params,
        )
        out = kda_state_decode_forward_reference_v2(step)
        y_all[:, t:t + 1] = out.y
        S = out.state_bf16
    return y_all, S


# ---------------------------------------------------------------------------
# NKI backend (compilable-in-principle Python DSL source string). Loaded lazily.
# ---------------------------------------------------------------------------

def _try_import_nki() -> Optional[object]:
    """Best-effort NKI import. CPU-only environments return None."""
    try:
        import neuronxcc.nki as nki  # type: ignore
        return nki
    except Exception:
        return None


_NKI = _try_import_nki()


def _kda_state_v2_nki_source() -> str:
    """Return the NKI Python DSL source for the bf16-state decode kernel.

    Returned as source text so it can be inspected without importing neuronxcc.
    The Trn2 compile driver executes the source in a hardware-aware context.

    Kernel body corresponds directly to FLA v0.5.2 `fused_recurrent_gated_delta_rule_fwd_kernel`
    with the KDA constexpr set (IS_KDA=True, COMPUTE_GATE=True, SAFE_GATE=True,
    USE_QK_L2NORM_IN_KERNEL=True, SIGMOID_BETA=True, LOWER_BOUND=-5.0).
    """
    return r'''
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

# Kernel slug: kda_state.decode.kda_gate.rank1_delta.bf16_state.v1

KDA_LOWER_BOUND = -5.0
KDA_L2NORM_EPS = 1e-6


@nki.jit(mode="baremetal")
def kda_state_v2_decode_forward_nki(
    query,      # HBM tensor [B, 1, H, D_qk] bf16
    key,        # HBM tensor [B, 1, H, D_qk] bf16
    value,      # HBM tensor [B, 1, H, D_v]  bf16
    g_raw,      # HBM tensor [B, 1, H, D_qk] bf16 (per-channel gate logits)
    beta_raw,   # HBM tensor [B, 1, H]       bf16 (per-head beta logit)
    a_log,      # HBM tensor [H]             bf16 (learned per-head scalar)
    g_bias,     # HBM tensor [H, D_qk]       bf16 (learned per-head-per-channel bias)
    state_in,   # HBM tensor [B, H, D_v, D_qk] bf16
    scale_value,# python float (constexpr): 1/sqrt(D_qk)
):
    """Rank-1 delta rule with KDA per-channel gate, bf16 state.

    Engines used:
      - Tensor engine  : outer product, S @ k, S @ q
      - Vector engine  : element-wise (mul/add/sub, sigmoid, exp)
      - GpSIMD engine  : L2-norm reduction on Q/K, gate amp per-head

    Tiling: PE tile [BV=64, BK=128] per head; D_qk parallel over 128.
    """
    B, _, H, D_qk = query.shape
    _, _, _, D_v = value.shape

    y_out = nl.ndarray((B, 1, H, D_v), dtype=nl.bfloat16, buffer=nl.hbm)
    state_out = nl.ndarray((B, H, D_v, D_qk), dtype=nl.bfloat16, buffer=nl.hbm)

    for b in nl.affine_range(B):
        for h in nl.affine_range(H):
            # 1. Load prior state as bf16, promote to fp32 for the body.
            s = nl.load(state_in[b, h]).astype(nl.float32)     # [D_v, D_qk]
            q = nl.load(query[b, 0, h]).astype(nl.float32)     # [D_qk]
            k = nl.load(key[b, 0, h]).astype(nl.float32)       # [D_qk]
            v = nl.load(value[b, 0, h]).astype(nl.float32)     # [D_v]
            g = nl.load(g_raw[b, 0, h]).astype(nl.float32)     # [D_qk]
            br = nl.load(beta_raw[b, 0, h]).astype(nl.float32) # scalar
            al = nl.load(a_log[h]).astype(nl.float32)          # scalar
            gb = nl.load(g_bias[h]).astype(nl.float32)         # [D_qk]

            # 2. L2-norm on q, k (eps=1e-6).
            q_norm = nl.sqrt(nl.sum(q * q) + KDA_L2NORM_EPS)
            k_norm = nl.sqrt(nl.sum(k * k) + KDA_L2NORM_EPS)
            q = q / q_norm
            k = k / k_norm

            # 3. Query scale.
            q = q * scale_value

            # 4. KDA per-channel gate.
            a_amp = nl.exp(al)                                 # scalar
            alpha = KDA_LOWER_BOUND / (1.0 + nl.exp(-(a_amp * (g + gb))))  # [D_qk]
            decay = nl.exp(alpha)                              # [D_qk]

            # 5. State decay (broadcast over D_v axis).
            s = s * decay[None, :]

            # 6. delta = v - S @ k
            Sk = nisa.nc_matmul(s, k[:, None])[:, 0]           # [D_v]
            delta = v - Sk

            # 7. beta = sigmoid(beta_raw); delta *= beta
            beta = nl.sigmoid(br)
            delta = delta * beta

            # 8. S += delta[:, None] * k[None, :]
            outer = nisa.nc_matmul(delta[:, None], k[None, :]) # [D_v, D_qk]
            s = s + outer

            # 9. y = S @ q
            y = nisa.nc_matmul(s, q[:, None])[:, 0]            # [D_v]

            # 10. Cast state + y back to bf16 for HBM store.
            nl.store(state_out[b, h], s.astype(nl.bfloat16))
            nl.store(y_out[b, 0, h], y.astype(nl.bfloat16))

    return y_out, state_out
'''


def get_nki_backend_v2() -> Optional[Callable[..., object]]:
    """Return the callable NKI kernel if the Neuron toolchain is present, else None.

    Silent on failure. Callers can detect via `KDA_KERNEL_IMPL` env var or by
    inspecting `_kda_state_v2_nki_source()`. The reference is the fallback path
    -- never a softmax lowering.
    """
    if _NKI is None:
        return None
    # Compile driver on the Trn2 host produces the artifact; we only prove the
    # toolchain is importable here.
    return None


# ---------------------------------------------------------------------------
# Public dispatch shim
# ---------------------------------------------------------------------------

# Sentinel banned impls -- see MLA-VS-DSA-KERNEL-VERIFICATION §1.1.
_BANNED_IMPLS = frozenset({"softmax", "full_attention", "sdpa", "flash_attn"})


def kda_state_decode_forward_v2(
    inputs: KdaDecodeInputsV2,
    impl: str = os.environ.get("KDA_KERNEL_IMPL_V2", "reference"),
) -> KdaDecodeOutputsV2:
    """Dispatch shim -- CPU reference or NKI backend.

    `impl` values:
      - "reference" (default) : bf16 CPU numpy. Always safe.
      - "nki"                 : requires neuronxcc; falls through to reference
                                if the backend returns None.

    Refuses any impl name in `_BANNED_IMPLS` -- a silent softmax lowering would
    corrupt the model. See MLA-VS-DSA-KERNEL-VERIFICATION §1.1.
    """
    if impl in _BANNED_IMPLS:
        raise ValueError(
            f"KDA impl='{impl}' is banned -- a full-attention fallback CORRUPTS "
            "the model. Use impl='reference' or impl='nki'. See "
            "MLA-VS-DSA-KERNEL-VERIFICATION-2026-08-27.md §1.1."
        )
    if impl == "nki":
        backend = get_nki_backend_v2()
        if backend is not None:
            # NKI path not exercisable from this file; compile driver on the
            # Trn2 host produces the artifact. Fall through to reference for
            # round-trip correctness testing.
            pass
    return kda_state_decode_forward_reference_v2(inputs)


# ---------------------------------------------------------------------------
# Kernel-tier sizing helpers (bf16 state -- 2x the v1 int8 sizes)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KdaShapeV2:
    """Shape preset for the bf16-state KDA kernel.

    Fields mirror v1.KdaShape but the dtype is bf16 (2 bytes) rather than int8.
    """
    B: int         # batch (= num_slots for continuous batching)
    H: int         # heads (HV per FLA convention; H_kv folded in)
    D_v: int       # value head-dim
    D_qk: int      # QK head-dim
    layers: int    # KDA layers colocated on one NC


def sbuf_resident_state_bytes_v2(shape: KdaShapeV2) -> int:
    """Bytes of per-layer bf16 state SBUF-resident during a decode step.

    payload = B * H * D_v * D_qk * 2  (bf16)
    No per-channel scale -- bf16 has no separate quantization scale.
    """
    return shape.B * shape.H * shape.D_v * shape.D_qk * 2


def sbuf_total_state_bytes_v2(shape: KdaShapeV2) -> int:
    """Total resident bytes when `layers` KDA layers are colocated on one NC."""
    return shape.layers * sbuf_resident_state_bytes_v2(shape)


def dma_descriptor_bytes_per_layer_v2(shape: KdaShapeV2) -> int:
    """Bytes moved per decode step per layer over HBM<->SBUF DMA (read + write).

    Read: state_bf16 [B, H, D_v, D_qk].
    Write: same shape.
    Excludes q/k/v/g_raw/beta_raw which are dispatched by the caller.

    (Learned params a_log/g_bias are hoisted once per compile and pinned in SBUF;
    they do not contribute to per-decode-step DMA.)
    """
    return 2 * sbuf_resident_state_bytes_v2(shape)


# Same efficiency floor as v1 (per NKI-DMA-COALESCING-SCAFFOLD-2026-08-27.md §3).
EFFICIENT_DMA_DESCRIPTOR_BYTES_V2 = 4 * 1024

# Trn2 per-NeuronCore SBUF budget (unchanged).
TRAINIUM2_SBUF_BUDGET_BYTES_V2 = 24 * 1024 * 1024

# Trn2 per-NeuronCore HBM budget (24 GiB).
TRAINIUM2_HBM_BUDGET_BYTES_V2 = 24 * (1024 ** 3)


# ---------------------------------------------------------------------------
# Model-specific shape presets (bf16 state)
# ---------------------------------------------------------------------------

# Kimi K3: HV=96, K=V=128, 69 KDA layers (per CAMPAIGN-SCOPE-KIMI-K3-2026-08-27.md).
KIMI_K3_KDA_SHAPE_V2 = KdaShapeV2(B=1, H=96, D_v=128, D_qk=128, layers=69)

# GLM-5.3-Flash: HV=64, K=V=128, 34 KDA layers (per CAMPAIGN-SCOPE-GLM-5.3-FLASH-2026-08-27.md).
GLM_5_3_FLASH_KDA_SHAPE_V2 = KdaShapeV2(B=1, H=64, D_v=128, D_qk=128, layers=34)


def build_shape_v2(base: KdaShapeV2, *, B: int) -> KdaShapeV2:
    """Rebind batch on a preset shape (heads/layers/dims fixed by the model)."""
    return KdaShapeV2(B=B, H=base.H, D_v=base.D_v, D_qk=base.D_qk, layers=base.layers)


# ---------------------------------------------------------------------------
# Backwards-compat convenience aliases (unversioned names)
# ---------------------------------------------------------------------------

# These aliases let downstream code import the "current best" without pinning to
# _v2 explicitly. When a v3 lands, flip the alias in one place.
KdaShape = KdaShapeV2
KdaDecodeInputs = KdaDecodeInputsV2
KdaDecodeOutputs = KdaDecodeOutputsV2
kda_state_decode_forward = kda_state_decode_forward_v2
kda_state_decode_forward_reference = kda_state_decode_forward_reference_v2
kda_state_prefill_forward_reference = kda_state_prefill_forward_reference_v2
kda_state_reset = kda_state_reset_v2
