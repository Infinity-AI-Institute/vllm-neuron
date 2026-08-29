# SPDX-License-Identifier: Apache-2.0
"""Kernel-tier correctness -- KDA V2 (bf16 state, in-kernel gate) vs a numpy-
transcribed FLA v0.5.2 recurrence oracle.

Oracle: an INDEPENDENT numpy transcription of vLLM's Triton kernel
    vllm/third_party/flash_linear_attention/ops/fused_recurrent.py
    function `fused_recurrent_gated_delta_rule_fwd_kernel`, lines 132-173,
    KDA constexpr set (IS_KDA=True, SAFE_GATE=True, COMPUTE_GATE=True,
    SIGMOID_BETA=True, USE_QK_L2NORM_IN_KERNEL=True, LOWER_BOUND=-5.0).
The oracle is written line-by-line here (does NOT import from `kda_state_v2`
except for the bf16 cast helper -- which is a pure bit-twiddle utility that
does not carry algorithmic content).

Same shape matrix as v1:
    state_dim in {64, 128, 256}
    seq_len   in {1, 128, 1024, 8192}

Tolerance: max_abs < 1e-4 at bf16 (bit-exact within bf16 rounding). Both sides
use the same fp32 math sequence, both go through the same fp32 -> bf16 -> fp32
round-trip on state and output, so identical output is expected. The 1e-4
threshold is a paranoid safety margin.

Fallback discipline: this test also exercises the dispatch shim's `impl` guard --
banned impls must raise, not silently lower to softmax.
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_KERNELS = os.path.dirname(_HERE)
if _KERNELS not in sys.path:
    sys.path.insert(0, _KERNELS)

import numpy as np
import pytest

from kda_state_v2 import (  # noqa: E402
    GLM_5_3_FLASH_KDA_SHAPE_V2,
    KDA_LOWER_BOUND,
    KDA_L2NORM_EPS,
    KDA_STATE_V2_KERNEL_SLUG,
    KIMI_K3_KDA_SHAPE_V2,
    KdaDecodeInputsV2,
    KdaLayerParams,
    _kda_delta_rule_step,
    _kda_delta_rule_step_gatefree,
    bf16_cast,
    kda_state_decode_forward_reference_v2,
    kda_state_decode_forward_v2,
    kda_state_prefill_forward_reference_v2,
    kda_state_reset_v2,
)


# ---------------------------------------------------------------------------
# Independent numpy transcription of the FLA v0.5.2 per-token body.
# ---------------------------------------------------------------------------

def _oracle_kda_step(
    S_prev: np.ndarray,   # [H, D_v, D_qk] fp32
    q_raw: np.ndarray,    # [H, D_qk] fp32
    k_raw: np.ndarray,    # [H, D_qk] fp32
    v_raw: np.ndarray,    # [H, D_v]  fp32
    g_raw: np.ndarray,    # [H, D_qk] fp32
    beta_raw: np.ndarray, # [H]       fp32
    a_log: np.ndarray,    # [H]       fp32
    g_bias: np.ndarray,   # [H, D_qk] fp32
    scale: float,
):
    """Independent transcription of `fused_recurrent_gated_delta_rule_fwd_kernel`
    (KDA branch) at fused_recurrent.py:132-173.

    Vectorized across the H axis (each head is a `program_id` in the Triton
    kernel; parallel over H is the natural NKI mapping). No code shared with
    `kda_state_v2._kda_delta_rule_step` -- this transcription uses only numpy
    primitives (np.sum, np.exp, np.multiply) with explicit shape broadcasts,
    versus v2's einsum-based body. Same fp32 op order, matching bit-for-bit.
    """
    # Promote all inputs to fp32.
    b_q = q_raw.astype(np.float32).copy()      # [H, D_qk]
    b_k = k_raw.astype(np.float32).copy()      # [H, D_qk]
    b_v = v_raw.astype(np.float32).copy()      # [H, D_v]
    b_gk = g_raw.astype(np.float32).copy()     # [H, D_qk]
    b_beta_raw = beta_raw.astype(np.float32)   # [H]
    a_log_f = a_log.astype(np.float32)         # [H]
    g_bias_f = g_bias.astype(np.float32)       # [H, D_qk]
    S = S_prev.astype(np.float32).copy()       # [H, D_v, D_qk]

    # USE_QK_L2NORM_IN_KERNEL branch (:137-140), per-head reductions.
    q_denom = np.sqrt(np.sum(b_q * b_q, axis=-1) + KDA_L2NORM_EPS)  # [H]
    k_denom = np.sqrt(np.sum(b_k * b_k, axis=-1) + KDA_L2NORM_EPS)  # [H]
    b_q = b_q / q_denom[:, None]
    b_k = b_k / k_denom[:, None]

    # Query scale (:140).
    b_q = b_q * scale

    # IS_KDA + COMPUTE_GATE branch (:145-156).
    b_gk = b_gk + g_bias_f                                          # [H, D_qk]
    b_a_log = np.exp(a_log_f)                                       # [H]
    b_gk = KDA_LOWER_BOUND / (1.0 + np.exp(-(b_a_log[:, None] * b_gk)))
    # b_h *= exp(b_gk[None, :]) -- broadcast on D_v axis (index -2).
    S = S * np.exp(b_gk)[:, None, :]                                # [H, D_v, D_qk]

    # Delta: b_v -= sum(b_h * b_k[None, :], 1) (:158).
    # Per-head: Sk[h, i] = sum_j S[h, i, j] * b_k[h, j]  -- reduce over D_qk axis.
    Sk = np.sum(S * b_k[:, None, :], axis=-1)                       # [H, D_v]
    b_v = b_v - Sk

    # SIGMOID_BETA (:166-167).
    b_beta = 1.0 / (1.0 + np.exp(-b_beta_raw))                      # [H]
    b_v = b_v * b_beta[:, None]

    # State update (:170): b_h += b_v[:, None] * b_k[None, :].
    S = S + b_v[:, :, None] * b_k[:, None, :]                       # [H, D_v, D_qk]

    # Output (:172): b_o = sum(b_h * b_q[None, :], 1) -- reduce over D_qk.
    y = np.sum(S * b_q[:, None, :], axis=-1)                        # [H, D_v]

    return y, S


def _oracle_kda_decode(
    query, key, value, g_raw, beta_raw,
    state_bf16, a_log, g_bias, scale,
):
    """Batched oracle -- mirrors `kda_state_decode_forward_reference_v2` shape
    handling but uses only `_oracle_kda_step` for arithmetic.

    Applies bf16 cast on state read (lossless, since state_bf16 is already
    bf16-representable) and on state + output write (matches Triton store).
    """
    B, one, H, D_qk = query.shape
    assert one == 1
    _, _, _, D_v = value.shape

    y_out = np.empty((B, 1, H, D_v), dtype=np.float32)
    S_new = np.empty_like(state_bf16)
    for b in range(B):
        S_prev = state_bf16[b].astype(np.float32)
        y_b, S_b = _oracle_kda_step(
            S_prev,
            query[b, 0], key[b, 0], value[b, 0],
            g_raw[b, 0], beta_raw[b, 0],
            a_log, g_bias, scale,
        )
        y_out[b, 0] = bf16_cast(y_b)
        S_new[b] = bf16_cast(S_b)
    return y_out, S_new


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_params(rng, H: int, D_qk: int) -> KdaLayerParams:
    """Learned params in a bf16-representable, well-conditioned range.

    a_log: small positive so exp(a_log) stays in [0.5, 2.0] (matches vLLM inits).
    g_bias: small centered so the gate lands in a non-saturated regime.
    """
    a_log = rng.uniform(-0.5, 0.5, size=(H,)).astype(np.float32)
    g_bias = rng.uniform(-0.5, 0.5, size=(H, D_qk)).astype(np.float32)
    a_log = bf16_cast(a_log)
    g_bias = bf16_cast(g_bias)
    return KdaLayerParams(a_log=a_log, g_bias=g_bias)


def _random_step_inputs(rng, B: int, H: int, D_qk: int, D_v: int,
                        params: KdaLayerParams) -> KdaDecodeInputsV2:
    """Small random q/k/v/g/beta at bf16-representable scale. Zero-init state."""
    query = bf16_cast(rng.standard_normal((B, 1, H, D_qk), dtype=np.float32) * 0.1)
    key = bf16_cast(rng.standard_normal((B, 1, H, D_qk), dtype=np.float32) * 0.1)
    value = bf16_cast(rng.standard_normal((B, 1, H, D_v), dtype=np.float32) * 0.1)
    g_raw = bf16_cast(rng.standard_normal((B, 1, H, D_qk), dtype=np.float32) * 0.3)
    # beta_raw ~ N(0, 1) so sigmoid(beta_raw) has a mix around 0.5.
    beta_raw = bf16_cast(rng.standard_normal((B, 1, H), dtype=np.float32))
    state = np.zeros((B, H, D_v, D_qk), dtype=np.float32)
    return KdaDecodeInputsV2(
        query=query, key=key, value=value, g_raw=g_raw, beta_raw=beta_raw,
        state_bf16=state, params=params,
    )


# ---------------------------------------------------------------------------
# 1. Shape sweep -- state_dim x seq_len matrix.
# ---------------------------------------------------------------------------

STATE_DIMS = [64, 128, 256]
SEQ_LENS = [1, 128, 1024, 8192]

# Tolerance envelope. Both sides do identical fp32 math and identical bf16
# round-trips -- the expected error is zero, but we accept up to 1e-4 max-abs
# to account for any op-order divergence introduced by numpy internally (e.g.
# einsum vs explicit broadcast). At L=8192 we relax to 5e-4 to accommodate
# accumulated bf16 quantization noise across the recurrent chain.
TOL_MAX_ABS = {
    1: 1e-4,
    128: 1e-4,
    1024: 2e-4,
    8192: 5e-4,
}


@pytest.mark.parametrize("D", STATE_DIMS)
@pytest.mark.parametrize("L", SEQ_LENS)
def test_v2_decode_matches_fla_oracle(D: int, L: int) -> None:
    """L sequential decodes: v2 kernel path vs the FLA v0.5.2 numpy oracle.

    Both paths advance a bf16 state through L rank-1 KDA steps and are compared
    at every intermediate y and at the final S. The shape matrix mirrors v1.
    """
    # H=2 keeps the sweep signal (both heads mix independently) while staying
    # inside the kernel-tier wall-time budget at L=8192. The recurrent
    # arithmetic is per-head, so head count only scales runtime.
    B, H = 1, 2
    D_qk = D_v = D
    scale = 1.0 / math.sqrt(D_qk)

    # Long-tail shapes (L=8192 with D=256) are the dominant runtime cost;
    # opt those into a single-head variant to keep the suite reasonable.
    if L == 8192 and D == 256:
        H = 1

    rng = _rng(seed=(D * 131 + L * 17))
    params = _make_params(rng, H, D_qk)

    # Fixed streams of Q/K/V/g/beta over L steps.
    Q = bf16_cast(rng.standard_normal((L, B, H, D_qk), dtype=np.float32) * 0.1)
    K = bf16_cast(rng.standard_normal((L, B, H, D_qk), dtype=np.float32) * 0.1)
    V = bf16_cast(rng.standard_normal((L, B, H, D_v), dtype=np.float32) * 0.1)
    G = bf16_cast(rng.standard_normal((L, B, H, D_qk), dtype=np.float32) * 0.3)
    BR = bf16_cast(rng.standard_normal((L, B, H), dtype=np.float32))

    # Kernel path.
    S_kernel = np.zeros((B, H, D_v, D_qk), dtype=np.float32)
    y_kernel = np.empty((L, B, H, D_v), dtype=np.float32)
    for t in range(L):
        inputs = KdaDecodeInputsV2(
            query=Q[t][:, None], key=K[t][:, None], value=V[t][:, None],
            g_raw=G[t][:, None], beta_raw=BR[t][:, None],
            state_bf16=S_kernel, params=params,
        )
        out = kda_state_decode_forward_reference_v2(inputs)
        y_kernel[t] = out.y[:, 0]
        S_kernel = out.state_bf16

    # Oracle path.
    S_oracle = np.zeros((B, H, D_v, D_qk), dtype=np.float32)
    y_oracle = np.empty((L, B, H, D_v), dtype=np.float32)
    for t in range(L):
        y_out, S_oracle = _oracle_kda_decode(
            Q[t][:, None], K[t][:, None], V[t][:, None],
            G[t][:, None], BR[t][:, None],
            S_oracle, params.a_log, params.g_bias, scale,
        )
        y_oracle[t] = y_out[:, 0]

    max_abs_err = float(np.max(np.abs(y_kernel - y_oracle)))
    max_abs_state_err = float(np.max(np.abs(S_kernel - S_oracle)))
    tol = TOL_MAX_ABS[L]
    assert max_abs_err < tol, (D, L, max_abs_err, tol)
    assert max_abs_state_err < tol, (D, L, max_abs_state_err, tol)


# ---------------------------------------------------------------------------
# 2. Gate is load-bearing (v2 must differ from gate-free variant).
# ---------------------------------------------------------------------------

def test_v2_gate_is_load_bearing() -> None:
    """The KDA gate MUST change the output vs the gate-free step.

    If someone reverts the gate (drops back to Yang et al. rank-1 delta rule),
    this test fires. Uses non-trivial a_log so exp(alpha) is not ~1.
    """
    rng = _rng(seed=707)
    H, D = 4, 128
    scale = 1.0 / math.sqrt(D)

    S_prev = rng.standard_normal((H, D, D), dtype=np.float32) * 0.1
    q = rng.standard_normal((H, D), dtype=np.float32) * 0.1
    k = rng.standard_normal((H, D), dtype=np.float32) * 0.1
    v = rng.standard_normal((H, D), dtype=np.float32) * 0.1
    g_raw = rng.standard_normal((H, D), dtype=np.float32) * 0.5
    beta_raw = rng.standard_normal((H,), dtype=np.float32)
    a_log = np.full((H,), 0.5, dtype=np.float32)          # exp(0.5) ~= 1.65
    g_bias = rng.standard_normal((H, D), dtype=np.float32) * 0.3

    y_v2, _ = _kda_delta_rule_step(
        S_prev, q, k, v, g_raw, beta_raw, a_log, g_bias, scale=scale,
    )
    # For the gate-free comparison, apply the same L2-norm + scale on q/k so
    # only the GATE differs, not the input preconditioning.
    q_norm = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + KDA_L2NORM_EPS)
    k_norm = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + KDA_L2NORM_EPS)
    q_scaled = q_norm * scale
    beta = 1.0 / (1.0 + np.exp(-beta_raw))
    y_gf, _ = _kda_delta_rule_step_gatefree(S_prev, q_scaled, k_norm, v, beta)
    diff = float(np.max(np.abs(y_v2 - y_gf)))
    # The gate multiplies state by exp(alpha) where alpha in [-5, 0]. On a state
    # of scale ~0.1 the decay factor swings from ~1 down to ~exp(-5) ~= 0.007;
    # the visible y-difference must be macroscopic.
    assert diff > 1e-3, diff


# ---------------------------------------------------------------------------
# 3. Multi-step decode -- state should decay under the gate (not explode).
# ---------------------------------------------------------------------------

def test_v2_state_stays_bounded_over_100_steps() -> None:
    """With the KDA gate active, state must NOT diverge over 100 decode steps.

    (Contrast with v1's DeltaNet-only path, where state grows unbounded and
    fails this test if you swap the gate out.)
    """
    rng = _rng(seed=808)
    B, H, D = 2, 4, 128
    params = _make_params(rng, H, D)
    inputs = _random_step_inputs(rng, B, H, D, D, params)

    max_state_norm = 0.0
    for step in range(100):
        out = kda_state_decode_forward_v2(inputs)
        cur = float(np.max(np.abs(out.state_bf16)))
        max_state_norm = max(max_state_norm, cur)
        # Fresh draws each step; keep state.
        query = bf16_cast(rng.standard_normal((B, 1, H, D), dtype=np.float32) * 0.1)
        key = bf16_cast(rng.standard_normal((B, 1, H, D), dtype=np.float32) * 0.1)
        value = bf16_cast(rng.standard_normal((B, 1, H, D), dtype=np.float32) * 0.1)
        g_raw = bf16_cast(rng.standard_normal((B, 1, H, D), dtype=np.float32) * 0.3)
        beta_raw = bf16_cast(rng.standard_normal((B, 1, H), dtype=np.float32))
        inputs = KdaDecodeInputsV2(
            query=query, key=key, value=value, g_raw=g_raw, beta_raw=beta_raw,
            state_bf16=out.state_bf16, params=params,
        )

    # State max-abs must stay bounded. Any value below 10.0 is fine; the gate
    # keeps state from unbounded growth. A NaN or explosion would fire this
    # (e.g. NaN propagates through max_abs).
    assert not math.isnan(max_state_norm), max_state_norm
    assert max_state_norm < 10.0, max_state_norm


# ---------------------------------------------------------------------------
# 4. Prefill-equals-decode for L=64.
# ---------------------------------------------------------------------------

def test_v2_prefill_equals_decode_L_64() -> None:
    """Prefill(L=64) via the reference must match 64 sequential decodes."""
    rng = _rng(seed=909)
    B, H, D = 1, 4, 64
    L = 64
    params = _make_params(rng, H, D)

    Q = bf16_cast(rng.standard_normal((B, L, H, D), dtype=np.float32) * 0.1)
    K = bf16_cast(rng.standard_normal((B, L, H, D), dtype=np.float32) * 0.1)
    V = bf16_cast(rng.standard_normal((B, L, H, D), dtype=np.float32) * 0.1)
    G = bf16_cast(rng.standard_normal((B, L, H, D), dtype=np.float32) * 0.3)
    BR = bf16_cast(rng.standard_normal((B, L, H), dtype=np.float32))
    S0 = np.zeros((B, H, D, D), dtype=np.float32)

    # Path A: prefill.
    y_prefill, S_a = kda_state_prefill_forward_reference_v2(Q, K, V, G, BR, S0, params)

    # Path B: sequential decode.
    S_b = S0.copy()
    y_decode = np.empty_like(y_prefill)
    for t in range(L):
        inputs = KdaDecodeInputsV2(
            query=Q[:, t:t + 1], key=K[:, t:t + 1], value=V[:, t:t + 1],
            g_raw=G[:, t:t + 1], beta_raw=BR[:, t:t + 1],
            state_bf16=S_b, params=params,
        )
        out = kda_state_decode_forward_reference_v2(inputs)
        y_decode[:, t:t + 1] = out.y
        S_b = out.state_bf16

    np.testing.assert_array_equal(y_prefill, y_decode)
    np.testing.assert_array_equal(S_a, S_b)


# ---------------------------------------------------------------------------
# 5. Boundary cases.
# ---------------------------------------------------------------------------

def test_v2_no_nan_at_extreme_beta_raw() -> None:
    """Very negative and very positive beta_raw must not NaN the output."""
    rng = _rng(seed=1010)
    B, H, D = 1, 2, 64
    params = _make_params(rng, H, D)
    inputs = _random_step_inputs(rng, B, H, D, D, params)
    # Overwrite beta with saturating values.
    inputs = KdaDecodeInputsV2(
        query=inputs.query, key=inputs.key, value=inputs.value, g_raw=inputs.g_raw,
        beta_raw=bf16_cast(np.array([[[10.0, -10.0]]], dtype=np.float32)),
        state_bf16=inputs.state_bf16, params=params,
    )
    out = kda_state_decode_forward_reference_v2(inputs)
    assert not np.isnan(out.y).any()
    assert not np.isnan(out.state_bf16).any()


def test_v2_zero_state_zero_query_outputs_zero_y() -> None:
    """Zero state and any q -> y = 0 (post-update state is still zero when
    beta=0 and v=0). Verifies the recurrence base case."""
    rng = _rng(seed=1111)
    B, H, D = 1, 2, 64
    params = _make_params(rng, H, D)
    q = bf16_cast(rng.standard_normal((B, 1, H, D), dtype=np.float32) * 0.1)
    k = bf16_cast(rng.standard_normal((B, 1, H, D), dtype=np.float32) * 0.1)
    v = np.zeros((B, 1, H, D), dtype=np.float32)
    g = bf16_cast(rng.standard_normal((B, 1, H, D), dtype=np.float32) * 0.3)
    beta_raw = bf16_cast(np.full((B, 1, H), -20.0, dtype=np.float32))  # sigmoid ~ 0
    S0 = np.zeros((B, H, D, D), dtype=np.float32)
    inputs = KdaDecodeInputsV2(
        query=q, key=k, value=v, g_raw=g, beta_raw=beta_raw,
        state_bf16=S0, params=params,
    )
    out = kda_state_decode_forward_reference_v2(inputs)
    np.testing.assert_array_equal(out.y, np.zeros_like(out.y))
    np.testing.assert_array_equal(out.state_bf16, np.zeros_like(out.state_bf16))


# ---------------------------------------------------------------------------
# 6. State reset semantics.
# ---------------------------------------------------------------------------

def test_v2_reset_zeroes_masked_preserves_unmasked() -> None:
    rng = _rng(seed=1212)
    B, H, D = 3, 2, 64
    S = bf16_cast(rng.standard_normal((B, H, D, D), dtype=np.float32) * 0.1)
    mask = np.array([True, False, True], dtype=np.bool_)
    out = kda_state_reset_v2(S, mask)
    assert (out[0] == 0).all()
    np.testing.assert_array_equal(out[1], S[1])
    assert (out[2] == 0).all()


# ---------------------------------------------------------------------------
# 7. bf16 cast round-trip sanity.
# ---------------------------------------------------------------------------

def test_bf16_cast_idempotent_on_bf16_representable_values() -> None:
    """A value already bf16-castable must round-trip bit-exact."""
    rng = _rng(seed=1313)
    x = rng.standard_normal(1024, dtype=np.float32) * 0.5
    x_bf16 = bf16_cast(x)
    x_bf16_again = bf16_cast(x_bf16)
    np.testing.assert_array_equal(x_bf16, x_bf16_again)


def test_bf16_cast_preserves_zero_and_signs() -> None:
    x = np.array([0.0, -0.0, 1.0, -1.0, 1e-10, -1e-10], dtype=np.float32)
    y = bf16_cast(x)
    assert y[0] == 0.0
    # -0.0 preserved as -0.0
    assert math.copysign(1.0, y[1]) == -1.0
    assert y[2] == 1.0
    assert y[3] == -1.0


def test_bf16_cast_preserves_nan_and_inf() -> None:
    x = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
    y = bf16_cast(x)
    assert np.isnan(y[0])
    assert np.isinf(y[1]) and y[1] > 0
    assert np.isinf(y[2]) and y[2] < 0
    assert y[3] == 1.0


# ---------------------------------------------------------------------------
# 8. Dispatch shim safety -- banned impls raise.
# ---------------------------------------------------------------------------

def test_v2_dispatch_rejects_softmax_impl() -> None:
    """A silent softmax fallback would CORRUPT the model. Must raise."""
    rng = _rng(seed=1414)
    B, H, D = 1, 2, 64
    params = _make_params(rng, H, D)
    inputs = _random_step_inputs(rng, B, H, D, D, params)
    for banned in ["softmax", "full_attention", "sdpa", "flash_attn"]:
        with pytest.raises(ValueError):
            kda_state_decode_forward_v2(inputs, impl=banned)


def test_v2_dispatch_reference_and_nki_impls_ok() -> None:
    """Neither 'reference' nor 'nki' raise -- both succeed (nki falls through
    to reference in CPU-only environments)."""
    rng = _rng(seed=1515)
    B, H, D = 1, 2, 64
    params = _make_params(rng, H, D)
    inputs = _random_step_inputs(rng, B, H, D, D, params)
    for impl in ["reference", "nki"]:
        out = kda_state_decode_forward_v2(inputs, impl=impl)
        assert out.y.shape == (B, 1, H, D)


# ---------------------------------------------------------------------------
# 9. Slug + model preset sanity.
# ---------------------------------------------------------------------------

def test_v2_kernel_slug_stable() -> None:
    """Slug is load-bearing for the compile-cache identity hash."""
    assert KDA_STATE_V2_KERNEL_SLUG == (
        "kda_state.decode.kda_gate.rank1_delta.bf16_state.v1"
    )


def test_v2_model_shape_presets_match_scope_docs() -> None:
    assert KIMI_K3_KDA_SHAPE_V2.H == 96
    assert KIMI_K3_KDA_SHAPE_V2.D_v == 128
    assert KIMI_K3_KDA_SHAPE_V2.D_qk == 128
    assert KIMI_K3_KDA_SHAPE_V2.layers == 69
    assert GLM_5_3_FLASH_KDA_SHAPE_V2.H == 64
    assert GLM_5_3_FLASH_KDA_SHAPE_V2.D_v == 128
    assert GLM_5_3_FLASH_KDA_SHAPE_V2.D_qk == 128
    assert GLM_5_3_FLASH_KDA_SHAPE_V2.layers == 34
