# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for kda_state_nki_v2 -- dry-import, slug hygiene, source hygiene,
and dispatch fallback discipline.

Runs on any environment that has numpy (no neuronxcc required). On the Trn2
host the compile driver runs the real end-to-end NEFF vs CPU-golden test
separately; this file only covers the shim.

Companion: full CPU-golden correctness lives in
`test_kda_state_v2_correctness.py`. This file does NOT re-run those 25 tests
against the shim -- it's a smoke pass, per the task deliverable.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_KERNELS = os.path.dirname(_HERE)
if _KERNELS not in sys.path:
    sys.path.insert(0, _KERNELS)

import numpy as np  # noqa: E402

from kda_state_v2 import (  # noqa: E402
    KDA_LOWER_BOUND,
    KDA_L2NORM_EPS,
    KDA_STATE_V2_KERNEL_SLUG,
    KdaDecodeInputsV2,
    KdaLayerParams,
    bf16_cast,
    kda_state_decode_forward_reference_v2,
)


# ---------------------------------------------------------------------------
# Dry-import + surface
# ---------------------------------------------------------------------------

def test_module_imports():
    """The NKI shim module must import without a live neuronxcc install."""
    import kda_state_nki_v2  # noqa: F401
    assert kda_state_nki_v2.KDA_STATE_NKI_V2_KERNEL_SLUG


def test_public_surface_present():
    import kda_state_nki_v2 as m
    expected = [
        "KDA_STATE_NKI_V2_KERNEL_SLUG",
        "KDA_STATE_NKI_V2_SOURCE",
        "KDA_NKI_V2_TILING",
        "KdaNkiTilingV2",
        "kda_state_decode_forward_nki_v2",
        "kda_state_prefill_forward_nki_v2",
        "get_nki_backend_v2",
        "get_nki_source",
        "register_compiled_nki_kernel",
        "nki_source_matches_cpu_golden_constants",
        "KIMI_K3_KDA_SHAPE_NKI_V2",
        "GLM_5_3_FLASH_KDA_SHAPE_NKI_V2",
    ]
    for name in expected:
        assert hasattr(m, name), f"missing public symbol: {name}"


# ---------------------------------------------------------------------------
# Slug hygiene
# ---------------------------------------------------------------------------

def test_slug_differs_from_cpu_golden():
    """A stale NEFF from the CPU-golden slug served against the NKI shim
    (or vice versa) is a silent-correctness bug -- KDA has no numerically
    equivalent fallback. The slugs MUST diverge.
    """
    import kda_state_nki_v2 as m
    assert m.KDA_STATE_NKI_V2_KERNEL_SLUG != KDA_STATE_V2_KERNEL_SLUG
    assert m.KDA_STATE_NKI_V2_KERNEL_SLUG.endswith(".nki_v1")


def test_slug_contains_load_bearing_tokens():
    """Compile-cache key generator must include these substrings."""
    import kda_state_nki_v2 as m
    slug = m.KDA_STATE_NKI_V2_KERNEL_SLUG
    for tok in ("kda_gate", "rank1_delta", "bf16_state", "nki_v1"):
        assert tok in slug, f"slug missing load-bearing token: {tok}"


# ---------------------------------------------------------------------------
# NKI source hygiene
# ---------------------------------------------------------------------------

def test_source_is_nonempty_string():
    import kda_state_nki_v2 as m
    assert isinstance(m.KDA_STATE_NKI_V2_SOURCE, str)
    assert len(m.KDA_STATE_NKI_V2_SOURCE) > 500, "source suspiciously short"


def test_source_declares_kernel_body_and_imports():
    import kda_state_nki_v2 as m
    src = m.KDA_STATE_NKI_V2_SOURCE
    assert "import neuronxcc.nki as nki" in src
    assert "import neuronxcc.nki.language as nl" in src
    assert "import neuronxcc.nki.isa as nisa" in src
    assert "@nki.jit" in src
    assert "def kda_state_decode_forward_nki_v2_body(" in src


def test_source_contains_all_4_fla_parity_pieces():
    """The source must implement the four FLA v0.5.2 parity fixes documented
    in KDA-STATE-V2-STATUS-2026-08-28.md §3.
    """
    import kda_state_nki_v2 as m
    src = m.KDA_STATE_NKI_V2_SOURCE

    # (a) KDA per-channel gate: Diag(alpha_h) applied before delta step
    assert "KDA_LOWER_BOUND / (" in src, "KDA per-channel gate arithmetic"
    assert "1.0 + nl.exp(-(a_amp * g_plus))" in src
    assert "decay = nl.exp(alpha)" in src
    assert "s = s * decay[None, :]" in src, "state decay broadcast on D_v"

    # (b) In-kernel L2-norm on Q and K (eps=1e-6)
    assert "q_sq_sum = nl.sum(q * q)" in src
    assert "k_sq_sum = nl.sum(k * k)" in src
    assert "KDA_L2NORM_EPS" in src
    assert "q = q / q_denom" in src
    assert "k = k / k_denom" in src

    # (c) Query scale q *= 1/sqrt(D_qk) applied AFTER L2-norm
    assert "q = q * scale_value" in src
    # Ordering check: scale must come after L2-norm.
    scale_pos = src.index("q = q * scale_value")
    norm_pos = src.index("q = q / q_denom")
    assert norm_pos < scale_pos, "query scale must apply AFTER L2-norm"

    # (d) bf16 state layout [num_slots, HV, V, K] and bf16 store on state + y
    assert "state_in[b, h]" in src
    assert "state_out[b, h]" in src
    assert "s.astype(nl.bfloat16)" in src, "state stored bf16"
    assert "y.astype(nl.bfloat16)" in src, "y stored bf16"
    assert "dtype=nl.bfloat16" in src


def test_source_constants_match_cpu_golden():
    """Static sanity: the constants baked into the source match the CPU
    golden's constants.
    """
    import kda_state_nki_v2 as m
    assert m.nki_source_matches_cpu_golden_constants()


def test_source_omits_spec_decode_branch():
    """Operator hard rule `[No spec-decode methodology 2026-08-27]`: kernel
    must not include the IS_SPEC_DECODING branch.
    """
    import kda_state_nki_v2 as m
    assert "IS_SPEC_DECODING" not in m.KDA_STATE_NKI_V2_SOURCE
    assert "spec_decoding" not in m.KDA_STATE_NKI_V2_SOURCE
    assert "use_spec" not in m.KDA_STATE_NKI_V2_SOURCE


def test_source_documents_fla_reference_path():
    """Auditor must be able to jump from source to the FLA scratchpad mirror."""
    import kda_state_nki_v2 as m
    src = m.KDA_STATE_NKI_V2_SOURCE
    assert "fused_recurrent_gated_delta_rule_fwd_kernel" in src
    assert "132-173" in src  # per-token body line range


# ---------------------------------------------------------------------------
# Tile shape checks
# ---------------------------------------------------------------------------

def test_default_tiling_fits_sbuf():
    """Per-head SBUF residency (bf16 tile + fp32 working copy) must sit well
    under the 24 MiB Trn2 per-NC SBUF budget with headroom for the tensor-
    engine scratch.
    """
    import kda_state_nki_v2 as m
    t = m.KDA_NKI_V2_TILING
    bf16_tile = t.sbuf_state_bytes_per_head()
    fp32_working = t.sbuf_state_bytes_per_head_fp32_working()
    total = bf16_tile + fp32_working
    SBUF_BUDGET = 24 * 1024 * 1024
    assert total < SBUF_BUDGET // 32, (
        f"per-head SBUF residency {total} B too close to budget {SBUF_BUDGET}"
    )
    # BK must match the two target models' D_qk.
    assert t.BK == 128
    # BV must fit both HV=64 (GLM-5.3-Flash) and HV=96 (K3) evenly enough
    # that the head-parallel outer loop pipelines. BV=64 divides both.
    assert t.BV == 64


# ---------------------------------------------------------------------------
# Backend probe (no compile driver available on Windows)
# ---------------------------------------------------------------------------

def test_get_backend_returns_none_without_compile_driver():
    """On Windows the NKI toolchain is absent; the probe must return None,
    not raise.
    """
    import kda_state_nki_v2 as m
    assert m.get_nki_backend_v2() is None
    # And for a specific shape:
    assert m.get_nki_backend_v2((1, 96, 128, 128)) is None


def test_register_and_lookup_compiled_kernel_shape_scoped():
    """The registry hook must be shape-scoped so a K3 NEFF cannot serve
    GLM-5.3-Flash (and vice versa), matching the CPU-golden shape-hash
    divergence test.
    """
    import kda_state_nki_v2 as m
    called_with = {}

    def fake_neff(*args):
        called_with["args"] = args
        return "fake_y", "fake_state"

    k3_shape = (1, 96, 128, 128)
    glm_shape = (1, 64, 128, 128)
    m.register_compiled_nki_kernel(k3_shape, fake_neff)
    try:
        assert m.get_nki_backend_v2(k3_shape) is fake_neff
        # A different shape must NOT resolve to this NEFF.
        assert m.get_nki_backend_v2(glm_shape) is None
    finally:
        # Cleanup: don't leak the fake into the module's global cache.
        m._KDA_NKI_CALLABLES.pop(
            (m.KDA_STATE_NKI_V2_KERNEL_SLUG, k3_shape), None
        )


# ---------------------------------------------------------------------------
# Dispatch shim: fallback discipline + shape/dtype contract
# ---------------------------------------------------------------------------

def _make_inputs(B=1, H=4, D_v=128, D_qk=128, seed=0):
    """Build a minimal KdaDecodeInputsV2 with bf16-representable random data."""
    rng = np.random.default_rng(seed)
    def r(*shape):
        # Random normal then bf16-cast so state layout is consistent with the
        # CPU golden's expectations.
        arr = rng.standard_normal(shape).astype(np.float32) * 0.1
        return bf16_cast(arr)

    inputs = KdaDecodeInputsV2(
        query=r(B, 1, H, D_qk),
        key=r(B, 1, H, D_qk),
        value=r(B, 1, H, D_v),
        g_raw=r(B, 1, H, D_qk),
        beta_raw=r(B, 1, H),
        state_bf16=r(B, H, D_v, D_qk),
        params=KdaLayerParams(
            a_log=r(H),
            g_bias=r(H, D_qk),
        ),
    )
    return inputs


def test_dispatch_rejects_softmax_impl():
    """A silent softmax fallback CORRUPTS a KDA model. Must raise, not fall
    through. Locks in MLA-VS-DSA §1.1.
    """
    import kda_state_nki_v2 as m
    inputs = _make_inputs()
    for bad_impl in ("softmax", "full_attention", "sdpa", "flash_attn"):
        with pytest.raises(ValueError, match="banned"):
            m.kda_state_decode_forward_nki_v2(inputs, impl=bad_impl)


def test_dispatch_falls_through_to_cpu_golden_when_cold():
    """impl='auto' with a cold NKI cache must fall through to the CPU golden
    -- correct but slow. Output must match the CPU golden bit-for-bit.
    """
    import kda_state_nki_v2 as m
    inputs = _make_inputs()
    out_shim = m.kda_state_decode_forward_nki_v2(inputs, impl="auto")
    out_ref = kda_state_decode_forward_reference_v2(inputs)
    # bit-exact match: shim IS the reference on this path
    np.testing.assert_array_equal(out_shim.y, out_ref.y)
    np.testing.assert_array_equal(out_shim.state_bf16, out_ref.state_bf16)


def test_dispatch_reference_impl_matches_cpu_golden():
    """Explicit impl='reference' must bypass the NKI probe and route directly
    to the CPU golden.
    """
    import kda_state_nki_v2 as m
    inputs = _make_inputs()
    out_shim = m.kda_state_decode_forward_nki_v2(inputs, impl="reference")
    out_ref = kda_state_decode_forward_reference_v2(inputs)
    np.testing.assert_array_equal(out_shim.y, out_ref.y)
    np.testing.assert_array_equal(out_shim.state_bf16, out_ref.state_bf16)


def test_dispatch_nki_impl_raises_when_cold():
    """impl='nki' with a cold cache must raise loudly, never silently
    downgrade -- a caller who explicitly asked for NKI wants to know it
    wasn't served.
    """
    import kda_state_nki_v2 as m
    inputs = _make_inputs()
    with pytest.raises(RuntimeError, match="NKI backend is not warm|numpy missing"):
        m.kda_state_decode_forward_nki_v2(inputs, impl="nki")


def test_prefill_shim_rejects_softmax():
    """Same fallback discipline applies to the prefill shim."""
    import kda_state_nki_v2 as m
    L = 4
    B, H, D = 1, 4, 128
    rng = np.random.default_rng(0)
    def r(*shape):
        return bf16_cast(rng.standard_normal(shape).astype(np.float32) * 0.1)

    params = KdaLayerParams(a_log=r(H), g_bias=r(H, D))
    for bad_impl in ("softmax", "full_attention"):
        with pytest.raises(ValueError, match="banned"):
            m.kda_state_prefill_forward_nki_v2(
                query=r(B, L, H, D), key=r(B, L, H, D),
                value=r(B, L, H, D), g_raw=r(B, L, H, D),
                beta_raw=r(B, L, H), state_bf16=r(B, H, D, D),
                params=params, impl=bad_impl,
            )


def test_prefill_shim_falls_through_to_cpu_golden():
    """Chunk-KDA NKI body not landed yet -- shim must fall through cleanly."""
    import kda_state_nki_v2 as m
    L = 4
    B, H, D = 1, 4, 128
    rng = np.random.default_rng(1)
    def r(*shape):
        return bf16_cast(rng.standard_normal(shape).astype(np.float32) * 0.1)

    params = KdaLayerParams(a_log=r(H), g_bias=r(H, D))
    y, state_out = m.kda_state_prefill_forward_nki_v2(
        query=r(B, L, H, D), key=r(B, L, H, D),
        value=r(B, L, H, D), g_raw=r(B, L, H, D),
        beta_raw=r(B, L, H), state_bf16=r(B, H, D, D),
        params=params,
    )
    assert y.shape == (B, L, H, D)
    assert state_out.shape == (B, H, D, D)


# ---------------------------------------------------------------------------
# Shape presets (first plug-in path parity)
# ---------------------------------------------------------------------------

def test_kimi_k3_preset_shape():
    """First plug-in target: HV=96, D=128, 69 KDA layers."""
    import kda_state_nki_v2 as m
    p = m.KIMI_K3_KDA_SHAPE_NKI_V2
    assert p.H == 96
    assert p.D_v == 128
    assert p.D_qk == 128
    assert p.layers == 69


def test_glm53_flash_preset_shape():
    """Second plug-in target: HV=64, D=128, 34 KDA layers."""
    import kda_state_nki_v2 as m
    p = m.GLM_5_3_FLASH_KDA_SHAPE_NKI_V2
    assert p.H == 64
    assert p.D_v == 128
    assert p.D_qk == 128
    assert p.layers == 34
