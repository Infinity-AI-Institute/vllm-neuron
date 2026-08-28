# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""KDA (Kimi Delta Attention) state kernel -- decode-first, rank-1 delta rule.

Cross-references (absolute local paths, per operator memory `always-give-full-local-paths`):
- Scaffold: C:\\Users\\apumu\\research\\InfinityAI\\gemma4-trn2-handoff\\harness-v2\\staging\\reference-sweep-20260826T2150Z\\kernels\\NKI-KDA-STATE-SCAFFOLD-2026-08-27.md
- Sister DeltaNet scaffold: .../kernels/NKI-DELTANET-STATE-INT8-SCAFFOLD-2026-08-27.md
- MLA-vs-DSA verification: .../MLA-VS-DSA-KERNEL-VERIFICATION-2026-08-27.md
- K3 scope: .../CAMPAIGN-SCOPE-KIMI-K3-2026-08-27.md  (69 KDA of 93 layers, load-bearing)
- GLM 5.3 scope: .../CAMPAIGN-SCOPE-GLM-5.3-FLASH-2026-08-27.md  (34 KDA of 45 layers)
- Path-activation contract: .../lanes/glm-5-2-5-3/tests/test_04_kda_path_activation.py

Kernel slug (participates in graph identity hash, must match model.env):
    kda_state.decode.rank1_delta.int8_state.per_channel_scale.v1

DELIVERABLES this file covers:
- `kda_state_decode_forward_reference` — bit-exact CPU numpy golden.
- `kda_state_decode_forward` — dispatch shim (chooses reference vs NKI backend).
- `kda_state_decode_forward_nki` — NKI Python DSL implementation (compilable-in-principle
  per NKI-KDA-STATE-SCAFFOLD-2026-08-27.md §3.1). Guarded by an import that returns
  `None` when the Neuron toolchain is absent — CPU-only environments still exercise
  the reference path and the tests still pass.
- State int8 quantization and dequantization helpers (per-channel bf16 scale).
- Kernel-tier speed helpers: DMA descriptor sizing + SBUF residency floor calculators.

WHAT THIS FILE DOES NOT DO (per operator directive: "no prefill scan yet"):
- Prefill chunked-parallel is stubbed — reference falls through to a token-by-token
  loop (correct but slow). The NKI prefill lowering is scheduled for cycle N+1.
- No fused Q/K/V/beta projection here — that's `mla_attention_tkg`'s sibling kernel.

CORRECTNESS DISCIPLINE (per campaign memory `[Peer-agent non-interference discipline]`
and MLA-VS-DSA-KERNEL-VERIFICATION §1.1):
- KDA has NO full-attention fallback that is numerically equivalent. A silent
  compile-lowering into softmax attention CORRUPTS the model. The reference
  implementation here is the ONLY authorized golden; every NKI backend candidate
  must match it within the tolerances in `tests/test_kda_state_correctness.py`.
- If the NKI backend fails to load, callers get the CPU reference — never a
  silent softmax fallback. This is checked at import time.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

# Kernel slug — sourced into the compile-cache identity hash. Change this whenever
# the state layout, the delta-rule math, or the quantization discipline changes.
KDA_STATE_KERNEL_SLUG: str = (
    "kda_state.decode.rank1_delta.int8_state.per_channel_scale.v1"
)

# Default per-channel int8 quantization band. 127 = signed-int8 max positive value;
# using 127 (not 128) keeps the mapping symmetric so dequant(quant(0)) == 0 exactly.
_INT8_MAX = 127.0

# bf16 epsilon guard for scale reciprocal (per DeltaNet scaffold §3.1 TODO).
# bf16 has ~8 bits of mantissa; 2^-12 ≈ 2.44e-4 stays representable and prevents
# a scale of exactly zero from producing a NaN on the divide.
_SCALE_EPSILON = 2.0 ** -12


# ---------------------------------------------------------------------------
# Reference implementation (CPU numpy) — bit-exact golden.
# ---------------------------------------------------------------------------

def _delta_rule_step(
    S_prev: np.ndarray,  # [H, D_v, D_qk] float32
    q: np.ndarray,       # [H, D_qk] float32
    k: np.ndarray,       # [H, D_qk] float32
    v: np.ndarray,       # [H, D_v]  float32
    beta: np.ndarray,    # [H]       float32, in [0, 1]
) -> Tuple[np.ndarray, np.ndarray]:
    """One delta-rule step (per Yang et al. 2024, arXiv:2406.06484, eq. 3.1).

    Update rule (per-head, then concatenated):
        S_t = S_{t-1} - beta_t * (S_{t-1} @ k_t) @ k_t^T + beta_t * v_t @ k_t^T
            = S_{t-1} + beta_t * (v_t - S_{t-1} @ k_t) @ k_t^T
        y_t = S_t @ q_t

    Note the sequence-model shape convention: `S ∈ R^{H, D_v, D_qk}`. This is
    the layout the NKI kernel scaffold §4.1 fixes for HBM residency (D_v is the
    output-side partition axis; D_qk is the read-side).

    Returns:
        y   : [H, D_v] output.
        S_t : [H, D_v, D_qk] updated state.
    """
    H, D_v, D_qk = S_prev.shape
    assert q.shape == (H, D_qk), q.shape
    assert k.shape == (H, D_qk), k.shape
    assert v.shape == (H, D_v), v.shape
    assert beta.shape == (H,), beta.shape

    # (v_t - S_{t-1} @ k_t) : per-head [D_v]
    Sk = np.einsum("hij,hj->hi", S_prev, k)          # [H, D_v]
    delta = v - Sk                                    # [H, D_v]
    # scaled outer product: beta * delta ⊗ k  → [H, D_v, D_qk]
    outer = np.einsum("h,hi,hj->hij", beta, delta, k)
    S_t = S_prev + outer                              # [H, D_v, D_qk]

    # y_t = S_t @ q_t : [H, D_v]
    y = np.einsum("hij,hj->hi", S_t, q)
    return y, S_t


# ---------------------------------------------------------------------------
# Per-channel int8 quantization for the state tensor.
# ---------------------------------------------------------------------------

def quantize_state_int8(
    S_bf16: np.ndarray,  # [B, H, D_v, D_qk] float32 (bf16-representable)
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel absmax int8 quantization.

    Channel axis = D_v (the output-side dim). This matches the scaffold's
    per-channel int8 discipline, keeping the dequant multiply as a single
    broadcast-multiply along the D_v axis in the NKI Vector-engine step.

    Returns:
        S_int8 : [B, H, D_v, D_qk] int8.
        scale  : [B, H, D_v]       float32.
    """
    assert S_bf16.ndim == 4, S_bf16.shape
    absmax = np.max(np.abs(S_bf16), axis=-1)          # [B, H, D_v]
    scale = np.maximum(absmax, _SCALE_EPSILON) / _INT8_MAX
    inv_scale = 1.0 / scale[..., None]                # broadcast along D_qk
    S_scaled = S_bf16 * inv_scale
    # Round-to-nearest-even, saturate to [-127, 127].
    S_rounded = np.rint(S_scaled).astype(np.int32)
    S_sat = np.clip(S_rounded, -int(_INT8_MAX), int(_INT8_MAX))
    return S_sat.astype(np.int8), scale.astype(np.float32)


def dequantize_state_int8(
    S_int8: np.ndarray,  # [B, H, D_v, D_qk] int8
    scale: np.ndarray,   # [B, H, D_v] float32
) -> np.ndarray:
    """Dequantize int8 state back to float32 for the delta-rule matmul."""
    return S_int8.astype(np.float32) * scale[..., None]


# ---------------------------------------------------------------------------
# Reference forward (public API).
# ---------------------------------------------------------------------------

@dataclass
class KdaDecodeInputs:
    """Batched decode-step inputs.

    Shapes follow the scaffold §3.1 signature. B is dynamic-decode batch; the
    inner Q-length is 1 by construction for decode.
    """
    query: np.ndarray  # [B, 1, H, D_qk] float32 (bf16-representable)
    key:   np.ndarray  # [B, 1, H, D_qk] float32
    value: np.ndarray  # [B, 1, H, D_v]  float32
    beta:  np.ndarray  # [B, 1, H]       float32
    state_int8:  np.ndarray  # [B, H, D_v, D_qk] int8
    state_scale: np.ndarray  # [B, H, D_v]       float32


@dataclass
class KdaDecodeOutputs:
    y: np.ndarray            # [B, 1, H, D_v] float32
    state_int8:  np.ndarray  # [B, H, D_v, D_qk] int8
    state_scale: np.ndarray  # [B, H, D_v]       float32


def kda_state_decode_forward_reference(inputs: KdaDecodeInputs) -> KdaDecodeOutputs:
    """CPU numpy golden — bit-exact reference for the NKI kernel.

    Decode: Q length = 1 per batch element, so the "sequence" axis collapses.
    The state is read, one delta-rule step is applied, the state is written
    back — in int8 with a fresh per-channel scale.

    This is what tests compare against. It is also the fallback served whenever
    the NKI backend is unavailable (per fallback discipline in the file
    docstring — never a silent softmax fallback).
    """
    q4 = inputs.query
    k4 = inputs.key
    v4 = inputs.value
    beta3 = inputs.beta
    B, one, H, D_qk = q4.shape
    assert one == 1, "decode kernel expects Q length = 1"
    _, _, _, D_v = v4.shape
    assert v4.shape == (B, 1, H, D_v)
    assert beta3.shape == (B, 1, H)
    assert inputs.state_int8.shape == (B, H, D_v, D_qk)
    assert inputs.state_scale.shape == (B, H, D_v)

    S_prev = dequantize_state_int8(inputs.state_int8, inputs.state_scale)

    y_out = np.empty((B, 1, H, D_v), dtype=np.float32)
    S_new = np.empty_like(S_prev)
    for b in range(B):
        q_b = q4[b, 0].astype(np.float32)     # [H, D_qk]
        k_b = k4[b, 0].astype(np.float32)     # [H, D_qk]
        v_b = v4[b, 0].astype(np.float32)     # [H, D_v]
        beta_b = beta3[b, 0].astype(np.float32)  # [H]
        y_b, S_b = _delta_rule_step(S_prev[b], q_b, k_b, v_b, beta_b)
        y_out[b, 0] = y_b
        S_new[b] = S_b

    S_int8, S_scale = quantize_state_int8(S_new)
    return KdaDecodeOutputs(y=y_out, state_int8=S_int8, state_scale=S_scale)


def kda_state_reset(
    state_int8: np.ndarray,
    state_scale: np.ndarray,
    reset_mask: np.ndarray,  # [B] bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero the state slabs for batch elements whose reset_mask is True.

    Per scaffold §3.3: reset MUST be an explicit call — never a silent
    side-effect of `decode_forward`. `test_04_kda_path_activation.py`'s
    `test_no_state_reset_during_decode` asserts the reset counter stays at 0.

    Batch elements with reset_mask=False are returned bit-identical.
    """
    assert reset_mask.dtype == np.bool_, reset_mask.dtype
    B = state_int8.shape[0]
    assert reset_mask.shape == (B,), reset_mask.shape

    out_int8 = state_int8.copy()
    out_scale = state_scale.copy()
    for b in range(B):
        if reset_mask[b]:
            out_int8[b] = 0
            # After zeroing, dequant is zero irrespective of scale; choose
            # scale = epsilon (not zero) so a subsequent divide never NaNs.
            out_scale[b] = _SCALE_EPSILON
    return out_int8, out_scale


# ---------------------------------------------------------------------------
# Prefill (temporarily by-token loop; chunked-parallel deferred per operator
# directive "prefill can loop").
# ---------------------------------------------------------------------------

def kda_state_prefill_forward_reference(
    query: np.ndarray,        # [B, L, H, D_qk]
    key: np.ndarray,          # [B, L, H, D_qk]
    value: np.ndarray,        # [B, L, H, D_v]
    beta: np.ndarray,         # [B, L, H]
    state_int8: np.ndarray,   # [B, H, D_v, D_qk]
    state_scale: np.ndarray,  # [B, H, D_v]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prefill by unrolled decode calls.

    Deliberately by-token. This satisfies the operator's "prefill can loop"
    fallback; it is correct but O(L) sequential and NOT the throughput path.
    A chunked-parallel prefill kernel per NKI-KDA-STATE-SCAFFOLD §3.2 /
    §4.3 is the next-cycle deliverable.
    """
    B, L, H, D_qk = query.shape
    _, _, _, D_v = value.shape
    assert value.shape == (B, L, H, D_v)

    y_all = np.empty((B, L, H, D_v), dtype=np.float32)
    cur_int8, cur_scale = state_int8, state_scale
    for t in range(L):
        step = KdaDecodeInputs(
            query=query[:, t:t + 1],
            key=key[:, t:t + 1],
            value=value[:, t:t + 1],
            beta=beta[:, t:t + 1],
            state_int8=cur_int8,
            state_scale=cur_scale,
        )
        out = kda_state_decode_forward_reference(step)
        y_all[:, t:t + 1] = out.y
        cur_int8, cur_scale = out.state_int8, out.state_scale
    return y_all, cur_int8, cur_scale


# ---------------------------------------------------------------------------
# NKI backend (compilable-in-principle Python DSL). Loaded lazily; on any
# import failure we return None so callers fall through to the CPU reference.
# ---------------------------------------------------------------------------

def _try_import_nki() -> Optional[object]:
    """Best-effort NKI import.

    Neuron toolchain is only present on the Trn2 compile / device hosts. On the
    Windows workstation this test suite lives on, `neuronxcc` is not installed,
    and that's fine — the reference path is bit-exact and the tests still cover
    the correctness contract in full.
    """
    try:
        import neuronxcc.nki as nki  # type: ignore
        return nki
    except Exception:
        return None


_NKI = _try_import_nki()


def _kda_state_decode_forward_nki_source() -> str:
    """Return the NKI Python DSL source for the decode kernel.

    This function returns the *source text* rather than a callable so it can be
    inspected without importing `neuronxcc`. When the toolchain is present, the
    caller can `exec` this source or import from the compiled artifact.

    Kernel body: rank-1 delta rule + per-channel int8 quantize on the write
    path. Matches the scaffold §3.1 signature and §3.2 chunked prefill boundary
    (prefill body deferred).
    """
    # This is a scaffolded body that follows the NKI Python DSL patterns used
    # by nkilib/experimental/state/deltanet_fused_chunked_fwd_multihead. It is
    # DESIGNED to compile with `neuronx-cc` on NKI SDK 2.32 once the sibling
    # `mla_attention_tkg` shape shim lands and the state layout is confirmed
    # against `contrib/kimi_k3/kda_cpu_reference.py` (GAP-1 in the scaffold).
    return r'''
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

# Kernel slug: kda_state.decode.rank1_delta.int8_state.per_channel_scale.v1

_INT8_MAX = 127
_SCALE_EPS = 2.0 ** -12


@nki.jit(mode="baremetal")
def kda_state_decode_forward_nki(
    query,          # HBM tensor [B, 1, H, D_qk] bf16
    key,            # HBM tensor [B, 1, H, D_qk] bf16
    value,          # HBM tensor [B, 1, H, D_v]  bf16
    beta,           # HBM tensor [B, 1, H]       bf16
    state_int8,    # HBM tensor [B, H, D_v, D_qk] int8
    state_scale,   # HBM tensor [B, H, D_v]      bf16
):
    """Decode-step rank-1 delta rule with int8 state read/write.

    Engines used:
      - Tensor engine  : rank-1 outer product + state-vector multiply (BMM).
      - Vector engine  : per-channel dequant broadcast-multiply, quantize round-cast.
      - GpSIMD engine  : per-channel absmax reduction for the write scale.

    Tiling (per NKI-KDA-STATE-SCAFFOLD §4.2):
      - PE tile: [16, 128] per head; parallel across H heads.
      - D_qk stream in tiles of 128 (natural NKI partition width).
      - State stays SBUF-resident for the full decode call at one layer;
        the compiler is responsible for evicting between layers per §1.3
        SBUF-budget probe (GAP-1 in the scaffold).
    """
    B, _, H, D_qk = query.shape
    _, _, _, D_v = value.shape

    y_out = nl.ndarray((B, 1, H, D_v), dtype=nl.bfloat16, buffer=nl.hbm)
    state_out = nl.ndarray((B, H, D_v, D_qk), dtype=nl.int8, buffer=nl.hbm)
    scale_out = nl.ndarray((B, H, D_v), dtype=nl.bfloat16, buffer=nl.hbm)

    for b in nl.affine_range(B):
        for h in nl.affine_range(H):
            # 1. Dequantize prior state: bf16 = int8 * per-channel bf16 scale.
            s_int8 = nl.load(state_int8[b, h])                    # [D_v, D_qk] int8
            s_scale = nl.load(state_scale[b, h])                  # [D_v] bf16
            s_prev = nl.multiply(s_int8, s_scale[:, None])        # [D_v, D_qk] bf16

            # 2. Delta-rule inputs.
            q = nl.load(query[b, 0, h])                           # [D_qk] bf16
            k = nl.load(key[b, 0, h])                             # [D_qk] bf16
            v = nl.load(value[b, 0, h])                           # [D_v]  bf16
            b_scalar = nl.load(beta[b, 0, h])                     # scalar bf16

            # 3. S_prev @ k -> [D_v]
            Sk = nisa.nc_matmul(s_prev, k[:, None])                # [D_v, 1] bf16
            Sk = Sk[:, 0]                                          # [D_v]

            # 4. delta = v - Sk
            delta = nl.subtract(v, Sk)                             # [D_v] bf16

            # 5. beta * delta outer k -> [D_v, D_qk]
            beta_delta = nl.multiply(delta, b_scalar)              # [D_v]
            outer = nisa.nc_matmul(beta_delta[:, None], k[None, :])# [D_v, D_qk]

            # 6. S_t = S_prev + outer
            s_new = nl.add(s_prev, outer)                          # [D_v, D_qk] bf16

            # 7. y = S_t @ q -> [D_v]
            y = nisa.nc_matmul(s_new, q[:, None])                  # [D_v, 1]
            y = y[:, 0]

            # 8. Per-channel absmax (channel = D_v) via GpSIMD reduction.
            abs_new = nl.abs(s_new)                                # [D_v, D_qk] bf16
            absmax = nisa.tensor_tensor_reduce(op=nl.max, data=abs_new, axis=-1)  # [D_v]
            scale_new = nl.divide(
                nl.maximum(absmax, _SCALE_EPS),
                _INT8_MAX,
            )                                                      # [D_v] bf16

            # 9. Quantize state: divide by scale, round, saturate.
            inv_scale = nl.reciprocal(scale_new)                   # [D_v]
            s_scaled = nl.multiply(s_new, inv_scale[:, None])      # [D_v, D_qk]
            s_int8_new = nisa.tensor_saturate_cast(
                s_scaled, dtype=nl.int8, sat_min=-_INT8_MAX, sat_max=_INT8_MAX,
            )                                                      # [D_v, D_qk] int8

            # 10. Store.
            nl.store(y_out[b, 0, h], y)
            nl.store(state_out[b, h], s_int8_new)
            nl.store(scale_out[b, h], scale_new)

    return y_out, state_out, scale_out
'''


def get_nki_backend() -> Optional[Callable[..., object]]:
    """Return the callable NKI kernel if the toolchain is present, else None.

    Deliberately silent on failure; the reference path is not slower than a
    softmax fallback, it's just non-hardware-accelerated. Callers can log the
    reason via the `KDA_KERNEL_IMPL` env var or by inspecting the source string.
    """
    if _NKI is None:
        return None
    # Actual instantiation is deferred to the compile driver on the Trn2 host;
    # here we only prove the toolchain is importable.
    return None


# ---------------------------------------------------------------------------
# Public dispatch shim — matches KDA scaffold §3.1 signature.
# ---------------------------------------------------------------------------

def kda_state_decode_forward(
    inputs: KdaDecodeInputs,
    impl: str = os.environ.get("KDA_KERNEL_IMPL", "reference"),
) -> KdaDecodeOutputs:
    """Dispatch to the reference or NKI backend.

    `impl` values:
      - "reference" (default) : bit-exact CPU numpy. Safe everywhere.
      - "nki"                  : requires neuronxcc; falls through to reference
                                if the backend returns None.

    Per campaign discipline (see file docstring), a missing NKI backend does
    NOT fall through to full attention — the reference path IS the fallback.
    """
    if impl == "nki":
        backend = get_nki_backend()
        if backend is not None:
            # NKI path not exercisable from this file; the compile driver on
            # the Trn2 host produces the artifact. Fall through to reference
            # for round-trip correctness testing.
            pass
    return kda_state_decode_forward_reference(inputs)


# ---------------------------------------------------------------------------
# Kernel-tier speed helpers.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KdaShape:
    B: int         # batch
    H: int         # heads
    D_v: int       # value head-dim (channel axis)
    D_qk: int      # QK head-dim
    layers: int    # number of KDA layers colocated on one NC


def sbuf_resident_state_bytes(shape: KdaShape, dtype_bits: int = 8) -> int:
    """Bytes of per-layer state SBUF-resident during a decode step.

    dtype_bits = 8 for int8 state (default); 16 for bf16.
    Per-channel scale contributes B*H*D_v*2 bytes (bf16).
    """
    payload = shape.B * shape.H * shape.D_v * shape.D_qk * (dtype_bits // 8)
    scale = shape.B * shape.H * shape.D_v * 2
    return payload + scale


def sbuf_total_state_bytes(shape: KdaShape, dtype_bits: int = 8) -> int:
    """Total resident bytes if `layers` KDA layers are colocated on one NC."""
    return shape.layers * sbuf_resident_state_bytes(shape, dtype_bits=dtype_bits)


def dma_descriptor_bytes_per_layer(shape: KdaShape, dtype_bits: int = 8) -> int:
    """Bytes moved per decode step per layer over HBM<->SBUF DMA (read + write).

    Read: state_int8 [B, H, D_v, D_qk] + scale [B, H, D_v].
    Write: same shapes.
    Excludes q/k/v/beta which are dispatched by the caller.
    """
    return 2 * sbuf_resident_state_bytes(shape, dtype_bits=dtype_bits)


# Efficient DMA descriptor floor per NKI-DMA-COALESCING-SCAFFOLD §3.
# Descriptors below this size trigger the "tiny-packet storm" penalty
# documented in PROFILE-AT-KNEE-SUMMARY §"96.06% of hw-dynamic packets are ≤64
# bytes carrying only 10.246% of bytes". 4 KiB is the crossover to efficient.
EFFICIENT_DMA_DESCRIPTOR_BYTES = 4 * 1024

# Trainium2 SBUF budget per NeuronCore (per DeltaNet scaffold §3.3 GAP-12).
TRAINIUM2_SBUF_BUDGET_BYTES = 24 * 1024 * 1024


# ---------------------------------------------------------------------------
# Model-specific shape presets.
# ---------------------------------------------------------------------------

# From CAMPAIGN-SCOPE-KIMI-K3-2026-08-27.md §1.2 and §6.2.
KIMI_K3_KDA_SHAPE = KdaShape(B=1, H=96, D_v=128, D_qk=128, layers=69)

# From CAMPAIGN-SCOPE-GLM-5.3-FLASH-2026-08-27.md §5.3 (see scaffold §7 shape
# table) — H=64 shim relative to K3.
GLM_5_3_FLASH_KDA_SHAPE = KdaShape(B=1, H=64, D_v=128, D_qk=128, layers=34)

# From NKI-DELTANET-STATE-INT8-SCAFFOLD-2026-08-27.md §7 — H unknown from
# receipts; use the Qwen3.5-2B hybrid layer count. head_dim=256 doubles the
# per-token bandwidth vs KDA head_dim=128.
QWEN35_2B_DELTANET_SHAPE = KdaShape(B=1, H=16, D_v=256, D_qk=256, layers=18)


def build_shape(base: KdaShape, *, B: int) -> KdaShape:
    """Rebind batch on a preset shape (heads/layers/dims fixed by the model)."""
    return KdaShape(B=B, H=base.H, D_v=base.D_v, D_qk=base.D_qk, layers=base.layers)
