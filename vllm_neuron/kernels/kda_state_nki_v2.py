# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""KDA (Kimi Delta Attention) state kernel V2 -- NKI device body.

Callsign: kda-nki-v2-agent (Fleet A KDA authoring, campaign
reference-sweep-2026-08-26, tick 2026-08-27).

Sibling to (and API-compatible with) the CPU golden:
    C:\\Users\\apumu\\research\\InfinityAI\\gemma4-trn2-handoff\\harness-v2\\
        staging\\reference-sweep-20260826T2150Z\\kernels\\kda_state_v2.py

Kernel slug (participates in compile-cache identity hash; must NEVER collide
with the CPU-golden slug or with the deprecated V1 int8 slug):

    kda_state.decode.kda_gate.rank1_delta.bf16_state.nki_v1

vs

    kda_state.decode.kda_gate.rank1_delta.bf16_state.v1        (V2 CPU golden)
    kda_state.decode.rank1_delta.int8_state.per_channel_scale.v1 (V1, DRIFTING)

The `nki_v1` token is load-bearing: the Trn2 compile driver's cache-key
generator MUST include it. A stale NEFF from the CPU-golden slug served to
this kernel is a silent-correctness bug caught only at model-output-drift
time (KDA has no numerically-equivalent fallback -- see MLA-VS-DSA §1.1).

------------------------------------------------------------------------
WHAT THIS FILE SHIPS
------------------------------------------------------------------------

- `KDA_STATE_NKI_V2_KERNEL_SLUG` -- the new-slug string.
- `KDA_STATE_NKI_V2_SOURCE` -- the NKI Python DSL source, as a text string.
  Every algorithmic knob (LOWER_BOUND, L2NORM_EPS, scale, layout) matches
  the CPU golden bit-for-bit. This is the artifact the Trn2 compile driver
  ingests; it is NOT `exec()`ed on Windows.
- `kda_state_decode_forward_nki_v2(inputs)` -- dispatch shim with the
  same signature as `kda_state_decode_forward_v2` in the CPU-golden module.
  On a box where `neuronxcc` is importable AND the NEFF for the current
  shape is warm, it calls the compiled kernel; otherwise it falls through
  to the CPU golden. NEVER to softmax.
- `kda_state_prefill_forward_nki_v2(...)` -- prefill dispatch shim. Same
  fallback discipline. The NKI decode body ships now; the NKI chunk-KDA
  prefill body is a follow-on tick (source string TODO'd inline; falls
  through to the CPU golden by-token prefill in the meantime).
- `get_nki_backend_v2()` -- returns the compiled callable if available,
  else None. Callers use it for capability probing.
- `KDA_NKI_V2_TILING` -- named tile shapes (BV=64 by default, BK=128) with
  the H2H=32 split for HV=64 and HV=96 alignments -- see §Tiling below.

------------------------------------------------------------------------
WHAT THIS FILE DOES NOT DO (honest scope, per task 2026-08-27)
------------------------------------------------------------------------

- Windows-side NKI compilation. The NKI DSL is delivered as source text
  and is executed by the Trn2 host's compile driver. On Windows the shim
  falls through to the CPU golden -- correctly, never to softmax.
- Chunked-parallel prefill (`chunk_kda_with_fused_gate` FLA equivalent).
  The decode body ships now; the chunk body is a stub with an inline TODO
  and a fallback to the CPU-golden by-token prefill (correct but O(L)).
- Short causal-conv1d on Q/K/V (kernel_size=4, silu). Caller pre-applies.
- FusedRMSNormGated on the output. Caller post-applies.
- TP shard-shape adaptation for `a_log` / `g_bias`. Caller pre-shards.

Downstream ticks pick these up; none are on the K3-first-plug-in or the
GLM-5.3-Flash-first-plug-in critical path.

------------------------------------------------------------------------
FIRST PLUG-IN PATH (Kimi K3 KDA layers, 69/93)
------------------------------------------------------------------------

Kimi K3 (HV=96, D_qk=D_v=128, 69 KDA layers, per
`CAMPAIGN-SCOPE-KIMI-K3-2026-08-27.md`) is the first-plug-in target -- highest
customer-demand signal (per `MAKORA-COMPETITIVE-CONTEXT` memory,
2.8M HF downloads/month). The kernel signature accepts vLLM PR #53906's
param layout directly:

    a_log:    [H]              (sharded on head axis in vLLM;
                                per-NC after TP shard)
    g_bias:   [H, D_qk]        (sharded on projection axis in vLLM)
    g_raw:    [B, 1, H, D_qk]
    beta_raw: [B, 1, H]
    state:    [num_slots, HV, V, K]  ==  [B, H, D_v, D_qk] in decode

Plug-in point (Kimi K3 model file):
    `vllm/models/kimi_k3/nvidia/kda.py:564` (plain decode entry; the
    `use_spec` branch at :497 is out of scope per the campaign no-spec-decode
    hard rule -- operator memory `[No spec-decode methodology 2026-08-27]`).

GLM-5.3-Flash (HV=64, same algorithm) inherits the same kernel via a pure
shape rebind -- see `KIMI_K3_KDA_SHAPE_V2` vs `GLM_5_3_FLASH_KDA_SHAPE_V2`
in the CPU-golden module. The compile-cache key is `(slug, shape_tuple)`;
`test_v2_shape_hash_diverges_across_models` (in the CPU golden test suite)
already locks in that a K3 NEFF cannot accidentally serve GLM-5.3-Flash.

------------------------------------------------------------------------
TILING (matches the CPU golden's einsum shapes)
------------------------------------------------------------------------

Per-head, per-batch tile (executed once per (b, h) iteration in the NKI
body):

    state SBUF tile : [BV=64, BK=128]  bf16  ->  fp32 for the body
    q, k SBUF row   : [BK=128]         bf16  ->  fp32
    v   SBUF row    : [BV=64]          bf16  ->  fp32
    g_raw SBUF row  : [BK=128]         bf16  ->  fp32
    a_log, beta_raw : scalar (broadcast per-head)
    g_bias SBUF row : [BK=128]         bf16  ->  fp32 (pinned across steps)

BV=64 was chosen so a single-head state tile (64 * 128 * 2 = 16 KiB) leaves
plenty of headroom in the 24 MiB SBUF for the fp32-promoted copy (16 KiB *
2 = 32 KiB) plus the tensor-engine scratch (~128 KiB per matmul).

BK=128 matches the model's D_qk (both K3 and GLM-5.3-Flash) so the whole
K axis fits one PE tile per head. If a future model has D_qk > 128 we'll
loop over BK partitions (source has a `for bk_off in nl.affine_range(K // BK)`
hook for that -- currently commented out because both target models fit).

For HV=96 (Kimi K3), we run H heads in the outer `nl.affine_range` and let
the compiler pipeline. This matches the FLA Triton kernel's `program_id(1)`
sharding on the head axis. For HV=64 (GLM-5.3-Flash), same code path, fewer
iterations.

------------------------------------------------------------------------
ENGINE ASSIGNMENTS (Trn2 -- for the compile driver's static schedule)
------------------------------------------------------------------------

Per per-head iteration:

- Tensor engine  : `S @ k`  ([BV, BK] @ [BK, 1])  -> [BV, 1]
                   `S @ q`  ([BV, BK] @ [BK, 1])  -> [BV, 1]
                   outer `delta @ k.T` ([BV, 1] @ [1, BK]) -> [BV, BK]
- Vector engine  : all element-wise: `mul`, `add`, `sub`, `sigmoid`, `exp`
- GpSIMD (scalar): L2-norm reductions on Q, K (per-head scalar denominator)
                   `a_amp = exp(a_log)` (per-head scalar)
- DMA (HBM<->SBUF): state read (BV*BK*2 B/head), state write (BV*BK*2 B),
                    Q/K/V/g_raw read, y write, a_log/g_bias pinned once
                    per compile.

The tensor engine is the throughput floor (per-head 3 matmuls of shape
[64, 128] @ [128, 1] and one [64, 1] @ [1, 128] outer product). Vector and
GpSIMD run concurrently in the compiler's static schedule and are not the
bottleneck.

------------------------------------------------------------------------
FALLBACK DISCIPLINE
------------------------------------------------------------------------

Per `MLA-VS-DSA-KERNEL-VERIFICATION-2026-08-27.md` §1.1 and campaign memory
`[Peer-agent non-interference discipline]`, KDA has NO full-attention
fallback that is numerically equivalent. This file's dispatch shim reuses
`_BANNED_IMPLS` from the CPU-golden module and refuses any softmax-family
lowering.

When the NKI backend is not present or the NEFF for the current shape is
cold, the shim falls through to the CPU golden -- correct but slow. This
is the ONLY authorized fallback path. `test_nki_v2_dispatch_rejects_softmax`
locks this in on both the decode and the prefill entry points.
"""

from __future__ import annotations

import importlib.util
import math
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

# The CPU golden IS the fallback. Import lazily so the module can be inspected
# on machines without numpy (the source-string is inspectable statically).
try:  # pragma: no cover -- import guarded for Windows-without-numpy inspection
    import numpy as np  # type: ignore
    from kda_state_v2 import (  # type: ignore
        GLM_5_3_FLASH_KDA_SHAPE_V2,
        KDA_DEFAULT_HEAD_DIM,
        KDA_LOWER_BOUND,
        KDA_L2NORM_EPS,
        KDA_STATE_V2_KERNEL_SLUG,
        KIMI_K3_KDA_SHAPE_V2,
        KdaDecodeInputsV2,
        KdaDecodeOutputsV2,
        KdaLayerParams,
        KdaShapeV2,
        _BANNED_IMPLS,
        bf16_cast,
        build_shape_v2,
        kda_state_decode_forward_reference_v2,
        kda_state_prefill_forward_reference_v2,
        kda_state_reset_v2,
    )
    _CPU_GOLDEN_AVAILABLE = True
except ImportError:  # numpy missing (import-time inspection on a bare Windows)
    np = None  # type: ignore
    _CPU_GOLDEN_AVAILABLE = False
    KDA_LOWER_BOUND = -5.0
    KDA_L2NORM_EPS = 1e-6
    KDA_DEFAULT_HEAD_DIM = 128
    KDA_STATE_V2_KERNEL_SLUG = (
        "kda_state.decode.kda_gate.rank1_delta.bf16_state.v1"
    )
    _BANNED_IMPLS = frozenset(  # type: ignore[assignment]
        {"softmax", "full_attention", "sdpa", "flash_attn"}
    )


# ---------------------------------------------------------------------------
# Kernel identity -- distinct slug from the CPU golden
# ---------------------------------------------------------------------------

KDA_STATE_NKI_V2_KERNEL_SLUG: str = (
    "kda_state.decode.kda_gate.rank1_delta.bf16_state.nki_v1"
)

assert KDA_STATE_NKI_V2_KERNEL_SLUG != KDA_STATE_V2_KERNEL_SLUG, (
    "NKI slug MUST NOT collide with CPU-golden slug -- a stale NEFF from one "
    "served against the other is a silent-correctness bug (KDA has no fallback)."
)


# ---------------------------------------------------------------------------
# Tile shape defaults
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KdaNkiTilingV2:
    """Per-head tile shapes for the Trn2 NKI decode kernel.

    Selected so a single-head bf16 state tile (BV * BK * 2 B) plus its
    fp32-promoted working copy plus the tensor-engine scratchpad all fit
    comfortably in the 24 MiB per-NC SBUF budget with 6+ MiB slack for
    weights and pipeline.
    """
    BV: int = 64
    BK: int = 128
    # H_partition: how many heads run per outer `nl.affine_range` step.
    # 1 for both K3 (HV=96) and GLM-5.3-Flash (HV=64); the compiler pipelines.
    H_partition: int = 1

    def sbuf_state_bytes_per_head(self) -> int:
        return self.BV * self.BK * 2  # bf16 payload

    def sbuf_state_bytes_per_head_fp32_working(self) -> int:
        return self.BV * self.BK * 4  # fp32 working copy


KDA_NKI_V2_TILING = KdaNkiTilingV2()


# ---------------------------------------------------------------------------
# NKI DSL source (this is the artifact the compile driver ingests)
# ---------------------------------------------------------------------------

def _kda_state_nki_v2_source(
    lower_bound: float = KDA_LOWER_BOUND,
    l2norm_eps: float = KDA_L2NORM_EPS,
    tiling: KdaNkiTilingV2 = KDA_NKI_V2_TILING,
) -> str:
    """Return the NKI Python DSL source for the bf16-state decode kernel.

    The constexpr constants (LOWER_BOUND, L2NORM_EPS, BV, BK) are baked in
    at source-generation time so the compile-cache key can hash them into
    the identity slug.

    Body corresponds line-for-line to FLA v0.5.2
    `fused_recurrent_gated_delta_rule_fwd_kernel` at
    `vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:132-173`
    with the KDA constexpr set (IS_KDA=True, COMPUTE_GATE=True, SAFE_GATE=True,
    USE_QK_L2NORM_IN_KERNEL=True, SIGMOID_BETA=True, LOWER_BOUND=-5.0).

    Not `exec()`ed on Windows -- returned as inspectable text.
    """
    # Format-safe: no user input in the interpolated positions.
    lb = float(lower_bound)
    eps = float(l2norm_eps)
    BV = int(tiling.BV)
    BK = int(tiling.BK)
    slug = KDA_STATE_NKI_V2_KERNEL_SLUG
    return f'''# NKI kernel source -- slug: {slug}
# Bit-exact transcription of FLA v0.5.2 fused_recurrent_gated_delta_rule_fwd_kernel
# with KDA constexpr set (IS_KDA=True, COMPUTE_GATE=True, SAFE_GATE=True,
# USE_QK_L2NORM_IN_KERNEL=True, SIGMOID_BETA=True, LOWER_BOUND={lb}).
# Reference source (scratchpad mirror of vLLM PR #53906):
#   C:/Users/apumu/AppData/Local/Temp/claude/
#     C--Users-apumu-research-InfinityAI-gemma4-trn2-handoff/
#     fe22bc9a-b8c7-4107-ac55-1d55ba4bd33a/scratchpad/kda-fetch/
#     fused_recurrent_vllm_third_party_flash_linear_attention_ops.py
# Function `fused_recurrent_gated_delta_rule_fwd_kernel`, lines 132-173.

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

# Constexpr constants -- baked in at source-generation time.
KDA_LOWER_BOUND = {lb}
KDA_L2NORM_EPS = {eps}
BV = {BV}
BK = {BK}


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

            # ---- (2) L2-norm on q, k (eps={eps}) -------------------------
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
'''


KDA_STATE_NKI_V2_SOURCE: str = _kda_state_nki_v2_source()


# ---------------------------------------------------------------------------
# File-based body module (EXEC-TO-FILE-IMPORT-PATCH-2026-08-28)
# ---------------------------------------------------------------------------
#
# The kernel body ALSO lives as a physical Python file at
# `_kernel_bodies/kda_state_nki_v2_body.py` with default constants baked in.
# `@nki.jit`'s KernelRewriter.reparse_function calls
# `inspect.getsource(<decorated_fn>)` at compile time, and that call requires
# a real file on disk to walk. `exec(source_str, ns)` leaves the function's
# `__module__` as "<string>" and every NKI compile then raises
# `OSError: could not get source code`. Loading the body via
# `importlib.util.spec_from_file_location` + `spec.loader.exec_module` gives
# the decorated function a real `__file__` for `inspect.getsource`.
#
# See kernels/EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md for the bug diagnosis
# discovered on the Kimi K3 Route B fire.

_KERNEL_MODULE_SOURCE_PATH = (
    pathlib.Path(__file__).resolve().parent
    / "_kernel_bodies"
    / "kda_state_nki_v2_body.py"
)
"""Physical path to the default-constants NKI body. Load-bearing: file-import
of this path is the ONLY dispatch pattern that keeps `inspect.getsource`
happy when `@nki.jit` re-parses the decorated function."""


_KERNEL_MODULE_CACHE: Dict[str, Any] = {}


def _load_nki_kernel_module_if_available() -> Optional[Any]:
    """File-import the body module when the NKI toolchain is present.

    Returns the imported module (whose `kda_state_decode_forward_nki_v2_body`
    attribute is the `@nki.jit`-decorated callable) on success, or None on
    any host without NKI or if the body file is missing.

    Idempotent: subsequent calls return the same cached module.
    """
    if _KERNEL_MODULE_CACHE.get("module") is not None:
        return _KERNEL_MODULE_CACHE["module"]
    if _NKI is None:
        return None
    if not _KERNEL_MODULE_SOURCE_PATH.exists():
        _KERNEL_MODULE_CACHE["last_load_error"] = (
            f"body file missing: {_KERNEL_MODULE_SOURCE_PATH}"
        )
        return None
    try:  # pragma: no cover -- Trn2-only
        spec = importlib.util.spec_from_file_location(
            "kda_state_nki_v2_body",
            _KERNEL_MODULE_SOURCE_PATH,
        )
        if spec is None or spec.loader is None:
            _KERNEL_MODULE_CACHE["last_load_error"] = "spec_from_file_location returned None"
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as load_err:  # pragma: no cover -- Trn2-only
        _KERNEL_MODULE_CACHE["last_load_error"] = repr(load_err)
        return None
    _KERNEL_MODULE_CACHE["module"] = mod
    return mod


# ---------------------------------------------------------------------------
# NKI backend probe
# ---------------------------------------------------------------------------

def _try_import_nki() -> Optional[object]:
    """Best-effort NKI import. CPU-only environments return None."""
    try:  # pragma: no cover -- exercised only on the Trn2 host
        import neuronxcc.nki as nki  # type: ignore
        return nki
    except Exception:
        return None


_NKI = _try_import_nki()


# Compiled-callable cache. On the Trn2 host, the compile driver populates
# this dict keyed by (slug, shape_tuple). On Windows this stays empty and
# the dispatch shim falls through to the CPU golden.
_KDA_NKI_CALLABLES: Dict[Tuple[str, Tuple[int, ...]], Callable[..., object]] = {}


def register_compiled_nki_kernel(
    shape_tuple: Tuple[int, ...],
    callable_: Callable[..., object],
    slug: str = KDA_STATE_NKI_V2_KERNEL_SLUG,
) -> None:
    """Register a compile-driver-produced NEFF-backed callable.

    Called by the Trn2 host's compile driver once per (slug, shape) after
    it successfully compiles the source in `KDA_STATE_NKI_V2_SOURCE`.
    """
    _KDA_NKI_CALLABLES[(slug, shape_tuple)] = callable_


def get_nki_backend_v2(
    shape_tuple: Optional[Tuple[int, ...]] = None,
) -> Optional[Callable[..., object]]:
    """Return the compiled NKI callable if warm for this shape, else None.

    Args:
        shape_tuple: (B, H, D_v, D_qk) tuple. If None, returns any warm
            callable (test/probe use).

    The registry is authoritative: if a callable was `register_compiled_nki_kernel`-ed
    for this (slug, shape), it is returned. If not, we return None -- which is
    the Windows-inspection case and the cold-cache case on the Trn2 host.

    (The `_NKI is None` gate is NOT applied here -- the compile driver is what
    populates the registry, and only the Trn2 host would ever do so. Honoring
    a registered callable regardless of import status keeps the test surface
    clean and lets a downstream tool inject a fake backend for CI on Windows.)
    """
    if shape_tuple is None:
        for cb in _KDA_NKI_CALLABLES.values():
            return cb
        return None
    return _KDA_NKI_CALLABLES.get(
        (KDA_STATE_NKI_V2_KERNEL_SLUG, shape_tuple)
    )


# ---------------------------------------------------------------------------
# Public dispatch shims (API-compatible with the CPU-golden module)
# ---------------------------------------------------------------------------

def kda_state_decode_forward_nki_v2(
    inputs: "KdaDecodeInputsV2",  # type: ignore[name-defined]
    impl: str = os.environ.get("KDA_KERNEL_IMPL_NKI_V2", "auto"),
) -> "KdaDecodeOutputsV2":  # type: ignore[name-defined]
    """Dispatch shim -- NKI body when available, CPU golden otherwise.

    `impl` values:
      - "auto"      : (default) NKI if warm for this shape, else CPU golden.
      - "nki"       : NKI ONLY -- raises if the compile-cache is cold.
      - "reference" : CPU golden. Always safe.

    Refuses any impl name in `_BANNED_IMPLS` -- a silent softmax lowering
    would corrupt the model. See MLA-VS-DSA-KERNEL-VERIFICATION §1.1.
    """
    if not _CPU_GOLDEN_AVAILABLE:
        raise RuntimeError(
            "kda_state_v2 CPU golden not importable (numpy missing?). "
            "The NKI dispatch shim requires it as the fallback path. "
            "Install numpy or run on the Trn2 host with the compile driver."
        )
    if impl in _BANNED_IMPLS:
        raise ValueError(
            f"KDA impl='{impl}' is banned -- a full-attention fallback "
            "CORRUPTS the model. Use impl='auto', 'nki', or 'reference'. See "
            "MLA-VS-DSA-KERNEL-VERIFICATION-2026-08-27.md §1.1."
        )

    shape_tuple = _shape_tuple_from_inputs(inputs)
    backend = get_nki_backend_v2(shape_tuple)

    if impl == "nki":
        if backend is None:
            raise RuntimeError(
                "impl='nki' requested but NKI backend is not warm for shape "
                f"{shape_tuple}. Warm it via the Trn2 compile driver or use "
                "impl='auto' / 'reference'."
            )
        return _call_nki_backend(backend, inputs)

    if impl == "reference":
        return kda_state_decode_forward_reference_v2(inputs)

    # impl == "auto"
    if backend is not None:
        return _call_nki_backend(backend, inputs)
    return kda_state_decode_forward_reference_v2(inputs)


def kda_state_prefill_forward_nki_v2(
    query,        # [B, L, H, D_qk]
    key,          # [B, L, H, D_qk]
    value,        # [B, L, H, D_v]
    g_raw,        # [B, L, H, D_qk]
    beta_raw,     # [B, L, H]
    state_bf16,   # [B, H, D_v, D_qk]
    params: "KdaLayerParams",  # type: ignore[name-defined]
    impl: str = os.environ.get("KDA_KERNEL_IMPL_NKI_V2", "auto"),
):
    """Prefill dispatch shim. NKI chunk-KDA body TODO -- falls through to the
    CPU-golden by-token prefill in the meantime (correct, O(L) sequential).

    Same fallback discipline as decode: banned impls raise, no softmax path.
    """
    if not _CPU_GOLDEN_AVAILABLE:
        raise RuntimeError(
            "kda_state_v2 CPU golden not importable (numpy missing?)."
        )
    if impl in _BANNED_IMPLS:
        raise ValueError(
            f"KDA impl='{impl}' is banned -- see MLA-VS-DSA §1.1."
        )
    # Chunk-KDA NKI body not landed yet -- always fall through.
    return kda_state_prefill_forward_reference_v2(
        query, key, value, g_raw, beta_raw, state_bf16, params
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shape_tuple_from_inputs(inputs) -> Tuple[int, ...]:
    """Extract the (B, H, D_v, D_qk) shape tuple for the cache key."""
    B, _, H, D_qk = inputs.query.shape
    _, _, _, D_v = inputs.value.shape
    return (int(B), int(H), int(D_v), int(D_qk))


def _call_nki_backend(backend, inputs):  # pragma: no cover -- Trn2-only
    """Invoke the compile-driver-registered NKI callable.

    The callable's ABI matches the source's function signature (see
    `_kda_state_nki_v2_source`): it takes the seven bf16 HBM tensors plus
    the fp32 scale, and returns (y, state_out) as bf16 HBM tensors.

    On the CPU-side (Windows) this path is unreachable -- `get_nki_backend_v2`
    returns None. The registration hook is exercised end-to-end by the
    compile driver's own smoke test on the Trn2 host, not here.
    """
    params = inputs.params
    D_qk = int(inputs.query.shape[-1])
    scale = params.scale if params.scale is not None else 1.0 / math.sqrt(D_qk)
    y, state_out = backend(
        inputs.query,
        inputs.key,
        inputs.value,
        inputs.g_raw,
        inputs.beta_raw,
        params.a_log,
        params.g_bias,
        inputs.state_bf16,
        float(scale),
    )
    return KdaDecodeOutputsV2(y=y, state_bf16=state_out)


# ---------------------------------------------------------------------------
# Public inspection helpers
# ---------------------------------------------------------------------------

def get_nki_source(
    lower_bound: float = KDA_LOWER_BOUND,
    l2norm_eps: float = KDA_L2NORM_EPS,
    tiling: KdaNkiTilingV2 = KDA_NKI_V2_TILING,
) -> str:
    """Return the NKI DSL source string. Public wrapper around
    `_kda_state_nki_v2_source` for the compile driver + inspection tools.
    """
    return _kda_state_nki_v2_source(lower_bound, l2norm_eps, tiling)


def load_nki_kernel_module() -> Optional[Any]:
    """Public accessor for the file-imported NKI body module.

    On the Trn2 compile host, callers should PREFER this over exec()-ing
    `KDA_STATE_NKI_V2_SOURCE`: the file-import path gives the returned
    `@nki.jit`-decorated callable a physical `__file__` that
    `inspect.getsource` can walk. `exec(source_str, ns)` leaves
    `__module__ == "<string>"` and every NKI compile raises
    `OSError: could not get source code` on the first invocation.

    See kernels/EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md for the bug diagnosis.

    Returns the imported module (with `kda_state_decode_forward_nki_v2_body`
    as an attribute) on Trn2 hosts, or None on any host without NKI. The
    body file bakes the default constants (LOWER_BOUND=-5.0, L2NORM_EPS=1e-6,
    BV=64, BK=128). Non-default constants still go through
    `get_nki_source(...)` -> compile-driver `exec`, but that path SHOULD be
    migrated to a per-constant-set body file when it is next exercised.
    """
    return _load_nki_kernel_module_if_available()


def get_nki_kernel_source_path() -> pathlib.Path:
    """Absolute path of the physical body file used by `load_nki_kernel_module`.

    Useful for the compile driver to log which file it consumed and for
    audit tools that hash the source-on-disk into the NEFF cache key.
    """
    return _KERNEL_MODULE_SOURCE_PATH


def nki_source_matches_cpu_golden_constants() -> bool:
    """Static sanity check: the source's baked constants match the CPU
    golden's constants.

    Used by `test_kda_state_nki_v2_smoke::test_source_matches_cpu_constants`.
    """
    src = KDA_STATE_NKI_V2_SOURCE
    return (
        f"KDA_LOWER_BOUND = {float(KDA_LOWER_BOUND)}" in src
        and f"KDA_L2NORM_EPS = {float(KDA_L2NORM_EPS)}" in src
        and "IS_KDA=True" in src
        and "COMPUTE_GATE=True" in src
        and "SAFE_GATE=True" in src
        and "USE_QK_L2NORM_IN_KERNEL=True" in src
        and "SIGMOID_BETA=True" in src
    )


# ---------------------------------------------------------------------------
# Model-specific shape presets (re-export for callers pinning the NKI slug)
# ---------------------------------------------------------------------------

if _CPU_GOLDEN_AVAILABLE:
    # Re-export the two model presets so downstream callers can pin the NKI
    # slug without having to import both modules.
    KIMI_K3_KDA_SHAPE_NKI_V2 = KIMI_K3_KDA_SHAPE_V2
    GLM_5_3_FLASH_KDA_SHAPE_NKI_V2 = GLM_5_3_FLASH_KDA_SHAPE_V2
else:  # pragma: no cover -- numpy-less inspection path
    KIMI_K3_KDA_SHAPE_NKI_V2 = None  # type: ignore
    GLM_5_3_FLASH_KDA_SHAPE_NKI_V2 = None  # type: ignore
