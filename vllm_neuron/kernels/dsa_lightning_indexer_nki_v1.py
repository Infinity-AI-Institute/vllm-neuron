# SPDX-License-Identifier: Apache-2.0
"""DSA Lightning Indexer — NKI **v1** device-kernel author's draft.

Callsign: `dsa-nki-v1-agent` (2026-08-27 session).

Sibling of the CPU golden reference at
`kernels/dsa_lightning_indexer.py`. This module ships the *first* device
kernel authoring pass under the `dsa_sparse_attention.nki_v1` slug and
implements the operator-requested API

    dsa_sparse_attention_forward_nki_v1(Q, K, V, index_topk_idx, ...)

so it can plug into the GLM 5.2 correctness lane the moment the Trn2
host has NKI compilation available. Until then the runtime path falls
back to the v0 CPU golden reference so every gate that already gates
against v0 keeps working unchanged when a caller flips the slug.

Ship discipline (operator hard-rule 2026-08-27)
-----------------------------------------------
The v0 file documents the following non-negotiable:

    "Do NOT ship a broken kernel."

The Windows author session that produced this file **has no NKI
runtime** (`import neuronxcc.nki` -> ModuleNotFoundError, verified in
this session). Per the operator's prompt

    "If NKI runtime not accessible in Windows session, ship as
    SOURCE-STRING scaffold ready to compile when Trn2 host has NKI.
    Do NOT ship untested compiled code."

we honour that instruction by:

  1. Feature-detecting NKI at import time (`_NKI_AVAILABLE`). The
     module imports cleanly on any host, with or without
     `neuronxcc.nki`. The smoke test at
     `kernels/tests/test_dsa_lightning_indexer_nki_v1_smoke.py`
     verifies that.
  2. Keeping the **device kernel body as a SOURCE-STRING** in the
     `_NKI_KERNEL_SOURCE` module-level constant. The string is a
     structurally-complete NKI Python DSL draft targeting the
     `neuronxcc.nki` API surface documented in the campaign
     scaffold. It is NOT executed until we are on a Trn2 host with
     NKI, at which point `_compile_nki_kernel_if_available()` calls
     `exec(_NKI_KERNEL_SOURCE, ...)` into a module-scoped
     compilation namespace and pulls the resulting `@nki.jit`
     entrypoint out. This mirrors the SGLang / TileLang pattern of
     shipping kernel source and compiling on first call, which the
     LSE-fix analysis at `kernels/SGLANG-DSA-LSE-FIX-ANALYSIS-
     2026-08-28.md` §2.3 documents in detail.
  3. On any host where NKI is absent OR where the compile
     unexpectedly fails, `dsa_sparse_attention_forward_nki_v1`
     transparently invokes the v0 CPU golden reference
     `dsa_sparse_attention_forward` from
     `kernels/dsa_lightning_indexer.py`. That reference has 32/32
     tests passing per `DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md`
     §3.2, plus the LSE fix per
     `SGLANG-DSA-LSE-FIX-ANALYSIS-2026-08-28.md`, so the fallback
     path is safe to run.
  4. `LSE_BASE_CONVENTION` re-exports v0's `"natural"` constant
     (mirrored, not re-defined) so cross-verification between v0 and
     v1 is bit-check-able out of the box (see §5 of the LSE fix
     analysis for why natural log wins on Trainium2).

The kernel source in this file is an **author's design pass**. It is
structurally complete against the NKI DSL primitives named in the
operator's prompt (`nl.affine_range`, `nl.softmax`, `nisa.nc_matmul`,
`nl.top_k`) and honours the tile plan in
`NKI-DSA-SPARSE-ATTENTION-SCAFFOLD-2026-08-27.md` §4. It has NOT been
compiled or run on a device. Before it lands in production the
following GAPs from scaffold §9 must close:

  * GAP-1: neuron-cc --dump-neff on current 5.2 cache; confirm zero
    hits for `dsa_indexer_kernel` today.
  * GAP-6: prototype two-stage hierarchical top-K for L=1M input.
  * GAP-7: NEFF pattern-match; verify compiler doesn't lower
    gather+softmax back to full attention at S > topk.

Cache identity
--------------
`KERNEL_SLUG_V1_NKI = "dsa_sparse_attention.nki_v1"` — DIFFERENT from
v0's `nki_v0_reference_lightning_indexer` slug. Both slugs
participate in `DsaKernelConfig.cache_key()` (imported from v0), so a
v0 -> v1 swap is a fresh compile-cache line, never a silent replay.
The operator rule top-5 #2 (Gemma-4 lessons harvest: "every lever
names the graph + engine it changes") requires that.

First plug-in target
--------------------
GLM 5.2 correctness unblock at `lanes/glm-5-2-5-3/`, per
`DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md` §2. The lane's
`Glm52MlaAttention.forward` dispatches to
`dsa_lightning_indexer_forward` when
`os.environ.get("DSA_KERNEL_IMPL", ...) == "nki_v0_reference_..."`.
Flipping that env var to `dsa_sparse_attention.nki_v1` picks up this
module. On Windows or any NKI-less host the flip is a no-op (falls
back to v0); on the Trn2 host with NKI present it exercises the
compiled kernel.

Consumers today (once landing)
------------------------------
Same set as v0 (see v0 file docstring):
    1. lanes/glm-5-2-5-3/tests/test_03_dsa_path_activation.py
    2. lanes/deepseek-v4-flash/tests/test_kernel_correctness_dsv4_flash.py
    3. kernels/tests/test_dsa_lightning_indexer_correctness.py

Absolute path: C:\\Users\\apumu\\research\\InfinityAI\\gemma4-trn2-handoff\\harness-v2\\staging\\reference-sweep-20260826T2150Z\\kernels\\dsa_lightning_indexer_nki_v1.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import pathlib
from typing import Any, Optional, Tuple

# ---------------------------------------------------------------------------
# Public identity
# ---------------------------------------------------------------------------

KERNEL_SLUG_V1_NKI = "dsa_sparse_attention.nki_v1"
"""Cache identity for the first NKI device kernel authoring pass.

DIFFERENT from v0's `nki_v0_reference_lightning_indexer`. A swap between
v0 and v1 is a fresh compile-cache line — the slug participates in
`DsaKernelConfig.cache_key()` so `_GLM52_GRAPH_ID` (and its GLM 5.3
Flash / DSV4-Flash siblings) sees the change.
"""

LSE_BASE_CONVENTION = "natural"
"""LSE base convention mirrored from v0. See
`kernels/SGLANG-DSA-LSE-FIX-ANALYSIS-2026-08-28.md` §5 for rationale on
Trainium2 (no native base-2 intrinsic; `nl.exp`/`nl.log` lower cleanly)."""


# ---------------------------------------------------------------------------
# NKI runtime feature detection
# ---------------------------------------------------------------------------
#
# The Windows author session has no NKI runtime; a Trn2 host does. We
# import NKI lazily and record availability once. Do NOT raise at import
# time on any host, or the smoke test breaks.

_NKI_AVAILABLE = False
_NKI_IMPORT_ERROR: Optional[BaseException] = None
try:  # pragma: no cover - device-only path
    import neuronxcc.nki as _nki  # type: ignore
    import neuronxcc.nki.language as _nl  # type: ignore
    import neuronxcc.nki.isa as _nisa  # type: ignore

    _NKI_AVAILABLE = True
except Exception as _e:  # ImportError on Windows, some others on device
    _NKI_IMPORT_ERROR = _e
    _nki = None  # type: ignore
    _nl = None  # type: ignore
    _nisa = None  # type: ignore


def nki_runtime_available() -> bool:
    """True iff `neuronxcc.nki` (and .language, .isa) imported cleanly.

    Public helper for consumers and tests. On Windows this is always False;
    on Trn2 with the campaign container it is True.
    """
    return _NKI_AVAILABLE


# ---------------------------------------------------------------------------
# NKI kernel body (file-based import; EXEC-TO-FILE-IMPORT-PATCH-2026-08-28)
# ---------------------------------------------------------------------------
#
# The device kernel now lives as a physical Python file at
# `_kernel_bodies/dsa_lightning_indexer_nki_v1_body.py` and is
# `importlib.util.spec_from_file_location`-loaded when NKI is importable.
# The file-import path is load-bearing: `@nki.jit`'s
# `KernelRewriter.reparse_function` calls
# `inspect.getsource(<decorated_fn>)`, which requires the function to have a
# physical `__file__` to walk. The prior `exec(_NKI_KERNEL_SOURCE, ns)`
# dispatch left `__module__ == "<string>"` and every NKI compile raised
# `OSError: could not get source code`. See
# EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md for the full bug diagnosis
# discovered on the Kimi K3 Route B fire.
#
# The `_NKI_KERNEL_SOURCE` module-level constant is still exposed as the
# TEXT of the body file (read via `pathlib.Path.read_text`) so consumers
# and hygiene tests that grep the source keep working unchanged. The
# constant is no longer `exec()`'d anywhere -- the imported module is the
# single source of truth for the JIT-decorated entrypoint.
#
# Structural correspondence with the CPU reference:
#   * gather_selected_kv   <->  sparse_gather_kv        (v0 fn)
#   * fused score/softmax  <->  dsa_sparse_attention_forward (v0 fn)
#   * online-softmax /
#     LSE accumulator      <->  the natural-log LSE contract in v0
#                                +  fixup_zero_kv_rows sentinel
#                                (see SGLANG-DSA-LSE-FIX-ANALYSIS §4)
#
# Tile plan (scaffold §4):
#   * Q tile:      q_tile        = 16 queries
#   * K tile:      block_size    = 32 selected positions per DMA block
#   * accumulate over `topk // block_size` blocks per Q tile
#   * SBUF residency budgeted against `analytical_bounds(...)` in v0
#
# See `NKI-DSA-SPARSE-ATTENTION-SCAFFOLD-2026-08-27.md` §3.2 for the
# canonical API surface of this kernel.

_KERNEL_MODULE_SOURCE_PATH = (
    pathlib.Path(__file__).resolve().parent
    / "_kernel_bodies"
    / "dsa_lightning_indexer_nki_v1_body.py"
)
"""Physical path to the NKI body. Load-bearing: file-import of this path is
the ONLY dispatch pattern that keeps `inspect.getsource` happy when
`@nki.jit` re-parses the decorated function."""


def _read_body_source_text() -> str:
    """Read the body file as TEXT for source hygiene tests and consumers
    that grep the source (SGLang / TileLang / audit tools). Never exec()'d.

    Returns an empty string with a diagnostic prefix if the file is missing;
    hygiene tests will then fail loudly rather than pass on stale content.
    """
    try:
        return _KERNEL_MODULE_SOURCE_PATH.read_text(encoding="utf-8")
    except OSError as e:
        return (
            f"# ERROR: body file unreadable at {_KERNEL_MODULE_SOURCE_PATH}: {e}\n"
        )


_NKI_KERNEL_SOURCE = _read_body_source_text()
"""The body file's TEXT, exposed for source-inspection consumers. Never
`exec()`'d -- the module is loaded via `_compile_nki_kernel_if_available()`
using `importlib.util.spec_from_file_location`."""


# Retained for reference: the entrypoint signature (as it appears in the
# body file) is
#
#     @nki.jit(mode=_NKI_JIT_MODE)
#     def _dsa_sparse_attention_nki_v1_impl(
#         Q, K, V, index_topk_idx, q_pos, k_len, Out, Lse,
#         *,
#         scaling: float,
#         causal: bool = True,
#         topk_const: int = 2048,
#         q_tile: int = 16,
#         block_size: int = 32,
#     ):

# The imported body module lives here so the compiled entrypoint is
# reachable from `dsa_sparse_attention_forward_nki_v1`.
_NKI_KERNEL_NAMESPACE: dict = {}


def _compile_nki_kernel_if_available() -> Optional[Any]:
    """File-import the body module and return its JIT entrypoint, or None.

    Returns the `@nki.jit`-decorated `_dsa_sparse_attention_nki_v1_impl` on
    success; returns None on any host without NKI or on load failure. The
    module load is idempotent (`_NKI_KERNEL_NAMESPACE` caches the result).

    Uses `importlib.util.spec_from_file_location` + `spec.loader.exec_module`
    so the decorated function has a physical `__file__`. The prior
    `exec(_NKI_KERNEL_SOURCE, ns)` pattern set `__module__ == "<string>"`
    and every NKI compile then raised `OSError: could not get source code`
    inside `KernelRewriter.reparse_function` (which calls
    `inspect.getsource(<decorated_fn>)`). See
    kernels/EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md for the diagnosis
    from the Kimi K3 Route B fire.
    """
    if not _NKI_AVAILABLE:
        return None
    if "_dsa_sparse_attention_nki_v1_impl" in _NKI_KERNEL_NAMESPACE:
        return _NKI_KERNEL_NAMESPACE["_dsa_sparse_attention_nki_v1_impl"]

    if not _KERNEL_MODULE_SOURCE_PATH.exists():
        _NKI_KERNEL_NAMESPACE["last_compile_error"] = (
            f"body file missing: {_KERNEL_MODULE_SOURCE_PATH}"
        )
        return None

    try:  # pragma: no cover - device-only path
        spec = importlib.util.spec_from_file_location(
            "dsa_lightning_indexer_nki_v1_body",
            _KERNEL_MODULE_SOURCE_PATH,
        )
        if spec is None or spec.loader is None:
            _NKI_KERNEL_NAMESPACE["last_compile_error"] = (
                "spec_from_file_location returned None"
            )
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as compile_err:
        # Keep the module importable and the fallback path healthy; the
        # smoke test surfaces the compile failure via `last_compile_error`
        # so a device-side agent sees it without crashing consumers.
        _NKI_KERNEL_NAMESPACE["last_compile_error"] = repr(compile_err)
        return None

    entrypoint = getattr(mod, "_dsa_sparse_attention_nki_v1_impl", None)
    _NKI_KERNEL_NAMESPACE["_dsa_sparse_attention_nki_v1_impl"] = entrypoint
    _NKI_KERNEL_NAMESPACE["_nki_gather_kv_block"] = getattr(
        mod, "_nki_gather_kv_block", None
    )
    _NKI_KERNEL_NAMESPACE["module"] = mod
    return entrypoint


# ---------------------------------------------------------------------------
# CPU fallback plumbing (imports v0 lazily so this module imports on a
# host that has neither NKI nor torch).
# ---------------------------------------------------------------------------


def _v0_forward(*args, **kwargs):
    """Lazy import + dispatch to v0 CPU golden reference."""
    _here = pathlib.Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    # Local import: torch is only needed on the fallback path, not at
    # module-import time.
    from dsa_lightning_indexer import dsa_sparse_attention_forward  # noqa: E402
    return dsa_sparse_attention_forward(*args, **kwargs)


# ---------------------------------------------------------------------------
# Public v1 entrypoint — API-parallel to v0's dsa_sparse_attention_forward
# ---------------------------------------------------------------------------


def dsa_sparse_attention_forward_nki_v1(
    Q,                           # [B, S_q, H, D]     bf16
    K,                           # [B, S_kv, H, D]    bf16
    V,                           # [B, S_kv, H, D]    bf16
    index_topk_idx,              # [B, S_q, topk]     int32
    position_ids=None,           # [B, S_q]           int64  (fallback path req'd)
    key_lengths=None,            # [B]                int64  (fallback path req'd)
    *,
    topk: Optional[int] = None,
    scaling: Optional[float] = None,
    causal: bool = True,
    return_lse: bool = False,
    block_size: int = 32,
    q_tile: int = 16,
    force_fallback: bool = False,
):
    """v1 entrypoint — matches v0's `dsa_sparse_attention_forward` API.

    Positional arg order tracks the operator's request:
        (Q, K, V, index_topk_idx, ...)

    The extra kwargs (`position_ids`, `key_lengths`, `topk`, `scaling`,
    `causal`, `return_lse`) mirror v0 so a caller can flip the
    implementation via the `DSA_KERNEL_IMPL` env var without touching
    the call site.

    Dispatch order:
      1. If `force_fallback=True` OR NKI is not importable OR the
         source-string compile failed on a prior call: dispatch to v0.
         Returns exactly what v0 returns (`(out, lse)` if
         `return_lse=True`, else `out`).
      2. Otherwise: invoke the compiled `_dsa_sparse_attention_nki_v1_impl`
         with the tile constants and return device tensors.

    Consumers wire this into `Glm52MlaAttention.forward` gated by

        os.environ.get("DSA_KERNEL_IMPL", "nki_v0_reference_lightning_indexer")
            == KERNEL_SLUG_V1_NKI

    so flipping the slug on Trn2 picks up this kernel; on Windows the
    same flip is a safe no-op (falls through to v0).
    """
    # ---- Argument sanity (independent of NKI availability) --------------
    if topk is None:
        # Recover from the index tensor shape when not passed.
        topk = int(index_topk_idx.shape[-1])
    if scaling is None:
        # Recover from Q shape.
        D = int(Q.shape[-1])
        # Avoid importing math at module top for the fallback path.
        import math as _math
        scaling = 1.0 / _math.sqrt(D)

    # ---- Dispatch -------------------------------------------------------
    if force_fallback or not _NKI_AVAILABLE:
        return _v0_forward(
            Q, K, V,
            index_topk_idx,
            position_ids,
            key_lengths,
            topk=topk,
            scaling=scaling,
            causal=causal,
            return_lse=return_lse,
        )

    entrypoint = _compile_nki_kernel_if_available()
    if entrypoint is None:  # pragma: no cover - device-only path
        return _v0_forward(
            Q, K, V,
            index_topk_idx,
            position_ids,
            key_lengths,
            topk=topk,
            scaling=scaling,
            causal=causal,
            return_lse=return_lse,
        )

    # pragma: no cover -- device-only path from here on. -----------------
    # Out and LSE are out-parameters in the NKI kernel body; allocate on
    # the caller side and pass in. Shapes match v0's return tensors.
    B = int(Q.shape[0])
    S_q = int(Q.shape[1])
    H = int(Q.shape[2])
    D = int(Q.shape[3])

    Out = _nl.ndarray((B, S_q, H, D), dtype=Q.dtype)  # type: ignore[union-attr]
    Lse = _nl.ndarray((B, S_q, H), dtype=_nl.float32)  # type: ignore[union-attr]

    entrypoint(
        Q, K, V,
        index_topk_idx,
        position_ids,
        key_lengths,
        Out, Lse,
        scaling=scaling,
        causal=causal,
        topk_const=topk,
        q_tile=q_tile,
        block_size=block_size,
    )

    if return_lse:
        return Out, Lse
    return Out


# ---------------------------------------------------------------------------
# Compile-cache identity helper
# ---------------------------------------------------------------------------


def build_v1_cache_key(
    *,
    topk: int,
    block_size: int = 32,
    index_n_heads: int,
    index_head_dim: int,
    index_pool: int = 1,
    causal: bool = True,
    return_topk_for_indexshare: bool = False,
    return_lse: bool = False,
) -> str:
    """Assemble the v1 compile-cache subkey to append to model graph_id.

    Wraps `DsaKernelConfig` from v0 with `impl=KERNEL_SLUG_V1_NKI` and
    layers on `return_lse` — v1's `return_lse=True` variant should get
    its own NEFF (see LSE fix analysis §7 action item #1).
    """
    _here = pathlib.Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from dsa_lightning_indexer import DsaKernelConfig  # local import

    cfg = DsaKernelConfig(
        topk=topk,
        block_size=block_size,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_pool=index_pool,
        causal=causal,
        return_topk_for_indexshare=return_topk_for_indexshare,
        impl=KERNEL_SLUG_V1_NKI,
    )
    base = cfg.cache_key()
    return f"{base}|return_lse={int(return_lse)}"


__all__ = [
    "KERNEL_SLUG_V1_NKI",
    "LSE_BASE_CONVENTION",
    "build_v1_cache_key",
    "dsa_sparse_attention_forward_nki_v1",
    "nki_runtime_available",
]
