"""NKI v1 device kernel for DMA descriptor coalescing.

Slug     : dma_coalesced_gather.nki_v1
Callsign : dma-coalescing-nki-v1-agent
Date     : 2026-08-27
Sibling  : dma_coalescing_transform.py (v0 - Python planner + K=1 wrapper)
Scaffold : NKI-DMA-COALESCING-SCAFFOLD-2026-08-27.md (design)
Status   : SOURCE-STRING scaffold. The @nki.jit function is materialized only
           when the NKI toolchain is importable in this environment; on a
           non-Trn2 box (this Windows host) the module still imports cleanly
           and `is_available()` returns False. The source string carries the
           full kernel body verbatim so the Trn2 compile host can trace and
           validate it without any additional intermediate step.

WHY V1 EXISTS
-------------
v0 (`dma_coalescing_transform.py`) shipped the Python-side machinery:
    - Path A K=1 passthrough wrapper (byte-identical to the un-wrapped
      `nisa.dma_copy` at the baseline call site);
    - Path B `plan_kv_slab_layout(...)` KV-slab pre-allocation planner that
      gates the existing NKI `k_dma_batch_n_batches` heuristic
      (`harness-v2/staging/cycle630/remote-core.py:2313, :2371, :2421`);
    - Path C descriptor-stream analyzer that reads
      `neuron-profile view --output-format summary-json` outputs.

v1 lands the actual NKI JIT body for Path A K>=2. The address-pattern shape
mirrors the K-way batched indirect DMA that attention_tkg already emits in
its `k_prior_reshaped.ap(...)` call at `remote-core.py:2421`:

    src=bufs.k_prior_reshaped.ap(
        [
            [atp.block_len * cfg.d_head, TC.p_max],   # outer: TC.p_max records
            [1, atp.block_len * cfg.d_head],          # inner: per-record bytes
        ],
        offset=0,
        vector_offset=cur_blks,      # TC.p_max-wide vector of indices
        indirect_dim=0,
    )

v1 generalizes this shape to arbitrary (K, B) with a compile-time-baked K.
Because the emission pattern is EXACTLY what the existing kernel already fires
(same `[[stride, count], [1, stride]]` shape with `vector_offset` +
`indirect_dim=0`), the risk surface is bounded to the (K, B) constant folding
inside `.ap(...)` - not to any new address-pattern primitive.

FIRST-FIRE LANE
---------------
GPT-OSS-20B TP=8 C=128 at K=8, B=650 (per
`lanes/gpt-oss-20b-tp8/PROFILE-C128-KNEE-2026-08-27.md`):
    - K*B = 5.2 KiB, inside the 4-8 KiB efficient descriptor window.
    - Well under the 2 MiB per-call-site SBUF budget (v0 s.2.3).
    - Baseline NEFF + 3-exec profile receipt already banked at
      `/mnt/scratch/tkg-profile-gpt-oss-20b-tp8-b128-20260827T052556Z/` -
      direct A/B does not require a fresh baseline compile.
    - Projected uplift 764 -> 1050-1090 tok/s/card (1.4-2.0x).

CORRECTNESS DISCIPLINE
----------------------
Same 4-gate stack as v0:
    1. Tier-1 CPU battery.
    2. NEFF-content diff (require-different at K>=2; require-identical at K=1).
    3. `verify_splice --tokens 10` bit-identical vs K=1 baseline.
    4. Trn2 profile capture at matched knee.

FALLBACK
--------
When NKI is not importable (this Windows box, most non-Trn2 hosts), the
symbol `dma_coalesced_gather_nki_v1` is defined as a stub that raises
`NotImplementedError` with a clear message pointing callers to v0's CPU-side
planners (`dma_coalescing_transform.plan_coalesce_factor`,
`analyze_descriptor_stream`, `plan_kv_slab_layout`). The source-string
scaffold is always exposed as `DMA_COALESCED_GATHER_NKI_V1_SOURCE` and can be
`exec()`-ed on any Trn2 host where `nki` is importable.

CAMPAIGN CONSTRAINTS RESPECTED
------------------------------
    - No spec-decode (hard operator rule, MEMORY.md).
    - Card 12 never referenced.
    - Container digests: NxDI attention_tkg NKI validation uses
      `sha256:be11c204f419a63e2487b2124005156dad091fb9edbfcadf42d81b745e284c12`.
    - Tier-3 profile-at-knee discipline: this file lands the device kernel;
      profile capture at the new knee is a separate Neuron Explorer call.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
from typing import Any, Optional

# --------------------------------------------------------------------------
# 0.  Module identity + design constants (kept small; deeper constants live
#     in the v0 module, imported lazily where needed to avoid cross-file
#     import-time coupling).
# --------------------------------------------------------------------------

SLUG = "dma_coalesced_gather.nki_v1"
CALLSIGN = "dma-coalescing-nki-v1-agent"

# Efficient descriptor window on Trn2 NeuronCore-V3 HBM DMA engine (bytes).
# 4 KiB / 8 KiB. See NKI-DMA-COALESCING-SCAFFOLD-2026-08-27.md s.1.
EFFICIENT_WINDOW_BYTES_MIN = 4 * 1024
EFFICIENT_WINDOW_BYTES_MAX = 8 * 1024

# Per-invocation SBUF budget headroom for coalesced destinations. Full SBUF
# is 24 MiB per NeuronCore; we hold to 2 MiB per call site to leave the
# fa_tile inner-loop working set intact (scaffold s.2.3).
SBUF_BUDGET_BYTES_PER_CALL_SITE = 2 * 1024 * 1024


# --------------------------------------------------------------------------
# 1.  NKI import guard (env-dependent; failure is not an error).
# --------------------------------------------------------------------------

try:  # pragma: no cover - env-dependent
    import nki                                        # type: ignore[import]
    import nki.isa as nisa                             # type: ignore[import]
    import nki.language as nl                          # type: ignore[import]
    from nki.isa import dge_mode, dma_engine, oob_mode  # type: ignore[import]

    _NKI_AVAILABLE = True
except Exception:                                      # pragma: no cover - env-dependent
    nki = None                                         # type: ignore[assignment]
    nisa = None                                        # type: ignore[assignment]
    nl = None                                          # type: ignore[assignment]
    dge_mode = None                                    # type: ignore[assignment]
    dma_engine = None                                  # type: ignore[assignment]
    oob_mode = None                                    # type: ignore[assignment]
    _NKI_AVAILABLE = False


def is_available() -> bool:
    """True iff the NKI toolchain (`nki`, `nki.isa`, `nki.language`) is importable.

    Callers gate Path A on this. On the non-Trn2 host (this Windows box) this
    returns False; the source-string scaffold below is still accessible.
    """
    return _NKI_AVAILABLE


# --------------------------------------------------------------------------
# 2.  Body module (file-import; EXEC-TO-FILE-IMPORT-PATCH-2026-08-28).
#
#     The actual kernel body now lives at a physical file
#     `_kernel_bodies/dma_coalescing_nki_v1_body.py` and is
#     `importlib.util.spec_from_file_location`-loaded when NKI is present.
#     The file-import path is load-bearing: `@nki.jit`'s
#     `KernelRewriter.reparse_function` calls
#     `inspect.getsource(<decorated_fn>)`, which requires the function to
#     have a physical `__file__` to walk. The prior
#     `exec(DMA_COALESCED_GATHER_NKI_V1_SOURCE, ns)` dispatch left
#     `__module__ == "<string>"` and every NKI compile raised
#     `OSError: could not get source code`. See
#     EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md for the diagnosis discovered
#     on the Kimi K3 Route B fire.
#
#     `DMA_COALESCED_GATHER_NKI_V1_SOURCE` is still exposed as the body
#     file's TEXT (read via `pathlib.Path.read_text`) so consumers and
#     hygiene tests that grep the source keep working unchanged. The
#     constant is no longer `exec()`'d anywhere.
# --------------------------------------------------------------------------

_KERNEL_MODULE_SOURCE_PATH = (
    pathlib.Path(__file__).resolve().parent
    / "_kernel_bodies"
    / "dma_coalescing_nki_v1_body.py"
)
"""Physical path to the NKI body. Load-bearing: file-import of this path
is the ONLY dispatch pattern that keeps `inspect.getsource` happy when
`@nki.jit` re-parses the decorated function."""


def _read_body_source_text() -> str:
    """Read the body file as TEXT for source hygiene tests and consumers
    that grep the source. Never `exec()`'d.
    """
    try:
        return _KERNEL_MODULE_SOURCE_PATH.read_text(encoding="utf-8")
    except OSError as e:
        return (
            f"# ERROR: body file unreadable at {_KERNEL_MODULE_SOURCE_PATH}: {e}\n"
        )


DMA_COALESCED_GATHER_NKI_V1_SOURCE = _read_body_source_text()
"""The body file's TEXT, exposed for source-inspection consumers. Never
`exec()`'d -- the module is loaded via file-import in the block below."""




# --------------------------------------------------------------------------
# 3.  File-import the JIT function when NKI is present; expose a clean stub
#     otherwise.
#
#     Uses `importlib.util.spec_from_file_location` +
#     `spec.loader.exec_module` so the `@nki.jit`-decorated function has a
#     physical `__file__`. The prior `exec(SOURCE, ns)` pattern set
#     `__module__ == "<string>"` and every NKI compile then raised
#     `OSError: could not get source code` inside
#     `KernelRewriter.reparse_function`. See
#     EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md for the diagnosis.
# --------------------------------------------------------------------------

_KERNEL_BODY_MODULE: Optional[Any] = None
_KERNEL_LOAD_ERROR: Optional[str] = None

if _NKI_AVAILABLE:                                    # pragma: no cover - device-side
    if not _KERNEL_MODULE_SOURCE_PATH.exists():
        _KERNEL_LOAD_ERROR = f"body file missing: {_KERNEL_MODULE_SOURCE_PATH}"

        def dma_coalesced_gather_nki_v1(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "dma_coalesced_gather_nki_v1 body file missing at "
                f"{_KERNEL_MODULE_SOURCE_PATH}; cannot file-import the "
                "@nki.jit kernel."
            )
    else:
        try:
            _spec = importlib.util.spec_from_file_location(
                "dma_coalescing_nki_v1_body",
                _KERNEL_MODULE_SOURCE_PATH,
            )
            if _spec is None or _spec.loader is None:
                raise RuntimeError("spec_from_file_location returned None")
            _KERNEL_BODY_MODULE = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_KERNEL_BODY_MODULE)
            dma_coalesced_gather_nki_v1 = _KERNEL_BODY_MODULE.dma_coalesced_gather_nki_v1
        except Exception as _load_err:
            _KERNEL_LOAD_ERROR = repr(_load_err)

            def dma_coalesced_gather_nki_v1(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    "dma_coalesced_gather_nki_v1 body file failed to load "
                    f"from {_KERNEL_MODULE_SOURCE_PATH}: {_KERNEL_LOAD_ERROR}"
                )

else:

    def dma_coalesced_gather_nki_v1(*args: Any, **kwargs: Any) -> None:
        """Stub raised on hosts without an NKI runtime.

        Points callers to v0's CPU-side planners (Path B/C) which run without
        `nki`/`neuronxcc`. The Path A device kernel needs a Trn2 environment
        to fire; validate via the source-string scaffold at
        `DMA_COALESCED_GATHER_NKI_V1_SOURCE` on the compile host.
        """
        raise NotImplementedError(
            "dma_coalesced_gather_nki_v1 requires the NKI toolchain "
            "(container sha256:be11c204... on the Trn2 compile host). "
            "NKI not detected in this Python environment. "
            "For CPU-side coalescing analysis / KV-slab reshape, use "
            "dma_coalescing_transform.plan_coalesce_factor, "
            "analyze_descriptor_stream, or plan_kv_slab_layout. "
            "The full kernel body is exposed as "
            "DMA_COALESCED_GATHER_NKI_V1_SOURCE and can be exec()-ed on "
            "any host where `nki` imports cleanly."
        )


# --------------------------------------------------------------------------
# 4.  First-fire lane manifest - readable by orchestration + validation code.
# --------------------------------------------------------------------------

FIRST_FIRE_LANE: dict[str, Any] = {
    "lane": "gpt-oss-20b-tp8-c128",
    "K": 8,
    "per_transfer_size": 650,
    "coalesced_bytes": 8 * 650,
    "sbuf_budget_bytes": SBUF_BUDGET_BYTES_PER_CALL_SITE,
    "efficient_window_bytes_min": EFFICIENT_WINDOW_BYTES_MIN,
    "projected_multiplier": (1.4, 2.0),
    "current_tokps_per_card": 764.27,
    "projected_tokps_per_card": (1050.0, 1090.0),
    "baseline_receipt_root": (
        "/mnt/scratch/tkg-profile-gpt-oss-20b-tp8-b128-20260827T052556Z/"
    ),
    "profile_receipt_path": (
        "harness-v2/staging/reference-sweep-20260826T2150Z/lanes/"
        "gpt-oss-20b-tp8/PROFILE-C128-KNEE-2026-08-27.md"
    ),
    "container_digest": (
        "sha256:be11c204f419a63e2487b2124005156dad091fb9edbfcadf42d81b745e284c12"
    ),
    "correctness_gate_stack": (
        "Tier-1 CPU battery",
        "NEFF-content diff (require-different)",
        "verify_splice --tokens 10 bit-identical vs K=1 baseline",
        "Trn2 profile capture at matched knee",
    ),
    "success_gate": "dma_active_time_percent reduction >= 30%",
}


__all__ = [
    "SLUG",
    "CALLSIGN",
    "EFFICIENT_WINDOW_BYTES_MIN",
    "EFFICIENT_WINDOW_BYTES_MAX",
    "SBUF_BUDGET_BYTES_PER_CALL_SITE",
    "DMA_COALESCED_GATHER_NKI_V1_SOURCE",
    "FIRST_FIRE_LANE",
    "is_available",
    "dma_coalesced_gather_nki_v1",
]
