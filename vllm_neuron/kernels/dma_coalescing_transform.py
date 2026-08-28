"""DMA descriptor coalescing transform for Trainium2 NKI attention kernels.

Author  : Fleet A worker agent, Trainium2 campaign
Date    : 2026-08-27
Status  : NKI wrapper + Python-level KV-slab pre-allocation reshape helper. NOT yet
          compilable end-to-end - requires NKI toolchain (`neuronxcc`, container
          `sha256:be11c204f419a63e2487b2124005156dad091fb9edbfcadf42d81b745e284c12`)
          to validate the multi-source strided-descriptor emission path.
Sources : NKI-DMA-COALESCING-SCAFFOLD-2026-08-27.md (design)
          harness-v2/staging/cycle630/remote-core.py:2313-2434 (existing `k_dma_batch_n_batches`
              path that already does K-way batching under `use_dma_transpose`; this module
              generalizes it to all `nisa.dma_copy` call sites)
          PROFILE-AT-KNEE-SUMMARY-2026-08-27.md (three Tier-3 profiles)

WHY THIS FILE EXISTS
--------------------
Universal Fleet A finding across three Tier-3 profiles (GPT-OSS-20B TP8 C=128,
GPT-OSS-20B TP4 C=4, Qwen3-32B TP8 C=16): **73-80% wall-share is spent in DMA
active time on HW-dynamic packets of 94-967 B** - 4-85x below the ~4-8 KB
efficient window of the NeuronCore-V3 HBM DMA engine. Descriptor-issue overhead
dominates; coalescing K adjacent per-fold / per-batch loads into one wider
descriptor moves the workload from descriptor-issue-bound to bandwidth-bound.

Projected uplifts (per `PROFILE-AT-KNEE-SUMMARY-2026-08-27.md`):
    GPT-OSS-20B TP8 C=128 : 764 -> 1050-1090 tok/s/card   (1.37-1.43x)
    GPT-OSS-20B TP4 C=4   : 135 -> 150-165  tok/s/card   (1.15-1.30x)
    Qwen3-32B  TP8 C=16   : 145 -> 260-350  tok/s/card   (2.40-3.00x) <- largest headroom

WHAT THIS FILE PROVIDES
-----------------------
Two intervention paths, both callable from NxDI's attention_tkg shim:

    (A) NKI-level wrapper: `dma_coalesced_gather(...)` - a K-way coalescing wrapper
        around `nisa.dma_copy` that, at K=1, is a bit-identical no-op passthrough
        (NEFF-diff clean per GEMMA4-LESSONS-GENERALIZED-2026-08-27.md A6), and at
        K>=2 folds K adjacent gather transfers into `ceil(N/K)` descriptors of
        K*B bytes each. `oob_mode.skip` semantics preserved so KV-cache -1
        sentinels still short-circuit the destination write.

    (B) Python-level pre-allocation reshape: `plan_kv_slab_layout(...)` +
        `apply_kv_slab_layout(...)` - a NxDI-level helper that pre-sizes KV
        blocks to a K-multiple of the natural block_size on 4-KiB boundaries.
        This makes the existing attention_tkg `k_dma_batch_n_batches` path
        (remote-core.py line 2313) fire on ALL knees, not only when the
        heuristic randomly picks a batching factor >1. Runs BEFORE compile,
        no NKI change required. Intermediate unblock for lanes where (A) is
        blocked on the NKI toolchain gap.

    (C) Descriptor-stream analyzer: `analyze_descriptor_stream(...)` - reads a
        `neuron-profile view --output-format summary-json` output and reports
        per-site coalescing headroom. This is the CPU-side battery check that
        certifies the mechanism is present in the compiled artifact before
        spending device time.

FIRST-FIRE LANE
---------------
GPT-OSS-20B TP8 C=128 (banked 764.27 tok/s/card, ranked knee at
`lanes/gpt-oss-20b-tp8/PROFILE-C128-KNEE-2026-08-27.md`). Reason:
    - 650 B HW-dyn packets, 5.9 M packet count -> K=8 target of 5.2 KB is a
      moderate coalesce well inside SBUF budget (2 MiB per call site).
    - Baseline NEFF and 3-exec profile receipt exist for direct A/B diff.
    - 1.4-2x projected uplift is the highest absolute tok/s move (764 -> ~1090)
      of any measured knee.

Qwen3-32B TP8 C=16 is the LARGEST relative headroom (2.4-3x) but K=45 pushes
SBUF budget hard and its 94 B/packet is deep enough that Path B (padding) vs
Path B.1 (strided) selection needs empirical `neuron-profile view` per-descriptor
breakdown that Fleet A hasn't yet captured. Fire second.

CORRECTNESS DISCIPLINE (non-negotiable)
---------------------------------------
Every A/B fire from this module MUST pass the gate stack in this order:

    1. Tier-1 CPU battery (partition-cap + HBM + slug determinism).
    2. NEFF-content byte-diff (per GEMMA4-LESSONS A6): if TKG program bytes
       byte-identical to K=1 baseline, the knob did NOT land - ABORT before
       device time. `run_neff_content_check(...)` runs this.
    3. `verify_splice --tokens 10` bit-identical vs K=1 baseline
       (GEMMA4-LESSONS D3). K=1 self-insert = 10/10 PASS; K>=2 candidate =
       10/10 PASS or blocked lane.
    4. Only then: Trn2 profile capture at the matched knee.

CAMPAIGN CONSTRAINTS RESPECTED
------------------------------
    - No spec-decode (hard operator rule, MEMORY.md).
    - Card 12 never referenced.
    - Container digests: NxDI compile uses `sha256:011d49c7...` (MoE workaround
      applied); attention_tkg NKI validation uses `sha256:be11c204...`.
    - Tier-3 profile-at-knee discipline: this module lands the lever; the
      profile at the new knee is a separate call to Neuron Explorer.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
from typing import Any, Iterable, Optional

# --------------------------------------------------------------------------
# 0.  Module constants (from the scaffold, cross-referenced to receipts)
# --------------------------------------------------------------------------

# Efficient descriptor window on Trn2 NeuronCore-V3 HBM DMA engine.
# 4 KiB lower / 8 KiB upper. See PROFILE-AT-KNEE-SUMMARY-2026-08-27.md and the
# scaffold s.1.2 for the derivation.
EFFICIENT_WINDOW_BYTES_MIN = 4 * 1024
EFFICIENT_WINDOW_BYTES_MAX = 8 * 1024

# Per-invocation SBUF budget headroom for coalesced destinations (NKI-scaffold s.2.3).
# Full SBUF is 24 MiB per NeuronCore; we hold to 2 MiB per call site to leave the
# fa_tile inner-loop working set intact.
SBUF_BUDGET_BYTES_PER_CALL_SITE = 2 * 1024 * 1024

# HBM byte-bandwidth on NeuronCore-V3 (order-of-magnitude; used only for the
# break-even K planner - the direction of the lever is invariant to this constant).
HBM_BW_BYTES_PER_NS = 2900.0

# Descriptor-issue latency assumption (50-100 ns industry-typical). GAP #1 in
# the scaffold. Used only for break-even K math; not part of correctness.
DESCRIPTOR_ISSUE_LATENCY_NS = 75.0

# NEFF header length (bytes) before the compressed tar payload. Comes from
# GEMMA4-LESSONS A6 - the byte-diff check strips this prefix before diffing
# TKG program bytes.
NEFF_HEADER_BYTES = 1024


# --------------------------------------------------------------------------
# 1.  Descriptor-stream analysis (input: profile summary-json; no NKI needed)
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DescriptorSiteReport:
    """Per-site descriptor summary derived from `neuron-profile view` output."""

    site_name: str
    packet_count: int
    mean_bytes_per_packet: float
    total_bytes: int
    descriptor_class: str          # "hw_dyn", "sw_dyn", "static"
    coalesce_headroom_k: int       # K_target = EFFICIENT_WINDOW / mean_bytes
    projected_dma_active_reduction_pct: float
    fires_first: bool

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def plan_coalesce_factor(
    mean_bytes_per_packet: float,
    *,
    target_window_bytes: int = EFFICIENT_WINDOW_BYTES_MIN,
    sbuf_budget_bytes: int = SBUF_BUDGET_BYTES_PER_CALL_SITE,
) -> int:
    """Pick the smallest K that lifts the coalesced packet size to the efficient window.

    Break-even K is math.ceil(t_d * bw / B) from the scaffold s.1.3, but bounded by
    the SBUF budget so any K*B stays under `sbuf_budget_bytes` per call site.

    Returns K = 1 when the raw packet already sits inside the efficient window
    (no coalescing needed).
    """
    if mean_bytes_per_packet <= 0:
        return 1
    if mean_bytes_per_packet >= target_window_bytes:
        return 1
    k_by_window = int(math.ceil(target_window_bytes / mean_bytes_per_packet))
    # Cap by SBUF budget so K*B fits per call site.
    k_by_sbuf = max(1, int(sbuf_budget_bytes // max(1.0, mean_bytes_per_packet)))
    return max(1, min(k_by_window, k_by_sbuf))


def _projected_dma_active_reduction(
    current_bytes_per_packet: float, coalesce_k: int
) -> float:
    """Fraction of DMA-active wall-share removed by coalescing to K.

    Derived from T_dma = N * (t_d + B/bw) vs T_coalesced = (N/K)*(t_d + K*B/bw).
    Reduction ratio = 1 - T_coalesced / T_dma.
    """
    if coalesce_k <= 1:
        return 0.0
    t_d = DESCRIPTOR_ISSUE_LATENCY_NS
    b_over_bw = current_bytes_per_packet / HBM_BW_BYTES_PER_NS
    per_packet_baseline = t_d + b_over_bw
    per_packet_coalesced = (t_d + coalesce_k * b_over_bw) / coalesce_k
    return max(0.0, min(1.0, 1.0 - (per_packet_coalesced / per_packet_baseline)))


def analyze_descriptor_stream(
    summary_json_paths: Iterable[str | os.PathLike],
    *,
    fires_first_predicate: Optional[callable] = None,
) -> list[DescriptorSiteReport]:
    """Read per-shard `neuron-profile view --output-format summary-json` outputs,
    return a coalescing plan per hardware-descriptor class.

    Contract (matches the JSON schema Fleet A has been consuming since 2026-08-27):
        {
          "dma_hw_dyn": {"packet_count": ..., "mean_bytes_per_packet": ...},
          "dma_sw_dyn": {...},
          "dma_static": {...},
        }

    Any key that isn't present is skipped without raising - the analyzer treats
    missing classes as zero-headroom rather than as an error.
    """
    reports_by_site: dict[str, DescriptorSiteReport] = {}
    for path in summary_json_paths:
        with open(path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
        for cls_key, cls_label in (
            ("dma_hw_dyn", "hw_dyn"),
            ("dma_sw_dyn", "sw_dyn"),
            ("dma_static", "static"),
        ):
            entry = summary.get(cls_key)
            if not entry:
                continue
            n = int(entry.get("packet_count", 0))
            b = float(entry.get("mean_bytes_per_packet", 0.0))
            if n <= 0 or b <= 0:
                continue
            k = plan_coalesce_factor(b)
            projected = _projected_dma_active_reduction(b, k)
            site_name = f"{path}::{cls_key}"
            reports_by_site[site_name] = DescriptorSiteReport(
                site_name=site_name,
                packet_count=n,
                mean_bytes_per_packet=b,
                total_bytes=int(n * b),
                descriptor_class=cls_label,
                coalesce_headroom_k=k,
                projected_dma_active_reduction_pct=100.0 * projected,
                fires_first=False,
            )

    # Order sites by projected reduction (largest wins first). Optionally decorate
    # with a caller-provided "fires_first" predicate (e.g. name match against the
    # target lane's known knee).
    ranked = sorted(
        reports_by_site.values(),
        key=lambda r: r.projected_dma_active_reduction_pct,
        reverse=True,
    )
    if fires_first_predicate is not None and ranked:
        picked = None
        for r in ranked:
            if fires_first_predicate(r):
                picked = r
                break
        if picked is not None:
            # Replace with a fires_first=True copy.
            ranked = [
                dataclasses.replace(r, fires_first=(r is picked)) for r in ranked
            ]
    elif ranked:
        ranked = [
            dataclasses.replace(ranked[0], fires_first=True)
        ] + list(ranked[1:])
    return ranked


# --------------------------------------------------------------------------
# 2.  Path (B): Python-level KV-slab pre-allocation reshape (no NKI needed)
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class KvSlabLayoutPlan:
    """The output of `plan_kv_slab_layout` - a plan for NxDI's KV-cache allocator.

    The existing NKI attention_tkg (remote-core.py line 2313) already batches K
    folds/batches into one dma_transpose descriptor when both `k_dma_batch_n_folds`
    and `k_dma_batch_n_batches` are > 1. That path is heuristically selected today.
    This plan gates it deterministically by sizing KV blocks so K-way batching
    fires at every fa_tile.

    All quantities are compile-time constants that flow through `NeuronConfig`
    into the kernel selector.
    """

    coalesce_factor_k: int
    kv_block_size_source: int      # current block_size (from NeuronConfig.block_size)
    kv_block_size_target: int      # coalesce_factor_k * block_size, aligned to 4 KiB
    aligned_stride_bytes: int
    padding_bytes_per_shard: int   # cost of the layout in wasted HBM per shard
    fits_hbm: bool                 # False if padding pushes past HBM budget
    reason: str                    # human-readable derivation
    fires_on_kernels: tuple[str, ...] = (
        "attention_tkg.k_prior_block_load",         # remote-core.py:2421
        "attention_tkg.dma_transpose",              # remote-core.py:2381
        "attention_tkg.mask_load",                  # remote-core.py:2046, :2070
        "attention_tkg.pos_ids_load",               # remote-core.py:1979
        "attention_tkg.inv_freqs_load",             # remote-core.py:2161
    )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def plan_kv_slab_layout(
    *,
    tokens_per_block: int,
    d_head: int,
    dtype_bytes: int,
    hbm_budget_bytes_per_shard: int,
    total_blocks_per_shard: int,
    current_bytes_per_packet: float,
    target_bytes_per_packet: int = EFFICIENT_WINDOW_BYTES_MIN,
) -> KvSlabLayoutPlan:
    """Compute the KV-slab layout that gates the existing NKI k_dma_batch path.

    Approach (Path B intermediate unblock):
        - Choose K = ceil(target_bytes_per_packet / current_bytes_per_packet).
        - Set the KV block size to `K * tokens_per_block` (this is a NeuronConfig
          knob today).
        - Align the aggregate slab stride to 4 KiB (Trn2 DMA engine efficient
          window; the scaffold s.1 derives this).
        - Cost: per-block padding = align_up(K * tokens_per_block * d_head *
          dtype_bytes, 4 KiB) - K * tokens_per_block * d_head * dtype_bytes.
          Multiplied by total_blocks_per_shard.
        - `fits_hbm` is False if the padded slab overflows the shard HBM budget;
          the caller must either shrink K or fall back to Path A (NKI wrapper).

    This planner performs NO device-side work. It's the CPU-battery input to
    Tier-1 gates before compile submission.
    """
    if current_bytes_per_packet <= 0:
        raise ValueError("current_bytes_per_packet must be positive")
    k = max(1, math.ceil(target_bytes_per_packet / current_bytes_per_packet))

    base_slab_bytes = k * tokens_per_block * d_head * dtype_bytes
    aligned_slab_bytes = ((base_slab_bytes + EFFICIENT_WINDOW_BYTES_MIN - 1)
                         // EFFICIENT_WINDOW_BYTES_MIN) * EFFICIENT_WINDOW_BYTES_MIN
    padding_per_block = aligned_slab_bytes - base_slab_bytes
    padding_total = padding_per_block * total_blocks_per_shard

    total_padded_bytes = aligned_slab_bytes * total_blocks_per_shard
    fits = total_padded_bytes <= hbm_budget_bytes_per_shard

    reason_lines = [
        f"K = ceil({target_bytes_per_packet} / {current_bytes_per_packet:.2f}) = {k}",
        f"per-block payload = K*tokens_per_block*d_head*dtype_bytes = "
        f"{k}*{tokens_per_block}*{d_head}*{dtype_bytes} = {base_slab_bytes} B",
        f"per-block aligned = align_up({base_slab_bytes}, "
        f"{EFFICIENT_WINDOW_BYTES_MIN}) = {aligned_slab_bytes} B",
        f"per-block padding = {padding_per_block} B",
        f"total padded slab = {total_padded_bytes} B "
        f"(HBM budget {hbm_budget_bytes_per_shard} B)"
        + (" - FITS" if fits else " - OVERFLOWS, fall back to Path A"),
    ]

    return KvSlabLayoutPlan(
        coalesce_factor_k=k,
        kv_block_size_source=tokens_per_block,
        kv_block_size_target=k * tokens_per_block,
        aligned_stride_bytes=aligned_slab_bytes,
        padding_bytes_per_shard=padding_total,
        fits_hbm=fits,
        reason="\n  ".join(reason_lines),
    )


def apply_kv_slab_layout(neuron_config: Any, plan: KvSlabLayoutPlan) -> None:
    """Mutate NxDI's `NeuronConfig` to enact the plan (before compile submission).

    Contract (matches NxDI's current NeuronConfig surface, per
    `harness-v2/staging/cycle410-derep-g4-static-repair/modeling_gemma4.py`
    and MEMORY.md `NxDI container MoE blockwise-mm workaround`):

        neuron_config.block_size <- plan.kv_block_size_target
        neuron_config.dma_coalesce_factor <- plan.coalesce_factor_k
        neuron_config.use_shard_on_intermediate_dynamic_while <- True  (MoE fix)

    Kept as a duck-typed setter so it works against both the vLLM-Neuron and
    the direct NxDI config surface without importing either at module load time.
    """
    if not plan.fits_hbm:
        raise RuntimeError(
            "KV slab layout plan overflows HBM budget - refusing to apply. "
            "Fall back to Path A (NKI wrapper) or shrink K.\n"
            f"Plan reason:\n  {plan.reason}"
        )
    setattr(neuron_config, "block_size", plan.kv_block_size_target)
    setattr(neuron_config, "dma_coalesce_factor", plan.coalesce_factor_k)
    # Preserve the MoE blockwise workaround from MEMORY.md.
    if getattr(neuron_config, "use_shard_on_intermediate_dynamic_while", None) is None:
        setattr(neuron_config, "use_shard_on_intermediate_dynamic_while", True)


# --------------------------------------------------------------------------
# 3.  Path (A): NKI wrapper `dma_coalesced_gather`
# --------------------------------------------------------------------------
#
# This section is IMPORTED ONLY WHEN THE NKI TOOLCHAIN IS PRESENT. It lands
# inside the NxDI `HloTorchCompatibleAttentionBlockTkgKernel` shim. Without
# `nki`/`neuronxcc` installed (this Windows box, most compile hosts) the module
# still imports and Paths B and C stay usable.
#
# The API surface matches the existing `nisa.dma_copy(dst, src, ...)` calls in
# harness-v2/staging/cycle630/remote-core.py at lines 2046, 2070, 2179, 2421.
# --------------------------------------------------------------------------

try:  # pragma: no cover - NKI toolchain gate
    import nki                                   # type: ignore[import]
    import nki.isa as nisa                        # type: ignore[import]
    import nki.language as nl                     # type: ignore[import]
    from nki.isa import dma_engine, oob_mode      # type: ignore[import]

    _NKI_AVAILABLE = True
except Exception:                                 # pragma: no cover - env-dependent
    nki = None                                    # type: ignore[assignment]
    nisa = None                                   # type: ignore[assignment]
    nl = None                                     # type: ignore[assignment]
    dma_engine = None                             # type: ignore[assignment]
    oob_mode = None                               # type: ignore[assignment]
    _NKI_AVAILABLE = False


def nki_available() -> bool:
    """True iff the NKI toolchain (`nki`) imported successfully.

    Callers use this as the Tier-1 gate before enabling Path A. On the compile
    host without NKI (rare), fall back to Path B pre-allocation reshape.
    """
    return _NKI_AVAILABLE


if _NKI_AVAILABLE:                                # pragma: no cover - device-side

    @nki.jit  # type: ignore[misc]
    def dma_coalesced_gather(
        src_hbm,
        dst_sbuf,
        indices,
        coalesce_factor: int,
        per_transfer_size: int,
        num_transfers: int,
        engine=None,
        oob_behavior=None,
        name: str = "dma_coalesced_gather",
    ) -> None:
        """K-way coalescing wrapper around `nisa.dma_copy` for HBM->SBUF gather.

        Contract (guaranteed byte-identical when K=1):
            - coalesce_factor == 1: passthrough; emits N per-packet `nisa.dma_copy`
              calls with the same address-pattern shape the existing attention_tkg
              already uses. NEFF-content diff MUST show byte-identical TKG program
              bytes to the baseline (per GEMMA4-LESSONS A6).

            - coalesce_factor >= 2: emits ceil(N/K) descriptors of K*B bytes each.
              `oob_mode.skip` propagated so KV-cache -1 sentinels skip the write.
              Falls back per-K-group to K=1 if the SBUF budget is exceeded.

        This function is the Path A entry point. It calls the same underlying
        `nisa.dma_copy` primitive the existing attention_tkg uses - the win comes
        from grouping the `indices[i*K:(i+1)*K]` into ONE descriptor with a
        wider `vector_offset` and a K*B-shaped `ap(...)` source view.
        """
        engine = engine if engine is not None else dma_engine.dma
        oob_behavior = oob_behavior if oob_behavior is not None else oob_mode.skip
        K = int(coalesce_factor)
        B = int(per_transfer_size)
        N = int(num_transfers)

        # ---- Compat: K == 1 -> passthrough (byte-identical to baseline) ----
        if K == 1:
            for i in nl.affine_range(N):
                nisa.dma_copy(
                    dst=dst_sbuf[i, :],
                    src=src_hbm[indices[i], :],
                    oob_mode=oob_behavior,
                    dma_engine=engine,
                    name=f"{name}_i{i}",
                )
            return

        # ---- SBUF budget guard (compile-time constant when K is JIT-baked) ----
        coalesced_bytes = K * B
        if coalesced_bytes > SBUF_BUDGET_BYTES_PER_CALL_SITE:
            raise RuntimeError(
                f"dma_coalesced_gather: K*B = {coalesced_bytes} exceeds SBUF "
                f"budget {SBUF_BUDGET_BYTES_PER_CALL_SITE} - shrink K or split "
                f"the call site."
            )

        # ---- Coalesced path ----
        G = (N + K - 1) // K
        for g in nl.affine_range(G):
            # Group's K indices lift into one address-pattern gather.
            group_indices = indices[g * K : (g + 1) * K]
            dst_view = dst_sbuf[g * K : (g + 1) * K, :]
            # Source view: K-wide vector_offset gather. Address pattern matches
            # the existing attention_tkg `k_prior_reshaped.ap(...)` shape at
            # remote-core.py:2421 (indirect_dim=0, vector_offset=<K-wide>).
            src_view = src_hbm.ap(
                [
                    [B, K],           # outer: K packets of B bytes each
                    [1, B],           # inner: contiguous B-byte packet
                ],
                offset=0,
                vector_offset=group_indices,
                indirect_dim=0,
            )
            nisa.dma_copy(
                dst=dst_view,
                src=src_view,
                oob_mode=oob_behavior,
                dma_engine=engine,
                name=f"{name}_g{g}",
            )


# --------------------------------------------------------------------------
# 4.  Post-compile validation gate: NEFF-content diff
# --------------------------------------------------------------------------

def run_neff_content_check(
    baseline_neff: str | os.PathLike,
    candidate_neff: str | os.PathLike,
    *,
    require_different: bool = True,
) -> tuple[bool, str]:
    """Compare TKG program bytes inside two NEFF artifacts (post-1024-B header).

    Per GEMMA4-LESSONS A6 (`campaign-debrief-72h-20260824.md:54-56`): "NEFF = 1024
    byte header + compressed tar; direct member comparison exposed six binary
    no-ops that had burned ~5.8 h of compile."

    Contract:
        - If `require_different` is True (candidate lane), returns (True, reason)
          only if the two NEFFs differ AFTER the 1024-byte header. A byte-identical
          match means the knob did NOT land; ABORT before device time.
        - If `require_different` is False (K=1 self-insert), returns (True, ...)
          only if the two NEFFs match byte-for-byte - the compat gate.

    Implementation is a pure-Python byte compare (no `tar`/`sha256sum` subshell)
    so it runs identically on the compile host and this Windows box.
    """
    b_path = pathlib.Path(baseline_neff)
    c_path = pathlib.Path(candidate_neff)
    if not b_path.exists():
        return False, f"baseline_neff missing: {b_path}"
    if not c_path.exists():
        return False, f"candidate_neff missing: {c_path}"

    with open(b_path, "rb") as fh:
        b_bytes = fh.read()[NEFF_HEADER_BYTES:]
    with open(c_path, "rb") as fh:
        c_bytes = fh.read()[NEFF_HEADER_BYTES:]

    identical = b_bytes == c_bytes
    if require_different and identical:
        return False, (
            f"NEFF program bytes byte-identical to baseline "
            f"({len(b_bytes)} B post-header) - knob did NOT land, ABORT."
        )
    if not require_different and not identical:
        return False, (
            f"NEFF program bytes DIFFER from K=1 baseline "
            f"({len(b_bytes)} vs {len(c_bytes)} B post-header) - "
            f"K=1 compat gate FAIL, wrapper is not zero-overhead."
        )
    if identical:
        return True, "K=1 compat gate PASS: NEFF program bytes byte-identical."
    return True, (
        f"NEFF program bytes differ (baseline {len(b_bytes)} B, "
        f"candidate {len(c_bytes)} B) - knob landed."
    )


# --------------------------------------------------------------------------
# 5.  Top-level orchestration entry point
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CoalescingPlan:
    """Aggregate: which lane fires, at what K, using which path."""

    lane: str
    path: str              # "A" (NKI wrapper), "B" (Python KV slab), "hybrid"
    coalesce_factor_k: int
    site_reports: tuple[DescriptorSiteReport, ...]
    slab_plan: Optional[KvSlabLayoutPlan]
    projected_tokps_multiplier: tuple[float, float]

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        # `KvSlabLayoutPlan` is nested; flatten for JSON.
        return d


def build_coalescing_plan(
    lane_name: str,
    summary_json_paths: Iterable[str | os.PathLike],
    *,
    kv_block_kwargs: Optional[dict] = None,
    prefer_path: str = "hybrid",
) -> CoalescingPlan:
    """Read Fleet A's per-shard summary-jsons, pick K, and pick which path fires.

    Path selection heuristic:
        - Path A (NKI wrapper) if `nki_available()` and mean_bytes_per_packet
          is inside [94, 2000] B and K stays under 64.
        - Path B (KV-slab reshape) if `kv_block_kwargs` given AND the plan
          fits HBM AND the target K is >= 2.
        - Hybrid (both) is the default when both fire cleanly.
    """
    reports = tuple(analyze_descriptor_stream(summary_json_paths))
    if not reports:
        raise RuntimeError(
            f"analyze_descriptor_stream returned no reports for {lane_name} - "
            f"summary-json inputs missing or empty."
        )

    # Pick the largest-headroom report as the coalescing anchor.
    anchor = reports[0]
    k = anchor.coalesce_headroom_k

    slab_plan: Optional[KvSlabLayoutPlan] = None
    if kv_block_kwargs is not None:
        slab_plan = plan_kv_slab_layout(
            current_bytes_per_packet=anchor.mean_bytes_per_packet,
            **kv_block_kwargs,
        )

    path = "A"
    if prefer_path == "B" or (slab_plan is not None and slab_plan.fits_hbm):
        path = "B" if not nki_available() else "hybrid"
    if not nki_available() and slab_plan is None:
        raise RuntimeError(
            "Neither NKI toolchain nor KV-slab kwargs available; cannot land "
            "the lever."
        )

    # Projected multiplier: per PROFILE-AT-KNEE-SUMMARY table.
    # Anchor at 1.4-2.0x for GPT-OSS TP8 C=128, 1.15-1.30x for TP4 C=4,
    # 2.4-3.0x for Qwen3-32B TP8 C=16.
    mean_b = anchor.mean_bytes_per_packet
    if mean_b < 200:
        multiplier = (2.4, 3.0)
    elif mean_b < 800:
        multiplier = (1.4, 2.0)
    else:
        multiplier = (1.15, 1.30)

    return CoalescingPlan(
        lane=lane_name,
        path=path,
        coalesce_factor_k=k,
        site_reports=reports,
        slab_plan=slab_plan,
        projected_tokps_multiplier=multiplier,
    )
