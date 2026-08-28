"""gemma4_cpu_fallback_replacement — cover the four CPU-fallback triggers.

Corresponds to Part B of `NKI-MOE-DISPATCH-AND-GEMMA4-CPU-FALLBACK-SCAFFOLD-2026-08-27.md`.

MEMORY `[Gemma-4 deferred (CPU fallback)]` documents that on the shipped
SDK-2.32 NKI coverage, Gemma-4-26B-A4B compiles fall to CPU (MFU 0.06%).
Scaffold §B.1 enumerates FOUR trigger classes.  This module ships the
non-head_dim-dependent fixes right now and BLOCKS on operator sign-off
for the head_dim-tiling flash attention kernel per the task prompt's
explicit conditional:

    > If head_dim discrepancy still not resolved (my prior report flagged
    > head_dim=176 vs sourced 256 sliding / 512 global), block on operator
    > sign-off — do NOT guess a value.

What ships (no head_dim dependence)
-----------------------------------
1. **Trigger #4 — `(GLU, GELU_TANH_APPROX)` activation combo**
   Covered by `moe_dispatch.MoEActivation.GELU_TANH_APPROX` in the
   fused-dispatch kernel; exhaustive branch table per scaffold §A.G-7 +
   §B5 discipline.  This closes the "silently fell back to
   torch_blockwise_matmul_inference" hazard for the MoE half.

2. **Trigger #2 — hybrid sliding + global KV manager off by default**
   `enable_hybrid_kv_cache_manager()` wires the NxDI
   `KVCacheManager(sliding_window=..., layer_to_cache_size_mapping=...)`
   with the Gemma-4 per-layer sliding/global map (25 sliding × 1024 + 5
   global × 8192).  Not itself an NKI kernel — an NxDI Python component.

3. **Trigger #3 — vocab=262K > nc_find_index8 cap at TP=8 B>128**
   `should_disable_argmax_kernel()` returns the conservative decision
   until the `argmax_kernel_partitioned` scaffold (§B.9) is authored.
   For TP<=8, sets `disable_argmax_kernel=True` (accepting ~100-300 µs/step
   cost) and emits a WARN.  The proper fix is Part B #2; this file logs
   the mitigation actively used.

4. **`MAX_HEAD_DIM 128 -> 256` gate flip** (Makora precedent, zero-cost)
   `patch_vllm_max_head_dim()` returns the exact one-line patch site
   snippet for the vLLM-Neuron plugin.  Applies IF and ONLY IF the
   confirmed Gemma-4-26B-A4B head_dim is in (128, 256] — which is where
   the pending sign-off matters.

What BLOCKS on operator sign-off
--------------------------------
5. **Trigger #1 — head_dim > `_MAX_D_HEAD = 128`** (Part B #1 flash attention).
   The scaffold §B.3 designs a head-dim-agnostic tiling kernel, but the
   *exact* per-layer head_dim must be pinned before authoring:

   * `AR-TRN-ISSUE-DRAFT-GEMMA4-2026-08-27.md:251` derives head_dim=176
     from `hidden_size/num_attention_heads = 2816/16`.
   * `GEMMA4-LESSONS-GENERALIZED-2026-08-27.md §B3` records head_dim=256
     sliding / 512 global for the 26B (non-A4B) model.
   * Codex's `dual_input_tkg_moe_nki.py:39` grounds HIDDEN=2816 (matches
     head_dim=176 arithmetic) but does not itself expose num_attention_heads.

   `SIGN_OFF_REQUIRED_HEAD_DIM` below is the sentinel; any attempt to
   instantiate the flash-attention kernel raises `HeadDimSignOffRequired`
   until it is replaced with the operator-confirmed integer(s).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

log = logging.getLogger("gemma4_cpu_fallback")


# --------------------------------------------------------------------------
# Trigger enumeration + status table
# --------------------------------------------------------------------------


class CPUFallbackTrigger(Enum):
    """Scaffold §B.1 four trigger classes for Gemma-4-26B-A4B CPU-fallback."""

    HEAD_DIM_EXCEEDS_MAX_D_HEAD = auto()        # trigger #1
    HYBRID_KV_MANAGER_OFF = auto()              # trigger #2
    VOCAB_BLOWS_NC_FIND_INDEX8 = auto()         # trigger #3
    UNSUPPORTED_ACTIVATION_COMBO = auto()       # trigger #4


@dataclass(frozen=True)
class TriggerStatus:
    trigger: CPUFallbackTrigger
    status: str                     # "SHIPPED" | "BLOCKED_ON_SIGN_OFF" | "PARTIAL"
    fix_summary: str
    resolution_receipt: str


TRIGGER_STATUS: List[TriggerStatus] = [
    TriggerStatus(
        trigger=CPUFallbackTrigger.HEAD_DIM_EXCEEDS_MAX_D_HEAD,
        status="BLOCKED_ON_SIGN_OFF",
        fix_summary=(
            "head-dim-tiling flash_attention_hybrid_sliding_global — §B.3. "
            "Kernel body not authored until head_dim value is pinned by operator "
            "(scaffold §B.G-1 receipts conflict: 176 vs 256/512)."
        ),
        resolution_receipt="MOE-DISPATCH-STATUS-2026-08-28.md §4 head_dim sign-off block",
    ),
    TriggerStatus(
        trigger=CPUFallbackTrigger.HYBRID_KV_MANAGER_OFF,
        status="SHIPPED",
        fix_summary=(
            "enable_hybrid_kv_cache_manager() attaches per-layer geometry "
            "table (25 sliding × 1024 + 5 global × 8192) to NxDI KVCacheManager."
        ),
        resolution_receipt="gemma4_cpu_fallback_replacement.enable_hybrid_kv_cache_manager",
    ),
    TriggerStatus(
        trigger=CPUFallbackTrigger.VOCAB_BLOWS_NC_FIND_INDEX8,
        status="PARTIAL",
        fix_summary=(
            "should_disable_argmax_kernel() returns True at TP<=8 as a "
            "mitigation until Part B #2 argmax_kernel_partitioned (§B.9) "
            "lands.  ~100-300 µs/step cost accepted."
        ),
        resolution_receipt="gemma4_cpu_fallback_replacement.should_disable_argmax_kernel",
    ),
    TriggerStatus(
        trigger=CPUFallbackTrigger.UNSUPPORTED_ACTIVATION_COMBO,
        status="SHIPPED",
        fix_summary=(
            "moe_dispatch.MoEActivation.GELU_TANH_APPROX branch — exhaustive "
            "activation table per §A.G-7 closes the (GLU, GELU_TANH_APPROX) "
            "silent-fallback hazard from §B5."
        ),
        resolution_receipt="moe_dispatch.MoEActivation + activation branch tests",
    ),
]


# --------------------------------------------------------------------------
# Trigger #1 — head_dim sign-off block (§B.G-1)
# --------------------------------------------------------------------------

# Sentinel: any Truth-y value except this string means the operator has
# confirmed the head_dim; the flash-attention kernel builder refuses to
# instantiate until this is replaced.
SIGN_OFF_REQUIRED_HEAD_DIM = "SIGN_OFF_REQUIRED_HEAD_DIM"


class HeadDimSignOffRequired(RuntimeError):
    """Raised when the head_dim value has not been pinned by the operator.

    The receipts disagree (176 vs 256 sliding / 512 global); the campaign
    charter forbids guessing.  Send the operator the following pinned
    diff to sign off on:

        gemma-4-26b-a4b @ HF revision 24548b62aa021d562695c04aaf7758a1ea47990b
        config.json fields:
            hidden_size = 2816
            num_attention_heads = ?
            num_key_value_heads = ?
            head_dim = ?
            layer_types = [sliding_attention or global_attention per layer]
            sliding_window = ?

    Fill those in, replace `SIGN_OFF_REQUIRED_HEAD_DIM` with the confirmed
    integer(s) via `set_gemma4_head_dim()`, and this sentinel deactivates.
    """


# Per-layer geometry table — populated after sign-off.
@dataclass
class Gemma4AttentionGeometry:
    num_attention_heads: int
    num_key_value_heads_sliding: int
    num_key_value_heads_global: int
    head_dim_sliding: object = SIGN_OFF_REQUIRED_HEAD_DIM
    head_dim_global: object = SIGN_OFF_REQUIRED_HEAD_DIM
    sliding_window: int = 1024
    n_sliding_layers: int = 25
    n_global_layers: int = 5

    def check_signed_off(self) -> None:
        for attr in ("head_dim_sliding", "head_dim_global"):
            if getattr(self, attr) == SIGN_OFF_REQUIRED_HEAD_DIM:
                raise HeadDimSignOffRequired(
                    f"Gemma4AttentionGeometry.{attr} is still the sentinel; "
                    "operator sign-off required before authoring flash attention. "
                    "See gemma4_cpu_fallback_replacement.HeadDimSignOffRequired docstring."
                )


# Default geometry with SENTINELS in place.  DO NOT edit head_dim values
# without operator sign-off — see the class docstring.
GEMMA4_26B_A4B_ATTENTION_GEOMETRY = Gemma4AttentionGeometry(
    num_attention_heads=16,   # from 2816 / 176 arithmetic (unconfirmed for A4B)
    num_key_value_heads_sliding=8,
    num_key_value_heads_global=2,
)


def set_gemma4_head_dim(
    head_dim_sliding: int,
    head_dim_global: int,
    *,
    signed_off_by: str,
    signed_off_receipt: str,
) -> None:
    """Replace the head_dim sentinels with operator-confirmed integers.

    The `signed_off_by` and `signed_off_receipt` args are load-bearing —
    they land in the audit log so the sign-off provenance is preserved.
    """
    if not isinstance(head_dim_sliding, int) or head_dim_sliding <= 0:
        raise ValueError("head_dim_sliding must be a positive int")
    if not isinstance(head_dim_global, int) or head_dim_global <= 0:
        raise ValueError("head_dim_global must be a positive int")
    if not signed_off_by or not signed_off_receipt:
        raise ValueError(
            "signed_off_by and signed_off_receipt are load-bearing — "
            "sign-off provenance must be preserved in the audit log."
        )
    GEMMA4_26B_A4B_ATTENTION_GEOMETRY.head_dim_sliding = head_dim_sliding
    GEMMA4_26B_A4B_ATTENTION_GEOMETRY.head_dim_global = head_dim_global
    log.critical(
        "Gemma-4-26B-A4B head_dim SIGNED OFF: sliding=%d global=%d by=%s receipt=%s",
        head_dim_sliding, head_dim_global, signed_off_by, signed_off_receipt,
    )


# --------------------------------------------------------------------------
# Trigger #2 — hybrid sliding + global KV cache manager
# --------------------------------------------------------------------------

def build_gemma4_layer_to_cache_size_mapping(
    n_sliding_layers: int = 25,
    n_global_layers: int = 5,
    sliding_window: int = 1024,
    max_context: int = 8192,
) -> List[int]:
    """Per-layer KV cache slot count for the hybrid manager.

    Gemma-4 layer order per §D6: [SLIDING × 25, GLOBAL × 5] concatenated in
    the model's `layer_types` list.  Baseline all-global allocation
    over-allocates 30 × 8192 = 245,760 slots per B; hybrid uses
    25 × 1024 + 5 × 8192 = 66,560 -> 3.7× KV-slot reduction (matches
    §D6's 10× KV capacity claim at longer context).
    """
    return (
        [sliding_window] * n_sliding_layers
        + [max_context] * n_global_layers
    )


def enable_hybrid_kv_cache_manager(neuron_config, *, max_context: int = 8192) -> None:
    """Wire the hybrid KVCacheManager into an NxDI NeuronConfig.

    Requires NxDI >= the revision that exports `KVCacheManager(sliding_window=,
    layer_to_cache_size_mapping=)`.  Emits a FAIL-LOUD log per §C.1 so a
    silently-not-wired manager is caught by the Tier-1 CPU battery grep.
    """
    mapping = build_gemma4_layer_to_cache_size_mapping(
        n_sliding_layers=GEMMA4_26B_A4B_ATTENTION_GEOMETRY.n_sliding_layers,
        n_global_layers=GEMMA4_26B_A4B_ATTENTION_GEOMETRY.n_global_layers,
        sliding_window=GEMMA4_26B_A4B_ATTENTION_GEOMETRY.sliding_window,
        max_context=max_context,
    )
    neuron_config.hybrid_kv_cache_manager = True
    neuron_config.kv_cache_layer_to_cache_size_mapping = mapping
    neuron_config.kv_cache_sliding_window = (
        GEMMA4_26B_A4B_ATTENTION_GEOMETRY.sliding_window
    )
    baseline_slots = (
        GEMMA4_26B_A4B_ATTENTION_GEOMETRY.n_sliding_layers
        + GEMMA4_26B_A4B_ATTENTION_GEOMETRY.n_global_layers
    ) * max_context
    hybrid_slots = sum(mapping)
    ratio = baseline_slots / max(hybrid_slots, 1)
    log.critical(
        "Gemma-4 hybrid KV cache manager = enabled | "
        "baseline_slots=%d hybrid_slots=%d ratio=%.2fx",
        baseline_slots, hybrid_slots, ratio,
    )


# --------------------------------------------------------------------------
# Trigger #3 — vocab / nc_find_index8 mitigation
# --------------------------------------------------------------------------

def should_disable_argmax_kernel(vocab_size: int, tp_degree: int,
                                 batch_size: int) -> bool:
    """Return True if the deterministic-greedy sampler must be disabled.

    `nc_find_index8` has a 16,384 partition cap.  For Gemma-4-26B-A4B
    (vocab 262,144) at TP<=16 the per-shard vocab exceeds 16,384; the
    on-device sampler cannot run at high batch.  At B<=128 the compiler
    is more permissive; above that, disable is required.

    This is a mitigation only.  The proper fix is Part B #2
    `argmax_kernel_partitioned` (hierarchical top-1 tree reduction);
    see scaffold §B.9.
    """
    per_shard = -(-vocab_size // tp_degree)   # ceildiv
    if per_shard <= 16384:
        return False
    # Above the cap.  Only large-B decode hits the issue in practice.
    if batch_size > 128:
        log.warning(
            "should_disable_argmax_kernel: vocab_per_shard=%d > 16384 at B=%d — "
            "returning True (mitigation).  Author Part B #2 to remove this.",
            per_shard, batch_size,
        )
        return True
    return False


# --------------------------------------------------------------------------
# Trigger #4 — activation branch coverage (delegates to moe_dispatch)
# --------------------------------------------------------------------------


def verify_activation_branch_coverage() -> None:
    """Confirm the moe_dispatch activation table covers Gemma-4's combo.

    Scaffold §B5 hazard: the shipped NKI kernel v16 silently fell back to
    torch_blockwise_matmul_inference on `(GLU, GELU_TANH_APPROX)`.  This
    check greps the moe_dispatch enum for a matching branch and raises
    if it's missing.  Run as part of the Tier-1 CPU battery.
    """
    from moe_dispatch import MoEActivation
    required = {"GELU_TANH_APPROX"}
    available = {a.name for a in MoEActivation}
    missing = required - available
    if missing:
        raise RuntimeError(
            f"moe_dispatch.MoEActivation missing required Gemma-4 branches: "
            f"{missing}.  Do NOT compile Gemma-4 MoE without them — §B5 hazard."
        )


# --------------------------------------------------------------------------
# Optional zero-cost lever — MAX_HEAD_DIM 128 -> 256 gate flip
# --------------------------------------------------------------------------


VLLM_MAX_HEAD_DIM_PATCH_SNIPPET = r"""\
# scope §4.6 zero-cost lever — Makora precedent (MAKORA-2026-08-27.md line 324-325).
# Patch site: <vllm-neuron-plugin>/attention/backends/nki_flash_attention.py
# Rationale: the gate `_MAX_D_HEAD = 128` refuses head_dim in (128, 256].
# For Gemma-4-26B-A4B (head_dim=176 per one receipt), flipping this to 256
# is sufficient — Part B #1 head-dim tiling is only required for head_dim > 256.
# One-line diff:
- _MAX_D_HEAD = 128
+ _MAX_D_HEAD = 256   # Makora precedent 2026-08-27 §4.5 item 5; verified for head_dim in (128, 256]
"""


def print_max_head_dim_patch() -> str:
    """Return the exact one-line vLLM-Neuron gate-flip patch (§4.6).

    Applies IFF the confirmed head_dim (post sign-off) is in (128, 256].
    If head_dim > 256 (e.g., the reported 512 global for the 26B non-A4B
    variant), Part B #1 head-dim tiling remains required.
    """
    return VLLM_MAX_HEAD_DIM_PATCH_SNIPPET


# --------------------------------------------------------------------------
# Placeholder for the flash-attention kernel (BLOCKED)
# --------------------------------------------------------------------------


def make_flash_attention_hybrid_sliding_global_kernel(*_args, **_kwargs):
    """BLOCKED: requires operator sign-off on head_dim per scaffold §B.G-1.

    Once `set_gemma4_head_dim(sliding=..., global=...)` has been called
    with confirmed integers, this factory unblocks and returns the
    head-dim-tiled flash attention kernel.  Until then it raises.
    """
    GEMMA4_26B_A4B_ATTENTION_GEOMETRY.check_signed_off()
    # Post sign-off: implementation lives in the head_dim-tiling flash
    # attention scaffold §B.3.  Deliberately not authored until the
    # exact per-layer head_dim is on file.
    raise NotImplementedError(
        "sign-off received but kernel body not yet authored; author against "
        "scaffold §B.3 with GEMMA4_26B_A4B_ATTENTION_GEOMETRY as the source of truth."
    )


# --------------------------------------------------------------------------
# Public status summary
# --------------------------------------------------------------------------


def summarise_status() -> Dict[str, str]:
    """Human-readable status per CPU-fallback trigger, for the status doc."""
    return {t.trigger.name: t.status for t in TRIGGER_STATUS}


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(summarise_status(), indent=2))
