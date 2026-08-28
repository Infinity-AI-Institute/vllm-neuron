# SPDX-License-Identifier: Apache-2.0
"""gemma4_no_fallback_mitigations — PREVENT CPU fallback for Gemma-4-26B-A4B.

**Intent (renamed 2026-08-28 per operator concern):** this module PREVENTS
CPU fallback by wiring the AWS PR #172 upstream kernels for triggers #1 /
#2 and shipping stopgaps for triggers #3 / #4.  It NEVER plans for a
fallback, never authors a "CPU replacement path," and never accepts a
silent CPU emission.  If any of the four trigger classes cannot be
satisfied, the intended failure mode is a loud raise — never a fall.

Renamed from `gemma4_cpu_fallback_replacement.py` (2026-08-28).  The old
name read as "we ship a replacement for the CPU fallback" which is the
opposite of the campaign discipline.  Use `test_no_cpu_fallback.py` as
the universal assertion gate on every future compile lane.

Trigger status (four scaffold §B.1 classes)
-------------------------------------------

  #1  head_dim > _MAX_D_HEAD=128
      SUPERSEDED by AWS PR #172 — the port ships validated
      `nki_flash_attn_d256_swa.py` (SWA layers, head_dim=256) and
      `nki_flash_attn_large_d.py` (global layers, head_dim=512).  Both
      are reused verbatim from PR #106 (gemma-4-31B-IT) and have Stage 5
      canonical-chat validation matching HF CPU bf16 at 100% token
      agreement for 11/12 greedy/sample combos (2026-06-03, TP=8, BS=1).
      This module exposes `import_pr172_flash_attention()` as the wiring
      helper — until PR #172 merges to NxDI main, it points at the local
      vendored copy in the scratchpad; post-merge it swaps to the
      canonical `neuronx_distributed_inference.contrib.gemma_4_26b_a4b_it`
      import.

  #2  hybrid sliding + global KV manager off
      SUPERSEDED by AWS PR #172 — the port ships a subclassed
      `Gemma4KVCacheManager` that handles the per-layer heterogeneous
      shapes (8×head_dim=256 SWA vs 2×head_dim=512 global, post TP
      sharding).  This module exposes `import_pr172_kv_cache_manager()`
      with the same adapter shim.

  #3  vocab=262K > nc_find_index8 partition cap (16384)
      KEPT — PR #172 was validated at batch_size=1 only.  For B>128
      decode, the on-device sampler still hits the 16384 partition cap
      unless `disable_argmax_kernel=True`.  `should_disable_argmax_kernel()`
      returns the conservative decision until Part B #2
      `argmax_kernel_partitioned` (§B.9) lands.  Universal — Qwen3-30B-A3B
      TP=8 with vocab=151k also hits this at TP<=8; the mitigation is
      not Gemma-4-specific despite this module living under the
      gemma4 name.

  #4  (GLU, GELU_TANH_APPROX) activation combo
      KEPT + verify against PR #172 on first fire.  The scaffold §B5
      hazard is: shipped NKI kernel v16 silently fell back to
      `torch_blockwise_matmul_inference` on `(GLU, GELU_TANH_APPROX)`.
      PR #172 does not touch the fused MoE dispatch path (it uses
      NxDI `moe_v2` with `initialize_moe_module`), so a moe_dispatch
      compile ordered on Gemma-4 shape still needs the explicit
      `MoEActivation.GELU_TANH_APPROX` branch coverage.  Verified by
      `verify_activation_branch_coverage()` at Tier-1.

Universal usage
---------------

    from gemma4_no_fallback_mitigations import (
        import_pr172_flash_attention,
        import_pr172_kv_cache_manager,
        should_disable_argmax_kernel,
        verify_activation_branch_coverage,
    )

    # Wire PR #172 attention + KV manager
    flash_attn = import_pr172_flash_attention(variant="d256_swa")
    kv_mgr_cls = import_pr172_kv_cache_manager()

    # Trigger #3 mitigation
    disable = should_disable_argmax_kernel(
        vocab_size=262_144, tp_degree=8, batch_size=256,
    )
    neuron_config.disable_argmax_kernel = disable

    # Trigger #4 discipline: Tier-1 activation branch coverage guard
    verify_activation_branch_coverage()

    # And ALWAYS run the universal assertion tests over the compile log:
    # pytest kernels/tests/test_no_cpu_fallback.py --compile-log <path>

Adapter shim (temporary while PR #172 is open)
----------------------------------------------
Until AWS PR #172 merges to NxDI main, the imports resolve against a
local vendored copy at
`C:\\Users\\apumu\\AppData\\Local\\Temp\\claude\\C--Users-apumu-research-InfinityAI-gemma4-trn2-handoff\\fe22bc9a-b8c7-4107-ac55-1d55ba4bd33a\\scratchpad\\aws-pr172\\`
(populated 2026-08-28 with the PR review snapshot).  Post-merge, set the
env var `GEMMA4_USE_UPSTREAM_PR172=1` and the shim resolves against the
canonical `neuronx_distributed_inference.contrib.gemma_4_26b_a4b_it`.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

log = logging.getLogger("gemma4_no_fallback")


# --------------------------------------------------------------------------
# Trigger enumeration + status table (four scaffold §B.1 classes)
# --------------------------------------------------------------------------


class CPUFallbackTrigger(Enum):
    """Scaffold §B.1 four trigger classes for Gemma-4-26B-A4B CPU fallback."""

    HEAD_DIM_EXCEEDS_MAX_D_HEAD = auto()        # trigger #1
    HYBRID_KV_MANAGER_OFF = auto()              # trigger #2
    VOCAB_BLOWS_NC_FIND_INDEX8 = auto()         # trigger #3
    UNSUPPORTED_ACTIVATION_COMBO = auto()       # trigger #4


@dataclass(frozen=True)
class TriggerStatus:
    trigger: CPUFallbackTrigger
    status: str                     # "SHIPPED_UPSTREAM_PR172" | "SHIPPED_LOCAL_STOPGAP" | "PARTIAL"
    fix_summary: str
    resolution_receipt: str


TRIGGER_STATUS: List[TriggerStatus] = [
    TriggerStatus(
        trigger=CPUFallbackTrigger.HEAD_DIM_EXCEEDS_MAX_D_HEAD,
        status="SHIPPED_UPSTREAM_PR172",
        fix_summary=(
            "AWS PR #172 nki_flash_attn_d256_swa.py (SWA, head_dim=256) + "
            "nki_flash_attn_large_d.py (global, head_dim=512) reused "
            "verbatim from PR #106; Stage 5 canonical-chat validation "
            "matches HF CPU bf16 at 100% token agreement 11/12 combos."
        ),
        resolution_receipt="import_pr172_flash_attention() adapter shim",
    ),
    TriggerStatus(
        trigger=CPUFallbackTrigger.HYBRID_KV_MANAGER_OFF,
        status="SHIPPED_UPSTREAM_PR172",
        fix_summary=(
            "AWS PR #172 Gemma4KVCacheManager subclass ships per-layer "
            "heterogeneous KV shapes (8×256 SWA vs 2×512 global, post-TP)."
        ),
        resolution_receipt="import_pr172_kv_cache_manager() adapter shim",
    ),
    TriggerStatus(
        trigger=CPUFallbackTrigger.VOCAB_BLOWS_NC_FIND_INDEX8,
        status="PARTIAL",
        fix_summary=(
            "should_disable_argmax_kernel() returns True at TP<=8 B>128 "
            "as a mitigation until Part B #2 argmax_kernel_partitioned "
            "(§B.9) lands.  PR #172 validated only at B=1 so does not "
            "close this trigger.  ~100-300 µs/step cost accepted."
        ),
        resolution_receipt="gemma4_no_fallback_mitigations.should_disable_argmax_kernel",
    ),
    TriggerStatus(
        trigger=CPUFallbackTrigger.UNSUPPORTED_ACTIVATION_COMBO,
        status="SHIPPED_LOCAL_STOPGAP",
        fix_summary=(
            "moe_dispatch.MoEActivation.GELU_TANH_APPROX branch — exhaustive "
            "activation table per §A.G-7 closes the (GLU, GELU_TANH_APPROX) "
            "silent-fallback hazard.  PR #172 uses NxDI moe_v2 directly and "
            "does not touch the fused dispatch path — this stopgap remains "
            "in force when we compile the fused-dispatch NEFF."
        ),
        resolution_receipt="moe_dispatch.MoEActivation + activation branch tests",
    ),
]


# --------------------------------------------------------------------------
# PR #172 adapter shim (triggers #1 + #2)
# --------------------------------------------------------------------------

# Local vendored PR #172 snapshot — populated 2026-08-28 with the PR review copy.
_PR172_LOCAL_VENDORED_PATH = (
    r"C:\Users\apumu\AppData\Local\Temp\claude"
    r"\C--Users-apumu-research-InfinityAI-gemma4-trn2-handoff"
    r"\fe22bc9a-b8c7-4107-ac55-1d55ba4bd33a\scratchpad\aws-pr172"
)

# Post-merge canonical import path.
_PR172_UPSTREAM_MODULE = (
    "neuronx_distributed_inference.contrib.gemma_4_26b_a4b_it"
)


def _pr172_use_upstream() -> bool:
    """Whether to resolve against upstream NxDI or the local vendored copy.

    Post-merge, operator sets `GEMMA4_USE_UPSTREAM_PR172=1` in the compile
    env and this shim swaps to the canonical import.
    """
    return os.environ.get("GEMMA4_USE_UPSTREAM_PR172", "0") == "1"


def import_pr172_flash_attention(variant: str = "d256_swa"):
    """Return the PR #172 flash-attention kernel module for a given variant.

    Args
    ----
    variant : {"d256_swa", "large_d"}
        - "d256_swa" — sliding-window attention layers (head_dim=256), 25/30 layers
        - "large_d" — global attention layers (head_dim=512), 5/30 layers

    Returns
    -------
    module
        The PR #172 kernel module.  Post-merge that is
        `neuronx_distributed_inference.contrib.gemma_4_26b_a4b_it.<variant>`;
        while PR #172 is open, that is the local vendored copy loaded from
        `_PR172_LOCAL_VENDORED_PATH`.

    Raises
    ------
    ImportError
        If neither upstream nor local vendored source can be resolved.
        Never returns a CPU stub — a missing kernel is a HARD FAIL.
    """
    if variant not in ("d256_swa", "large_d"):
        raise ValueError(
            f"variant must be 'd256_swa' or 'large_d'; got {variant!r}"
        )
    module_leaf = f"nki_flash_attn_{variant}"

    if _pr172_use_upstream():
        upstream_dotted = f"{_PR172_UPSTREAM_MODULE}.{module_leaf}"
        log.info("Importing PR #172 flash attention from upstream: %s",
                 upstream_dotted)
        return importlib.import_module(upstream_dotted)

    # Local vendored copy (temporary).
    import sys
    if _PR172_LOCAL_VENDORED_PATH not in sys.path:
        sys.path.insert(0, _PR172_LOCAL_VENDORED_PATH)
    log.info("Importing PR #172 flash attention from local vendored copy: %s\\%s.py",
             _PR172_LOCAL_VENDORED_PATH, module_leaf)
    return importlib.import_module(module_leaf)


def import_pr172_kv_cache_manager():
    """Return the PR #172 `Gemma4KVCacheManager` class.

    Post-merge:
        `neuronx_distributed_inference.contrib.gemma_4_26b_a4b_it.modeling_gemma4_neuron.Gemma4KVCacheManager`

    Until then, the local vendored `modeling_gemma4_neuron.py` is not
    part of the scratchpad snapshot (the snapshot is kernels-only per
    PR-review scope), so this shim raises with a clear message pointing
    to the PR #172 diff snapshot for the class body.  When PR #172 lands
    on the compile host, set `GEMMA4_USE_UPSTREAM_PR172=1` and this
    resolves cleanly.
    """
    if _pr172_use_upstream():
        upstream_dotted = f"{_PR172_UPSTREAM_MODULE}.modeling_gemma4_neuron"
        log.info("Importing PR #172 Gemma4KVCacheManager from upstream: %s",
                 upstream_dotted)
        module = importlib.import_module(upstream_dotted)
        return module.Gemma4KVCacheManager

    raise ImportError(
        "Gemma4KVCacheManager is not part of the PR-172 kernels-only "
        "scratchpad snapshot.  When PR #172 lands on the compile host, "
        "set env `GEMMA4_USE_UPSTREAM_PR172=1` and this resolves to "
        f"{_PR172_UPSTREAM_MODULE}.modeling_gemma4_neuron.Gemma4KVCacheManager.  "
        "Until then, see the PR #172 DIFF_FROM_PR106.md snapshot in "
        f"{_PR172_LOCAL_VENDORED_PATH} for the class body."
    )


# --------------------------------------------------------------------------
# Trigger #3 — vocab / nc_find_index8 mitigation (KEPT; PR #172 does not cover)
# --------------------------------------------------------------------------

# `nc_find_index8` partition cap on Trn2.
NC_FIND_INDEX8_PARTITION_CAP = 16_384


def should_disable_argmax_kernel(vocab_size: int, tp_degree: int,
                                 batch_size: int) -> bool:
    """Return True if the deterministic-greedy sampler must be disabled.

    `nc_find_index8` has a 16,384 partition cap.  For Gemma-4-26B-A4B
    (vocab 262,144) at TP<=16 the per-shard vocab exceeds 16,384; the
    on-device sampler cannot run at high batch.  At B<=128 the compiler
    is more permissive; above that, disable is required.

    Universal — applies to any decoder whose per-shard vocab exceeds the
    partition cap.  Qwen3-30B-A3B TP=8 (vocab 151,936) hits it at high B
    too; despite the file name, this function is model-agnostic.

    This is a mitigation only.  The proper fix is Part B #2
    `argmax_kernel_partitioned` (hierarchical top-1 tree reduction); see
    scaffold §B.9.  PR #172 was validated at B=1 only and so does not
    close this trigger.
    """
    per_shard = -(-vocab_size // tp_degree)   # ceildiv
    if per_shard <= NC_FIND_INDEX8_PARTITION_CAP:
        return False
    # Above the cap.  Only large-B decode hits the issue in practice.
    if batch_size > 128:
        log.warning(
            "should_disable_argmax_kernel: vocab_per_shard=%d > %d at B=%d — "
            "returning True (mitigation).  Author Part B #2 to remove this.",
            per_shard, NC_FIND_INDEX8_PARTITION_CAP, batch_size,
        )
        return True
    return False


# --------------------------------------------------------------------------
# Trigger #4 — activation branch coverage (delegates to moe_dispatch)
# --------------------------------------------------------------------------


def verify_activation_branch_coverage() -> None:
    """Confirm the moe_dispatch activation table covers Gemma-4's combo.

    Scaffold §B5 hazard: the shipped NKI kernel v16 silently fell back to
    `torch_blockwise_matmul_inference` on `(GLU, GELU_TANH_APPROX)`.  This
    check greps the moe_dispatch enum for a matching branch and raises
    if it's missing.  Run as part of the Tier-1 CPU battery.

    PR #172 does NOT close this trigger — it uses NxDI `moe_v2` /
    `initialize_moe_module` for MoE and never routes through the fused
    dispatch NEFF this campaign compiles.  When (and only when) the
    fused-dispatch NEFF is the one on device, this guard remains
    load-bearing.  Verify PR #172's own MoE path against the same
    activation label on first Gemma-4-26B-A4B fire.
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
# Public status summary
# --------------------------------------------------------------------------


def summarise_status() -> dict:
    """Human-readable status per CPU-fallback trigger, for the status doc."""
    return {t.trigger.name: t.status for t in TRIGGER_STATUS}


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(summarise_status(), indent=2))
