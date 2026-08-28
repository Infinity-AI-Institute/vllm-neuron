# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""KDA int8-state study lane -- NOT SERVED.

This module is a thin alias over the v1 kernel at
`C:\\Users\\apumu\\research\\InfinityAI\\gemma4-trn2-handoff\\harness-v2\\staging\\reference-sweep-20260826T2150Z\\kernels\\kda_state.py`
and exists only for HBM-bandwidth studies (see NKI-DELTANET-STATE-INT8-SCAFFOLD-2026-08-27.md).

**NEVER USE IN THE MODEL SERVING LOOP.** The int8 per-channel absmax discipline
adds ~0.79% per-element quantization noise per decode step; over 34 KDA layers
of GLM-5.3-Flash (or 69 of Kimi K3) the accumulated RMS error destroys the
recurrent state within a request-length window (see
VLLM-KDA-KERNEL-FLAVOR-2026-08-28.md §4 table row "State dtype in HBM").

The serving path is `kda_state_v2.py`, slug
`kda_state.decode.kda_gate.rank1_delta.bf16_state.v1`.

WHY IT STAYS ON DISK:
  - v1's 22 correctness tests + speed-floor tests still document, on-disk,
    the algorithmic gap the v2 refactor addresses (missing per-channel gate,
    missing in-kernel L2-norm, missing query scale, int8 state noise).
  - The int8 layout is still a valid *quantized-KV-cache* study lane if we
    ever revisit int8 for a NON-KDA linear-attention variant where the state
    is not itself the sole memory of the sequence.

WHAT LIVES HERE:
  - Re-exports of the v1 API under the same names.
  - Explicit `SERVING_STATUS = "NOT_SERVED"` marker any importer can grep for.
  - A `require_study_ack()` guard callers must invoke to prove they read this
    docstring before wiring the module into anything.
"""

from __future__ import annotations

# Explicit no-serve marker. Any downstream tooling that checks module-level
# `SERVING_STATUS` must refuse to include this module in the served graph.
SERVING_STATUS: str = "NOT_SERVED"
STUDY_LANE_REASON: str = (
    "int8 per-channel absmax adds ~0.79%/step quantization noise; unbounded "
    "over a request-length KDA recurrence. See "
    "VLLM-KDA-KERNEL-FLAVOR-2026-08-28.md #4 and kda_state_v2.py."
)
STUDY_LANE_ACK_STRING: str = (
    "I have read kda_state_int8_study docstring and confirm this module is not "
    "for serving; it is a bandwidth study only."
)


def require_study_ack(ack: str) -> None:
    """Callable guard for callers importing this module.

    Downstream code that wires the int8 kernel into anything (a compile artifact,
    a benchmark harness, a study notebook) MUST call this with the exact
    STUDY_LANE_ACK_STRING to prove the docstring was read.
    """
    if ack != STUDY_LANE_ACK_STRING:
        raise RuntimeError(
            "kda_state_int8_study.require_study_ack: caller did not provide the "
            "acknowledgement string. Read the module docstring and pass "
            "STUDY_LANE_ACK_STRING. This module is NOT for serving -- see "
            "STUDY_LANE_REASON."
        )


# Re-export the v1 API so external references (documentation, test files, older
# harness code) continue to resolve. New code MUST import from `kda_state_v2`.
from kda_state import (  # noqa: E402, F401
    EFFICIENT_DMA_DESCRIPTOR_BYTES,
    GLM_5_3_FLASH_KDA_SHAPE,
    KDA_STATE_KERNEL_SLUG,
    KIMI_K3_KDA_SHAPE,
    KdaDecodeInputs,
    KdaDecodeOutputs,
    KdaShape,
    QWEN35_2B_DELTANET_SHAPE,
    TRAINIUM2_SBUF_BUDGET_BYTES,
    build_shape,
    dequantize_state_int8,
    dma_descriptor_bytes_per_layer,
    get_nki_backend,
    kda_state_decode_forward,
    kda_state_decode_forward_reference,
    kda_state_prefill_forward_reference,
    kda_state_reset,
    quantize_state_int8,
    sbuf_resident_state_bytes,
    sbuf_total_state_bytes,
)


# Local alias so grep for the study-lane symbol finds this file, not v1.
INT8_STUDY_KERNEL_SLUG: str = KDA_STATE_KERNEL_SLUG  # v1's slug
