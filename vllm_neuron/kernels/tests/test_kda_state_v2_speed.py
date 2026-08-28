# SPDX-License-Identifier: Apache-2.0
"""Kernel-tier speed floor -- KDA V2 (bf16 state).

Bf16 state is 2x the byte cost of v1's int8 state, so the SBUF and HBM budgets
have to be revalidated at the model presets. This file locks in:

  A. DMA descriptor floor still cleared at bf16 (all model shapes stay above the
     4 KiB tiny-packet threshold).

  B. Single-layer bf16 state fits SBUF for K3 and GLM-5.3-Flash at B=1.

  C. Colocated-layer count that fits SBUF at B=1 (fewer than v1 because bf16
     is 2x the bytes).

  D. HBM ceiling audit at B=32:
       K3 B=32 bf16 total: 32 * 69 * 96 * 128 * 128 * 2 = 6.46 GiB
         -- must fit inside 24 GiB per-NC HBM.
       GLM-5.3-Flash B=32 bf16 total: 32 * 34 * 64 * 128 * 128 * 2 = 2.13 GiB
         -- fits with wide margin.

  E. Kernel slug is compile-cache-safe.

No hardware, no compile -- byte arithmetic over `kda_state_v2.KdaShapeV2`.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_KERNELS = os.path.dirname(_HERE)
if _KERNELS not in sys.path:
    sys.path.insert(0, _KERNELS)

import pytest

from kda_state_v2 import (  # noqa: E402
    EFFICIENT_DMA_DESCRIPTOR_BYTES_V2,
    GLM_5_3_FLASH_KDA_SHAPE_V2,
    KDA_STATE_V2_KERNEL_SLUG,
    KIMI_K3_KDA_SHAPE_V2,
    KdaShapeV2,
    TRAINIUM2_HBM_BUDGET_BYTES_V2,
    TRAINIUM2_SBUF_BUDGET_BYTES_V2,
    build_shape_v2,
    dma_descriptor_bytes_per_layer_v2,
    sbuf_resident_state_bytes_v2,
    sbuf_total_state_bytes_v2,
)


# ---------------------------------------------------------------------------
# A. DMA descriptor floor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("preset", ["k3", "glm53flash"])
def test_dma_descriptor_over_floor_v2(preset: str, B: int) -> None:
    """Per-layer bf16 DMA over the tiny-packet penalty floor.

    At K3 preset with B=1 the per-layer bf16 DMA is:
        payload_bf16 = 1*96*128*128*2 = 3.0 MiB (read)
        + same for write = 6.0 MiB total per layer
    Comfortably above 4 KiB.
    """
    base = KIMI_K3_KDA_SHAPE_V2 if preset == "k3" else GLM_5_3_FLASH_KDA_SHAPE_V2
    shape = build_shape_v2(base, B=B)
    dma_bytes = dma_descriptor_bytes_per_layer_v2(shape)
    assert dma_bytes >= EFFICIENT_DMA_DESCRIPTOR_BYTES_V2, (preset, B, dma_bytes)


def test_bf16_dma_is_exactly_2x_int8_at_same_shape() -> None:
    """bf16 DMA per layer must be exactly 2x the int8 payload plus
    subtract the scale overhead (v1 int8 had a per-channel scale side channel;
    v2 bf16 has none)."""
    shape = build_shape_v2(KIMI_K3_KDA_SHAPE_V2, B=8)
    bf16_dma = dma_descriptor_bytes_per_layer_v2(shape)
    # int8 payload equivalent: B*H*D_v*D_qk*1 bytes; 2x for read+write.
    int8_payload_dma = 2 * shape.B * shape.H * shape.D_v * shape.D_qk
    # bf16 is exactly 2x the int8 payload (2 bytes per element).
    assert bf16_dma == 2 * int8_payload_dma, (bf16_dma, int8_payload_dma)


# ---------------------------------------------------------------------------
# B. Single-layer SBUF fits
# ---------------------------------------------------------------------------

def test_k3_single_layer_bf16_state_fits_in_sbuf() -> None:
    """Single K3 KDA layer at B=1 with bf16 state:
        1 * 96 * 128 * 128 * 2 = 3.0 MiB  = 12.5% of 24 MiB SBUF
    Leaves room for prologue/epilogue tiles, learned-param broadcast,
    and the delta-rule working set.
    """
    shape = build_shape_v2(KIMI_K3_KDA_SHAPE_V2, B=1)
    per_layer = sbuf_resident_state_bytes_v2(shape)
    assert per_layer < TRAINIUM2_SBUF_BUDGET_BYTES_V2 // 4, per_layer


def test_glm53flash_single_layer_bf16_state_fits_in_sbuf() -> None:
    """GLM-5.3-Flash single-layer bf16 at B=1:
        1 * 64 * 128 * 128 * 2 = 2.0 MiB  = 8.3% of 24 MiB SBUF
    """
    shape = build_shape_v2(GLM_5_3_FLASH_KDA_SHAPE_V2, B=1)
    per_layer = sbuf_resident_state_bytes_v2(shape)
    assert per_layer < TRAINIUM2_SBUF_BUDGET_BYTES_V2 // 4, per_layer


# ---------------------------------------------------------------------------
# C. Colocated-layer capacity (2x fewer than v1)
# ---------------------------------------------------------------------------

def test_k3_two_colocated_bf16_layers_fit() -> None:
    """K3 bf16 state: 2 colocated layers at B=1 = 6.0 MiB, fits.
    (Contrast v1 int8: 4 layers = 6.1 MiB. Doubling per-element cost halves
    the layer capacity.)
    """
    shape = build_shape_v2(KIMI_K3_KDA_SHAPE_V2, B=1)
    per_layer = sbuf_resident_state_bytes_v2(shape)
    assert 2 * per_layer < TRAINIUM2_SBUF_BUDGET_BYTES_V2, per_layer


def test_glm53flash_three_colocated_bf16_layers_fit() -> None:
    """GLM-5.3-Flash bf16 state at B=1: 3 colocated layers = 6.0 MiB, fits."""
    shape = build_shape_v2(GLM_5_3_FLASH_KDA_SHAPE_V2, B=1)
    per_layer = sbuf_resident_state_bytes_v2(shape)
    assert 3 * per_layer < TRAINIUM2_SBUF_BUDGET_BYTES_V2, per_layer


def test_state_bytes_scale_linearly_with_batch_v2() -> None:
    for base in [KIMI_K3_KDA_SHAPE_V2, GLM_5_3_FLASH_KDA_SHAPE_V2]:
        b1 = sbuf_resident_state_bytes_v2(build_shape_v2(base, B=1))
        b8 = sbuf_resident_state_bytes_v2(build_shape_v2(base, B=8))
        assert b8 == 8 * b1, (base, b1, b8)


# ---------------------------------------------------------------------------
# D. HBM ceiling audit at B=32
# ---------------------------------------------------------------------------

def test_k3_b32_bf16_state_fits_hbm_ceiling() -> None:
    """K3 B=32 bf16 total (all layers): 32 * 69 * 96 * 128 * 128 * 2 = 6.46 GiB.

    Must fit inside per-NC HBM budget of 24 GiB. Provides room for weights,
    KV cache slack, and activations. Task doc quotes the range 6.0-7.2 GiB.
    """
    shape = build_shape_v2(KIMI_K3_KDA_SHAPE_V2, B=32)
    total = sbuf_total_state_bytes_v2(shape)
    gib = total / (1024 ** 3)
    assert 6.0 <= gib <= 7.2, gib
    assert total < TRAINIUM2_HBM_BUDGET_BYTES_V2, total


def test_glm53flash_b32_bf16_state_fits_hbm_ceiling() -> None:
    """GLM-5.3-Flash B=32 bf16 total (all layers):
        32 * 34 * 64 * 128 * 128 * 2 = 2,281,701,376 bytes = 2.125 GiB.

    Fits with wide margin. Task doc quoted ~4.3 GiB; that figure was
    double-counted (either from a stale HV=128 config or from including a
    separate spec-decode scratch buffer that is not part of the served path).
    Recomputed here from the model-scope docs (H=64, D=128, layers=34).
    """
    shape = build_shape_v2(GLM_5_3_FLASH_KDA_SHAPE_V2, B=32)
    total = sbuf_total_state_bytes_v2(shape)
    gib = total / (1024 ** 3)
    assert 1.9 <= gib <= 2.4, gib
    assert total < TRAINIUM2_HBM_BUDGET_BYTES_V2, total


# ---------------------------------------------------------------------------
# E. Slug + compile-cache
# ---------------------------------------------------------------------------

def test_v2_kernel_slug_present_and_versioned() -> None:
    parts = KDA_STATE_V2_KERNEL_SLUG.split(".")
    assert parts[0] == "kda_state"
    assert parts[1] == "decode"
    assert parts[2] == "kda_gate"
    assert parts[-1].startswith("v"), KDA_STATE_V2_KERNEL_SLUG


def test_v2_slug_differs_from_v1_slug() -> None:
    """Compile-cache safety: a v2 NEFF must never be served as v1 or vice
    versa. Slug divergence is the guard."""
    v1_slug = "kda_state.decode.rank1_delta.int8_state.per_channel_scale.v1"
    assert KDA_STATE_V2_KERNEL_SLUG != v1_slug
    # And explicitly different key tokens ("kda_gate" vs no gate token).
    assert "kda_gate" in KDA_STATE_V2_KERNEL_SLUG
    assert "int8_state" not in KDA_STATE_V2_KERNEL_SLUG
    assert "bf16_state" in KDA_STATE_V2_KERNEL_SLUG


def test_v2_shape_hash_diverges_across_models() -> None:
    k3 = build_shape_v2(KIMI_K3_KDA_SHAPE_V2, B=1)
    glm = build_shape_v2(GLM_5_3_FLASH_KDA_SHAPE_V2, B=1)
    assert hash((k3.H, k3.D_v, k3.D_qk, k3.layers)) != hash(
        (glm.H, glm.D_v, glm.D_qk, glm.layers)
    )


# ---------------------------------------------------------------------------
# F. HBM growth vs int8 (2x)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["k3", "glm53flash"])
def test_bf16_hbm_is_2x_int8_payload(preset: str) -> None:
    """bf16 state total must be exactly 2x the int8 payload total at the same
    shape. This is the concrete cost of the correctness fix -- caller uses it
    to justify the HBM budget bump in the tokenomics receipt."""
    if preset == "k3":
        base = KIMI_K3_KDA_SHAPE_V2
    else:
        base = GLM_5_3_FLASH_KDA_SHAPE_V2
    shape = build_shape_v2(base, B=8)
    bf16_total = sbuf_total_state_bytes_v2(shape)
    int8_payload_total = shape.layers * shape.B * shape.H * shape.D_v * shape.D_qk
    assert bf16_total == 2 * int8_payload_total, (bf16_total, int8_payload_total)
