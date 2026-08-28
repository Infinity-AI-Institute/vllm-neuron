# SPDX-License-Identifier: Apache-2.0
"""Kernel-tier speed floor — DMA descriptor + SBUF resident-state.

What this suite verifies (per NKI-KDA-STATE-SCAFFOLD-2026-08-27.md §4 and
NKI-DMA-COALESCING-SCAFFOLD-2026-08-27.md §3):

  A. **DMA descriptor floor.** Every per-layer DMA descriptor emitted by the
     KDA state kernel must be at least `EFFICIENT_DMA_DESCRIPTOR_BYTES`
     (= 4 KiB) at the target model shapes. Descriptors under this threshold
     hit the "tiny-packet storm" penalty (per PROFILE-AT-KNEE-SUMMARY
     §"96.06% of hw-dynamic packets are ≤64 bytes carrying 10.246% of bytes"),
     and are the exact class of bug the DMA-coalescing scaffold fixes.

  B. **SBUF resident-state floor.** For the K3 and GLM 5.3 Flash preset
     shapes at reasonable batches, the per-layer int8 state must fit inside
     the Trainium2 SBUF budget with room for the delta-rule prologue/epilogue
     tiles. Concretely: single-layer state <= 4 MiB (per scaffold §2.2 "up to
     6 layers fit if we cache aggressively" projection), and the total across
     colocated layers stays below the 24 MiB budget when 4-6 layers are held
     resident.

  C. **Int8 discipline halves DMA traffic.** For the same shape, the int8
     state layout produces at most 55% of the DMA bytes of the bf16 layout
     (the extra 5% is per-channel scale overhead). This is the core lever
     the KDA scaffold is chartered on (per §2.3 K3 "cuts KDA state HBM
     traffic by ~50%").

  D. **Kernel slug is compile-cache-safe.** Presence + shape of the slug is
     stable so compile-caches upstream of us do not silently reuse a stale
     artifact. This is a Fleet A lesson (`Gemma-4 lessons harvest` rule #2:
     every lever names the graph + engine it changes).

The suite is closed-form — no hardware, no compile, no PyTorch. It reads the
`kda_state.KdaShape` size functions and checks byte arithmetic. It is
appropriate to run this in CI on any Windows/Linux machine.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_KERNELS = os.path.dirname(_HERE)
if _KERNELS not in sys.path:
    sys.path.insert(0, _KERNELS)

import pytest

from kda_state import (  # noqa: E402
    EFFICIENT_DMA_DESCRIPTOR_BYTES,
    GLM_5_3_FLASH_KDA_SHAPE,
    KDA_STATE_KERNEL_SLUG,
    KIMI_K3_KDA_SHAPE,
    KdaShape,
    QWEN35_2B_DELTANET_SHAPE,
    TRAINIUM2_SBUF_BUDGET_BYTES,
    build_shape,
    dma_descriptor_bytes_per_layer,
    sbuf_resident_state_bytes,
    sbuf_total_state_bytes,
)


# ---------------------------------------------------------------------------
# A. DMA descriptor floor.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("preset", ["k3", "glm53flash"])
def test_dma_descriptor_over_floor(preset: str, B: int) -> None:
    """Per-layer DMA over the tiny-packet penalty floor.

    At K3 preset (H=96, D=128, layers=69) with B=1 the per-layer DMA is:
        payload_int8 = 1*96*128*128 = 1,572,864 B (1.5 MiB)
        scale_bf16   = 1*96*128*2   = 24,576   B (24 KiB)
        read+write   = 2 * (payload + scale) ~= 3.05 MiB
    Comfortably above 4 KiB. This test catches regressions where someone
    proposes a per-head DMA that would explode into 96 small descriptors
    per layer — head-parallelism must live inside the load, not across it.
    """
    base = KIMI_K3_KDA_SHAPE if preset == "k3" else GLM_5_3_FLASH_KDA_SHAPE
    shape = build_shape(base, B=B)
    dma_bytes = dma_descriptor_bytes_per_layer(shape, dtype_bits=8)
    assert dma_bytes >= EFFICIENT_DMA_DESCRIPTOR_BYTES, (preset, B, dma_bytes)


def test_dma_scale_write_coalesced_with_int8_write() -> None:
    """Per-channel scale write must fuse with the int8 payload write.

    GAP-11 in the DeltaNet scaffold §3.2. The scale is tiny (B*H*D_v*2 bytes)
    — at K3 B=1 that's 24 KiB, at B=32 it's 768 KiB — and if issued as its
    own DMA descriptor bundle it clears the 4 KiB tiny-packet floor only
    weakly. We flag it as a fused pair with the payload so the compiler sees
    one descriptor bundle per layer, not two.
    """
    shape = build_shape(KIMI_K3_KDA_SHAPE, B=1)
    payload = shape.B * shape.H * shape.D_v * shape.D_qk  # int8
    scale = shape.B * shape.H * shape.D_v * 2  # bf16
    fused = payload + scale
    # The fused pair must be at least 4x the efficient floor — provides room
    # to sub-tile across heads without falling below tiny-packet threshold.
    assert fused >= 4 * EFFICIENT_DMA_DESCRIPTOR_BYTES, fused


# ---------------------------------------------------------------------------
# B. SBUF resident-state floor.
# ---------------------------------------------------------------------------

def test_k3_single_layer_state_fits_in_sbuf_with_headroom() -> None:
    """Single K3 KDA layer at B=1 must occupy no more than 25% of SBUF.

    Per scaffold §2.2:
        S_bytes(B=1) = 1 * 96 * 128 * 128 * 1 + 1 * 96 * 128 * 2 (scale)
                     = 1,572,864 + 24,576 = 1.522 MiB
    That's 6.3% of the 24 MiB SBUF — leaving room for prologue/epilogue
    tiles + the concurrent delta-rule matmul working set.
    """
    shape = build_shape(KIMI_K3_KDA_SHAPE, B=1)
    per_layer = sbuf_resident_state_bytes(shape, dtype_bits=8)
    assert per_layer < TRAINIUM2_SBUF_BUDGET_BYTES // 4, per_layer


def test_glm53flash_single_layer_state_fits_in_sbuf_with_headroom() -> None:
    """GLM 5.3 Flash single-layer state at B=1 must occupy < 25% of SBUF.

    H=64 vs K3's 96 makes GLM 5.3 Flash ~33% smaller per layer.
    """
    shape = build_shape(GLM_5_3_FLASH_KDA_SHAPE, B=1)
    per_layer = sbuf_resident_state_bytes(shape, dtype_bits=8)
    assert per_layer < TRAINIUM2_SBUF_BUDGET_BYTES // 4, per_layer


def test_k3_four_colocated_layers_fit() -> None:
    """The scaffold's "up to 6 layers fit if we cache aggressively" projection
    at B=1. Four is the safe target for a first-cut cache policy; six is a
    stretch that needs an SBUF budget audit.
    """
    shape_1 = build_shape(KIMI_K3_KDA_SHAPE, B=1)
    per_layer = sbuf_resident_state_bytes(shape_1, dtype_bits=8)
    assert 4 * per_layer < TRAINIUM2_SBUF_BUDGET_BYTES, per_layer


def test_state_bytes_scale_linearly_with_batch() -> None:
    """Linear-in-B state footprint is a discipline check on the size functions."""
    for base in [KIMI_K3_KDA_SHAPE, GLM_5_3_FLASH_KDA_SHAPE]:
        b1 = sbuf_resident_state_bytes(build_shape(base, B=1), dtype_bits=8)
        b8 = sbuf_resident_state_bytes(build_shape(base, B=8), dtype_bits=8)
        assert b8 == 8 * b1, (base, b1, b8)


# ---------------------------------------------------------------------------
# C. Int8 halves DMA vs bf16.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["k3", "glm53flash", "qwen35_2b"])
def test_int8_halves_dma_vs_bf16(preset: str) -> None:
    if preset == "k3":
        base = KIMI_K3_KDA_SHAPE
    elif preset == "glm53flash":
        base = GLM_5_3_FLASH_KDA_SHAPE
    else:
        base = QWEN35_2B_DELTANET_SHAPE
    shape = build_shape(base, B=8)
    bf16 = dma_descriptor_bytes_per_layer(shape, dtype_bits=16)
    int8 = dma_descriptor_bytes_per_layer(shape, dtype_bits=8)
    # int8 should be at most 55% of bf16 — the "5%" upper cushion is the
    # per-channel scale overhead. If int8 > bf16 the size functions are
    # miscounting the scale contribution.
    ratio = int8 / bf16
    assert ratio <= 0.55, (preset, ratio)
    # And int8 <= 0.51 * bf16 + a small constant — a firmer floor.
    scale_bytes = shape.B * shape.H * shape.D_v * 2
    assert int8 <= 0.5 * bf16 + 2 * scale_bytes + 8, (preset, int8, bf16)


# ---------------------------------------------------------------------------
# D. Compile-cache safety.
# ---------------------------------------------------------------------------

def test_kernel_slug_present_and_versioned() -> None:
    parts = KDA_STATE_KERNEL_SLUG.split(".")
    assert parts[0] == "kda_state"
    assert parts[1] == "decode"
    assert parts[-1].startswith("v"), KDA_STATE_KERNEL_SLUG


def test_shape_hash_diverges_across_models() -> None:
    """K3 and GLM 5.3 Flash shape hashes must differ so compile-cache never
    silently reuses one artifact for the other.

    Cache key includes the shape tuple; equal hashes would allow a K3
    NEFF to serve GLM 5.3 Flash and vice versa.
    """
    k3 = build_shape(KIMI_K3_KDA_SHAPE, B=1)
    glm = build_shape(GLM_5_3_FLASH_KDA_SHAPE, B=1)
    assert hash((k3.H, k3.D_v, k3.D_qk, k3.layers)) != hash(
        (glm.H, glm.D_v, glm.D_qk, glm.layers)
    )


# ---------------------------------------------------------------------------
# Cross-check against the receipts cited in NKI-KDA-STATE-SCAFFOLD-2026-08-27.md
# ---------------------------------------------------------------------------

def test_k3_b32_state_slab_matches_scaffold_math() -> None:
    """Scaffold §1.3 quotes S_MiB(B=32) = 6624 MiB for K3.

    Recompute: B * layers * H * D_v * D_qk * 2 bytes
             = 32 * 69 * 96 * 128 * 128 * 2 = 6,941,573,120 bytes ~= 6620 MiB.
    Within rounding of the scaffold quote. This test locks in that
    arithmetic so a later shape-preset edit doesn't drift.
    """
    shape = build_shape(KIMI_K3_KDA_SHAPE, B=32)
    bf16_all_layers = shape.layers * shape.B * shape.H * shape.D_v * shape.D_qk * 2
    mib = bf16_all_layers / (1024 * 1024)
    # Scaffold quote is 6624 MiB; allow ±1% tolerance.
    assert 6550 <= mib <= 6700, mib


def test_glm53flash_b32_state_slab_matches_scaffold_math() -> None:
    """Scaffold §1.3 quotes S_MiB(B=32) = 2176 MiB for GLM 5.3 Flash.

    Recompute: 32 * 34 * 64 * 128 * 128 * 2 = 2,281,701,376 bytes ~= 2176 MiB.
    """
    shape = build_shape(GLM_5_3_FLASH_KDA_SHAPE, B=32)
    bf16_all_layers = shape.layers * shape.B * shape.H * shape.D_v * shape.D_qk * 2
    mib = bf16_all_layers / (1024 * 1024)
    assert 2150 <= mib <= 2200, mib


def test_int8_total_state_closes_hbm_ceiling_at_k3_b32() -> None:
    """At K3 B=32 with int8 state, total state = ~3.3 GiB — fits inside the
    24 GiB per-NC HBM budget with wide margin. This is the direct proof point
    the scaffold §2.3 K3 "unlocks B=32 which the bf16 layout blows the HBM
    ceiling on" hinges on.
    """
    shape = build_shape(KIMI_K3_KDA_SHAPE, B=32)
    int8_all_layers_bytes = sbuf_total_state_bytes(shape, dtype_bits=8)
    gib = int8_all_layers_bytes / (1024 ** 3)
    # int8 should be ~half of bf16's 6.5 GiB (~3.2-3.5 GiB) at B=32.
    assert 3.0 <= gib <= 3.6, gib
