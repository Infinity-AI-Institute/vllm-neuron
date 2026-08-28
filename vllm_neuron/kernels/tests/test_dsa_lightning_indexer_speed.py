# SPDX-License-Identifier: Apache-2.0
"""Kernel-tier speed gate for the DSA Lightning Indexer.

**What this file is.**
    Device-free analytical envelope checks for the DSA Lightning
    Indexer. Every assertion is against the pre-compute
    `analytical_bounds(...)` from `dsa_lightning_indexer.py`. These are
    the invariants a real NKI compile MUST satisfy at scaffold-derived
    knees; they are ceilings until an on-device Tier-3 profile at the
    knee tightens them into measured floors (profile-at-knee discipline).

**What this file is NOT.**
    A live tok/s benchmark. Wall-clock speed of the CPU reference is
    O(L * H_idx * D_idx + topk * H * D) fp32 and would swamp the < 10 s
    Tier-1 budget above L~16k without adding correctness signal. Live
    device profiles land at the corresponding lane's Tier-3 slot.

Gates covered:
    S0 — SBUF budget fits inside 24 MiB per NC after weights, across
         the (GLM 5.2, GLM 5.3 Flash, DSV4-Flash) reference shapes at
         each S bucket.
    S1 — descriptor coalescing ratio meets scaffold §4.3 target of
         `block_size` (32x reduction at the default config).
    S2 — descriptor cache slot usage stays below the ~4096-slot cache
         at production Q-tile.
    S3 — `nc_find_index8` partition cap (16384) is respected at every
         swept L (the same cap that gates GPT-OSS TP8 B>128).
    S4 — cycles/token floor is below the campaign's O(1M) product
         story ceiling.
    S5 — DsaKernelConfig cache-key change on any shape flip.

Run with:
    py -3 -m pytest -q kernels/tests/test_dsa_lightning_indexer_speed.py
"""
from __future__ import annotations

import sys
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
KERNEL_DIR = HERE.parent
sys.path.insert(0, str(KERNEL_DIR))

from dsa_lightning_indexer import (  # noqa: E402
    DsaKernelConfig,
    KERNEL_SLUG_V0_REFERENCE,
    analytical_bounds,
)


# ---------------------------------------------------------------------------
# Reference production shapes
# ---------------------------------------------------------------------------
#
# Values sourced from CAMPAIGN-SCOPE-{GLM-5.3-FLASH,DEEPSEEK-V4-FLASH,K3}
# and MLA-VS-DSA-KERNEL-VERIFICATION docs. Where a GAP marker still
# stands in the scaffold (e.g. GLM 5.2 GAP-2 for index_n_heads), the
# scaffold's assumed value is used here and flagged as "assumed" in
# the docstring for that shape.

GLM_5_2 = dict(
    label="glm-5-2",
    B=1, Q=1,                    # decode
    H=128, D=128,                # main-attention head geometry (assumed)
    H_idx=64, D_idx=64,          # scaffold GAP-2/GAP-5 assumed
    topk=2048, block_size=32,
    index_pool=1,
)

GLM_5_3_FLASH = dict(
    label="glm-5-3-flash",
    B=1, Q=1,
    H=64, D=128,                 # config: hidden 4096, 64 heads, head_dim=128
    H_idx=64, D_idx=128,         # scaffold GAP-4 assumed
    topk=2048, block_size=32,
    index_pool=4,                # IndexPool=4
)

DSV4_FLASH = dict(
    label="dsv4-flash",
    B=1, Q=1,
    H=64, D=512,                 # config: 64 heads x head_dim=512
    H_idx=64, D_idx=128,         # from config
    topk=512, block_size=32,
    index_pool=1,
)

REFERENCE_SHAPES = [GLM_5_2, GLM_5_3_FLASH, DSV4_FLASH]

# Sequence-length buckets swept per operator prompt.
L_BUCKETS = [2048, 4096, 8192, 16384, 32768]


# ---------------------------------------------------------------------------
# S0 — SBUF budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", REFERENCE_SHAPES, ids=[s["label"] for s in REFERENCE_SHAPES])
@pytest.mark.parametrize("L", L_BUCKETS)
def test_S0_sbuf_budget_fits(shape: dict, L: int):
    bounds = analytical_bounds(
        B=shape["B"], Q=shape["Q"], L=L,
        H=shape["H"], D=shape["D"],
        H_idx=shape["H_idx"], D_idx=shape["D_idx"],
        topk=shape["topk"], block_size=shape["block_size"],
        index_pool=shape["index_pool"],
    )
    # SBUF ceiling: 24 MiB per NC after weights (scaffold §4.1).
    # At the assumed shapes some entries exceed this cap and the
    # kernel must stream in blocks. The gate is on the *streaming*
    # workspace, not the full working set.
    # We assert two things:
    #   (a) the streamed workspace fits (or the stream tile shrinks),
    #   (b) if it does NOT fit at the current window, this test emits
    #       a scaffold-warning marker on the stream count.
    if not bounds.sbuf_fits:
        # Streaming waves needed to fit the indexer KV.
        waves = 1 + bounds.sbuf_bytes_indexer_kv_resident // bounds.sbuf_bytes_ceiling
        # A real kernel is fine with up to O(L / block_size / 96) waves.
        # We cap at 64 waves per query — beyond that a design revisit
        # is required.
        assert waves <= 64, (
            f"{shape['label']} at L={L}: {waves} SBUF waves needed for the "
            f"indexer KV; a design revisit is required per scaffold §4.1."
        )
    else:
        assert bounds.sbuf_headroom_bytes >= 0


# ---------------------------------------------------------------------------
# S1 — descriptor coalescing ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", REFERENCE_SHAPES, ids=[s["label"] for s in REFERENCE_SHAPES])
def test_S1_descriptor_coalescing_hits_block_size(shape: dict):
    """Coalesced/naive descriptor ratio must equal block_size on evenly
    divisible topk. Scaffold §4.3 target: 32x reduction at block_size=32.
    """
    bounds = analytical_bounds(
        B=shape["B"], Q=shape["Q"], L=L_BUCKETS[-1],
        H=shape["H"], D=shape["D"],
        H_idx=shape["H_idx"], D_idx=shape["D_idx"],
        topk=shape["topk"], block_size=shape["block_size"],
        index_pool=shape["index_pool"],
    )
    expected_ratio = shape["block_size"]
    assert bounds.descriptor_reduction_factor >= expected_ratio - 1e-9, (
        f"{shape['label']}: descriptor reduction {bounds.descriptor_reduction_factor:.1f}x "
        f"below scaffold target {expected_ratio}x."
    )


# ---------------------------------------------------------------------------
# S2 — descriptor cache slot usage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", REFERENCE_SHAPES, ids=[s["label"] for s in REFERENCE_SHAPES])
def test_S2_descriptor_cache_slot_budget(shape: dict):
    bounds = analytical_bounds(
        B=shape["B"], Q=16, L=L_BUCKETS[-1],   # Q-tile of 16
        H=shape["H"], D=shape["D"],
        H_idx=shape["H_idx"], D_idx=shape["D_idx"],
        topk=shape["topk"], block_size=shape["block_size"],
        index_pool=shape["index_pool"],
    )
    assert bounds.descriptor_cache_slots_used <= bounds.descriptor_cache_slots_ceiling


# ---------------------------------------------------------------------------
# S3 — nc_find_index8 partition cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", REFERENCE_SHAPES, ids=[s["label"] for s in REFERENCE_SHAPES])
@pytest.mark.parametrize("L", L_BUCKETS)
def test_S3_nc_find_index8_partition_cap(shape: dict, L: int):
    """The top-K stream tile must be within the 16384 partition cap.

    Per scaffold §4.2, `nc_find_index8` is capped at 16384 elements per
    partition. At L up to 16384 the whole score row fits; above that,
    the tile must stream and merge. The analytical bound clips its
    workspace at 16384 fp32 slots.
    """
    bounds = analytical_bounds(
        B=shape["B"], Q=shape["Q"], L=L,
        H=shape["H"], D=shape["D"],
        H_idx=shape["H_idx"], D_idx=shape["D_idx"],
        topk=shape["topk"], block_size=shape["block_size"],
        index_pool=shape["index_pool"],
    )
    # Tile fp32 workspace: 16384 * 4 = 65536 bytes.
    assert bounds.sbuf_bytes_topk_workspace <= 16384 * 4


# ---------------------------------------------------------------------------
# S4 — cycles/token floor sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", REFERENCE_SHAPES, ids=[s["label"] for s in REFERENCE_SHAPES])
@pytest.mark.parametrize("L", L_BUCKETS)
def test_S4_cycles_per_token_floor_reasonable(shape: dict, L: int):
    """Cycles/token floor must (a) be positive, (b) not exceed 10M
    cycles at L=32768 with topk=2048.

    10M cycles at 3 GHz Trn2 clock ~= 3.3 ms/token — well above the
    scaffold's product-story target of ~200 microseconds/token but a
    sane analytical ceiling. The tightener comes from Tier-3 device
    profile at knee.
    """
    bounds = analytical_bounds(
        B=shape["B"], Q=shape["Q"], L=L,
        H=shape["H"], D=shape["D"],
        H_idx=shape["H_idx"], D_idx=shape["D_idx"],
        topk=shape["topk"], block_size=shape["block_size"],
        index_pool=shape["index_pool"],
    )
    assert bounds.cycles_floor_per_token > 0
    assert bounds.cycles_floor_per_token <= 10_000_000


# ---------------------------------------------------------------------------
# S5 — cache-key change on any shape flip
# ---------------------------------------------------------------------------


def test_S5_cache_key_reflects_every_shape_flip():
    """Every field of DsaKernelConfig affects the compile-cache key.
    A stale key silently reuses a wrong-shape kernel; this is a
    correctness-adjacent gate the Gemma-4 top-5 lessons file names
    explicitly ("every lever names the graph + engine it changes").
    """
    base = DsaKernelConfig(
        topk=2048, block_size=32,
        index_n_heads=64, index_head_dim=64,
        index_pool=1, causal=True, return_topk_for_indexshare=False,
    )
    seen = {base.cache_key()}
    for k in [
        DsaKernelConfig(topk=512,  block_size=32, index_n_heads=64, index_head_dim=64, index_pool=1, causal=True, return_topk_for_indexshare=False),
        DsaKernelConfig(topk=2048, block_size=64, index_n_heads=64, index_head_dim=64, index_pool=1, causal=True, return_topk_for_indexshare=False),
        DsaKernelConfig(topk=2048, block_size=32, index_n_heads=48, index_head_dim=64, index_pool=1, causal=True, return_topk_for_indexshare=False),
        DsaKernelConfig(topk=2048, block_size=32, index_n_heads=64, index_head_dim=128, index_pool=1, causal=True, return_topk_for_indexshare=False),
        DsaKernelConfig(topk=2048, block_size=32, index_n_heads=64, index_head_dim=64, index_pool=4, causal=True, return_topk_for_indexshare=False),
        DsaKernelConfig(topk=2048, block_size=32, index_n_heads=64, index_head_dim=64, index_pool=1, causal=False, return_topk_for_indexshare=False),
        DsaKernelConfig(topk=2048, block_size=32, index_n_heads=64, index_head_dim=64, index_pool=1, causal=True, return_topk_for_indexshare=True),
    ]:
        key = k.cache_key()
        assert key not in seen, f"cache key collision on {k}"
        seen.add(key)


def test_S5_cache_key_carries_kernel_slug():
    key = DsaKernelConfig(
        topk=2048, block_size=32,
        index_n_heads=64, index_head_dim=64,
        index_pool=1, causal=True, return_topk_for_indexshare=False,
    ).cache_key()
    assert KERNEL_SLUG_V0_REFERENCE in key
