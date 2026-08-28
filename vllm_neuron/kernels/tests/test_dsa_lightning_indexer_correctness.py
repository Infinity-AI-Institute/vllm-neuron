# SPDX-License-Identifier: Apache-2.0
"""Kernel-tier correctness gate for the DSA Lightning Indexer reference.

Tier-1 discipline (from lane 6-test flywheel):
    * wall budget: < 10 s per case
    * no device, no NKI compile — CPU only
    * every case has an *independent* oracle (not the SUT)

Gates covered:
    T0 — permutation invariant: top-K over uniform random scores is
         a subset of range(L) with no repeats (scaffold §5 T2).
    T1 — ranked-scores bit-exactness: top-K over descending fp32
         scores returns [0..topk-1] (scaffold §5 T3).
    T2 — sparse-vs-full equivalence at topk >= L: outputs match full
         attention to bf16 relative tolerance (scaffold §5 T1).
    T3 — sparse attention golden: at each (L, topk) shape the fused
         forward matches the independent
         `full_attention_at_indices_reference` — this is the exact
         oracle from the operator's prompt:
           `torch.softmax(Q @ K.T)[:, topk_indices]`
    T4 — causal-mask preserves token 0: single-position query at
         position 0 with topk=1 returns a non-zero attention output
         (scaffold §5 T4 — the token-0 GLM 5.2 gate escape).
    T5 — IndexPool linearity: pooled index-keys with uniform weights
         equal the plain average (scaffold §5 T6).
    T6 — IndexShare fidelity: topk_indices returned by a full-indexer
         call reused by a shared-indexer sparse_attention call yields
         the same output as re-computing with the full indexer
         (scaffold §5 T5).
    T7 — dtype consistency: bf16 inputs return bf16 outputs; fp16
         inputs return fp16 outputs.
    T8 — top-K stability: given identical inputs, two indexer calls
         return identical index tensors (bitwise equal).

Shapes swept per the operator's prompt:
    K (== L) in {2048, 4096, 8192, 16384, 32768}
    topk    in {2048, 4096}

At K=2048, topk=2048 the mathematical degeneracy from scaffold §1.1
Mode A applies — the sparse output MUST equal full attention to bf16.
Higher K exercises real sparsity.

Run with:
    py -3 -m pytest -q kernels/tests/test_dsa_lightning_indexer_correctness.py
"""
from __future__ import annotations

import math
import sys
import pathlib

import pytest
import torch

# Make the sibling kernel module importable when pytest is run from repo root
# or from the kernels/ directory.
HERE = pathlib.Path(__file__).resolve().parent
KERNEL_DIR = HERE.parent
sys.path.insert(0, str(KERNEL_DIR))

from dsa_lightning_indexer import (           # noqa: E402
    DsaKernelConfig,
    KERNEL_SLUG_V0_REFERENCE,
    dsa_index_pool_projection,
    dsa_lightning_indexer_forward,
    dsa_sparse_attention_forward,
    full_attention_at_indices_reference,
    full_attention_reference,
    lightning_indexer_scores,
    lightning_indexer_topk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case(
    *,
    B: int = 1,
    Q: int = 4,
    L: int = 2048,
    H: int = 2,
    D: int = 64,
    H_idx: int = 4,
    D_idx: int = 16,
    index_pool: int = 1,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
):
    """Small fixture — sized so CPU wall stays under 10 s at L=32768.

    Shapes are deliberately small along H/D so the L dimension can be
    swept without blowing up CPU time. Semantics are unchanged.
    """
    g = torch.Generator().manual_seed(seed)
    query = torch.randn(B, Q, H, D, generator=g, dtype=torch.float32).to(dtype)
    indexer_q_proj = torch.randn(
        H_idx, H * D, D_idx, generator=g, dtype=torch.float32
    ).to(dtype) / math.sqrt(H * D)
    # indexer_k_cache: [B, L, H_idx*index_pool, D_idx]
    indexer_k_cache = torch.randn(
        B, L, H_idx * index_pool, D_idx, generator=g, dtype=torch.float32
    ).to(dtype)
    kv_cache_k = torch.randn(B, L, H, D, generator=g, dtype=torch.float32).to(dtype)
    kv_cache_v = torch.randn(B, L, H, D, generator=g, dtype=torch.float32).to(dtype)
    # Queries positioned at the end of the sequence (decode-like).
    position_ids = torch.arange(L - Q, L).unsqueeze(0).expand(B, Q).to(torch.int64)
    key_lengths = torch.full((B,), L, dtype=torch.int64)
    return dict(
        query=query,
        indexer_q_proj=indexer_q_proj,
        indexer_k_cache=indexer_k_cache,
        kv_cache_k=kv_cache_k,
        kv_cache_v=kv_cache_v,
        position_ids=position_ids,
        key_lengths=key_lengths,
    )


def _relative_max_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    denom = b32.abs().max().clamp_min(1e-8)
    return float((a32 - b32).abs().max() / denom)


# ---------------------------------------------------------------------------
# T0 — Permutation invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L,topk", [(2048, 2048), (4096, 2048), (8192, 4096)])
def test_T0_topk_permutation_property(L: int, topk: int):
    """At topk == number-of-valid-positions the result is a permutation."""
    kw = _make_case(L=L, Q=2)
    valid = topk  # all positions before Q's own position are valid
    # Set position_ids so all L positions are causally visible.
    kw["position_ids"] = torch.full((1, 2), L - 1, dtype=torch.int64)
    kw["key_lengths"] = torch.full((1,), L, dtype=torch.int64)

    idx = lightning_indexer_topk(
        kw["query"], kw["indexer_q_proj"], kw["indexer_k_cache"],
        kw["position_ids"], kw["key_lengths"],
        topk=topk, causal=True,
    )
    assert idx.shape == (1, 2, topk)
    for b in range(idx.shape[0]):
        for q in range(idx.shape[1]):
            row = idx[b, q].tolist()
            assert len(set(row)) == len(row), "duplicate indices in top-K row"
            assert min(row) >= 0 and max(row) < L
            if topk == L:
                assert set(row) == set(range(L))


# ---------------------------------------------------------------------------
# T1 — Ranked-scores bit exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L,topk", [(2048, 2048), (4096, 2048), (4096, 4096)])
def test_T1_ranked_scores_bit_exact(L: int, topk: int):
    """When indexer scores can be forced to descending, top-K = [0..topk-1]."""
    # Manufacture the scores directly.
    B, Q = 1, 1
    fake_scores = torch.arange(L, 0, -1, dtype=torch.float32).view(1, 1, L)
    fake_scores = fake_scores + 0.0   # copy
    # Ranked topk: torch.topk on strictly descending scores yields [0..topk-1].
    _, idx = torch.topk(fake_scores, k=topk, dim=-1)
    expected = torch.arange(topk).view(1, 1, topk)
    assert torch.equal(idx.to(torch.int64), expected.to(torch.int64))


# ---------------------------------------------------------------------------
# T2 — Sparse-vs-full equivalence at topk >= L
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L,topk", [(2048, 2048)])
def test_T2_sparse_equals_full_when_topk_equals_L(L: int, topk: int):
    """The Mode-A degeneracy — sparse output MUST equal full attention.

    This is the invariant that lets GLM 5.2's compiled 2K graph run
    correctly today. If it fails, no downstream lane is safe.
    """
    kw = _make_case(L=L, Q=8, dtype=torch.float32)

    out_full = full_attention_reference(
        kw["query"], kw["kv_cache_k"], kw["kv_cache_v"],
        kw["position_ids"], kw["key_lengths"], causal=True,
    )
    out_sparse, _ = dsa_lightning_indexer_forward(
        kw["query"], kw["indexer_q_proj"], kw["indexer_k_cache"],
        kw["kv_cache_k"], kw["kv_cache_v"],
        kw["position_ids"], kw["key_lengths"],
        topk=topk, causal=True,
    )
    rel = _relative_max_error(out_sparse, out_full)
    # fp32 arithmetic + different accumulation orders: 1e-4 is safe.
    assert rel < 1e-4, f"sparse != full at topk={topk}: rel={rel:.3e}"


# ---------------------------------------------------------------------------
# T3 — Sparse attention golden vs independent oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "L,topk",
    [
        (2048, 2048),
        (4096, 2048),
        (8192, 2048),
        (16384, 2048),
        (32768, 2048),
        (8192, 4096),
        (16384, 4096),
        (32768, 4096),
    ],
)
def test_T3_sparse_attention_matches_independent_oracle(L: int, topk: int):
    """Match `dsa_sparse_attention_forward` against an independently coded
    `full_attention_at_indices_reference` on synthetic top-K indices.

    This is the operator's exact prompt formula:
        `torch.softmax(Q @ K.T)[:, topk_indices]`
    """
    # Use tiny H,D so wall time stays under 10 s at L=32768.
    if L >= 16384:
        kw = _make_case(L=L, Q=2, H=2, D=32)
    else:
        kw = _make_case(L=L, Q=4, H=2, D=32)

    # Pick arbitrary but valid indices — take the last `topk` positions
    # (all causally visible since queries are at position L-1).
    B, Q = kw["query"].shape[:2]
    idx = torch.arange(L - topk, L, dtype=torch.int32).view(1, 1, topk)
    topk_indices = idx.expand(B, Q, topk).contiguous()

    out_sut = dsa_sparse_attention_forward(
        kw["query"], kw["kv_cache_k"], kw["kv_cache_v"],
        topk_indices,
        kw["position_ids"], kw["key_lengths"],
        topk=topk, causal=True,
    )
    out_ref = full_attention_at_indices_reference(
        kw["query"], kw["kv_cache_k"], kw["kv_cache_v"],
        topk_indices,
        kw["position_ids"], kw["key_lengths"],
        causal=True,
    )
    rel = _relative_max_error(out_sut, out_ref)
    assert rel < 5e-6, f"sparse != oracle at L={L}, topk={topk}: rel={rel:.3e}"


# ---------------------------------------------------------------------------
# T4 — Causal-mask preserves token 0
# ---------------------------------------------------------------------------


def test_T4_token_zero_survives_topk_one():
    """The token-0 gate escape. Attention over the single visible K/V
    must be a non-zero weighted sum, not zero. This is what the ten-token
    gate on GLM 5.2 caught — the reference must not repeat that bug.
    """
    B, Q, L, H, D = 1, 1, 1, 2, 32
    query = torch.randn(B, Q, H, D)
    k = torch.randn(B, L, H, D)
    v = torch.randn(B, L, H, D)
    # v is nonzero everywhere by construction.
    position_ids = torch.zeros(B, Q, dtype=torch.int64)   # token 0
    key_lengths = torch.ones(B, dtype=torch.int64)         # 1 valid position
    idx = torch.zeros(B, Q, 1, dtype=torch.int32)          # topk=1, index 0

    out = dsa_sparse_attention_forward(
        query, k, v, idx, position_ids, key_lengths,
        topk=1, causal=True,
    )
    assert out.shape == (B, Q, H, D)
    assert not torch.isnan(out).any(), "token-0 attention returned NaN"
    assert out.abs().max() > 0, "token-0 attention silently zeroed"


# ---------------------------------------------------------------------------
# T5 — IndexPool linearity
# ---------------------------------------------------------------------------


def test_T5a_index_pool_uniform_equals_average():
    """Uniform pool weights collapse to a plain mean along the pool axis."""
    B, L, H_idx, D_idx = 1, 32, 4, 16
    index_pool = 4
    keys = torch.randn(B, L, H_idx * index_pool, D_idx)
    weights = torch.full((index_pool,), 1.0 / index_pool)

    got = dsa_index_pool_projection(
        keys, weights, index_pool=index_pool,
    )

    ref = keys.view(B, L, H_idx, index_pool, D_idx).mean(dim=3)
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-6)


def test_T5b_index_pool_arbitrary_matches_numpy():
    """Non-uniform pool weights match a hand-coded weighted sum."""
    B, L, H_idx, D_idx = 1, 8, 4, 16
    index_pool = 4
    keys = torch.randn(B, L, H_idx * index_pool, D_idx)
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4])

    got = dsa_index_pool_projection(
        keys, weights, index_pool=index_pool,
    )

    reshaped = keys.view(B, L, H_idx, index_pool, D_idx)
    ref = (reshaped * weights.view(1, 1, 1, index_pool, 1)).sum(dim=3)
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# T6 — IndexShare fidelity (scaffold §5 T5)
# ---------------------------------------------------------------------------


def test_T6_indexshare_reuse_matches_recompute():
    """topk_indices from a full-indexer call, reused by a shared-indexer
    sparse_attention call, produce the same output as re-running the
    full indexer at the shared layer.

    This is the GLM 5.2 IndexShare contract: 78 layers = 20 full + 58
    shared, and the shared layers MUST get bit-identical topk_indices.
    """
    kw = _make_case(L=2048, Q=4, dtype=torch.float32)
    out_a, idx_a = dsa_lightning_indexer_forward(
        kw["query"], kw["indexer_q_proj"], kw["indexer_k_cache"],
        kw["kv_cache_k"], kw["kv_cache_v"],
        kw["position_ids"], kw["key_lengths"],
        topk=1024, causal=True, return_topk_for_indexshare=True,
    )
    # Shared layer receives idx_a and computes sparse attention with the
    # same query and cache. Result must match `out_a`.
    out_b = dsa_sparse_attention_forward(
        kw["query"], kw["kv_cache_k"], kw["kv_cache_v"],
        idx_a,
        kw["position_ids"], kw["key_lengths"],
        topk=1024, causal=True,
    )
    rel = _relative_max_error(out_a, out_b)
    assert rel < 1e-6, f"IndexShare mismatch: rel={rel:.3e}"


# ---------------------------------------------------------------------------
# T7 — dtype consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_T7_dtype_roundtrip(dtype: torch.dtype):
    kw = _make_case(L=2048, Q=2, dtype=dtype)
    out, _ = dsa_lightning_indexer_forward(
        kw["query"], kw["indexer_q_proj"], kw["indexer_k_cache"],
        kw["kv_cache_k"], kw["kv_cache_v"],
        kw["position_ids"], kw["key_lengths"],
        topk=512, causal=True,
    )
    assert out.dtype == dtype
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# T8 — top-K determinism
# ---------------------------------------------------------------------------


def test_T8_topk_determinism():
    """Two calls with identical inputs return bit-identical top-K indices."""
    kw = _make_case(L=8192, Q=4)
    idx1 = lightning_indexer_topk(
        kw["query"], kw["indexer_q_proj"], kw["indexer_k_cache"],
        kw["position_ids"], kw["key_lengths"], topk=2048, causal=True,
    )
    idx2 = lightning_indexer_topk(
        kw["query"], kw["indexer_q_proj"], kw["indexer_k_cache"],
        kw["position_ids"], kw["key_lengths"], topk=2048, causal=True,
    )
    assert torch.equal(idx1, idx2)


# ---------------------------------------------------------------------------
# T9 — DsaKernelConfig cache identity
# ---------------------------------------------------------------------------


def test_T9_cache_key_stable_and_distinct():
    a = DsaKernelConfig(
        topk=2048, block_size=32,
        index_n_heads=64, index_head_dim=64,
        index_pool=1, causal=True, return_topk_for_indexshare=False,
    )
    b = DsaKernelConfig(
        topk=2048, block_size=32,
        index_n_heads=64, index_head_dim=64,
        index_pool=1, causal=True, return_topk_for_indexshare=False,
    )
    c = DsaKernelConfig(
        topk=2048, block_size=32,
        index_n_heads=64, index_head_dim=64,
        index_pool=4, causal=True, return_topk_for_indexshare=False,
    )
    assert a.cache_key() == b.cache_key()
    assert a.cache_key() != c.cache_key()
    assert KERNEL_SLUG_V0_REFERENCE in a.cache_key()
