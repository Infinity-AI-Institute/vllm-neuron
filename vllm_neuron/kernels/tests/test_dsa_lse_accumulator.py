# SPDX-License-Identifier: Apache-2.0
"""Tier-1 LSE-accumulator correctness gate for the DSA CPU reference.

Author: Trn2 reference-sweep 2026-08-27
Motivated by: `kernels/SGLANG-DSA-LSE-FIX-ANALYSIS-2026-08-28.md`

Why this gate exists
--------------------
SGLang's TileLang DSA kernel emits base-2 LSE internally but does NOT
expose a **final combined LSE** from its split-K decode combine — that
gap is why the DCP DSA feature PR (sgl-project/sglang#31821) is
"trtllm backends only". TRT-LLM's `ComputeLSEFromMD` returns LSE natively
so DCP's `cp_lse_ag_out_rs_mla` can consume it directly.

Any future NKI v1 DSA kernel that must (a) split-K across neuron cores,
(b) shard KV across TP/DCP-style groups, or (c) compose with any
cross-shard softmax combine has the same requirement: emit an LSE
tensor with a documented base convention, and honor the all-masked
sentinel so combines don't NaN-propagate.

This gate is the tier-1 (CPU, no device, no NKI compile) invariant that
future NKI split-K / cross-shard kernels must pass to be considered
correct against the CPU golden reference in `dsa_lightning_indexer.py`.

Gates
-----
    L0 -- LSE base convention: `LSE_BASE_CONVENTION == "natural"`
          (constant, not a function of inputs). The NKI author's contract.
    L1 -- LSE numeric ground truth: `lse` returned by
          `dsa_sparse_attention_forward(return_lse=True)` matches
          `torch.logsumexp(masked_scores, dim=K)` on an independent
          numpy oracle to fp32 rel tol 1e-5.
    L2 -- Split-recombine invariant (the exact TileLang DSA LSE-fix
          need): given two disjoint splits of the top-K indices
          (`topk_a`, `topk_b`), the LSE-weighted recombine of
          `(out_a, lse_a)` and `(out_b, lse_b)` bit-exactly reconstructs
          the single-pass `out_full` at topk = a + b. This is the
          precise math that `cp_lse_ag_out_rs_mla` performs across DCP
          ranks in SGLang PR #31821.
    L3 -- All-masked sentinel: if every gathered index is masked out
          (causal + key_length rejects all), `lse[b, q, h] == -inf`
          and `exp(lse - any_finite) == 0`. Same contract as SGLang
          `fixup_zero_kv_rows` (PR #31821).
    L4 -- Dtype: `lse` is fp32 regardless of input dtype (bf16/fp16
          inputs return fp32 LSE) so downstream combines have full
          precision.

Wall budget: < 5 s total, CPU-only.
Run with:
    py -3 -m pytest -q kernels/tests/test_dsa_lse_accumulator.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
KERNEL_DIR = HERE.parent
sys.path.insert(0, str(KERNEL_DIR))

from dsa_lightning_indexer import (      # noqa: E402
    LSE_BASE_CONVENTION,
    dsa_lightning_indexer_forward,
    dsa_sparse_attention_forward,
    full_attention_at_indices_reference,
    lightning_indexer_topk,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _make_case(
    *,
    B: int = 2,
    Q: int = 3,
    L: int = 512,
    H: int = 4,
    D: int = 32,
    H_idx: int = 4,
    D_idx: int = 16,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
):
    g = torch.Generator().manual_seed(seed)
    query = torch.randn(B, Q, H, D, generator=g, dtype=torch.float32).to(dtype)
    indexer_q_proj = torch.randn(
        H_idx, H * D, D_idx, generator=g, dtype=torch.float32
    ).to(dtype) / math.sqrt(H * D)
    indexer_k_cache = torch.randn(
        B, L, H_idx, D_idx, generator=g, dtype=torch.float32
    ).to(dtype)
    kv_k = torch.randn(B, L, H, D, generator=g, dtype=torch.float32).to(dtype)
    kv_v = torch.randn(B, L, H, D, generator=g, dtype=torch.float32).to(dtype)
    position_ids = torch.arange(L - Q, L).unsqueeze(0).expand(B, Q).to(torch.int64)
    key_lengths = torch.full((B,), L, dtype=torch.int64)
    return dict(
        query=query,
        indexer_q_proj=indexer_q_proj,
        indexer_k_cache=indexer_k_cache,
        kv_k=kv_k,
        kv_v=kv_v,
        position_ids=position_ids,
        key_lengths=key_lengths,
    )


# ---------------------------------------------------------------------------
# L0 -- Base-convention constant
# ---------------------------------------------------------------------------


def test_L0_lse_base_convention_is_natural():
    """The base convention must be a compile-time constant, not
    inputs-dependent. Any future NKI kernel emits LSE in this base.
    """
    assert LSE_BASE_CONVENTION == "natural", (
        "LSE_BASE_CONVENTION drifted -- if a device kernel switches to "
        "base-2, update the constant, the docstring, AND every combine "
        "site simultaneously (see SGLang PR #35045 for the class of "
        "silent-corruption bug caused by a mismatch)."
    )


# ---------------------------------------------------------------------------
# L1 -- Independent LSE oracle
# ---------------------------------------------------------------------------


def test_L1_lse_matches_naive_logsumexp_oracle():
    case = _make_case(L=512)
    topk = 128
    idx = lightning_indexer_topk(
        case["query"],
        case["indexer_q_proj"],
        case["indexer_k_cache"],
        case["position_ids"],
        case["key_lengths"],
        topk=topk,
    )
    _, lse = dsa_sparse_attention_forward(
        case["query"],
        case["kv_k"],
        case["kv_v"],
        idx,
        case["position_ids"],
        case["key_lengths"],
        topk=topk,
        return_lse=True,
    )
    # Independent numpy-style oracle: gather K rows, dot with Q, mask,
    # logsumexp over the K axis in fp32.
    B, Q, H, D = case["query"].shape
    scaling = 1.0 / math.sqrt(D)
    idx64 = idx.to(torch.int64)
    batch_idx = torch.arange(B).view(B, 1, 1).expand_as(idx64)
    k_sel = case["kv_k"][batch_idx, idx64].to(torch.float32)      # [B,Q,K,H,D]
    q = case["query"].unsqueeze(2).to(torch.float32)              # [B,Q,1,H,D]
    scores = (q * k_sel).sum(dim=-1) * scaling                    # [B,Q,K,H]
    # Mask by causal + key_lengths (case has key_lengths = L, causal on).
    q_pos = case["position_ids"].view(B, Q, 1).to(torch.int64)
    ok = (idx64 <= q_pos) & (idx64 < case["key_lengths"].view(B, 1, 1))
    scores = scores.masked_fill(~ok.unsqueeze(-1), float("-inf"))
    lse_oracle = torch.logsumexp(scores, dim=2)                    # [B,Q,H]

    err = (lse - lse_oracle).abs().max().item()
    denom = lse_oracle.abs().max().clamp_min(1e-8).item()
    rel = err / denom
    assert rel < 1e-5, (
        f"LSE oracle mismatch: rel={rel:.2e}, abs={err:.2e}. "
        "This is the LSE fix contract -- the returned LSE MUST be "
        "log(sum(exp(masked_scores))) in natural log."
    )


# ---------------------------------------------------------------------------
# L2 -- Split-recombine invariant (the actual TileLang DCP LSE fix)
# ---------------------------------------------------------------------------


def test_L2_split_lse_recombine_reconstructs_single_pass_output():
    """The precise math the LSE fix enables: two disjoint index shards
    each produce (out_shard, lse_shard); the LSE-weighted recombine
    reproduces the single-pass output at the union of indices.

    Reconstruction formula (`cp_lse_ag_out_rs_mla`):
        global_lse = logsumexp([lse_a, lse_b])
        w_a = exp(lse_a - global_lse); w_b = exp(lse_b - global_lse)
        out = w_a * out_a + w_b * out_b
    """
    case = _make_case(L=256, B=1, Q=2, H=2, D=16)
    topk_full = 64
    # Compute the full top-K once so the two shards use the same "true"
    # selected positions (this is how DCP works: same indexer output,
    # different rank-owned shards).
    idx_full = lightning_indexer_topk(
        case["query"],
        case["indexer_q_proj"],
        case["indexer_k_cache"],
        case["position_ids"],
        case["key_lengths"],
        topk=topk_full,
    )
    # Single-pass reference.
    out_full = full_attention_at_indices_reference(
        case["query"],
        case["kv_k"],
        case["kv_v"],
        idx_full,
        case["position_ids"],
        case["key_lengths"],
    )

    # Split idx along the K axis into two disjoint halves.
    K_half = topk_full // 2
    idx_a = idx_full[:, :, :K_half].contiguous()
    idx_b = idx_full[:, :, K_half:].contiguous()

    out_a, lse_a = dsa_sparse_attention_forward(
        case["query"], case["kv_k"], case["kv_v"], idx_a,
        case["position_ids"], case["key_lengths"],
        topk=K_half, return_lse=True,
    )
    out_b, lse_b = dsa_sparse_attention_forward(
        case["query"], case["kv_k"], case["kv_v"], idx_b,
        case["position_ids"], case["key_lengths"],
        topk=topk_full - K_half, return_lse=True,
    )

    # LSE-weighted recombine.
    stacked_lse = torch.stack([lse_a, lse_b], dim=0)              # [2, B, Q, H]
    global_lse = torch.logsumexp(stacked_lse, dim=0)              # [B, Q, H]
    w_a = torch.exp(lse_a - global_lse).unsqueeze(-1).to(torch.float32)
    w_b = torch.exp(lse_b - global_lse).unsqueeze(-1).to(torch.float32)
    recomb = (w_a * out_a.to(torch.float32) + w_b * out_b.to(torch.float32))

    err = (recomb - out_full.to(torch.float32)).abs().max().item()
    denom = out_full.abs().max().clamp_min(1e-8).item()
    rel = err / denom
    assert rel < 1e-5, (
        f"Split-recombine mismatch: rel={rel:.2e}, abs={err:.2e}. "
        "This is the correctness contract for cross-shard sparse-attn "
        "combines -- any future NKI kernel that splits softmax MUST "
        "reproduce this reconstruction bit-exactly."
    )


# ---------------------------------------------------------------------------
# L3 -- All-masked sentinel
# ---------------------------------------------------------------------------


def test_L3_all_masked_row_returns_lse_neg_inf_sentinel():
    """If every gathered index is causal/pad-invalid for a row, LSE
    must be -inf so a downstream combine's `exp(lse - global_lse)`
    contributes zero mass. Same behavior as SGLang `fixup_zero_kv_rows`.
    """
    B, Q, H, D, L = 1, 1, 2, 8, 32
    query = torch.randn(B, Q, H, D)
    kv_k = torch.randn(B, L, H, D)
    kv_v = torch.randn(B, L, H, D)
    # Query at position 0, causal on: only index 0 is valid. Force all
    # picked indices to L-1 so ALL are masked (index > q_pos).
    position_ids = torch.zeros(B, Q, dtype=torch.int64)
    key_lengths = torch.full((B,), L, dtype=torch.int64)
    topk = 4
    idx = torch.full((B, Q, topk), L - 1, dtype=torch.int32)     # all invalid

    out, lse = dsa_sparse_attention_forward(
        query, kv_k, kv_v, idx, position_ids, key_lengths,
        topk=topk, return_lse=True,
    )
    # Output is forced to zero for all-masked rows (existing contract).
    assert torch.all(out == 0.0)
    # LSE must be -inf so cross-shard combines cleanly discard this row.
    assert torch.isneginf(lse).all(), (
        f"All-masked row must return lse=-inf, got: {lse}"
    )
    # exp(-inf - any_finite) must be exactly 0 (no NaN).
    weights = torch.exp(lse - torch.zeros_like(lse))
    assert torch.all(weights == 0.0)
    assert not torch.isnan(weights).any()


# ---------------------------------------------------------------------------
# L4 -- Dtype invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_L4_lse_is_always_fp32(dtype):
    case = _make_case(L=64, dtype=dtype)
    idx = lightning_indexer_topk(
        case["query"], case["indexer_q_proj"], case["indexer_k_cache"],
        case["position_ids"], case["key_lengths"], topk=16,
    )
    _, lse = dsa_sparse_attention_forward(
        case["query"], case["kv_k"], case["kv_v"], idx,
        case["position_ids"], case["key_lengths"],
        topk=16, return_lse=True,
    )
    assert lse.dtype == torch.float32, (
        f"lse dtype={lse.dtype} but downstream combines require fp32 "
        "for stability (SGLang PR #31821 pre-registers lse_buf as "
        "torch.float32 in symmetric memory)."
    )


# ---------------------------------------------------------------------------
# One-shot forward plumbing check (indexer + attention)
# ---------------------------------------------------------------------------


def test_one_shot_forward_return_lse_shapes():
    case = _make_case(L=128, B=1, Q=2, H=2, D=16)
    result = dsa_lightning_indexer_forward(
        case["query"], case["indexer_q_proj"], case["indexer_k_cache"],
        case["kv_k"], case["kv_v"],
        case["position_ids"], case["key_lengths"],
        topk=32, return_lse=True,
    )
    assert len(result) == 3
    attn, topk_ret, lse = result
    assert topk_ret is None                          # not asked for indexshare
    B, Q, H, D = case["query"].shape
    assert attn.shape == (B, Q, H, D)
    assert lse.shape == (B, Q, H)
    assert lse.dtype == torch.float32

    # With both return_topk_for_indexshare and return_lse:
    result2 = dsa_lightning_indexer_forward(
        case["query"], case["indexer_q_proj"], case["indexer_k_cache"],
        case["kv_k"], case["kv_v"],
        case["position_ids"], case["key_lengths"],
        topk=32,
        return_topk_for_indexshare=True,
        return_lse=True,
    )
    assert len(result2) == 3
    attn2, topk2, lse2 = result2
    assert topk2.shape == (B, Q, 32)
    assert lse2.shape == (B, Q, H)
