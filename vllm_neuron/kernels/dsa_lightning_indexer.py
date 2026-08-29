# SPDX-License-Identifier: Apache-2.0
"""DSA (Deep Sparse Attention) Lightning Indexer — reference implementation.

This module is the **campaign-canonical CPU golden reference** for the
DSA Lightning Indexer + sparse gather + sparse attention chain used by:

    - GLM 5.2       (index_topk=2048, index_pool=1, all 78 layers)
    - GLM 5.3 Flash (index_topk=2048, index_pool=4, 11 of 45 layers)
    - DeepSeek-V4-Flash-Base (index_topk=512,  index_pool=1, 21 of 43 CSA layers)

**Contract with the campaign.**
    Downstream lanes gate against `dsa_lightning_indexer_forward` and
    `dsa_sparse_attention_forward` in this file. When a future NKI kernel
    lands, it is bit-exact against this reference at bf16 tolerance
    (~1e-3 rel) or a defect is filed against the kernel, not this file.

**Why CPU-only today.**
    Per operator's 2026-08-27 "do NOT ship a broken kernel" rule, a real
    NKI kernel requires SBUF sizing, descriptor coalescing, and per-tile
    validation against a live device — none of which are safe to
    hand-wave in a single-turn deliverable. The NKI skeleton at the
    bottom of this file (`_nki_kernel_stub_...`) is a design placeholder
    only; every function raises `NotImplementedError` until an author
    with device access closes the GAP markers in
    `kernels/NKI-DSA-SPARSE-ATTENTION-SCAFFOLD-2026-08-27.md`.

**Kernel slug.**
    `nki_v0_reference_lightning_indexer` — the "v0" prefix marks this as
    the *reference* implementation, not a device kernel. When device kernels
    land they get `nki_v1_lightning_indexer` (or successor). Both must
    match this v0 to within bf16 rel tolerance.

    Set `DSA_KERNEL_IMPL=nki_v0_reference_lightning_indexer` in `model.env`
    to select this path; that env var participates in the compile-cache
    graph_id so a swap is a fresh cache line, not a silent replay.

**Consumers today:**
    1. `harness-v2/staging/reference-sweep-20260826T2150Z/lanes/glm-5-2-5-3/
        tests/test_03_dsa_path_activation.py` — telemetry contract
        against this reference at S=2048, topk=2048.
    2. `harness-v2/staging/reference-sweep-20260826T2150Z/lanes/deepseek-v4-flash/
        tests/test_kernel_correctness_dsv4_flash.py` — 10-tok exact gate
        against this reference at S=8192, topk=512.
    3. `harness-v2/staging/reference-sweep-20260826T2150Z/kernels/tests/
        test_dsa_lightning_indexer_correctness.py` — Tier-1 kernel-tier
        correctness across K in {2048, 4096, 8192, 16384, 32768} and
        topk in {2048, 4096}. Author's own gate.

**References.**
    - `kernels/NKI-DSA-SPARSE-ATTENTION-SCAFFOLD-2026-08-27.md` — design
      doc this file implements.
    - `MLA-VS-DSA-KERNEL-VERIFICATION-2026-08-27.md` — cross-model
      applicability + rationale for making DSA the highest-leverage
      shared kernel.
    - `apuroop/glm5-2-enablement-v2@cdc0f435c8:
        third_party/vllm-neuron@a57a4e82:vllm_neuron/model/glm52_moe_dsa/
        {indexer.py,attention.py,mla.py}` — the PyTorch reference from
      which the CPU behavior is ported. Cross-check semantics before
      shipping any NKI kernel against this file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Public identity
# ---------------------------------------------------------------------------

KERNEL_SLUG_V0_REFERENCE = "nki_v0_reference_lightning_indexer"
"""Cache identity for the reference path. Participates in graph_id."""

KERNEL_SLUG_V1_NKI = "nki_v1_lightning_indexer"
"""Cache identity reserved for the first landed NKI device kernel."""


# ---------------------------------------------------------------------------
# LSE base convention (see `kernels/SGLANG-DSA-LSE-FIX-ANALYSIS-2026-08-28.md`)
# ---------------------------------------------------------------------------
#
# This CPU reference emits LSE in the **natural-log** convention when
# `return_lse=True` is passed to `dsa_sparse_attention_forward` or
# `dsa_lightning_indexer_forward`. That is:
#
#     lse[b, q, h] = log( sum_k exp( scores[b, q, k, h] ) )    (base-e)
#
# where `scores` are the pre-softmax scaled scores (`(Q @ K.T) / sqrt(D)`)
# after causal + key-length masking (out-of-range positions set to -inf).
#
# This mirrors SGLang's `flashmla`/`cutedsl_mla` LSE convention (their
# `is_mla_dcp_lse_base_on_e()` returns True for those backends). The
# alternative base-2 convention that TRT-LLM DSA and TileLang DSA use
# internally is FASTER on GPU (native `ex2.approx`), but there is no such
# intrinsic on Trainium2 — the natural-log convention lowers cleanly to
# `nl.exp`/`nl.log` in NKI. See §5 of the SGLang LSE fix analysis for
# rationale.
#
# Any future NKI v1 device kernel that emits LSE MUST match this base
# convention or file a defect against `test_dsa_lse_accumulator`.
LSE_BASE_CONVENTION = "natural"   # "natural" or "base2"


@dataclass(frozen=True)
class DsaKernelConfig:
    """Immutable shape config baked into an NKI compile.

    Fields mirror `dsa_lightning_indexer_forward`'s `nl.constant` args in
    the scaffold's §3.1. This dataclass is the ONLY safe way to identify
    a compiled kernel — two DsaKernelConfigs with the same field values
    are guaranteed to lower to the same NEFF (given the same neuronx-cc).
    """

    topk: int
    block_size: int
    index_n_heads: int
    index_head_dim: int
    index_pool: int
    causal: bool
    return_topk_for_indexshare: bool
    # `impl` participates in the cache identity but is not a shape.
    impl: str = KERNEL_SLUG_V0_REFERENCE

    def cache_key(self) -> str:
        """Compile-cache subkey — append to model graph_id."""
        return (
            f"{self.impl}|topk={self.topk}|block={self.block_size}"
            f"|H_idx={self.index_n_heads}|D_idx={self.index_head_dim}"
            f"|pool={self.index_pool}|causal={int(self.causal)}"
            f"|return_topk={int(self.return_topk_for_indexshare)}"
        )


# ---------------------------------------------------------------------------
# Core: Lightning Indexer (top-K position selection)
# ---------------------------------------------------------------------------


def _causal_mask_scores(
    scores: torch.Tensor,
    position_ids: torch.Tensor,
    key_lengths: torch.Tensor,
) -> torch.Tensor:
    """Apply causal + key-length mask to raw indexer scores.

    Positions strictly greater than the query position (causal violation),
    OR positions greater-or-equal to `key_lengths[b]` (padding), are set to
    -inf so the following top-K skips them.

    scores        [B, Q, L]  float32
    position_ids  [B, Q]     int64  — absolute position of each query
    key_lengths   [B]        int64  — valid K length per batch
    """
    B, Q, L = scores.shape
    device = scores.device

    key_idx = torch.arange(L, device=device).view(1, 1, L)     # [1,1,L]
    q_pos = position_ids.view(B, Q, 1).to(torch.int64)          # [B,Q,1]
    causal_ok = key_idx <= q_pos                                # [B,Q,L] bool

    valid_len = key_lengths.view(B, 1, 1).to(torch.int64)       # [B,1,1]
    len_ok = key_idx < valid_len                                 # [B,1,L] bool

    ok = causal_ok & len_ok                                     # [B,Q,L] bool
    return scores.masked_fill(~ok, float("-inf"))


def lightning_indexer_scores(
    query: torch.Tensor,          # [B, Q, H, D]  — the main-attention query
    indexer_q_proj: torch.Tensor, # [H_idx, H*D, D_idx]  or projection weights
    indexer_k_cache: torch.Tensor,# [B, L, H_idx, D_idx]
    *,
    index_pool: int = 1,
    pool_weights: Optional[torch.Tensor] = None,  # [index_pool] fp32
) -> torch.Tensor:
    """Compute the indexer's dense scores over ALL L positions.

    Reference implementation only — no top-K yet, no masking yet. The
    output is `[B, Q, L]` fp32 scores; masking + top-K happens downstream.

    `indexer_q_proj` projects the main-attention query into the smaller
    (H_idx, D_idx) indexer space. `indexer_k_cache` holds the indexer-side
    keys, which in the paged design are computed at KV-cache write time.

    IndexPool: when `index_pool > 1`, `indexer_k_cache` is assumed to
    already carry `index_pool` sub-vectors per (b, l, h) position, i.e.
    the stored shape is `[B, L, H_idx * index_pool, D_idx]` and this
    function collapses the pool axis via `pool_weights` before scoring.
    That collapse is exactly `dsa_index_pool_projection` in scaffold §3.3.
    """
    B, Q, H, D = query.shape
    L = indexer_k_cache.shape[1]
    H_idx = indexer_k_cache.shape[2] // max(index_pool, 1)
    D_idx = indexer_k_cache.shape[3]

    # Project query into indexer space.
    q_flat = query.reshape(B, Q, H * D).to(torch.float32)
    # indexer_q_proj: [H_idx, H*D, D_idx]  → contract on H*D
    # q_idx: [B, Q, H_idx, D_idx]
    q_idx = torch.einsum("bqf,hfd->bqhd", q_flat, indexer_q_proj.to(torch.float32))

    # Collapse the pool if needed.
    if index_pool > 1:
        if pool_weights is None:
            pool_weights = torch.full(
                (index_pool,), 1.0 / index_pool, dtype=torch.float32,
                device=indexer_k_cache.device,
            )
        k_pooled = _apply_index_pool(indexer_k_cache, index_pool, pool_weights)
    else:
        k_pooled = indexer_k_cache.to(torch.float32)

    # Score: [B, Q, L] = sum over (H_idx, D_idx) of q_idx * k_pooled
    scores = torch.einsum("bqhd,blhd->bql", q_idx, k_pooled.to(torch.float32))

    # Scale by 1/sqrt(D_idx) so scores are unit-variance.
    scores = scores * (1.0 / math.sqrt(D_idx))
    return scores


def _apply_index_pool(
    indexer_k_cache: torch.Tensor,  # [B, L, H_idx*index_pool, D_idx]
    index_pool: int,
    pool_weights: torch.Tensor,     # [index_pool] fp32
) -> torch.Tensor:
    """Weighted-sum collapse of the pool axis. See scaffold §3.3."""
    B, L, HP, D_idx = indexer_k_cache.shape
    assert HP % index_pool == 0, "pool dim must divide H_idx*index_pool"
    H_idx = HP // index_pool
    reshaped = indexer_k_cache.reshape(B, L, H_idx, index_pool, D_idx).to(torch.float32)
    weights = pool_weights.to(torch.float32).view(1, 1, 1, index_pool, 1)
    pooled = (reshaped * weights).sum(dim=3)   # [B, L, H_idx, D_idx]
    return pooled


def dsa_index_pool_projection(
    index_keys: torch.Tensor,   # [B, L, index_pool * H_idx, D_idx]
    pool_weights: torch.Tensor, # [index_pool]
    *,
    index_pool: int,
) -> torch.Tensor:
    """Public alias for the IndexPool weighted-sum. Scaffold §3.3 API."""
    return _apply_index_pool(index_keys, index_pool, pool_weights)


def lightning_indexer_topk(
    query: torch.Tensor,           # [B, Q, H, D]
    indexer_q_proj: torch.Tensor,  # [H_idx, H*D, D_idx]
    indexer_k_cache: torch.Tensor, # [B, L, H_idx*index_pool, D_idx]
    position_ids: torch.Tensor,    # [B, Q] int64
    key_lengths: torch.Tensor,     # [B]    int64
    *,
    topk: int,
    causal: bool = True,
    index_pool: int = 1,
    pool_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return `[B, Q, topk]` int32 top-K indices per query.

    Ties broken by lower index first (torch.topk default). Invalid
    positions (masked to -inf) never appear in the top-K IF `topk` is
    less than or equal to the number of valid positions; when `topk`
    exceeds valid count, invalid indices ARE returned and the caller's
    downstream softmax mask (via `key_lengths`) suppresses their
    contribution. Callers must not read the returned indices without
    passing them through the same key_lengths mask.
    """
    B, Q = position_ids.shape
    scores = lightning_indexer_scores(
        query,
        indexer_q_proj,
        indexer_k_cache,
        index_pool=index_pool,
        pool_weights=pool_weights,
    )
    if causal:
        scores = _causal_mask_scores(scores, position_ids, key_lengths)

    # torch.topk returns (values, indices) sorted descending.
    _, indices = torch.topk(scores, k=topk, dim=-1, largest=True, sorted=True)
    return indices.to(torch.int32)


# ---------------------------------------------------------------------------
# Sparse gather (paged KV read at selected positions)
# ---------------------------------------------------------------------------


def sparse_gather_kv(
    kv_cache: torch.Tensor,        # [B, L, H, D]
    topk_indices: torch.Tensor,    # [B, Q, topk] int32
) -> torch.Tensor:
    """Gather K or V rows at `topk_indices`.

    Reference implementation — a bare `torch.gather`. The NKI-side win
    is DMA descriptor coalescing on this exact pattern (per scaffold §4.3
    and `NKI-DMA-COALESCING-SCAFFOLD-2026-08-27.md`); the CPU reference
    is O(topk) descriptors per query, the NKI implementation should
    coalesce to O(topk / block_size).
    """
    B, L, H, D = kv_cache.shape
    _, Q, K = topk_indices.shape
    idx = topk_indices.to(torch.int64).clamp_(0, L - 1)         # [B, Q, K]
    # Expand for gather over dim=1 (the L axis).
    idx_exp = idx.view(B, Q, K, 1, 1).expand(B, Q, K, H, D)      # [B,Q,K,H,D]
    kv_exp = kv_cache.view(B, L, 1, H, D).expand(B, L, Q, H, D)
    # gather along dim=1 of kv_exp gives [B, K_selected, Q, H, D] — but we
    # want per-query independent gathers; do it with advanced indexing.
    batch_idx = torch.arange(B, device=kv_cache.device).view(B, 1, 1).expand(B, Q, K)
    selected = kv_cache[batch_idx, idx]                          # [B, Q, K, H, D]
    return selected


# ---------------------------------------------------------------------------
# Sparse attention (softmax over selected K/V rows)
# ---------------------------------------------------------------------------


def dsa_sparse_attention_forward(
    query: torch.Tensor,          # [B, Q, H, D]
    kv_cache_k: torch.Tensor,     # [B, L, H, D]
    kv_cache_v: torch.Tensor,     # [B, L, H, D]
    topk_indices: torch.Tensor,   # [B, Q, topk] int32
    position_ids: torch.Tensor,   # [B, Q] int64
    key_lengths: torch.Tensor,    # [B]    int64
    *,
    topk: int,
    scaling: Optional[float] = None,
    causal: bool = True,
    return_lse: bool = False,
):
    """Softmax attention restricted to the gathered `topk` rows.

    Numeric semantics:
      * scale = 1/sqrt(D) unless overridden
      * causal + key_lengths mask applied to the gathered positions;
        gathered positions that violate causal OR are >= key_lengths[b]
        are set to -inf pre-softmax
      * softmax is numerically stable (max-subtract)

    LSE plumbing (opt-in via `return_lse=True`, see file header):
      * Returns `(out, lse)` where `lse[B, Q, H]` fp32 is the
        natural-log log-sum-exp of the masked pre-softmax scores:
          lse = log( sum_k exp( scores[b, q, k, h] ) )
      * All-masked rows (indexer selected only causal/pad-invalid
        positions) return `lse = -inf` — sentinel that makes any
        downstream cross-shard combine treat this shard's row as
        zero-mass (`exp(-inf - global_lse) = 0`). This mirrors
        `fixup_zero_kv_rows` in SGLang PR #31821.
      * This is the correctness contract any future NKI split-K /
        DCP / cross-rank kernel must uphold. See
        `SGLANG-DSA-LSE-FIX-ANALYSIS-2026-08-28.md`.
    """
    B, Q, H, D = query.shape
    _, _, K = topk_indices.shape
    if scaling is None:
        scaling = 1.0 / math.sqrt(D)

    # Gather K and V per selected index.
    k_sel = sparse_gather_kv(kv_cache_k, topk_indices)   # [B, Q, K, H, D]
    v_sel = sparse_gather_kv(kv_cache_v, topk_indices)   # [B, Q, K, H, D]

    q = query.unsqueeze(2).to(torch.float32)              # [B, Q, 1, H, D]
    k = k_sel.to(torch.float32)                           # [B, Q, K, H, D]
    v = v_sel.to(torch.float32)

    # Scores: [B, Q, K, H] = sum_D q * k
    scores = (q * k).sum(dim=-1) * scaling               # [B, Q, K, H]

    # Build the mask over the K axis. Same rule as the indexer.
    device = query.device
    q_pos = position_ids.view(B, Q, 1).to(torch.int64)   # [B,Q,1]
    idx_int = topk_indices.to(torch.int64)                # [B,Q,K]

    valid_len = key_lengths.view(B, 1, 1).to(torch.int64) # [B,1,1]
    len_ok = idx_int < valid_len
    if causal:
        causal_ok = idx_int <= q_pos
        ok = causal_ok & len_ok
    else:
        ok = len_ok

    scores = scores.masked_fill(~ok.unsqueeze(-1), float("-inf"))

    # Guard: rows with ZERO valid positions produce all -inf, which the
    # softmax below turns into NaN. This can happen at token 0 with
    # topk=1 IF the indexer's single choice was masked out — a real
    # correctness escape hatch. Detect and replace with zero output
    # (attention contributes nothing) rather than NaN-propagate.
    all_neg_inf = (~ok).all(dim=-1)                       # [B, Q]
    if all_neg_inf.any():
        # Replace those rows' scores with a single sentinel 0.0 in slot 0
        # so softmax returns a one-hot on the (masked) row 0; the
        # gathered V at that row is still masked-in below.
        pass  # softmax will still be all zeros; we handle post-softmax.

    # Numerically-stable softmax over K.
    scores_max = scores.detach().amax(dim=2, keepdim=True)
    scores_max = torch.where(
        torch.isfinite(scores_max), scores_max, torch.zeros_like(scores_max)
    )
    scores_shifted = scores - scores_max
    exp = torch.exp(scores_shifted)
    # Zero out the -inf rows properly.
    exp = torch.where(torch.isfinite(scores), exp, torch.zeros_like(exp))
    denom = exp.sum(dim=2, keepdim=True)
    # Guard against all-masked queries -> denom=0 -> nan.
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    attn = exp / denom                                    # [B, Q, K, H]

    # Weighted sum of V:  out = sum_K attn * v
    out = (attn.unsqueeze(-1) * v).sum(dim=2)             # [B, Q, H, D]

    # Force all-masked rows to zero output, not the softmax fallback.
    if all_neg_inf.any():
        out = out.masked_fill(all_neg_inf.view(B, Q, 1, 1), 0.0)

    if return_lse:
        # LSE (natural log) = scores_max + log(sum(exp(scores - scores_max))).
        # `scores_max` is the finite-guarded per-row max ([B, Q, 1, H]);
        # `denom` is the guarded sum-of-exps in the same shape. For all-
        # finite (non-degenerate) rows this is the exact fp32 LSE.
        # For all-masked rows we forced denom=1 above -> log(denom)=0 and
        # scores_max=0, which would yield lse=0. Override with -inf so a
        # cross-shard combine treats this row as zero-mass (matches
        # SGLang `fixup_zero_kv_rows` at PR #31821).
        lse = (scores_max + torch.log(denom)).squeeze(2)   # [B, Q, H]
        if all_neg_inf.any():
            lse = lse.masked_fill(
                all_neg_inf.unsqueeze(-1),
                float("-inf"),
            )
        return out.to(query.dtype), lse
    return out.to(query.dtype)


# ---------------------------------------------------------------------------
# One-shot forward: indexer + sparse attention
# ---------------------------------------------------------------------------


def dsa_lightning_indexer_forward(
    query: torch.Tensor,             # [B, Q, H, D]
    indexer_q_proj: torch.Tensor,    # [H_idx, H*D, D_idx]
    indexer_k_cache: torch.Tensor,   # [B, L, H_idx*index_pool, D_idx]
    kv_cache_k: torch.Tensor,        # [B, L, H, D]
    kv_cache_v: torch.Tensor,        # [B, L, H, D]
    position_ids: torch.Tensor,      # [B, Q]  int64
    key_lengths: torch.Tensor,       # [B]     int64
    *,
    topk: int,
    index_pool: int = 1,
    pool_weights: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    causal: bool = True,
    return_topk_for_indexshare: bool = False,
    return_lse: bool = False,
):
    """One-shot: indexer top-K -> sparse gather -> sparse attention.

    Matches scaffold §3.1's API exactly. See file docstring for cache
    identity, semantics, and consumers.

    Return shape:
      * default:                     (attn, None)
      * return_topk_for_indexshare:  (attn, topk_indices)
      * return_lse:                  (attn, None, lse)  -- 3-tuple
      * both:                        (attn, topk_indices, lse)

    See `dsa_sparse_attention_forward` for LSE semantics (natural log,
    all-masked rows -> -inf sentinel).
    """
    topk_indices = lightning_indexer_topk(
        query,
        indexer_q_proj,
        indexer_k_cache,
        position_ids,
        key_lengths,
        topk=topk,
        causal=causal,
        index_pool=index_pool,
        pool_weights=pool_weights,
    )
    attn_result = dsa_sparse_attention_forward(
        query,
        kv_cache_k,
        kv_cache_v,
        topk_indices,
        position_ids,
        key_lengths,
        topk=topk,
        scaling=scaling,
        causal=causal,
        return_lse=return_lse,
    )
    if return_lse:
        attn, lse = attn_result
        topk_ret = topk_indices if return_topk_for_indexshare else None
        return attn, topk_ret, lse
    if return_topk_for_indexshare:
        return attn_result, topk_indices
    return attn_result, None


# ---------------------------------------------------------------------------
# Naive full-attention reference (for the correctness gate at topk >= L)
# ---------------------------------------------------------------------------


def full_attention_reference(
    query: torch.Tensor,          # [B, Q, H, D]
    kv_cache_k: torch.Tensor,     # [B, L, H, D]
    kv_cache_v: torch.Tensor,     # [B, L, H, D]
    position_ids: torch.Tensor,   # [B, Q] int64
    key_lengths: torch.Tensor,    # [B]    int64
    *,
    scaling: Optional[float] = None,
    causal: bool = True,
) -> torch.Tensor:
    """Dense full-attention. Used ONLY by the T1 invariant test that
    asserts sparse-vs-full equivalence at topk >= L.
    """
    B, Q, H, D = query.shape
    _, L, _, _ = kv_cache_k.shape
    if scaling is None:
        scaling = 1.0 / math.sqrt(D)

    q = query.to(torch.float32)
    k = kv_cache_k.to(torch.float32)
    v = kv_cache_v.to(torch.float32)

    # Scores: [B, Q, L, H]
    scores = torch.einsum("bqhd,blhd->bqlh", q, k) * scaling

    device = query.device
    key_idx = torch.arange(L, device=device).view(1, 1, L, 1)
    q_pos = position_ids.view(B, Q, 1, 1).to(torch.int64)
    valid_len = key_lengths.view(B, 1, 1, 1).to(torch.int64)
    len_ok = key_idx < valid_len
    if causal:
        causal_ok = key_idx <= q_pos
        ok = causal_ok & len_ok
    else:
        ok = len_ok
    scores = scores.masked_fill(~ok, float("-inf"))

    scores_max = scores.detach().amax(dim=2, keepdim=True)
    scores_max = torch.where(
        torch.isfinite(scores_max), scores_max, torch.zeros_like(scores_max)
    )
    scores_shifted = scores - scores_max
    exp = torch.exp(scores_shifted)
    exp = torch.where(torch.isfinite(scores), exp, torch.zeros_like(exp))
    denom = exp.sum(dim=2, keepdim=True)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    attn = exp / denom                                # [B, Q, L, H]

    out = torch.einsum("bqlh,blhd->bqhd", attn, v)
    return out.to(query.dtype)


# ---------------------------------------------------------------------------
# Gated-indices reference — for testing sparse-attention independently
# ---------------------------------------------------------------------------


def full_attention_at_indices_reference(
    query: torch.Tensor,          # [B, Q, H, D]
    kv_cache_k: torch.Tensor,     # [B, L, H, D]
    kv_cache_v: torch.Tensor,     # [B, L, H, D]
    topk_indices: torch.Tensor,   # [B, Q, topk] int32
    position_ids: torch.Tensor,   # [B, Q]       int64
    key_lengths: torch.Tensor,    # [B]          int64
    *,
    scaling: Optional[float] = None,
    causal: bool = True,
) -> torch.Tensor:
    """Independent softmax-over-selected-indices reference — matches the
    formula `torch.softmax(Q @ K.T)[:, topk_indices]` from the operator
    prompt when position_ids and key_lengths permit all indices.

    This exists so the sparse-attention correctness test can be authored
    without threading through the indexer's top-K. It is functionally
    equivalent to `dsa_sparse_attention_forward` but independently coded
    to catch bugs in either.
    """
    B, Q, H, D = query.shape
    if scaling is None:
        scaling = 1.0 / math.sqrt(D)

    # Materialise selected K, V per query.
    idx = topk_indices.to(torch.int64)
    batch_idx = torch.arange(B, device=query.device).view(B, 1, 1).expand_as(idx)
    k_sel = kv_cache_k[batch_idx, idx]      # [B, Q, K, H, D]
    v_sel = kv_cache_v[batch_idx, idx]      # [B, Q, K, H, D]

    q = query.unsqueeze(2).to(torch.float32)
    k = k_sel.to(torch.float32)
    v = v_sel.to(torch.float32)
    scores = (q * k).sum(dim=-1) * scaling  # [B, Q, K, H]

    q_pos = position_ids.view(B, Q, 1).to(torch.int64)
    valid_len = key_lengths.view(B, 1, 1).to(torch.int64)
    len_ok = idx < valid_len
    if causal:
        causal_ok = idx <= q_pos
        ok = causal_ok & len_ok
    else:
        ok = len_ok
    scores = scores.masked_fill(~ok.unsqueeze(-1), float("-inf"))

    scores_max = scores.detach().amax(dim=2, keepdim=True)
    scores_max = torch.where(
        torch.isfinite(scores_max), scores_max, torch.zeros_like(scores_max)
    )
    exp = torch.exp(scores - scores_max)
    exp = torch.where(torch.isfinite(scores), exp, torch.zeros_like(exp))
    denom = exp.sum(dim=2, keepdim=True)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    attn = exp / denom                       # [B, Q, K, H]
    out = (attn.unsqueeze(-1) * v).sum(dim=2)  # [B, Q, H, D]
    return out.to(query.dtype)


# ---------------------------------------------------------------------------
# Analytical bounds (for the speed-tier gate) — DEVICE-FREE
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelResourceBounds:
    """Analytical resource ceiling for one call to the fused kernel.

    ALL fields are derived from the config + shapes, not measured. Speed
    gates check that these ceilings are honoured; a real device profile
    at the knee tightens them into measured floors (per profile-at-knee
    discipline). This dataclass is the pre-fire analytical gate.
    """

    # Bytes / MiB
    sbuf_bytes_indexer_kv_resident: int
    sbuf_bytes_topk_workspace: int
    sbuf_bytes_gather_workspace: int
    sbuf_bytes_attention_workspace: int
    sbuf_bytes_total: int
    sbuf_bytes_ceiling: int          # Trn2 per-NC SBUF ~24 MiB after weights
    sbuf_fits: bool                  # sbuf_bytes_total <= sbuf_bytes_ceiling
    sbuf_headroom_bytes: int

    # DMA descriptors
    descriptors_per_query_naive: int    # 1 per topk position
    descriptors_per_query_coalesced: int # 1 per (topk // block_size)
    descriptor_reduction_factor: float   # naive / coalesced
    descriptor_cache_slots_used: int
    descriptor_cache_slots_ceiling: int  # ~4096

    # Compute floor
    flops_indexer: int    # 2*L*H_idx*D_idx + L*log(L)   (top-K sort)
    flops_sparse_attn: int  # 4*topk*H*D
    cycles_floor_per_token: int  # very optimistic — TensorE peak
    cycles_ceiling_per_token: int  # pessimistic — full serial


def analytical_bounds(
    *,
    B: int,
    Q: int,
    L: int,
    H: int,
    D: int,
    H_idx: int,
    D_idx: int,
    topk: int,
    block_size: int,
    index_pool: int = 1,
    dtype_bytes: int = 2,   # bf16 default
) -> KernelResourceBounds:
    """Compute the analytical SBUF/DMA/cycles envelope for one kernel call.

    Numbers match the scaffold §4 arithmetic. Ceilings are the campaign's
    documented Trn2 partition limits.
    """
    # ----- SBUF pieces -------------------------------------------------
    # Indexer KV resident window: keep `sbuf_indexer_window_blocks` blocks
    # of the indexer cache SBUF-resident during the top-K pass. This is
    # the smallest tile allowed by the descriptor-coalescing scaffold.
    sbuf_indexer_window_blocks = 96   # scaffold §4.1
    sbuf_indexer_bytes = (
        sbuf_indexer_window_blocks * block_size * H_idx * D_idx * dtype_bytes
    )
    # Top-K workspace: `L` fp32 scores per (b, q) held during selection.
    # We stream in `min(L, 16384)` slots at a time to stay under the
    # `nc_find_index8` cap. Workspace is the STREAM tile, not the full L.
    topk_stream_width = min(L, 16384)   # nc_find_index8 partition cap
    sbuf_topk_bytes = topk_stream_width * 4   # fp32
    # Gather workspace: one block of K + V held per Q-tile.
    q_tile = min(Q, 16)   # scaffold §4.3
    sbuf_gather_bytes = 2 * q_tile * block_size * H * D * dtype_bytes
    # Attention workspace: one Q-tile of the K-fold accumulator.
    sbuf_attn_bytes = q_tile * topk * H * dtype_bytes    # scores at fp16
    sbuf_total = (
        sbuf_indexer_bytes + sbuf_topk_bytes + sbuf_gather_bytes + sbuf_attn_bytes
    )
    sbuf_ceiling = 24 * 1024 * 1024    # ~24 MiB per NC after weights
    sbuf_fits = sbuf_total <= sbuf_ceiling
    sbuf_headroom = sbuf_ceiling - sbuf_total

    # ----- DMA descriptors --------------------------------------------
    desc_naive = topk                                     # 1 per position
    desc_coalesced = max(1, topk // block_size)           # 1 per block
    desc_reduction = desc_naive / max(desc_coalesced, 1)
    # Total across a q_tile of 16 queries: 16 * desc_coalesced. Compare
    # to descriptor cache (~4096 slots per scaffold §4.3).
    desc_slots_used = q_tile * desc_coalesced
    desc_slots_ceiling = 4096

    # ----- Compute floor ----------------------------------------------
    flops_indexer_kv = 2 * L * H_idx * D_idx
    flops_indexer_sort = int(L * math.log2(max(L, 2)))
    flops_index = flops_indexer_kv + flops_indexer_sort
    flops_sparse_attn = 4 * topk * H * D
    # Cycles floor: TensorE @ 8k FLOPs/cycle (Trn2 peak), one NC.
    tensor_e_flops_per_cycle = 8192
    cycles_floor = max(1, (flops_index + flops_sparse_attn) // tensor_e_flops_per_cycle)
    cycles_ceiling = cycles_floor * 4    # pessimistic serial factor

    return KernelResourceBounds(
        sbuf_bytes_indexer_kv_resident=sbuf_indexer_bytes,
        sbuf_bytes_topk_workspace=sbuf_topk_bytes,
        sbuf_bytes_gather_workspace=sbuf_gather_bytes,
        sbuf_bytes_attention_workspace=sbuf_attn_bytes,
        sbuf_bytes_total=sbuf_total,
        sbuf_bytes_ceiling=sbuf_ceiling,
        sbuf_fits=sbuf_fits,
        sbuf_headroom_bytes=sbuf_headroom,
        descriptors_per_query_naive=desc_naive,
        descriptors_per_query_coalesced=desc_coalesced,
        descriptor_reduction_factor=desc_reduction,
        descriptor_cache_slots_used=desc_slots_used,
        descriptor_cache_slots_ceiling=desc_slots_ceiling,
        flops_indexer=flops_index,
        flops_sparse_attn=flops_sparse_attn,
        cycles_floor_per_token=cycles_floor,
        cycles_ceiling_per_token=cycles_ceiling,
    )


# ---------------------------------------------------------------------------
# NKI kernel skeleton (DEFERRED - do NOT ship as-is)
# ---------------------------------------------------------------------------


def _nki_kernel_stub_dsa_lightning_indexer_forward(*args, **kwargs):
    """Placeholder for the NKI device kernel.

    A real implementation must, at minimum, close every GAP in
    `kernels/NKI-DSA-SPARSE-ATTENTION-SCAFFOLD-2026-08-27.md` §9:

      * GAP-1: neuron-cc --dump-neff on current 5.2 cache, confirm no
        `dsa_indexer_kernel` invocation today.
      * GAP-2: read full GLM-5.2 config.json for index_n_heads.
      * GAP-3: confirm IndexPool pool_weights is learned per-layer.
      * GAP-4: confirm GLM-5.3-Flash index_n_heads and index_head_dim.
      * GAP-5: confirm GLM-5.2 index_head_dim.
      * GAP-6: prototype two-stage hierarchical top-K for L=1M.
      * GAP-7: NEFF pattern-match — verify compiler doesn't lower
        `gather + softmax` back to full attention at S > topk.

    Until closed, this function refuses to run to avoid shipping a
    broken kernel.
    """
    raise NotImplementedError(
        "NKI device kernel is deferred. Use the CPU golden reference "
        "(`dsa_lightning_indexer_forward`) as the correctness oracle "
        "and see DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md for the "
        "path to the v1 device kernel."
    )


__all__ = [
    "KERNEL_SLUG_V0_REFERENCE",
    "KERNEL_SLUG_V1_NKI",
    "LSE_BASE_CONVENTION",
    "DsaKernelConfig",
    "KernelResourceBounds",
    "analytical_bounds",
    "dsa_index_pool_projection",
    "dsa_lightning_indexer_forward",
    "dsa_sparse_attention_forward",
    "full_attention_at_indices_reference",
    "full_attention_reference",
    "lightning_indexer_scores",
    "lightning_indexer_topk",
    "sparse_gather_kv",
]
