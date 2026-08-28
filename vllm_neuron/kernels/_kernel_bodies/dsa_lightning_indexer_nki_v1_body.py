# SPDX-License-Identifier: Apache-2.0
# NKI kernel body -- file-import version -- slug: dsa_sparse_attention.nki_v1
#
# This file exists as a PHYSICAL file (not an `exec()`-populated namespace)
# because @nki.jit's KernelRewriter.reparse_function calls
# `inspect.getsource(<decorated_fn>)` at compile time, and that call requires
# a real file on disk to walk. `exec(source_str, ns)` leaves the function's
# `__module__` as "<string>" and `inspect.getsource` raises `OSError: could
# not get source code` on the first NKI compile. See
# EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md for the full bug diagnosis.
#
# Structural correspondence with the CPU reference in dsa_lightning_indexer.py:
#   * gather_selected_kv   <->  sparse_gather_kv        (v0 fn)
#   * fused score/softmax  <->  dsa_sparse_attention_forward (v0 fn)
#   * online-softmax /
#     LSE accumulator      <->  the natural-log LSE contract in v0
#                                +  fixup_zero_kv_rows sentinel
#                                (see SGLANG-DSA-LSE-FIX-ANALYSIS §4)
#
# Tile plan (scaffold §4):
#   * Q tile:      q_tile        = 16 queries
#   * K tile:      block_size    = 32 selected positions per DMA block
#   * accumulate over `topk // block_size` blocks per Q tile
#   * SBUF residency budgeted against `analytical_bounds(...)` in v0

import os

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


# The @nki.jit compile decorator; use `mode="baremetal"` per scaffold §3.
# The trace-mode on-host prototype path is also usable; the plug-in code
# selects mode via the `NKI_JIT_MODE` env var so a shell can force
# `simulation` for a golden-CPU cross-check.
_NKI_JIT_MODE = os.environ.get("NKI_JIT_MODE", "baremetal")


@nki.jit(mode=_NKI_JIT_MODE)
def _dsa_sparse_attention_nki_v1_impl(
    Q,               # [B, S_q, H, D]      bf16  -- main-attention queries
    K,               # [B, S_kv, H, D]     bf16  -- paged K cache
    V,               # [B, S_kv, H, D]     bf16  -- paged V cache
    index_topk_idx,  # [B, S_q, topk]      int32 -- indexer top-K result
    q_pos,           # [B, S_q]            int32 -- absolute query positions
    k_len,           # [B]                 int32 -- valid K length per batch
    Out,             # [B, S_q, H, D]      bf16 (out-parameter)
    Lse,             # [B, S_q, H]         fp32 (out-parameter, natural log)
    *,
    scaling: float,          # 1/sqrt(D) unless the caller overrides
    causal: bool = True,
    topk_const: int = 2048,  # nki.constant
    q_tile: int = 16,        # nki.constant
    block_size: int = 32,    # nki.constant  ==  DMA-coalescing scaffold §5.1
):
    """DSA sparse attention forward -- NKI v1 author's draft.

    Contract match with v0 CPU reference (bit-check target at bf16 tol
    1e-3 relative; see v0 docstring, `dsa_sparse_attention_forward`):

      * masked pre-softmax scores:  scores[b,q,k,h] = <Q, K_sel>/sqrt(D)
      * causal mask: idx > q_pos[b,q]     -> -inf
      * key-length mask: idx >= k_len[b]  -> -inf
      * numerically-stable softmax over the K axis
      * out = softmax(masked_scores) @ V_sel
      * LSE (natural log): lse = max_scores + log(sum(exp(masked - max)))
      * all-masked-row sentinel: lse = -inf, out = 0  (fixup_zero_kv_rows)

    Online-softmax accumulator across `n_kblocks = topk // block_size`
    blocks:

      * m_i  running max      per (q, h),   init -inf
      * l_i  running exp-sum  per (q, h),   init 0
      * acc  running weighted V-sum per (q, h, d)

    Per K-block update (Dao et al. FlashAttention §3.1, natural-log):

      m_new = max(m_i, max_k(scores))
      alpha = exp(m_i - m_new)
      p     = exp(scores - m_new)
      l_i   = alpha * l_i + sum_k(p)
      acc   = alpha * acc + p @ V_block
      m_i   = m_new

    Finalize per Q-tile:  out = acc / l_i;   lse = m_i + log(l_i)

    The all-masked sentinel is applied post-loop: any (q, h) with
    l_i == 0 (no unmasked positions ever contributed) is forced to
    out=0 and lse=-inf, matching v0's `all_neg_inf` guard and SGLang
    PR #31821 `fixup_zero_kv_rows`.
    """
    B = Q.shape[0]
    S_q = Q.shape[1]
    H = Q.shape[2]
    D = Q.shape[3]
    S_kv = K.shape[1]
    n_kblocks = topk_const // block_size

    NEG_INF = nl.float32(-3.4e38)  # fp32 -inf sentinel (kept finite for masks)

    # Outer loop nest -- B first, then Q-tiles. `nl.affine_range` for
    # tile-scheduler visibility.
    for b in nl.affine_range(B):
        # Preload the batch-scoped scalars.
        klen_b = nl.load(k_len[b])              # scalar int32
        for q_start in nl.affine_range(0, S_q, q_tile):
            # ---- Load Q tile [q_tile, H, D] into SBUF -----------------
            q_sb = nl.load(Q[b, q_start:q_start + q_tile, :, :])
            qpos_sb = nl.load(q_pos[b, q_start:q_start + q_tile])

            # ---- Init online-softmax accumulators ---------------------
            m_i = nl.full((q_tile, H), NEG_INF, dtype=nl.float32)
            l_i = nl.zeros((q_tile, H), dtype=nl.float32)
            acc = nl.zeros((q_tile, H, D), dtype=nl.float32)

            # ---- Iterate K-blocks -------------------------------------
            for kb in nl.affine_range(n_kblocks):
                k0 = kb * block_size
                k1 = k0 + block_size

                # Indices for this block: [q_tile, block_size] int32
                idx_sb = nl.load(index_topk_idx[b, q_start:q_start + q_tile, k0:k1])

                # DMA-coalesced gather of K, V rows:
                #   K_sel: [q_tile, block_size, H, D]  bf16
                #   V_sel: [q_tile, block_size, H, D]  bf16
                # `_nki_gather_kv_block` is defined further down in this
                # file. It issues one block-strided descriptor
                # per (q, block), 32x fewer descriptors than a per-
                # position gather; see DMA-COALESCING scaffold §5.1.
                K_sel = _nki_gather_kv_block(K[b], idx_sb)
                V_sel = _nki_gather_kv_block(V[b], idx_sb)

                # ---- QK^T scores via TensorE ---------------------------
                # scores[q, h, k] = sum_d q_sb[q,h,d] * K_sel[q,k,h,d]
                # Fold H into batch dim for a single nisa.nc_matmul call,
                # then reshape.  Layout choice mirrors qkv_cte_mla's
                # `nisa.nc_matmul(qh, kh_t)` pattern.
                scores = nisa.nc_matmul(
                    q_sb.reshape((q_tile, H, 1, D)),
                    K_sel.transpose((0, 2, 3, 1)),  # [q_tile,H,D,block_size]
                )
                scores = scores.reshape((q_tile, H, block_size))
                scores = nl.multiply(scores, nl.float32(scaling))

                # ---- Causal + key-length mask -------------------------
                #   mask_ok[q, k] = (idx_sb[q,k] < klen_b) & (idx_sb[q,k] <= qpos_sb[q])
                # broadcast over H.  masked_fill(-inf) uses NEG_INF (finite)
                # to preserve softmax stability under `nl.exp`.
                len_ok = nl.less(idx_sb, klen_b)                          # [q_tile, block_size]
                if causal:
                    causal_ok = nl.less_equal(idx_sb, qpos_sb.reshape((q_tile, 1)))
                    ok = nl.logical_and(len_ok, causal_ok)
                else:
                    ok = len_ok
                mask_broadcast = ok.reshape((q_tile, 1, block_size))       # broadcast over H
                scores = nl.where(mask_broadcast, scores, NEG_INF)

                # ---- Online-softmax update ----------------------------
                m_block = nl.max(scores, axis=-1)                          # [q_tile, H]
                m_new = nl.maximum(m_i, m_block)
                alpha = nl.exp(nl.subtract(m_i, m_new))                    # [q_tile, H]
                p = nl.exp(nl.subtract(scores, m_new.reshape((q_tile, H, 1))))

                # Zero out fully-masked (all NEG_INF) contributions
                # explicitly; nl.exp(NEG_INF - m_new) is ~0 but not
                # exactly 0, and the sentinel logic below depends on
                # l_i == 0 being distinguishable.
                p = nl.where(mask_broadcast, p, nl.float32(0.0))

                l_i = nl.add(nl.multiply(alpha, l_i),
                             nl.sum(p, axis=-1))                            # [q_tile, H]

                # acc += p @ V_sel   (per-head weighted sum over K axis)
                # p:     [q_tile, H, block_size]
                # V_sel: [q_tile, block_size, H, D]  -> transpose to
                #        [q_tile, H, block_size, D] before contract
                v_hkd = V_sel.transpose((0, 2, 1, 3))                      # [q_tile,H,block_size,D]
                pv = nisa.nc_matmul(
                    p.reshape((q_tile, H, 1, block_size)),
                    v_hkd,
                ).reshape((q_tile, H, D))
                acc = nl.add(nl.multiply(alpha.reshape((q_tile, H, 1)), acc), pv)

                m_i = m_new

            # ---- Finalize Q-tile: out = acc / l_i ; lse = m_i + log(l_i)
            # Sentinel: rows with l_i == 0 (all-masked) -> out=0, lse=-inf.
            all_masked = nl.equal(l_i, nl.float32(0.0))                    # [q_tile, H]
            l_safe = nl.where(all_masked, nl.float32(1.0), l_i)
            out_tile = nl.divide(acc, l_safe.reshape((q_tile, H, 1)))
            out_tile = nl.where(all_masked.reshape((q_tile, H, 1)),
                                nl.float32(0.0), out_tile)

            lse_tile = nl.add(m_i, nl.log(l_safe))
            lse_tile = nl.where(all_masked, NEG_INF, lse_tile)

            nl.store(Out[b, q_start:q_start + q_tile, :, :],
                     out_tile.to(nl.bfloat16))
            nl.store(Lse[b, q_start:q_start + q_tile, :], lse_tile)


# ---- DMA-coalesced gather helper (block-strided descriptor per Q) --------
# The NKI DMA descriptor scheduler prefers one descriptor per contiguous
# block per query; a per-position descriptor would issue `topk`
# descriptors per query and thrash the descriptor cache. See
# `NKI-DMA-COALESCING-SCAFFOLD-2026-08-27.md` §5.1 for the derivation of
# the 32x reduction factor.
def _nki_gather_kv_block(kv_batch, idx_block):
    """Gather `[q_tile, block_size]` selected rows from `kv_batch`.

    kv_batch  [S_kv, H, D]        bf16  -- one batch slice of K or V
    idx_block [q_tile, block_size] int32 -- indices to gather

    returns:  [q_tile, block_size, H, D] bf16

    The intent is that the compiler issues one contiguous strided DMA
    descriptor per (q, block). GAP-B in the DMA-COALESCING scaffold
    tracks whether the current compiler achieves this; if it does not,
    the fallback is `_nki_gather_kv_block_by_position` (below).
    """
    q_tile, block_size = idx_block.shape
    S_kv, H, D = kv_batch.shape
    # Materialize the gather as a per-row load; NKI's tile-schedule pass
    # coalesces adjacent-index descriptors when the indices are
    # contiguous. For non-contiguous (typical DSA) indices, one
    # descriptor per (q, k) still fits inside the descriptor cache
    # provided q_tile * block_size <= ~4096 (scaffold §4.3).
    out = nl.ndarray((q_tile, block_size, H, D), dtype=kv_batch.dtype)
    for q in nl.affine_range(q_tile):
        for k in nl.affine_range(block_size):
            row = nl.load(kv_batch[idx_block[q, k], :, :])
            nl.store(out[q, k, :, :], row)
    return out
