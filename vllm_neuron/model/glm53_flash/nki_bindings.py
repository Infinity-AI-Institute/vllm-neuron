# SPDX-License-Identifier: Apache-2.0
"""Round-3 device-traceable kernel bindings for the GLM-5.3-Flash NxDI wrapper.

Round 2 left three ``forward()`` methods raising ``NotImplementedError``
(``_KDABlock``, ``_DSAIndexerBlock``, ``_MoEBlock`` routed branch).  This
module supplies the bindings that unblock them.

Why this file exists at all
---------------------------
The canonical CPU goldens live in the handoff bundle and stay the single
source of truth (loaded through ``_reference_kernels.load_reference_kernel``,
never vendor-copied).  But they are not uniformly usable from an NxDI
compile graph:

===============  ===============  =========================================
golden           implementation   usable directly from a traced forward?
===============  ===============  =========================================
dsa_lightning_   torch            YES — called straight through.
  indexer.py
kda_state_v2.py  numpy            NO — ``np.einsum`` on detached arrays is
                                  invisible to the XLA tracer.  This module
                                  carries a *line-for-line torch port* of
                                  ``_kda_delta_rule_step``, gated by an
                                  equivalence test against the numpy golden
                                  (``tests/test_kda_torch_parity.py``).
moe_dispatch.py  ``@nki.jit``     NO CPU path at all — it is the *device*
                                  kernel builder.  This module carries the
                                  traceable torch gather-dispatch that the
                                  device kernel replaces, plus the
                                  ``MoEDispatchConfig`` identity for
                                  GLM-5.3-Flash so the fused path can be
                                  enabled when the container supports it.
===============  ===============  =========================================

Fallback discipline
-------------------
Per the campaign no-fallback rule, nothing here may silently degrade to
``softmax`` / ``full_attention`` / ``sdpa`` / ``flash_attn``.  Every entry
point routes through :func:`assert_impl_not_banned`.  A missing kernel raises;
it never quietly becomes dense attention.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from ._reference_kernels import load_reference_kernel

# Mirrors `kda_state_v2._BANNED_IMPLS` and `moe_dispatch` §B5 discipline.
BANNED_IMPLS = frozenset({"softmax", "full_attention", "sdpa", "flash_attn"})

KDA_KERNEL_SLUG_V2 = "kda_state.decode.kda_gate.rank1_delta.bf16_state.v1"
DSA_KERNEL_SLUG_V0 = "dsa.lightning_indexer.topk_gather.v0"
MOE_KERNEL_SLUG_V1 = "moe.dispatch.nc_find_index8.blockwise.v1"


def assert_impl_not_banned(impl: str, where: str) -> None:
    """Refuse any dense-attention fall-through by name."""
    if impl in BANNED_IMPLS:
        raise ValueError(
            f"{where}: impl={impl!r} is banned — a dense/full-attention "
            "fallback CORRUPTS GLM-5.3-Flash (KDA is a linear-attention "
            "recurrence and DSA is sparse-by-construction). Use the bound "
            "kernel or let this raise."
        )


# ---------------------------------------------------------------------------
# KDA — torch port of the numpy golden `_kda_delta_rule_step`
# ---------------------------------------------------------------------------

def bf16_round(x: torch.Tensor) -> torch.Tensor:
    """Round-trip through bf16, matching the golden's `bf16_cast` store."""
    return x.to(torch.bfloat16).to(x.dtype)


def kda_delta_step_torch(
    state: torch.Tensor,      # [B, H, D_v, D_qk] fp32 (bf16-representable)
    q_raw: torch.Tensor,      # [B, H, D_qk] (post-conv, pre-L2norm)
    k_raw: torch.Tensor,      # [B, H, D_qk]
    v: torch.Tensor,          # [B, H, D_v]
    g_raw: torch.Tensor,      # [B, H, D_qk] raw per-channel gate logits
    beta_raw: torch.Tensor,   # [B, H]       raw per-head beta logit
    a_log: torch.Tensor,      # [H]          learned per-head
    g_bias: torch.Tensor,     # [H, D_qk]    learned per-head-per-channel
    *,
    lower_bound: float = -5.0,
    l2norm_eps: float = 1e-6,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One KDA step — torch transcription of ``_kda_delta_rule_step``.

    Preserves all four FLA v0.5.2 parity pieces called out in the Round-3
    contract:

      1. per-channel gate ``alpha = lower_bound * sigmoid(exp(A_log)*(g+g_bias))``
         (written as the golden's ``lower_bound / (1 + exp(-a_amp*g))`` so the
         float rounding matches term-for-term),
      2. in-kernel L2-norm on q and k with ``eps=1e-6``, spelled
         ``x / sqrt(sum(x*x) + eps)`` — NOT ``rsqrt``, which differs in the
         last mantissa bit,
      3. query scale ``*= D_qk ** -0.5``,
      4. bf16 state (caller quantizes at the HBM boundary; see
         :func:`kda_state_forward_torch`).

    Returns ``(y, state_new)`` with ``y = S_post @ q`` — the *post-update*
    state, matching the vLLM/Triton store order.
    """
    D_qk = q_raw.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D_qk)

    S = state.to(torch.float32)
    q = q_raw.to(torch.float32)
    k = k_raw.to(torch.float32)
    v = v.to(torch.float32)

    # (1) L2-norm on q, k — eps=1e-6, per-head. `/ sqrt(...)`, not rsqrt.
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True) + l2norm_eps)
    k = k / torch.sqrt((k * k).sum(dim=-1, keepdim=True) + l2norm_eps)

    # (2) Query scale.
    q = q * scale

    # (3) KDA per-channel gate.
    a_amp = torch.exp(a_log.to(torch.float32)).view(1, -1, 1)   # [1, H, 1]
    g = g_raw.to(torch.float32) + g_bias.to(torch.float32).unsqueeze(0)
    alpha = lower_bound / (1.0 + torch.exp(-(a_amp * g)))       # [B, H, D_qk]
    decay = torch.exp(alpha)

    # (4) State decay — broadcast over the D_v axis.
    S = S * decay.unsqueeze(-2)                                  # [B,H,D_v,D_qk]

    # (5) delta = v - S @ k
    Sk = torch.einsum("bhij,bhj->bhi", S, k)                     # [B, H, D_v]
    delta = v - Sk

    # (6)(7) beta = sigmoid(beta_raw); delta *= beta
    beta = torch.sigmoid(beta_raw.to(torch.float32))             # [B, H]
    delta = delta * beta.unsqueeze(-1)

    # (8) rank-1 update S += delta outer k
    S = S + delta.unsqueeze(-1) * k.unsqueeze(-2)

    # (9) y = S_post @ q
    y = torch.einsum("bhij,bhj->bhi", S, q)
    return y, S


def kda_state_forward_torch(
    state: torch.Tensor,      # [B, H, D_v, D_qk]
    query: torch.Tensor,      # [B, L, H, D_qk]
    key: torch.Tensor,        # [B, L, H, D_qk]
    value: torch.Tensor,      # [B, L, H, D_v]
    g_raw: torch.Tensor,      # [B, L, H, D_qk]
    beta_raw: torch.Tensor,   # [B, L, H]
    a_log: torch.Tensor,      # [H]
    g_bias: torch.Tensor,     # [H, D_qk]
    *,
    lower_bound: float = -5.0,
    l2norm_eps: float = 1e-6,
    scale: float | None = None,
    impl: str = "torch",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Traceable KDA over a full (short) sequence — decode is ``L == 1``.

    Mirrors ``kda_state_decode_forward_reference_v2`` for ``L == 1`` and
    ``kda_state_prefill_forward_reference_v2`` for ``L > 1`` (an unrolled
    sequential scan; the chunked-parallel prefill kernel is a follow-on).
    The Python loop is over a *static* bucket length, so it unrolls cleanly
    into the traced graph.

    Returns ``(y[B, L, H, D_v], state[B, H, D_v, D_qk])`` — both bf16-rounded
    at the store boundary exactly as the golden does.
    """
    assert_impl_not_banned(impl, "kda_state_forward_torch")
    length = query.shape[1]
    S = state
    outputs = []
    for t in range(length):
        y_t, S = kda_delta_step_torch(
            S,
            query[:, t],
            key[:, t],
            value[:, t],
            g_raw[:, t],
            beta_raw[:, t],
            a_log,
            g_bias,
            lower_bound=lower_bound,
            l2norm_eps=l2norm_eps,
            scale=scale,
        )
        # Store-boundary bf16 quantization, per-step (matches the golden's
        # per-decode-call cast — prefill re-quantizes every token).
        S = bf16_round(S)
        outputs.append(bf16_round(y_t).unsqueeze(1))
    y = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
    return y, S


def kda_reference_parity_check(
    *, batch: int = 2, heads: int = 4, d_qk: int = 8, d_v: int = 8, seed: int = 0
) -> float:
    """Max abs error of the torch port vs the numpy golden. 0.0 == bit-exact.

    Kept in-module (not just in tests) so the compile driver can assert parity
    on the compile host, where the harness test suite is not installed.
    """
    import numpy as np

    golden = load_reference_kernel("kda")
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    q, k = r(batch, 1, heads, d_qk), r(batch, 1, heads, d_qk)
    v = r(batch, 1, heads, d_v)
    g = r(batch, 1, heads, d_qk)
    beta = r(batch, 1, heads)
    state = golden.bf16_cast(r(batch, heads, d_v, d_qk))
    a_log, g_bias = r(heads), r(heads, d_qk)

    params = golden.KdaLayerParams(a_log=a_log, g_bias=g_bias)
    out = golden.kda_state_decode_forward_reference_v2(
        golden.KdaDecodeInputsV2(
            query=q, key=k, value=v, g_raw=g, beta_raw=beta,
            state_bf16=state, params=params,
        )
    )
    y_t, s_t = kda_state_forward_torch(
        torch.from_numpy(state),
        torch.from_numpy(q).permute(0, 1, 2, 3),
        torch.from_numpy(k),
        torch.from_numpy(v),
        torch.from_numpy(g),
        torch.from_numpy(beta),
        torch.from_numpy(a_log),
        torch.from_numpy(g_bias),
    )
    err_y = float(torch.abs(y_t - torch.from_numpy(out.y)).max())
    err_s = float(torch.abs(s_t - torch.from_numpy(out.state_bf16)).max())
    return max(err_y, err_s)


# ---------------------------------------------------------------------------
# DSA — straight-through to the torch golden
# ---------------------------------------------------------------------------

def dsa_scores_from_qidx(
    q_idx: torch.Tensor,           # [B, Q, H_idx, D_idx] — already TP-reduced
    indexer_k_cache: torch.Tensor, # [B, L, H_idx*index_pool, D_idx]
    *,
    index_pool: int,
    pool_weights: torch.Tensor,
) -> torch.Tensor:
    """Indexer scores from a pre-projected, TP-reduced query.

    Split out of the golden's ``lightning_indexer_scores`` because under
    tensor parallelism the ``q_flat @ indexer_q_proj`` contraction runs over
    the *main-attention* head axis, which is sharded.  Each rank can only
    produce a partial ``q_idx``; the shards must be summed before scoring, or
    every rank would rank positions differently and select a different sparse
    set — and the sparse KV gather has to agree across ranks that hold
    different head shards of the same KV.  The reduce happens in the caller
    (``_DSAIndexerBlock.forward``); this function takes the reduced result.

    The pooling collapse and the ``1/sqrt(D_idx)`` scale are the golden's own
    (``_apply_index_pool``), so the numerics stay owned by the golden.
    """
    golden = load_reference_kernel("dsa")
    d_idx = q_idx.shape[-1]
    if index_pool > 1:
        k_pooled = golden._apply_index_pool(
            indexer_k_cache, index_pool, pool_weights
        )
    else:
        k_pooled = indexer_k_cache.to(torch.float32)
    scores = torch.einsum(
        "bqhd,blhd->bql", q_idx.to(torch.float32), k_pooled.to(torch.float32)
    )
    return scores * (1.0 / math.sqrt(d_idx))


def dsa_attend_from_scores(
    scores: torch.Tensor,          # [B, Q, L] fp32
    query: torch.Tensor,           # [B, Q, H_local, D]
    kv_cache_k: torch.Tensor,      # [B, L, H_local, D]
    kv_cache_v: torch.Tensor,      # [B, L, H_local, D]
    position_ids: torch.Tensor,    # [B, Q] int64
    key_lengths: torch.Tensor,     # [B]    int64
    *,
    topk: int,
    causal: bool = True,
    return_lse: bool = False,
    impl: str = "reference",
):
    """Mask -> top-k -> sparse gather -> sparse attention, all via the golden.

    ``topk`` is clamped to the available context length: ``torch.topk``
    requires ``k <= L`` and GLM-5.3-Flash's ``index_topk=2048`` exceeds the
    short contexts used by the smoke and the first bucket.  At ``topk >= L``
    the selection is every valid position, which is the mathematically
    identical dense case the golden's own gate asserts against
    ``full_attention_reference`` — not a fallback, just a degenerate top-k.
    """
    assert_impl_not_banned(impl, "dsa_attend_from_scores")
    golden = load_reference_kernel("dsa")
    context_len = scores.shape[-1]
    effective_topk = min(topk, context_len)
    masked = (
        golden._causal_mask_scores(scores, position_ids, key_lengths)
        if causal
        else scores
    )
    _, indices = torch.topk(
        masked, k=effective_topk, dim=-1, largest=True, sorted=True
    )
    return golden.dsa_sparse_attention_forward(
        query,
        kv_cache_k,
        kv_cache_v,
        indices.to(torch.int32),
        position_ids,
        key_lengths,
        topk=effective_topk,
        causal=causal,
        return_lse=return_lse,
    )


def dsa_sparse_forward(
    query: torch.Tensor,           # [B, Q, H, D]
    indexer_q_proj: torch.Tensor,  # [H_idx, H*D, D_idx]
    indexer_k_cache: torch.Tensor, # [B, L, H_idx*index_pool, D_idx]
    kv_cache_k: torch.Tensor,      # [B, L, H, D]
    kv_cache_v: torch.Tensor,      # [B, L, H, D]
    position_ids: torch.Tensor,    # [B, Q] int64
    key_lengths: torch.Tensor,     # [B]    int64
    *,
    topk: int,
    index_pool: int,
    pool_weights: torch.Tensor,
    causal: bool = True,
    return_lse: bool = False,
    impl: str = "reference",
):
    """Bind ``dsa_lightning_indexer_forward`` (torch golden, natural-log LSE).

    ``topk`` is clamped to the available context length: the golden's
    ``torch.topk`` requires ``k <= L``, and GLM-5.3-Flash runs
    ``index_topk=2048`` against contexts that are shorter than that during the
    smoke and the first bucket sweep.  Clamping is *not* a fallback — the
    selection is still exactly "every position the indexer ranks highest",
    and at ``topk >= L`` it degenerates to the mathematically identical dense
    case, which is what the golden's own correctness gate asserts
    (``full_attention_reference`` at ``topk >= L``).
    """
    assert_impl_not_banned(impl, "dsa_sparse_forward")
    golden = load_reference_kernel("dsa")
    context_len = indexer_k_cache.shape[1]
    effective_topk = min(topk, context_len)
    return golden.dsa_lightning_indexer_forward(
        query,
        indexer_q_proj,
        indexer_k_cache,
        kv_cache_k,
        kv_cache_v,
        position_ids,
        key_lengths,
        topk=effective_topk,
        index_pool=index_pool,
        pool_weights=pool_weights,
        causal=causal,
        return_lse=return_lse,
    )


# ---------------------------------------------------------------------------
# MoE — GLM-5.3 routing + traceable gather-dispatch
# ---------------------------------------------------------------------------

def build_glm53_moe_dispatch_config(
    *,
    hidden: int,
    num_experts: int,
    top_k: int,
    intermediate_global: int,
    tp_degree: int,
    renormalize_topk: bool,
) -> Any:
    """GLM-5.3-Flash's ``MoEDispatchConfig`` identity + Tier-1 CPU battery.

    Running ``validate()`` here is the pre-fire gate from the Gemma-4 lessons
    (partition cap, ``I_TP % 16``, top-k in the tested set).  A GLM-5.3-Flash
    compile must never be submitted with a config that fails it.
    """
    moe = load_reference_kernel("moe")
    cfg = moe.MoEDispatchConfig(
        name=f"glm53-flash-tp{tp_degree}",
        hidden=hidden,
        num_experts=num_experts,
        top_k=top_k,
        intermediate_global=intermediate_global,
        tp_degree=tp_degree,
        activation=moe.MoEActivation.SILU,
        renormalize_topk=renormalize_topk,
    )
    cfg.validate()
    return cfg


def glm53_route(
    hidden_states: torch.Tensor,   # [B, L, hidden]
    router_weight: torch.Tensor,   # [num_experts, hidden]
    *,
    top_k: int,
    scoring_func: str,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
    correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GLM routing: sigmoid scores, selection-only correction bias.

    Mirrors ``glm52_moe_dsa.moe.select_glm52_experts``: the learned correction
    bias moves *which* experts win top-k but must never leak into the routing
    weights, which are gathered from the raw sigmoid scores.

    Returns ``(expert_indices[B, L, top_k] int64, weights[B, L, top_k] fp32)``.
    """
    if scoring_func != "sigmoid":
        raise NotImplementedError(
            f"GLM-5.3-Flash router scoring_func={scoring_func!r}; only "
            "'sigmoid' is qualified. Refusing to guess a softmax equivalent."
        )
    logits = F.linear(
        hidden_states.to(torch.float32), router_weight.to(torch.float32)
    )
    scores = torch.sigmoid(logits)
    selection = scores
    if correction_bias is not None:
        selection = scores + correction_bias.to(torch.float32)
    indices = torch.topk(selection, k=top_k, dim=-1, sorted=False).indices
    indices = indices.to(torch.int64)
    weights = torch.gather(scores, -1, indices)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
    weights = weights * routed_scaling_factor
    return indices, weights


def moe_gather_dispatch_torch(
    hidden_states: torch.Tensor,   # [B, L, hidden]
    expert_indices: torch.Tensor,  # [B, L, top_k] int64
    routing_weights: torch.Tensor, # [B, L, top_k] fp32
    gate: torch.Tensor,            # [E, hidden, inter]
    up: torch.Tensor,              # [E, hidden, inter]
    down: torch.Tensor,            # [E, inter, hidden]
    *,
    swiglu_limit: float,
    impl: str = "gather",
) -> torch.Tensor:
    """Traceable routed-expert dispatch by top-k expert-weight gather.

    This is the *graph-level* reference the fused NKI ``moe_dispatch`` kernel
    replaces.  It gathers only the ``top_k`` selected expert weight slabs per
    token rather than evaluating all ``E`` experts, so it is O(top_k) FLOPs,
    not O(E) — the same asymptotics as the fused kernel, without the
    ``nc_find_index8`` capacity-dispatch machinery.

    Note the memory shape: the gather materialises
    ``[B*L*top_k, hidden, inter]`` expert slabs.  That is fine for decode
    (``B*L == 1``) and for the small prefill buckets this contract compiles;
    the fused kernel is what makes large ``B*L`` tractable, which is why
    ``enable_moe_fused_dispatch`` is wired alongside it.
    """
    assert_impl_not_banned(impl, "moe_gather_dispatch_torch")
    batch, length, hidden = hidden_states.shape
    top_k = expert_indices.shape[-1]

    tokens = hidden_states.reshape(batch * length, hidden)
    idx = expert_indices.reshape(batch * length, top_k)
    wts = routing_weights.reshape(batch * length, top_k).to(tokens.dtype)

    # [N, K, hidden, inter] / [N, K, inter, hidden]
    g_sel = gate.index_select(0, idx.reshape(-1)).view(
        batch * length, top_k, hidden, -1
    )
    u_sel = up.index_select(0, idx.reshape(-1)).view(
        batch * length, top_k, hidden, -1
    )
    d_sel = down.index_select(0, idx.reshape(-1)).view(
        batch * length, top_k, -1, hidden
    )

    x = tokens.unsqueeze(1).unsqueeze(2)                     # [N, 1, 1, hidden]
    gate_out = torch.matmul(x, g_sel).squeeze(2)             # [N, K, inter]
    up_out = torch.matmul(x, u_sel).squeeze(2)               # [N, K, inter]
    gate_out = gate_out.clamp(max=swiglu_limit)
    up_out = up_out.clamp(-swiglu_limit, swiglu_limit)
    act = F.silu(gate_out) * up_out                          # [N, K, inter]
    expert_out = torch.matmul(act.unsqueeze(2), d_sel).squeeze(2)  # [N, K, hidden]

    combined = (expert_out * wts.unsqueeze(-1)).sum(dim=1)   # [N, hidden]
    return combined.view(batch, length, hidden)


__all__ = [
    "BANNED_IMPLS",
    "DSA_KERNEL_SLUG_V0",
    "KDA_KERNEL_SLUG_V2",
    "MOE_KERNEL_SLUG_V1",
    "assert_impl_not_banned",
    "bf16_round",
    "build_glm53_moe_dispatch_config",
    "dsa_attend_from_scores",
    "dsa_scores_from_qidx",
    "dsa_sparse_forward",
    "glm53_route",
    "kda_delta_step_torch",
    "kda_reference_parity_check",
    "kda_state_forward_torch",
    "moe_gather_dispatch_torch",
]
