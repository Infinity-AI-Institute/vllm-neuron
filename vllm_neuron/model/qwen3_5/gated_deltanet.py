# SPDX-License-Identifier: Apache-2.0
"""
Gated DeltaNet math for Qwen3.5 on Neuron
=========================================

Pure torch, no vLLM imports, so every function here runs and is tested on CPU.

Ported from transformers' ``Qwen3_5GatedDeltaNet`` (v5.17.0, Apache-2.0):
``torch_chunk_gated_delta_rule``, ``torch_recurrent_gated_delta_rule``,
``causal_conv1d_fn`` / ``causal_conv1d_update`` and ``Qwen3_5RMSNormGated``.

Constraints from vllm-neuron, which ``torch.compile``s the model through its
``vllm_neuron`` backend:

- No Python control flow depends on tensor *values*. Loops run over static
  shapes only (sequence length, chunk count, chunk size), so a compiled
  bucket unrolls to a fixed graph.
- The within-chunk triangular solve uses transformers' forward-substitution
  form, not ``torch.linalg.solve_triangular``, which may not lower through the
  Neuron compiler.

Batching model
--------------
vLLM flattens the tokens of every scheduled request into one row. The
recurrence is sequential per request, so tokens are regrouped to a padded
``[requests, max_len]`` layout. Three rules keep that exact:

1. Padded positions get ``g = 0`` and ``beta = 0``: no decay and no write, so
   the recurrent state passes through them unchanged. With right padding a
   real row never references a padded column in the chunked triangular
   system, so padding is an exact no-op there too.
2. The conv state is gathered at each request's true length, never taken from
   the padded tail.
3. A request whose prompt spans several scheduler steps resets its state only
   on its first chunk; ``reset`` carries that per request.
"""

import torch
import torch.nn.functional as F


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Matches the FLA / transformers l2norm: ``x * rsqrt(sum(x^2) + eps)``."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def rmsnorm_gated(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """``Qwen3_5RMSNormGated``: norm before gate, plain weight, SiLU gate.

    Unlike the decoder's zero-centred ``Qwen3_5RMSNorm`` (``x * (1 + w)``),
    this norm multiplies by ``w`` directly.
    """
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = weight * h.to(dtype)
    h = h * F.silu(gate.to(torch.float32))
    return h.to(dtype)


def valid_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """``[B, max_len]`` bool: True where the position holds a real token."""
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal conv1d with SiLU, continuing from a carried state.

    Args:
        x: ``[B, C, T]`` inputs, right-padded past ``lengths``.
        weight: ``[C, K]`` depthwise kernel (no bias in Qwen3.5).
        state: ``[B, C, K-1]`` the previous ``K-1`` inputs; zeros for a new
            request, which reproduces transformers' ``padding=K-1`` prefill.
        lengths: ``[B]`` number of real tokens per request.

    Returns:
        ``(out, new_state)``: ``out`` is ``[B, C, T]``; ``new_state`` holds the
        last ``K-1`` *real* inputs of ``concat(state, x)``.
    """
    batch, channels, _ = x.shape
    width = weight.shape[-1] - 1
    full = torch.cat([state.to(x.dtype), x], dim=-1)
    out = F.conv1d(full.to(weight.dtype), weight.unsqueeze(1), groups=channels)
    out = F.silu(out).to(x.dtype)
    # Real inputs end at full index width + lengths - 1, so the last `width`
    # of them sit at indices lengths .. lengths + width - 1.
    index = lengths.view(batch, 1, 1) + torch.arange(width, device=x.device).view(
        1, 1, width
    )
    new_state = torch.gather(full, 2, index.expand(batch, channels, width))
    return out, new_state


def _prepare(query, key, value, g, beta, lengths):
    """Shared by both delta-rule forms: mask padding, normalize, move heads."""
    valid = valid_mask(lengths, key.shape[1]).unsqueeze(-1)
    g = torch.where(valid, g, torch.zeros_like(g))
    beta = torch.where(valid, beta, torch.zeros_like(beta))
    query, key, value, beta, g = [
        t.transpose(1, 2).to(torch.float32) for t in (query, key, value, beta, g)
    ]
    query = l2norm(query)
    key = l2norm(key)
    query = query * (query.shape[-1] ** -0.5)
    return query, key, value, g, beta


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token gated delta rule; the decode path and the ground truth.

    Args:
        query, key: ``[B, T, H, Dk]``, already repeated to the value heads.
        value: ``[B, T, H, Dv]``.
        g: ``[B, T, H]`` log decay (``<= 0``).
        beta: ``[B, T, H]`` write strength in ``(0, 1)``.
        initial_state: ``[B, H, Dk, Dv]`` F32.
        lengths: ``[B]`` real tokens per request.

    Returns:
        ``(out [B, T, H, Dv], final_state [B, H, Dk, Dv] F32)``.
    """
    out_dtype = query.dtype
    query, key, value, g, beta = _prepare(query, key, value, g, beta, lengths)
    state = initial_state.to(torch.float32)
    outputs = []
    for t in range(key.shape[2]):
        q_t, k_t, v_t = query[:, :, t], key[:, :, t], value[:, :, t]
        state = state * g[:, :, t].exp()[..., None, None]
        predicted = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - predicted) * beta[:, :, t].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
    out = torch.stack(outputs, dim=2).transpose(1, 2).contiguous()
    return out.to(out_dtype), state


def _unit_lower_inverse(system: torch.Tensor) -> torch.Tensor:
    """``(I + strictly_lower(system))^{-1}`` by forward substitution.

    transformers' export-safe path; avoids ``torch.linalg.solve_triangular``.
    Loops over the static chunk size, so it unrolls under ``torch.compile``.
    """
    size = system.shape[-1]
    inverse = -system.tril(-1)
    for i in range(1, size):
        row = inverse[..., i, :i].clone()
        sub = inverse[..., :i, :i].clone()
        inverse[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    eye = torch.eye(size, dtype=system.dtype, device=system.device)
    return inverse + eye


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    lengths: torch.Tensor,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked gated delta rule; the prefill path. Same contract as
    :func:`recurrent_gated_delta_rule`, equal to it up to F32 summation order.
    """
    out_dtype = query.dtype
    sequence_length = key.shape[1]
    query, key, value, g, beta = _prepare(query, key, value, g, beta, lengths)

    pad = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(t, (0, 0, 0, pad)) for t in (query, key, value))
    beta, g = (F.pad(t, (0, pad)) for t in (beta, g))
    num_chunks = (sequence_length + pad) // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        t.reshape(t.shape[0], t.shape[1], num_chunks, chunk_size, t.shape[-1])
        for t in (query, key, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], num_chunks, chunk_size)

    upper = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=query.device
    ).triu(1)
    cum = g.cumsum(dim=-1)
    pairwise = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).masked_fill(upper, float("-inf"))
    pairwise = pairwise.exp()

    system = (k_beta @ key.transpose(-1, -2)) * pairwise
    intra = (query @ key.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum.exp().unsqueeze(-1)

    inverse = _unit_lower_inverse(system)
    new_values = inverse @ v_beta
    k_cumdecay = inverse @ decayed_k_beta

    query = query * cum.exp().unsqueeze(-1)
    key = key * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]

    state = initial_state.to(torch.float32)
    chunks = []
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        chunks.append(query[:, :, i] @ state + intra[:, :, i] @ v_new)
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    out = torch.cat(chunks, dim=2)[:, :, :sequence_length]
    return out.transpose(1, 2).contiguous().to(out_dtype), state


def tokens_to_padded(
    x: torch.Tensor, offsets: torch.Tensor, counts: torch.Tensor, max_len: int
) -> torch.Tensor:
    """Flat ``[N, ...]`` tokens to ``[B, max_len, ...]``, zero-padded.

    Tokens of one request are contiguous and requests appear in order, which
    is how vllm-neuron lays out ``req_indices`` (``np.repeat``).
    """
    positions = torch.arange(max_len, device=x.device)
    index = offsets.unsqueeze(1) + positions.unsqueeze(0)
    valid = positions.unsqueeze(0) < counts.unsqueeze(1)
    index = torch.where(valid, index, torch.zeros_like(index))
    padded = x[index]
    shape = valid.shape + (1,) * (padded.dim() - 2)
    return padded * valid.view(shape).to(padded.dtype)


def padded_to_tokens(
    x: torch.Tensor, request_of_token: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """``[B, max_len, ...]`` back to flat ``[N, ...]``; static output shape.

    A segment longer than ``max_len`` (only the sink row, which holds the
    batch's trailing bucket padding) reads its last padded position for the
    overflow; those outputs are discarded.
    """
    tokens = torch.arange(request_of_token.shape[0], device=x.device)
    position = (tokens - offsets[request_of_token]).clamp(max=x.shape[1] - 1)
    return x[request_of_token, position]


def segment(request_of_token: torch.Tensor, num_requests: int):
    """Per-request ``(counts, offsets)`` from each token's request index."""
    counts = torch.zeros(
        num_requests, dtype=torch.int64, device=request_of_token.device
    ).index_add_(
        0,
        request_of_token.to(torch.int64),
        torch.ones_like(request_of_token, dtype=torch.int64),
    )
    offsets = torch.cumsum(counts, dim=0) - counts
    return counts, offsets


def gated_deltanet_mixer(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    lengths: torch.Tensor,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float,
    decode: bool,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole ``Qwen3_5GatedDeltaNet`` token mixer on a padded batch.

    ``weights`` holds the checkpoint's tensors under their transformers
    names, with ``in_proj_*`` / ``out_proj`` as ``[out, in]`` matrices and
    ``conv1d`` as ``[C, K]``. Returns ``(out, conv_state, recurrent_state)``.
    """
    batch, length, _ = hidden.shape
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim

    mixed = F.linear(hidden, weights["in_proj_qkv"]).transpose(1, 2)
    z = F.linear(hidden, weights["in_proj_z"]).reshape(
        batch, length, num_v_heads, head_v_dim
    )
    b = F.linear(hidden, weights["in_proj_b"])
    a = F.linear(hidden, weights["in_proj_a"])

    mixed, conv_state = causal_conv1d(mixed, weights["conv1d"], conv_state, lengths)
    mixed = mixed.transpose(1, 2)
    query, key, value = torch.split(mixed, [key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(batch, length, num_k_heads, head_k_dim)
    key = key.reshape(batch, length, num_k_heads, head_k_dim)
    value = value.reshape(batch, length, num_v_heads, head_v_dim)

    beta = b.sigmoid()
    g = -weights["A_log"].float().exp() * F.softplus(a.float() + weights["dt_bias"])
    repeat = num_v_heads // num_k_heads
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    rule = recurrent_gated_delta_rule if decode else chunk_gated_delta_rule
    extra = {} if decode else {"chunk_size": chunk_size}
    core, recurrent_state = rule(
        query, key, value, g, beta, recurrent_state, lengths, **extra
    )

    core = rmsnorm_gated(
        core.reshape(-1, head_v_dim), z.reshape(-1, head_v_dim), weights["norm"], eps
    ).reshape(batch, length, value_dim)
    return F.linear(core, weights["out_proj"]), conv_state, recurrent_state
