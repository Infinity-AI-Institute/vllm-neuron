# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 attention math against transformers 5.17.0 (skipped without it)."""

import pytest
import torch

from vllm_neuron.model.qwen3_5 import attention_math as am

transformers = pytest.importorskip("transformers")
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5TextConfig,
)

torch.manual_seed(0)
HIDDEN, HEADS, KV_HEADS, HEAD_DIM, ROTARY, THETA, EPS = 64, 4, 2, 32, 8, 1e7, 1e-6


def test_zero_centred_rmsnorm_matches_transformers():
    norm = hf.Qwen3_5RMSNorm(HEAD_DIM, eps=EPS)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(HEAD_DIM) * 0.1)
    x = torch.randn(5, HEAD_DIM, dtype=torch.bfloat16)
    torch.testing.assert_close(am.zero_centred_rmsnorm(x, norm.weight, EPS), norm(x))


def test_zero_centred_is_not_plain_rmsnorm():
    """Guard against silently reusing Qwen3's norm: w=0 must be the identity scale."""
    x = torch.randn(3, HEAD_DIM)
    out = am.zero_centred_rmsnorm(x, torch.zeros(HEAD_DIM), EPS)
    plain = x / (x.pow(2).mean(-1, keepdim=True) + EPS).sqrt()
    torch.testing.assert_close(out, plain)


def _hf_attention():
    config = Qwen3_5TextConfig(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=EPS,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        partial_rotary_factor=ROTARY / HEAD_DIM,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": THETA,
            "partial_rotary_factor": ROTARY / HEAD_DIM,
        },
    )
    config._attn_implementation = "eager"
    layer = hf.Qwen3_5Attention(config, layer_idx=0).eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    return config, layer


def _manual_attention(layer, hidden, positions):
    """Our helpers plus a plain causal softmax attention, nothing borrowed."""
    tokens = hidden.shape[0]
    query, gate = am.split_query_gate(hidden @ layer.q_proj.weight.T, HEADS, HEAD_DIM)
    key = (hidden @ layer.k_proj.weight.T).view(tokens, KV_HEADS, HEAD_DIM)
    value = (hidden @ layer.v_proj.weight.T).view(tokens, KV_HEADS, HEAD_DIM)
    query = am.zero_centred_rmsnorm(query, layer.q_norm.weight, EPS).transpose(0, 1)
    key = am.zero_centred_rmsnorm(key, layer.k_norm.weight, EPS).transpose(0, 1)
    cos, sin = am.rotary_cos_sin(positions, ROTARY, THETA, hidden.dtype)
    query, key = (
        am.apply_partial_rotary(query, cos, sin),
        am.apply_partial_rotary(key, cos, sin),
    )
    groups = HEADS // KV_HEADS
    key = key.repeat_interleave(groups, 0)
    value = value.transpose(0, 1).repeat_interleave(groups, 0)
    scores = (query @ key.transpose(-1, -2)) * HEAD_DIM**-0.5
    mask = torch.ones(tokens, tokens, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(mask, float("-inf")).softmax(-1)
    attention = (scores @ value).transpose(0, 1).reshape(tokens, HEADS * HEAD_DIM)
    return am.gate_output(attention, gate) @ layer.o_proj.weight.T


def test_gated_partial_rotary_attention_matches_transformers():
    config, layer = _hf_attention()
    hidden = torch.randn(1, 9, HIDDEN)
    positions = torch.arange(9)
    rotary = hf.Qwen3_5TextRotaryEmbedding(config)
    # M-RoPE takes (3, batch, T) position ids. For text all three axes are the
    # same position, so its interleaved recomposition reduces to standard RoPE
    # over the rotary channels, which is exactly what attention_math computes.
    cos, sin = rotary(hidden, positions[None, None, :].expand(3, 1, -1))
    mask = torch.full((1, 1, 9, 9), float("-inf")).triu(1)
    with torch.no_grad():
        expected, _ = layer(hidden, (cos, sin), mask)
        ours = _manual_attention(layer, hidden[0], positions)
    torch.testing.assert_close(ours, expected[0], atol=1e-5, rtol=1e-4)


def test_rotary_leaves_passthrough_channels_untouched():
    x = torch.randn(2, 5, HEAD_DIM)
    cos, sin = am.rotary_cos_sin(torch.arange(5), ROTARY, THETA, x.dtype)
    out = am.apply_partial_rotary(x, cos, sin)
    torch.testing.assert_close(out[..., ROTARY:], x[..., ROTARY:], atol=0, rtol=0)
    assert not torch.allclose(out[..., :ROTARY], x[..., :ROTARY])


def test_query_and_gate_stay_paired_per_head():
    """Sharding q_proj rows by whole heads must keep each query with its gate."""
    q_proj = torch.randn(1, HEADS * 2 * HEAD_DIM)
    query, gate = am.split_query_gate(q_proj, HEADS, HEAD_DIM)
    for h in range(HEADS):
        block = q_proj[0, h * 2 * HEAD_DIM : (h + 1) * 2 * HEAD_DIM]
        torch.testing.assert_close(query[0, h], block[:HEAD_DIM])
        torch.testing.assert_close(
            gate[0, h * HEAD_DIM : (h + 1) * HEAD_DIM], block[HEAD_DIM:]
        )
