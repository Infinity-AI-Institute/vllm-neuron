# SPDX-License-Identifier: Apache-2.0
"""CPU numerics for the Qwen3.5 Gated DeltaNet port.

Two references, deliberately independent of the code under test:

- ``reference_mixer`` below: a float64, one-token-at-a-time transcription of
  transformers' ``Qwen3_5GatedDeltaNet`` semantics, sharing no code with
  ``gated_deltanet.py``.
- transformers' own ``Qwen3_5GatedDeltaNet`` layer, when transformers is
  installed (skipped otherwise).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from vllm_neuron.model.qwen3_5 import gated_deltanet as gdn
from vllm_neuron.model.qwen3_5.state import RecurrentStateBuffers, SlotAllocator

torch.manual_seed(0)

# Small but structurally faithful: two value heads per key head, as in the 9B.
K_HEADS, V_HEADS, DK, DV, CONV, HIDDEN, EPS = 2, 4, 8, 8, 4, 32, 1e-6
CONV_DIM = 2 * K_HEADS * DK + V_HEADS * DV


def make_weights(dtype=torch.float32):
    return {
        "in_proj_qkv": torch.randn(CONV_DIM, HIDDEN, dtype=dtype) * 0.2,
        "in_proj_z": torch.randn(V_HEADS * DV, HIDDEN, dtype=dtype) * 0.2,
        "in_proj_b": torch.randn(V_HEADS, HIDDEN, dtype=dtype) * 0.2,
        "in_proj_a": torch.randn(V_HEADS, HIDDEN, dtype=dtype) * 0.2,
        "conv1d": torch.randn(CONV_DIM, CONV, dtype=dtype) * 0.3,
        "A_log": torch.log(torch.empty(V_HEADS).uniform_(0.5, 8.0)),
        "dt_bias": torch.randn(V_HEADS) * 0.3,
        "norm": 1.0 + 0.1 * torch.randn(DV, dtype=dtype),
        "out_proj": torch.randn(HIDDEN, V_HEADS * DV, dtype=dtype) * 0.2,
    }


def reference_mixer(hidden, w, conv_state=None, state=None):
    """Float64, token by token, straight from the transformers semantics."""
    h = hidden.double()
    w = {k: v.double() for k, v in w.items()}
    length = h.shape[0]
    mixed = h @ w["in_proj_qkv"].T
    z = (h @ w["in_proj_z"].T).reshape(length, V_HEADS, DV)
    b = h @ w["in_proj_b"].T
    a = h @ w["in_proj_a"].T
    history = (
        torch.zeros(CONV - 1, CONV_DIM, dtype=torch.float64)
        if conv_state is None
        else conv_state.double().T.clone()
    )
    conv_out = torch.empty_like(mixed)
    for t in range(length):
        window = torch.cat([history, mixed[t : t + 1]], dim=0)
        conv_out[t] = F.silu((window * w["conv1d"].T).sum(0))
        history = window[1:]
    kd = K_HEADS * DK
    q, k, v = conv_out[:, :kd], conv_out[:, kd : 2 * kd], conv_out[:, 2 * kd :]
    q = q.reshape(length, K_HEADS, DK).repeat_interleave(V_HEADS // K_HEADS, 1)
    k = k.reshape(length, K_HEADS, DK).repeat_interleave(V_HEADS // K_HEADS, 1)
    v = v.reshape(length, V_HEADS, DV)
    beta = torch.sigmoid(b)
    g = -w["A_log"].exp() * F.softplus(a + w["dt_bias"])
    s = (
        torch.zeros(V_HEADS, DK, DV, dtype=torch.float64)
        if state is None
        else state.double().clone()
    )
    out = torch.empty(length, V_HEADS, DV, dtype=torch.float64)
    for t in range(length):
        qt = q[t] / (q[t].pow(2).sum(-1, keepdim=True) + 1e-6).sqrt() / math.sqrt(DK)
        kt = k[t] / (k[t].pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        s = s * g[t].exp()[:, None, None]
        predicted = torch.einsum("hkv,hk->hv", s, kt)
        s = s + torch.einsum("hk,hv->hkv", kt, (v[t] - predicted) * beta[t][:, None])
        out[t] = torch.einsum("hkv,hk->hv", s, qt)
    normed = out / (out.pow(2).mean(-1, keepdim=True) + EPS).sqrt()
    normed = w["norm"] * normed * F.silu(z)
    return normed.reshape(length, V_HEADS * DV) @ w["out_proj"].T, history.T, s


def run_mixer(hidden_batch, lengths, w, decode=False, conv=None, state=None, chunk=64):
    batch = hidden_batch.shape[0]
    conv = torch.zeros(batch, CONV_DIM, CONV - 1) if conv is None else conv
    state = torch.zeros(batch, V_HEADS, DK, DV) if state is None else state
    return gdn.gated_deltanet_mixer(
        hidden_batch,
        w,
        conv,
        state,
        lengths,
        num_k_heads=K_HEADS,
        num_v_heads=V_HEADS,
        head_k_dim=DK,
        head_v_dim=DV,
        eps=EPS,
        decode=decode,
        chunk_size=chunk,
    )


# ── the delta rule itself ────────────────────────────────────────────────


@pytest.mark.parametrize("length", [1, 5, 16, 17, 40])
@pytest.mark.parametrize("chunk", [4, 16])
def test_chunked_matches_recurrent(length, chunk):
    q, k = torch.randn(2, length, V_HEADS, DK), torch.randn(2, length, V_HEADS, DK)
    v = torch.randn(2, length, V_HEADS, DV)
    g = -F.softplus(torch.randn(2, length, V_HEADS))
    beta = torch.rand(2, length, V_HEADS)
    s0 = torch.randn(2, V_HEADS, DK, DV) * 0.1
    lengths = torch.tensor([length, length])
    out_r, s_r = gdn.recurrent_gated_delta_rule(q, k, v, g, beta, s0, lengths)
    out_c, s_c = gdn.chunk_gated_delta_rule(
        q, k, v, g, beta, s0, lengths, chunk_size=chunk
    )
    torch.testing.assert_close(out_c, out_r, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(s_c, s_r, atol=2e-5, rtol=2e-5)


def test_unit_lower_inverse_matches_solve():
    system = torch.randn(3, 16, 16) * 0.3
    inverse = gdn._unit_lower_inverse(system.clone())
    unit = system.tril(-1) + torch.eye(16)
    torch.testing.assert_close(
        inverse @ unit, torch.eye(16).expand(3, 16, 16), atol=1e-5, rtol=0
    )


# ── the whole mixer against the independent float64 reference ───────────


@pytest.mark.parametrize("length", [1, 3, 9, 20])
def test_mixer_prefill_matches_reference(length):
    w = make_weights()
    hidden = torch.randn(length, HIDDEN)
    expected, conv_ref, state_ref = reference_mixer(hidden, w)
    out, conv, state = run_mixer(hidden[None], torch.tensor([length]), w, chunk=4)
    torch.testing.assert_close(out[0].double(), expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(conv[0].double(), conv_ref, atol=1e-6, rtol=0)
    torch.testing.assert_close(state[0].double(), state_ref, atol=1e-5, rtol=1e-4)


def test_prefill_then_decode_equals_one_long_prefill():
    """The decode path continues exactly where prefill left its state."""
    w = make_weights()
    hidden = torch.randn(12, HIDDEN)
    expected, _, _ = reference_mixer(hidden, w)
    out, conv, state = run_mixer(hidden[None, :8], torch.tensor([8]), w, chunk=4)
    pieces = [out[0]]
    for t in range(8, 12):
        step, conv, state = run_mixer(
            hidden[None, t : t + 1],
            torch.tensor([1]),
            w,
            decode=True,
            conv=conv,
            state=state,
        )
        pieces.append(step[0])
    torch.testing.assert_close(
        torch.cat(pieces).double(), expected, atol=1e-5, rtol=1e-4
    )


def test_split_prefill_equals_whole_prefill():
    """Chunked-prefill across scheduler steps carries both states correctly."""
    w = make_weights()
    hidden = torch.randn(15, HIDDEN)
    whole, conv_w, state_w = run_mixer(hidden[None], torch.tensor([15]), w, chunk=4)
    first, conv, state = run_mixer(hidden[None, :6], torch.tensor([6]), w, chunk=4)
    second, conv, state = run_mixer(
        hidden[None, 6:], torch.tensor([9]), w, conv=conv, state=state, chunk=4
    )
    torch.testing.assert_close(
        torch.cat([first[0], second[0]]), whole[0], atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(state, state_w, atol=1e-5, rtol=1e-4)
    # The conv state holds raw in_proj_qkv projections. F32 matmul is not
    # bit-reproducible across input shapes (BLAS blocking differs for 6, 9 and
    # 15 rows), so this is F32 rounding, not bit equality. The gather index
    # itself is guarded by test_conv_state_is_gathered_from_real_inputs.
    torch.testing.assert_close(conv, conv_w, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("decode", [False, True])
def test_padding_is_an_exact_no_op(decode):
    """A short request padded next to a long one matches running it alone."""
    w = make_weights()
    long_len, short_len = (1, 1) if decode else (11, 4)
    long_h, short_h = torch.randn(long_len, HIDDEN), torch.randn(short_len, HIDDEN)
    batch = torch.zeros(2, long_len, HIDDEN)
    batch[0, :long_len], batch[1, :short_len] = long_h, short_h
    out, conv, state = run_mixer(
        batch, torch.tensor([long_len, short_len]), w, decode=decode, chunk=4
    )
    alone, conv_a, state_a = run_mixer(
        short_h[None], torch.tensor([short_len]), w, decode=decode, chunk=4
    )
    torch.testing.assert_close(out[1, :short_len], alone[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(state[1], state_a[0], atol=1e-6, rtol=1e-5)
    # F32 rounding, not bit equality: a [2, L, H] projection and a [1, L, H]
    # one take different BLAS paths. See test_conv_state_is_gathered_from_real_inputs.
    torch.testing.assert_close(conv[1], conv_a[0], atol=1e-6, rtol=1e-5)


def test_conv_state_is_gathered_from_real_inputs():
    """The carried conv state is the last K-1 *real* inputs, not the padded tail.

    Discriminating on purpose: a correct gather lands at F32 rounding (~1e-7)
    from the true inputs, while gathering the padded tail misses by the full
    magnitude of the data. Bit-equality tests cannot make that distinction.
    """
    channels, width = 48, CONV
    projection = torch.randn(channels, HIDDEN) * 0.2
    kernel = torch.randn(channels, width) * 0.3
    long_len, short_len = 11, 4
    batch = torch.zeros(2, long_len, HIDDEN)
    batch[0] = torch.randn(long_len, HIDDEN)
    batch[1, :short_len] = torch.randn(short_len, HIDDEN)
    mixed = F.linear(batch, projection).transpose(1, 2)
    _, state = gdn.causal_conv1d(
        mixed,
        kernel,
        torch.zeros(2, channels, width - 1),
        torch.tensor([long_len, short_len]),
    )
    truth = (
        batch[1, short_len - (width - 1) : short_len].double() @ projection.double().T
    ).T
    tail = mixed[1, :, long_len - (width - 1) : long_len].double()
    assert (state[1].double() - truth).abs().max() < 1e-6
    assert (state[1].double() - tail).abs().max() > 0.1 * truth.abs().max()


def test_zero_length_row_leaves_state_untouched():
    w = make_weights()
    conv0 = torch.randn(1, CONV_DIM, CONV - 1)
    state0 = torch.randn(1, V_HEADS, DK, DV)
    _, conv, state = run_mixer(
        torch.zeros(1, 3, HIDDEN),
        torch.tensor([0]),
        w,
        conv=conv0,
        state=state0,
        chunk=4,
    )
    torch.testing.assert_close(state, state0, atol=0, rtol=0)
    torch.testing.assert_close(conv, conv0, atol=0, rtol=0)


# ── flat tokens <-> padded requests ──────────────────────────────────────


def test_token_layout_round_trips():
    counts_list = [3, 1, 5]
    request_of_token = torch.repeat_interleave(
        torch.arange(3), torch.tensor(counts_list)
    )
    x = torch.randn(sum(counts_list), 7)
    counts, offsets = gdn.segment(request_of_token, 3)
    assert counts.tolist() == counts_list
    padded = gdn.tokens_to_padded(x, offsets, counts, max_len=5)
    assert padded.shape == (3, 5, 7)
    assert torch.all(padded[1, 1:] == 0)
    torch.testing.assert_close(
        gdn.padded_to_tokens(padded, request_of_token, offsets), x
    )


# ── slots survive condense() ─────────────────────────────────────────────


def test_state_follows_the_request_not_the_row():
    """Simulate condense(): requests move rows, their state must not."""
    buffers = RecurrentStateBuffers(
        [0],
        num_slots=4,
        conv_dim=CONV_DIM,
        conv_width=CONV - 1,
        num_v_heads=V_HEADS,
        head_k_dim=DK,
        head_v_dim=DV,
        conv_dtype=torch.float32,
    )
    alloc = SlotAllocator(4)
    for rid in ("a", "b", "c"):
        alloc.acquire(rid)
    marks = {rid: float(i + 1) for i, rid in enumerate(("a", "b", "c"))}
    slots = torch.tensor(alloc.slots_for_rows(["a", "b", "c"], buffers.sink))
    buffers.write(
        0,
        slots,
        torch.stack([torch.full((CONV_DIM, CONV - 1), marks[r]) for r in "abc"]),
        torch.stack([torch.full((V_HEADS, DK, DV), marks[r]) for r in "abc"]),
    )
    # "b" finishes; condense() moves "c" into row 1.
    alloc.release("b")
    rows = ["a", "c"]
    slots = torch.tensor(alloc.slots_for_rows(rows, buffers.sink))
    _, recurrent = buffers.read(0, slots, torch.tensor([False, False]))
    assert recurrent[0].unique().item() == marks["a"]
    assert recurrent[1].unique().item() == marks["c"]


def test_new_request_starts_from_zero_even_in_a_reused_slot():
    buffers = RecurrentStateBuffers(
        [0],
        num_slots=2,
        conv_dim=CONV_DIM,
        conv_width=CONV - 1,
        num_v_heads=V_HEADS,
        head_k_dim=DK,
        head_v_dim=DV,
        conv_dtype=torch.float32,
    )
    alloc = SlotAllocator(2)
    slot = alloc.acquire("old")
    buffers.write(
        0,
        torch.tensor([slot]),
        torch.ones(1, CONV_DIM, CONV - 1),
        torch.ones(1, V_HEADS, DK, DV),
    )
    alloc.release("old")
    assert alloc.acquire("new") == slot
    conv, recurrent = buffers.read(0, torch.tensor([slot]), torch.tensor([True]))
    assert torch.count_nonzero(conv) == 0 and torch.count_nonzero(recurrent) == 0


def test_allocator_refuses_rather_than_overwrites():
    alloc = SlotAllocator(1)
    alloc.acquire("x")
    with pytest.raises(RuntimeError, match="no free Gated DeltaNet state slot"):
        alloc.acquire("y")


def test_9b_state_budget():
    per_slot = RecurrentStateBuffers.bytes_per_slot(
        num_layers=24,
        conv_dim=8192,
        conv_width=3,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        conv_itemsize=2,
    )
    assert 48 * 2**20 < per_slot < 50 * 2**20


# ── transformers' own layer, when available ──────────────────────────────


def test_against_transformers_layer():
    pytest.importorskip("transformers")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        hidden_size=HIDDEN,
        linear_num_key_heads=K_HEADS,
        linear_num_value_heads=V_HEADS,
        linear_key_head_dim=DK,
        linear_value_head_dim=DV,
        linear_conv_kernel_dim=CONV,
        rms_norm_eps=EPS,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        hidden_act="silu",
    )
    layer = hf.Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    w = make_weights()
    with torch.no_grad():
        layer.in_proj_qkv.weight.copy_(w["in_proj_qkv"])
        layer.in_proj_z.weight.copy_(w["in_proj_z"])
        layer.in_proj_b.weight.copy_(w["in_proj_b"])
        layer.in_proj_a.weight.copy_(w["in_proj_a"])
        layer.conv1d.weight.copy_(w["conv1d"].unsqueeze(1))
        layer.A_log.copy_(w["A_log"])
        layer.dt_bias.copy_(w["dt_bias"])
        layer.norm.weight.copy_(w["norm"])
        layer.out_proj.weight.copy_(w["out_proj"])
        hidden = torch.randn(1, 13, HIDDEN)
        expected = layer(hidden)
    out, _, _ = run_mixer(hidden, torch.tensor([13]), w, chunk=4)
    torch.testing.assert_close(out, expected, atol=2e-5, rtol=2e-4)
