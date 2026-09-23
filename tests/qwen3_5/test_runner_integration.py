# SPDX-License-Identifier: Apache-2.0
"""The runner contract and the math together, on CPU.

Reproduces vllm-neuron's exact flat layout, ``[req0_real, req0_pad, req1_real,
req1_pad, ..., bucket_pad]``, and checks that every request comes out exactly
as if it had run alone, across steps and across condense() moves.
"""

import numpy as np
import pytest
import torch

from vllm_neuron.model.qwen3_5.layer_step import gdn_layer_step
from vllm_neuron.model.qwen3_5.runner_state import GdnRunnerState
from vllm_neuron.model.qwen3_5.state import RecurrentStateBuffers

from .test_gated_deltanet import (
    CONV,
    CONV_DIM,
    DK,
    DV,
    EPS,
    HIDDEN,
    K_HEADS,
    V_HEADS,
    make_weights,
    run_mixer,
)

torch.manual_seed(1)
LAYER = 0
DIMS = {
    "num_k_heads": K_HEADS,
    "num_v_heads": V_HEADS,
    "head_k_dim": DK,
    "head_v_dim": DV,
    "eps": EPS,
    "chunk_size": 4,
}


def buffers(slots=8):
    return RecurrentStateBuffers(
        [LAYER],
        num_slots=slots,
        conv_dim=CONV_DIM,
        conv_width=CONV - 1,
        num_v_heads=V_HEADS,
        head_k_dim=DK,
        head_v_dim=DV,
        conv_dtype=torch.float32,
    )


def flat_layout(real, padded_sizes, bucket_pad):
    """Flatten per-request real tokens the way the runner does, garbage in the pads."""
    pieces = []
    for tokens, size in zip(real, padded_sizes):
        pieces.append(tokens)
        pieces.append(torch.randn(size - tokens.shape[0], HIDDEN) * 50)  # loud garbage
    pieces.append(torch.randn(bucket_pad, HIDDEN) * 50)
    return torch.cat(pieces)


def real_rows(flat, real, padded_sizes):
    out, start = [], 0
    for tokens, size in zip(real, padded_sizes):
        out.append(flat[start : start + tokens.shape[0]])
        start += size
    return out


def alone(tokens, w, decode=False, conv=None, state=None):
    out, conv, state = run_mixer(
        tokens[None],
        torch.tensor([tokens.shape[0]]),
        w,
        decode=decode,
        conv=conv,
        state=state,
        chunk=4,
    )
    return out[0], conv, state


def test_interleaved_padding_matches_each_request_alone():
    """Padding tokens are attributed to their own request; they must not leak."""
    w, buf, runner = make_weights(), buffers(), GdnRunnerState(8)
    real = [torch.randn(n, HIDDEN) for n in (5, 1, 3)]
    padded_sizes = [8, 4, 4]
    ids = ["a", "b", "c"]
    for rid in ids:
        runner.acquire(rid)
    flat = flat_layout(real, padded_sizes, bucket_pad=3)
    step = runner.step_inputs(
        ids,
        np.zeros(3),
        np.array([5, 1, 3]),
        np.array(padded_sizes),
        num_reqs_padded=4,
        num_tokens=flat.shape[0],
        decode=False,
    )
    out = gdn_layer_step(flat, w, buf, LAYER, step, decode=False, **DIMS)
    for rid, tokens, got in zip(ids, real, real_rows(out, real, padded_sizes)):
        expected, _, state = alone(tokens, w)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
        slot = runner.allocator.slot(rid)
        torch.testing.assert_close(
            buf.recurrent[LAYER][slot], state[0], atol=1e-5, rtol=1e-4
        )


def test_state_survives_condense_between_steps():
    """Step 1 prefills a,b,c. 'b' finishes; condense moves 'c' to row 1. Step 2
    decodes a and c; each must continue from its own state."""
    w, buf, runner = make_weights(), buffers(), GdnRunnerState(8)
    prompts = {rid: torch.randn(n, HIDDEN) for rid, n in (("a", 4), ("b", 6), ("c", 3))}
    for rid in "abc":
        runner.acquire(rid)
    real = [prompts[r] for r in "abc"]
    sizes = [4, 8, 4]
    flat = flat_layout(real, sizes, bucket_pad=0)
    step = runner.step_inputs(
        list("abc"),
        np.zeros(3),
        np.array([4, 6, 3]),
        np.array(sizes),
        num_reqs_padded=3,
        num_tokens=flat.shape[0],
        decode=False,
    )
    gdn_layer_step(flat, w, buf, LAYER, step, decode=False, **DIMS)

    runner.release("b")
    nxt = {r: torch.randn(1, HIDDEN) for r in "ac"}
    rows = ["a", "c"]  # 'c' moved from row 2 to row 1
    flat = torch.cat([nxt[r] for r in rows])
    step = runner.step_inputs(
        rows,
        np.array([4, 3]),
        np.array([1, 1]),
        np.array([1, 1]),
        num_reqs_padded=2,
        num_tokens=2,
        decode=True,
    )
    out = gdn_layer_step(flat, w, buf, LAYER, step, decode=True, **DIMS)

    for row, rid in enumerate(rows):
        _, conv, state = alone(prompts[rid], w)
        expected, _, _ = alone(nxt[rid], w, decode=True, conv=conv, state=state)
        torch.testing.assert_close(out[row : row + 1], expected, atol=1e-5, rtol=1e-4)


def test_chunked_prefill_across_steps_resets_only_once():
    w, buf, runner = make_weights(), buffers(), GdnRunnerState(8)
    prompt = torch.randn(10, HIDDEN)
    runner.acquire("a")
    first = runner.step_inputs(
        ["a"],
        np.array([0]),
        np.array([6]),
        np.array([8]),
        num_reqs_padded=1,
        num_tokens=8,
        decode=False,
    )
    flat = flat_layout([prompt[:6]], [8], bucket_pad=0)
    out1 = gdn_layer_step(flat, w, buf, LAYER, first, decode=False, **DIMS)
    second = runner.step_inputs(
        ["a"],
        np.array([6]),
        np.array([4]),
        np.array([4]),
        num_reqs_padded=1,
        num_tokens=4,
        decode=False,
    )
    assert bool(first.state_reset[0]) and not bool(second.state_reset[0])
    out2 = gdn_layer_step(prompt[6:], w, buf, LAYER, second, decode=False, **DIMS)
    expected, _, _ = alone(prompt, w)
    torch.testing.assert_close(
        torch.cat([out1[:6], out2]), expected, atol=1e-5, rtol=1e-4
    )


def test_prefix_caching_is_refused_not_silently_resumed():
    runner = GdnRunnerState(4)
    runner.acquire("a")
    with pytest.raises(RuntimeError, match="no-enable-prefix-caching"):
        runner.step_inputs(
            ["a"],
            np.array([16]),
            np.array([4]),
            np.array([4]),
            num_reqs_padded=1,
            num_tokens=4,
            decode=False,
        )


def test_reused_slot_does_not_inherit_a_finished_request():
    w, buf, runner = make_weights(), buffers(slots=1), GdnRunnerState(1)
    runner.acquire("old")
    step = runner.step_inputs(
        ["old"], np.array([0]), np.array([5]), np.array([5]), 1, 5, decode=False
    )
    gdn_layer_step(torch.randn(5, HIDDEN), w, buf, LAYER, step, decode=False, **DIMS)
    runner.release("old")
    runner.acquire("new")
    prompt = torch.randn(3, HIDDEN)
    step = runner.step_inputs(
        ["new"], np.array([0]), np.array([3]), np.array([3]), 1, 3, decode=False
    )
    out = gdn_layer_step(prompt, w, buf, LAYER, step, decode=False, **DIMS)
    expected, _, _ = alone(prompt, w)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)


def test_warmup_never_touches_live_state():
    w, buf, runner = make_weights(), buffers(), GdnRunnerState(8)
    runner.acquire("live")
    slot = runner.allocator.slot("live")
    buf.recurrent[LAYER][slot].fill_(3.0)
    step = runner.warmup_inputs(num_reqs_padded=4, num_tokens=16, decode=False)
    gdn_layer_step(torch.randn(16, HIDDEN), w, buf, LAYER, step, decode=False, **DIMS)
    assert torch.all(buf.recurrent[LAYER][slot] == 3.0)


def test_decode_bucket_padding_keeps_segment_length_one():
    """Decode pads 2 live requests to a 4-row bucket. The two tail tokens go to
    the sink, which overflows max_len=1; that must be harmless."""
    w, buf, runner = make_weights(), buffers(), GdnRunnerState(8)
    prompts = {rid: torch.randn(n, HIDDEN) for rid, n in (("a", 3), ("b", 5))}
    for rid in "ab":
        runner.acquire(rid)
    for rid in "ab":  # prefill each alone so both slots hold real state
        n = prompts[rid].shape[0]
        step = runner.step_inputs(
            [rid], np.array([0]), np.array([n]), np.array([n]), 1, n, decode=False
        )
        gdn_layer_step(prompts[rid], w, buf, LAYER, step, decode=False, **DIMS)
    nxt = torch.randn(2, HIDDEN)
    flat = torch.cat([nxt, torch.randn(2, HIDDEN) * 50])
    step = runner.step_inputs(
        list("ab"),
        np.array([3, 5]),
        np.array([1, 1]),
        np.array([1, 1]),
        num_reqs_padded=4,
        num_tokens=4,
        decode=True,
    )
    assert step.max_len == 1
    out = gdn_layer_step(flat, w, buf, LAYER, step, decode=True, **DIMS)
    for row, rid in enumerate("ab"):
        _, conv, state = alone(prompts[rid], w)
        expected, _, _ = alone(nxt[row : row + 1], w, decode=True, conv=conv, state=state)
        torch.testing.assert_close(out[row : row + 1], expected, atol=1e-5, rtol=1e-4)


def test_max_len_depends_only_on_the_compiled_shape():
    runner = GdnRunnerState(8)
    for rid in "ab":
        runner.acquire(rid)
    lens = set()
    for real in ([1, 1], [7, 2], [16, 0]):
        step = runner.step_inputs(
            list("ab"),
            np.zeros(2),
            np.array(real),
            np.array([max(r, 1) for r in real]),
            num_reqs_padded=2,
            num_tokens=32,
            decode=False,
        )
        lens.add(step.max_len)
        runner._fresh.update("ab")
    assert lens == {32}
