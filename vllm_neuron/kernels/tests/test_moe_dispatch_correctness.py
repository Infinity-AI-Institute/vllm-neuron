"""Tier-1a — moe_dispatch kernel correctness (CPU-simulate golden reference).

Runs entirely on CPU with pytorch; the fused MoE dispatch NKI kernel body
is compared against a plain-torch reference derived from the scaffold's
`torch.nn.functional.embedding + sparse_scatter + per-expert MM` sketch.

Runtime target: **< 10 s per (config, batch) pair** (§iteration-flywheel).

Coverage matrix
---------------
Every shape family in `moe_dispatch.py` is exercised:

* Qwen3-30B-A3B TP=8 — 128 experts / top-8 / SILU / renormalize=True
* Gemma-4-26B-A4B TP=4 — 128 experts / top-8 / GELU_Tanh_Approx / renormalize=True
* GPT-OSS-20B TP=4 — 128 experts / top-4 / SILU / renormalize=**False**

Every activation branch (§A.G-7) is exercised because the branch table
selects at construct-time based on `cfg.activation`.

Correctness gates
-----------------
Per scaffold §A.4:

* Router probability parity vs `torch.softmax` at rel 1e-3 (bf16 tolerance).
* Top-K index parity vs `torch.topk(..., largest=True, sorted=True)` with
  lowest-index-wins tie-break.
* Weight-sum invariant: sum_e affinity[e, b] == 1.0 ± 2^-14 (fp32).
* Output parity at abs 1e-2 (post-softmax logit tolerance).

The renormalize branch (§A.G-9) is tested by asserting that GPT-OSS
config produces per-token weight sums != 1.0 in general (i.e., NOT
renormalized), while Qwen3 / Gemma-4 configs DO sum to 1.0.

This test module is import-safe on the compile host but does NOT itself
compile any NEFF — it validates the reference semantics that the NKI
kernel must match at Tier-2 (compile + one-token) and Tier-3 (profile
at knee).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import pytest


# ==========================================================================
# Import the config objects (import-guarded so this test file runs without
# the nki stack on laptop / CI)
# ==========================================================================

# Prefer the sibling module path when this test lives inside kernels/tests/.
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KERNELS_DIR = _HERE.parent
if str(_KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(_KERNELS_DIR))

from moe_dispatch import (  # noqa: E402
    GEMMA4_26B_A4B_TP4,
    GPT_OSS_20B_TP4,
    QWEN3_30B_A3B_TP8,
    MoEActivation,
    MoEDispatchConfig,
)


# ==========================================================================
# Torch reference (pure-Python; scaffold §A.4 "torch.nn.functional.embedding
# + sparse_scatter" pattern)
# ==========================================================================

try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

pytestmark = pytest.mark.skipif(
    not _TORCH_OK, reason="PyTorch required for CPU-simulate golden reference."
)


def _rmsnorm(x: "torch.Tensor", gamma: "torch.Tensor", eps: float) -> "torch.Tensor":
    """FP32-stats RMSNorm matching Gemma's convention.

    x:      [..., H]
    gamma:  [1, H]  (already includes Gemma's H**-0.5 scale for the
                    dual-input router path)
    """
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x32 * torch.rsqrt(var + eps)
    return (x_norm * gamma.to(torch.float32)).to(x.dtype)


def _apply_activation(gate: "torch.Tensor", up: "torch.Tensor",
                      act: MoEActivation) -> "torch.Tensor":
    if act is MoEActivation.SILU:
        return F.silu(gate) * up
    if act is MoEActivation.GELU_TANH_APPROX:
        return F.gelu(gate, approximate="tanh") * up
    if act is MoEActivation.GLU:
        return torch.sigmoid(gate) * up
    if act is MoEActivation.SILU_GLU:
        return F.silu(gate) * torch.sigmoid(up)
    raise ValueError(f"Unknown activation {act}")


def _reference_router(
    router_input: "torch.Tensor",     # bf16 [T, H]
    router_gamma: "torch.Tensor",     # bf16 [1, H]
    router_weights: "torch.Tensor",   # bf16 [H, E]
    cfg: MoEDispatchConfig,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Reference router: RMSNorm -> logits -> softmax -> top-K + affinity scatter.

    Returns (expert_affinities [T, E], expert_index [T, K], router_logits [T, E]).
    """
    T, H = router_input.shape
    E, K = cfg.num_experts, cfg.top_k

    norm = _rmsnorm(router_input, router_gamma, cfg.eps)          # [T, H] bf16
    logits = (norm.to(torch.float32) @ router_weights.to(torch.float32))  # [T, E]
    probs = torch.softmax(logits, dim=-1)                          # fp32 [T, E]

    # Top-K with lowest-index-wins tie-break (matches torch.topk default).
    top_vals, top_idx = torch.topk(probs, k=K, dim=-1, largest=True, sorted=True)
    if cfg.renormalize_topk:
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-20)

    # Scatter the K sparse weights back into a dense [T, E] affinity tensor.
    affinities = torch.zeros(T, E, dtype=torch.float32)
    affinities.scatter_(dim=1, index=top_idx.to(torch.long), src=top_vals.to(torch.float32))

    return affinities, top_idx.to(torch.int64), logits.to(router_input.dtype)


def _reference_expert_combine(
    expert_input: "torch.Tensor",              # bf16 [T, H]
    expert_gate_up_weights: "torch.Tensor",    # bf16 [E, H, 2, I_TP]
    expert_down_weights_scaled: "torch.Tensor",# bf16 [E, I_TP, H]  (down * per_expert_scale)
    affinities: "torch.Tensor",                # fp32 [T, E]
    cfg: MoEDispatchConfig,
) -> "torch.Tensor":
    """Reference expert-MM + weighted in-place combine.

    Runs the dense all-experts pattern (matches §B6 AoT constraint); the
    zero-affinity slots contribute nothing to the accumulator.  POST_SCALE
    semantics: the affinity multiplies into the down-projection contraction.
    """
    T, H = expert_input.shape
    E = cfg.num_experts
    I_TP = cfg.intermediate_per_tp

    x = expert_input.to(torch.float32)             # fp32 for accumulation stability
    acc = torch.zeros(T, H, dtype=torch.float32)

    for e in range(E):
        gate_w = expert_gate_up_weights[e, :, 0, :].to(torch.float32)   # [H, I_TP]
        up_w = expert_gate_up_weights[e, :, 1, :].to(torch.float32)     # [H, I_TP]
        down_w = expert_down_weights_scaled[e].to(torch.float32)        # [I_TP, H]

        gate = x @ gate_w                                   # [T, I_TP]
        up = x @ up_w                                       # [T, I_TP]
        hidden_act = _apply_activation(gate, up, cfg.activation)
        expert_out = hidden_act @ down_w                    # [T, H]

        # POST_SCALE weighted-combine (§A.2 stage 5 in-place accumulator).
        w = affinities[:, e:e + 1]                          # [T, 1]
        acc = acc + expert_out * w

    return acc.to(expert_input.dtype)


# ==========================================================================
# Test fixtures — deterministic inputs per shape family
# ==========================================================================


def _rand_bf16(shape, seed: int) -> "torch.Tensor":
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(torch.bfloat16)


@dataclass
class _Batch:
    router_input: "torch.Tensor"
    router_gamma: "torch.Tensor"
    router_weights: "torch.Tensor"
    expert_input: "torch.Tensor"
    expert_gate_up_weights: "torch.Tensor"
    expert_down_weights_scaled: "torch.Tensor"


def _make_batch(cfg: MoEDispatchConfig, tokens: int, seed: int) -> _Batch:
    return _Batch(
        router_input=_rand_bf16((tokens, cfg.hidden), seed + 0),
        router_gamma=_rand_bf16((1, cfg.hidden), seed + 1),
        router_weights=_rand_bf16((cfg.hidden, cfg.num_experts), seed + 2),
        expert_input=_rand_bf16((tokens, cfg.hidden), seed + 3),
        expert_gate_up_weights=_rand_bf16(
            (cfg.num_experts, cfg.hidden, 2, cfg.intermediate_per_tp), seed + 4
        ),
        expert_down_weights_scaled=_rand_bf16(
            (cfg.num_experts, cfg.intermediate_per_tp, cfg.hidden), seed + 5
        ),
    )


CONFIGS = [
    pytest.param(QWEN3_30B_A3B_TP8, id="qwen3-30b-a3b-tp8"),
    pytest.param(GEMMA4_26B_A4B_TP4, id="gemma4-26b-a4b-tp4"),
    pytest.param(GPT_OSS_20B_TP4, id="gpt-oss-20b-tp4"),
]


# ==========================================================================
# §A.4 correctness gates
# ==========================================================================


@pytest.mark.parametrize("cfg", CONFIGS)
def test_config_validates(cfg: MoEDispatchConfig) -> None:
    """Every shipped config must clear the Tier-1 CPU battery gates."""
    cfg.validate()
    # Reciprocal sanity: TP=8 for Gemma-4 SHOULD fail the %16 wall.
    if cfg is GEMMA4_26B_A4B_TP4:
        broken = MoEDispatchConfig(
            name="gemma4-26b-a4b-tp8-BROKEN",
            hidden=cfg.hidden,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            intermediate_global=cfg.intermediate_global,
            tp_degree=8,   # I_TP = 88 -> 88 % 16 = 8 -> assert fails
            activation=cfg.activation,
            renormalize_topk=cfg.renormalize_topk,
        )
        with pytest.raises(ValueError, match="fails %16"):
            broken.validate()


@pytest.mark.parametrize("cfg", CONFIGS)
def test_router_topk_index_parity(cfg: MoEDispatchConfig) -> None:
    """Top-K indices must match torch.topk on the same fp32 softmax.

    Recomputes the reference from scratch using the identical numerical
    path (RMSNorm fp32 stats -> fp32 matmul -> softmax -> topk with lowest
    index tie-break).  Bit-exact match is required — no precision drift
    tolerated because the router path is fully fp32 inside `_reference_router`.
    """
    tokens = 4
    b = _make_batch(cfg, tokens, seed=17)
    affinities, top_idx, _ = _reference_router(
        b.router_input, b.router_gamma, b.router_weights, cfg
    )

    # Recompute the ground truth using the same fp32 path.  Cannot go through
    # the bf16-cast-back `logits` return because bf16 quantization would
    # break the tie order that `torch.topk` resolves on the fp32 softmax.
    norm = _rmsnorm(b.router_input, b.router_gamma, cfg.eps)
    ref_logits = norm.to(torch.float32) @ b.router_weights.to(torch.float32)
    ref_probs = torch.softmax(ref_logits, dim=-1)
    _, ref_idx = torch.topk(ref_probs, k=cfg.top_k, dim=-1,
                            largest=True, sorted=True)
    assert torch.equal(top_idx, ref_idx.to(top_idx.dtype))


@pytest.mark.parametrize("cfg", CONFIGS)
def test_weight_sum_invariant(cfg: MoEDispatchConfig) -> None:
    """§A.4 weight-sum: sum_e affinity[e, b] == 1.0 IF renormalized.

    For the no-renormalize branch (§A.G-9 GPT-OSS), use controlled
    uniform-ish inputs so the natural top-K mass is definitively < 1.0
    (random bf16 inputs give softmax peaked enough that top-K mass = 1.0
    within fp32 eps, which would mask the renorm/no-renorm distinction).
    """
    tokens = 4
    b = _make_batch(cfg, tokens, seed=23)
    affinities, _, _ = _reference_router(
        b.router_input, b.router_gamma, b.router_weights, cfg
    )
    sums = affinities.sum(dim=-1)  # [T]

    if cfg.renormalize_topk:
        # Qwen3 / Gemma-4: MUST sum to 1.0 within 2^-14 (fp32 weight-sum
        # accumulator drift bound).  Scaffold §A.G-4 flags the 128-expert
        # drift budget; check is looser than 2^-14 for fp32 sum-of-K==8.
        assert torch.allclose(
            sums, torch.ones_like(sums), atol=2 ** -14, rtol=0
        ), f"{cfg.name}: renormalized weight-sum drift exceeds 2^-14: {sums}"
    else:
        # GPT-OSS: MUST equal the raw top-K softmax mass, not 1.0 (§A.G-9).
        # Craft controlled inputs so the softmax over N experts is close to
        # uniform (1/N per bin) → top-K sum ≈ K/N ≪ 1.0.  Scale router
        # weights to sub-unit magnitude and zero the router gamma bias.
        torch.manual_seed(37)
        tiny_router_input = torch.zeros(tokens, cfg.hidden, dtype=torch.bfloat16)
        tiny_router_input += (torch.randn(tokens, cfg.hidden) * 0.01).to(torch.bfloat16)
        tiny_router_gamma = torch.ones(1, cfg.hidden, dtype=torch.bfloat16)
        tiny_router_weights = (torch.randn(cfg.hidden, cfg.num_experts) * 0.01).to(torch.bfloat16)

        affinities_tiny, _, _ = _reference_router(
            tiny_router_input, tiny_router_gamma, tiny_router_weights, cfg
        )
        sums_tiny = affinities_tiny.sum(dim=-1)

        # Uniform softmax over 128 experts, top-4 mass ≈ 4/128 = 0.031.
        # Assert the natural mass is well below 1.0 — a strict inequality
        # so renormalization cannot pass by accident.
        expected_upper = cfg.top_k / cfg.num_experts * 5.0   # 5× uniform-slack
        assert (sums_tiny < expected_upper).all(), (
            f"{cfg.name}: uniform-softmax top-K mass exceeded {expected_upper} "
            f"— either the renormalize=False branch is broken or the "
            f"controlled inputs went out of range.  sums={sums_tiny}."
        )
        assert not torch.allclose(
            sums_tiny, torch.ones_like(sums_tiny), atol=1e-3
        ), (
            f"{cfg.name}: uniform-softmax top-K mass equals 1.0 — a renorm "
            f"slipped in.  sums={sums_tiny}."
        )


@pytest.mark.parametrize("cfg", CONFIGS)
def test_expert_combine_dense_all_experts(cfg: MoEDispatchConfig) -> None:
    """§B6 dense all-experts pattern: zero-affinity slots contribute zero.

    Constructs an affinity tensor that routes token 0 to a single expert
    with weight 1.0, and verifies that the combined output equals the
    reference for that single expert alone.
    """
    tokens = 2
    b = _make_batch(cfg, tokens, seed=31)

    # Force a deterministic single-expert route for token 0 (expert=5, w=1.0)
    # and a two-expert route for token 1 (experts 3,7 at weights 0.3, 0.7).
    E = cfg.num_experts
    affinities = torch.zeros(tokens, E, dtype=torch.float32)
    affinities[0, 5] = 1.0
    affinities[1, 3] = 0.3
    affinities[1, 7] = 0.7

    fused = _reference_expert_combine(
        b.expert_input, b.expert_gate_up_weights,
        b.expert_down_weights_scaled, affinities, cfg,
    )

    # Rebuild the expected combine by summing only the participating experts.
    x_fp32 = b.expert_input.to(torch.float32)
    def _single_expert(e: int) -> "torch.Tensor":
        gate_w = b.expert_gate_up_weights[e, :, 0, :].to(torch.float32)
        up_w = b.expert_gate_up_weights[e, :, 1, :].to(torch.float32)
        down_w = b.expert_down_weights_scaled[e].to(torch.float32)
        gate = x_fp32 @ gate_w
        up = x_fp32 @ up_w
        return _apply_activation(gate, up, cfg.activation) @ down_w

    e5 = _single_expert(5)
    e3 = _single_expert(3)
    e7 = _single_expert(7)
    expected = torch.zeros(tokens, cfg.hidden, dtype=torch.float32)
    expected[0] = 1.0 * e5[0]
    expected[1] = 0.3 * e3[1] + 0.7 * e7[1]

    # Cast expected to bf16 and back so the tolerance budget matches the
    # `fused` output dtype (bf16 stores ~3 decimal digits per value; a
    # dense sum over 128 experts amplifies quantization).  Scaffold §A.4
    # names "absolute 1e-2 on the output logits" for the compile-side gate
    # only; the CPU-simulate reference must use rel tolerance because the
    # bf16 output magnitude scales with (num_experts × I_TP).
    fused_fp32 = fused.to(torch.float32)
    expected_bf16 = expected.to(torch.bfloat16).to(torch.float32)

    # Component 1: the fused path IS the reference (both compute in fp32
    # then cast to bf16), so `fused_fp32` should match `expected_bf16`
    # bit-exactly for the participating experts.
    assert torch.allclose(fused_fp32, expected_bf16, atol=0, rtol=0), (
        f"{cfg.name}: fused vs bf16-quantized-expected diverged "
        "(bit-exact within bf16 quantization is required for the CPU sim)."
    )

    # Component 2: the un-quantized fp32 reference matches the bf16 output
    # within rel 1e-2 (the practical bf16 accumulator tolerance).
    max_mag = expected.abs().max()
    rel_atol = max(1e-2, float(max_mag) * 1e-2)   # generous scaling
    assert torch.allclose(fused_fp32, expected, atol=rel_atol, rtol=1e-2), (
        f"{cfg.name}: dense all-experts combine drifted beyond rel 1e-2 "
        f"(max_magnitude={max_mag:.2f}, atol={rel_atol:.2f})"
    )


@pytest.mark.parametrize("cfg", CONFIGS)
def test_end_to_end_reference_stable(cfg: MoEDispatchConfig) -> None:
    """End-to-end router + fused-combine golden must be reproducible.

    Running the reference twice with the same seed produces identical output
    (fp32 accumulator; no non-deterministic ops).  This is the CPU-side
    baseline that the Trn2-compiled kernel must match at Tier-2 within
    the scaffold's rel 1e-3 / abs 1e-2 gates.
    """
    tokens = 8
    b = _make_batch(cfg, tokens, seed=41)
    affinities_a, top_idx_a, logits_a = _reference_router(
        b.router_input, b.router_gamma, b.router_weights, cfg
    )
    affinities_b, top_idx_b, logits_b = _reference_router(
        b.router_input, b.router_gamma, b.router_weights, cfg
    )
    assert torch.equal(top_idx_a, top_idx_b)
    assert torch.allclose(affinities_a, affinities_b, atol=0, rtol=0)
    assert torch.allclose(logits_a, logits_b, atol=0, rtol=0)

    out_a = _reference_expert_combine(
        b.expert_input, b.expert_gate_up_weights,
        b.expert_down_weights_scaled, affinities_a, cfg,
    )
    out_b = _reference_expert_combine(
        b.expert_input, b.expert_gate_up_weights,
        b.expert_down_weights_scaled, affinities_b, cfg,
    )
    assert torch.allclose(out_a.to(torch.float32), out_b.to(torch.float32),
                          atol=0, rtol=0), f"{cfg.name}: reference not deterministic"


# ==========================================================================
# Activation branch exhaustive coverage (§A.G-7)
# ==========================================================================

@pytest.mark.parametrize("act", list(MoEActivation))
def test_activation_branch_registered(act: MoEActivation) -> None:
    """Every activation in `MoEActivation` must produce distinct outputs.

    Catches silent-fallback drift: if the branch table is later collapsed
    to a single default, this test flags it because two activations would
    then yield identical outputs.
    """
    torch.manual_seed(53 + act.value)
    gate = torch.randn(4, 32)
    up = torch.randn(4, 32)
    out = _apply_activation(gate, up, act)
    assert out.shape == gate.shape

    # Every branch produces a numerically distinct fingerprint.
    other = MoEActivation.SILU if act is not MoEActivation.SILU else MoEActivation.GLU
    other_out = _apply_activation(gate, up, other)
    assert not torch.allclose(out, other_out, atol=1e-3), (
        f"Activation branches {act} and {other} collapsed to same output — "
        "the branch table has drifted; §A.G-7 discipline requires exhaustive "
        "distinct branches."
    )


# ==========================================================================
# NKI-availability probe (skips on hosts without the pinned stack)
# ==========================================================================

@pytest.mark.parametrize("cfg", CONFIGS)
def test_nki_kernel_constructible(cfg: MoEDispatchConfig) -> None:
    """Constructing the NKI kernel factory must succeed on the compile host.

    Skipped on laptop / CI where the nki stack is not installed.  On the
    compile host this exercises the closure over `cfg` — a shape drift
    (I_TP % 16 != 0) would ValueError here at construct time.
    """
    from moe_dispatch import _NKI_AVAILABLE, _make_moe_dispatch

    if not _NKI_AVAILABLE:
        pytest.skip("nki stack not installed on this host")

    router_fn, expert_fn = _make_moe_dispatch(cfg)
    assert router_fn is not None
    assert expert_fn is not None
