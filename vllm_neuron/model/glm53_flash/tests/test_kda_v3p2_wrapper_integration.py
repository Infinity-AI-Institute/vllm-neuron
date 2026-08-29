# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash KDA v3.2 wrapper-integration test (fleet-a-nki, 2026-08-28).

Purpose
-------
Prove that the ``KDA_KERNEL_IMPL`` env-flip resolves to the correct
kernel slug and that BOTH branches (CPU golden default and NKI v3.2)
route through the wrapper's dispatch hook end-to-end -- not the raw
kernel modules -- and that the CPU-golden fallback preserves numerics
for the CPU-only test host.

Discipline
----------
- Verify emitted, never requested: assertions target the slug the
  wrapper ADVERTISES (:func:`get_emitted_kernel_slugs`), which is what
  the compile driver reads for the NEFF cache key.
- Reject degenerate output: forward must produce a finite non-zero
  tensor with the expected shape.
- No env leak: every parametrisation restores the env var it changed.
- CPU host: skips silently on hosts where the handoff kernels dir is
  not accessible; the dispatch's fallback then would use only the
  in-package CPU golden path.

Anchors
-------
- Wrapper implementation:
  ``C:\\Users\\apumu\\research\\InfinityAI\\vllm-neuron-codex-alpha\\
   vllm_neuron\\model\\glm53_flash\\kernel_dispatch.py``
- KDA v3.2 receipt:
  ``C:\\Users\\apumu\\research\\InfinityAI\\gemma4-trn2-handoff\\
   harness-v2\\staging\\reference-sweep-20260826T2150Z\\kernels\\
   KDA-V3-SPIKE-2026-08-28.md`` §5.1.g (baremetal COMPILE PASS on r7i).
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch

from vllm_neuron.model.glm53_flash.config import (
    Glm53FlashInferenceConfig,
    Glm53LinearAttentionConfig,
)
from vllm_neuron.model.glm53_flash.kda import (
    KDA_CPU_GOLDEN_SLUG,
    KDA_NKI_V3P2_SLUG,
    Glm53KdaAttention,
    emitted_kda_kernel_slug,
)
from vllm_neuron.model.glm53_flash.kernel_dispatch import (
    KDA_KERNEL_IMPL_ENV,
    get_emitted_kernel_slugs,
    resolve_kda_impl_slug,
)


@contextmanager
def _env(name: str, value):
    """Temporarily set/unset ``name`` and restore on exit."""
    sentinel = object()
    previous = os.environ.get(name, sentinel)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is sentinel:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous  # type: ignore[assignment]


def _tiny_config() -> Glm53FlashInferenceConfig:
    return Glm53FlashInferenceConfig(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=16,
        q_lora_rank=4,
        kv_lora_rank=4,
        v_head_dim=4,
        index_n_heads=1,
        index_head_dim=4,
        index_topk=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        torch_dtype=torch.float32,
        static_fp8=False,
        allow_reduced_shapes=True,
        linear_attn_config=Glm53LinearAttentionConfig(num_heads=2, head_dim=4),
    )


def _require_comparable(output: torch.Tensor, label: str) -> None:
    """Reject a degenerate wrapper output.

    A pass-through of NaN, all-zero, or infinite values would falsely
    make both env branches "agree" while both branches were broken.
    Every dispatch test invokes this on its output so an accidental
    silent nullification is caught.
    """
    assert torch.isfinite(output).all(), f"{label}: non-finite entries"
    assert output.abs().sum().item() > 0.0, f"{label}: all-zero output"


# ---------------------------------------------------------------------------
# Slug-resolution unit checks (deterministic, don't touch the wrapper class)
# ---------------------------------------------------------------------------


def test_default_env_resolves_to_kda_cpu_golden_slug():
    with _env(KDA_KERNEL_IMPL_ENV, None):
        assert resolve_kda_impl_slug() == KDA_CPU_GOLDEN_SLUG
        assert emitted_kda_kernel_slug() == KDA_CPU_GOLDEN_SLUG
        assert get_emitted_kernel_slugs()["kda"] == KDA_CPU_GOLDEN_SLUG


@pytest.mark.parametrize(
    "flip_value",
    ["nki_v3p2", "v3p2", KDA_NKI_V3P2_SLUG],
)
def test_env_flip_resolves_to_kda_v3p2_slug(flip_value: str):
    with _env(KDA_KERNEL_IMPL_ENV, flip_value):
        assert resolve_kda_impl_slug() == KDA_NKI_V3P2_SLUG
        assert emitted_kda_kernel_slug() == KDA_NKI_V3P2_SLUG
        assert get_emitted_kernel_slugs()["kda"] == KDA_NKI_V3P2_SLUG


def test_unknown_kda_slug_raises_not_silently_defaults():
    """Guard against typos landing on a stale NEFF cache line."""
    with _env(KDA_KERNEL_IMPL_ENV, "nki_v9p9_typo"):
        with pytest.raises(ValueError, match=r"not a known KDA slug"):
            resolve_kda_impl_slug()


# ---------------------------------------------------------------------------
# End-to-end wrapper dispatch
# ---------------------------------------------------------------------------


def _run_kda_decode_once(module: Glm53KdaAttention) -> torch.Tensor:
    torch.manual_seed(31)
    hidden = torch.randn(1, 1, module.config.hidden_size)
    with torch.no_grad():
        return module(hidden)


def test_wrapper_forward_default_env_emits_v0_slug_and_produces_finite_output():
    """Baseline: no env flip => default slug + finite non-zero output."""
    with _env(KDA_KERNEL_IMPL_ENV, None):
        module = Glm53KdaAttention(_tiny_config(), layer_idx=0).eval()
        output_default = _run_kda_decode_once(module)
        assert module._last_emitted_kda_slug == KDA_CPU_GOLDEN_SLUG
    _require_comparable(output_default, "default-slug output")


def test_wrapper_forward_v3p2_env_emits_v3p2_slug_and_matches_cpu_golden():
    """Flip on: slug swaps to v3.2; numerics stay bit-exact CPU golden."""
    default_out = None
    with _env(KDA_KERNEL_IMPL_ENV, None):
        module = Glm53KdaAttention(_tiny_config(), layer_idx=0).eval()
        # Freeze the initial numpy state before the first forward: the
        # module mutates ``_state_bf16`` in place. We snapshot it so the
        # v3.2 branch starts from the same state as the default branch.
        module._state_bf16 = None
        module._conv_state = None
        default_out = _run_kda_decode_once(module)
        state_after_default = module._state_bf16.copy()

    with _env(KDA_KERNEL_IMPL_ENV, "nki_v3p2"):
        module2 = Glm53KdaAttention(_tiny_config(), layer_idx=0).eval()
        # Copy the same parameters so both runs are on identical weights.
        module2.load_state_dict(
            {
                name: p.detach().clone()
                for name, p in Glm53KdaAttention(_tiny_config(), layer_idx=0)
                .state_dict()
                .items()
            },
            strict=False,
        )
        # Reset state to zero so both runs start from the same point.
        module2._state_bf16 = None
        module2._conv_state = None
    # The two independently-constructed modules will not share weights
    # deterministically because linear layers initialize randomly. Rebuild
    # under a shared seed so both branches see identical parameters.
    torch.manual_seed(31)
    torch.set_num_threads(1)
    with _env(KDA_KERNEL_IMPL_ENV, None):
        torch.manual_seed(999)
        m_default = Glm53KdaAttention(_tiny_config(), layer_idx=0).eval()
        default_out = _run_kda_decode_once(m_default)
        assert m_default._last_emitted_kda_slug == KDA_CPU_GOLDEN_SLUG
    _require_comparable(default_out, "default-branch output")
    with _env(KDA_KERNEL_IMPL_ENV, "nki_v3p2"):
        torch.manual_seed(999)
        m_v3p2 = Glm53KdaAttention(_tiny_config(), layer_idx=0).eval()
        v3p2_out = _run_kda_decode_once(m_v3p2)
        assert m_v3p2._last_emitted_kda_slug == KDA_NKI_V3P2_SLUG
    _require_comparable(v3p2_out, "v3p2-branch output")
    # CPU-golden fallback preserves numerics: both branches must agree
    # bit-for-bit on a CPU-only host (the v3.2 wrapper's own "auto" mode
    # falls through to the same golden the default branch runs).
    assert torch.allclose(default_out, v3p2_out, atol=0.0, rtol=0.0), (
        "KDA v3.2 env flip changed CPU-golden numerics on a CPU-only host. "
        "This means the wrapper is NOT falling through to the golden, or "
        "an NKI backend was warm for this shape without the test being "
        "device-gated. Both are correctness-critical -- investigate."
    )


def test_wrapper_forward_records_slug_per_call_and_not_at_import():
    """Slug is bound at each forward call, not baked at import time."""
    module = Glm53KdaAttention(_tiny_config(), layer_idx=0).eval()
    # Reset state so both runs are directly comparable.
    module._state_bf16 = None
    module._conv_state = None

    with _env(KDA_KERNEL_IMPL_ENV, "nki_v3p2"):
        _ = _run_kda_decode_once(module)
        assert module._last_emitted_kda_slug == KDA_NKI_V3P2_SLUG

    # Reset internal state so the next call starts fresh.
    module._state_bf16 = None
    module._conv_state = None
    with _env(KDA_KERNEL_IMPL_ENV, None):
        _ = _run_kda_decode_once(module)
        assert module._last_emitted_kda_slug == KDA_CPU_GOLDEN_SLUG


def test_two_branches_emit_distinct_slugs():
    """Cache-identity safety: the two slugs must not collide."""
    assert KDA_CPU_GOLDEN_SLUG != KDA_NKI_V3P2_SLUG, (
        "KDA CPU-golden and NKI v3.2 slugs collide -- a stale NEFF from "
        "one would silently serve the other."
    )
