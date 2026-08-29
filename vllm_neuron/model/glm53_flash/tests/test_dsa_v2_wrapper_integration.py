# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash DSA v2 wrapper-integration test (fleet-a-nki, 2026-08-28).

Purpose
-------
Prove that the ``DSA_KERNEL_IMPL`` env-flip resolves to the correct
kernel slug and that BOTH branches (v0 CPU golden default and NKI v2)
route through the wrapper's dispatch hook end-to-end, and that the
CPU-golden fallback preserves numerics for the CPU-only test host.

Discipline mirrors ``test_kda_v3p2_wrapper_integration.py``.

Anchors
-------
- Wrapper implementation:
  ``C:\\Users\\apumu\\research\\InfinityAI\\vllm-neuron-codex-alpha\\
   vllm_neuron\\model\\glm53_flash\\kernel_dispatch.py``
- DSA v2 receipt:
  ``C:\\Users\\apumu\\research\\InfinityAI\\gemma4-trn2-handoff\\
   harness-v2\\staging\\reference-sweep-20260826T2150Z\\kernels\\
   DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md`` §3C (baremetal COMPILE
   PASS on r7i, 29.91s).
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch

from vllm_neuron.model.glm53_flash.attention import (
    Glm53DsaAttention,
    emitted_dsa_kernel_slug,
)
from vllm_neuron.model.glm53_flash.config import (
    Glm53FlashInferenceConfig,
    Glm53LinearAttentionConfig,
)
from vllm_neuron.model.glm53_flash.kernel_dispatch import (
    DSA_CPU_GOLDEN_SLUG,
    DSA_KERNEL_IMPL_ENV,
    DSA_NKI_V2_SLUG,
    get_emitted_kernel_slugs,
    resolve_dsa_impl_slug,
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
    """Reject a degenerate wrapper output."""
    assert torch.isfinite(output).all(), f"{label}: non-finite entries"
    assert output.abs().sum().item() > 0.0, f"{label}: all-zero output"


def _run_dsa_once(module: Glm53DsaAttention) -> torch.Tensor:
    torch.manual_seed(41)
    hidden = torch.randn(1, 2, module.config.hidden_size)
    positions = torch.arange(2, dtype=torch.int64).unsqueeze(0)
    with torch.no_grad():
        return module(hidden, positions)


# ---------------------------------------------------------------------------
# Slug-resolution unit checks
# ---------------------------------------------------------------------------


def test_default_env_resolves_to_dsa_v0_golden_slug():
    with _env(DSA_KERNEL_IMPL_ENV, None):
        assert resolve_dsa_impl_slug() == DSA_CPU_GOLDEN_SLUG
        assert emitted_dsa_kernel_slug() == DSA_CPU_GOLDEN_SLUG
        assert get_emitted_kernel_slugs()["dsa"] == DSA_CPU_GOLDEN_SLUG


@pytest.mark.parametrize(
    "flip_value",
    ["nki_v2", "v2", DSA_NKI_V2_SLUG],
)
def test_env_flip_resolves_to_dsa_v2_slug(flip_value: str):
    with _env(DSA_KERNEL_IMPL_ENV, flip_value):
        assert resolve_dsa_impl_slug() == DSA_NKI_V2_SLUG
        assert emitted_dsa_kernel_slug() == DSA_NKI_V2_SLUG
        assert get_emitted_kernel_slugs()["dsa"] == DSA_NKI_V2_SLUG


def test_unknown_dsa_slug_raises_not_silently_defaults():
    """Guard against typos landing on a stale NEFF cache line."""
    with _env(DSA_KERNEL_IMPL_ENV, "nki_v9_typo"):
        with pytest.raises(ValueError, match=r"not a known DSA slug"):
            resolve_dsa_impl_slug()


# ---------------------------------------------------------------------------
# End-to-end wrapper dispatch
# ---------------------------------------------------------------------------


def test_wrapper_forward_default_env_emits_v0_slug_and_produces_finite_output():
    """Baseline: no env flip -> default slug + finite non-zero output."""
    with _env(DSA_KERNEL_IMPL_ENV, None):
        torch.manual_seed(999)
        module = Glm53DsaAttention(_tiny_config()).eval()
        output_default = _run_dsa_once(module)
        assert module._last_emitted_dsa_slug == DSA_CPU_GOLDEN_SLUG
    _require_comparable(output_default, "default-slug DSA output")


def test_wrapper_forward_v2_env_emits_v2_slug_and_matches_cpu_golden():
    """Flip on: slug swaps to v2; CPU numerics stay bit-exact CPU golden."""
    with _env(DSA_KERNEL_IMPL_ENV, None):
        torch.manual_seed(999)
        m_default = Glm53DsaAttention(_tiny_config()).eval()
        default_out = _run_dsa_once(m_default)
        assert m_default._last_emitted_dsa_slug == DSA_CPU_GOLDEN_SLUG
    _require_comparable(default_out, "default-branch DSA output")

    with _env(DSA_KERNEL_IMPL_ENV, "nki_v2"):
        torch.manual_seed(999)
        m_v2 = Glm53DsaAttention(_tiny_config()).eval()
        v2_out = _run_dsa_once(m_v2)
        assert m_v2._last_emitted_dsa_slug == DSA_NKI_V2_SLUG
    _require_comparable(v2_out, "v2-branch DSA output")

    # CPU-golden fallback preserves numerics: both branches must agree
    # bit-for-bit on a CPU-only host (the v2 wrapper's own dispatch
    # falls through to the same golden the default branch runs).
    assert torch.allclose(default_out, v2_out, atol=0.0, rtol=0.0), (
        "DSA v2 env flip changed CPU-golden numerics on a CPU-only host. "
        "The v2 wrapper's `_v0_forward` fallback should be exercised on "
        "any host without an NKI runtime -- a divergence means the "
        "dispatch fell into an unexpected branch."
    )


def test_wrapper_forward_records_slug_per_call_and_not_at_import():
    """Slug is bound at each forward call, not baked at import time."""
    module = Glm53DsaAttention(_tiny_config()).eval()

    with _env(DSA_KERNEL_IMPL_ENV, "nki_v2"):
        _ = _run_dsa_once(module)
        assert module._last_emitted_dsa_slug == DSA_NKI_V2_SLUG

    module.reset_state()
    with _env(DSA_KERNEL_IMPL_ENV, None):
        _ = _run_dsa_once(module)
        assert module._last_emitted_dsa_slug == DSA_CPU_GOLDEN_SLUG


def test_two_branches_emit_distinct_slugs():
    """Cache-identity safety: the two slugs must not collide."""
    assert DSA_CPU_GOLDEN_SLUG != DSA_NKI_V2_SLUG, (
        "DSA CPU-golden and NKI v2 slugs collide -- a stale NEFF from "
        "one would silently serve the other."
    )


def test_combined_env_flip_produces_both_v3p2_and_v2_slugs():
    """Both env flips can be set together (independent axes)."""
    from vllm_neuron.model.glm53_flash.kernel_dispatch import (
        KDA_KERNEL_IMPL_ENV,
        KDA_NKI_V3P2_SLUG,
    )

    with _env(KDA_KERNEL_IMPL_ENV, "nki_v3p2"), _env(DSA_KERNEL_IMPL_ENV, "nki_v2"):
        slugs = get_emitted_kernel_slugs()
        assert slugs["kda"] == KDA_NKI_V3P2_SLUG
        assert slugs["dsa"] == DSA_NKI_V2_SLUG
