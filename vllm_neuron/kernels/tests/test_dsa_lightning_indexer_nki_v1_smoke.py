# SPDX-License-Identifier: Apache-2.0
"""Tier-1 dry-import + shape smoke test for the DSA Lightning Indexer NKI v1.

Author: Trn2 reference-sweep 2026-08-27 (Callsign: `dsa-nki-v1-agent`).

Sibling of `kernels/tests/test_dsa_lightning_indexer_correctness.py`
(v0 correctness gate, 32/32 passing per DSA-LIGHTNING-INDEXER-STATUS-
2026-08-28 §3.2). This file is the **first-touch smoke** for the v1
NKI author's draft at `kernels/dsa_lightning_indexer_nki_v1.py`.

Scope kept intentionally narrow — the v0 file already carries every
numeric-correctness invariant. Here we prove:

  * S0 -- v1 module imports cleanly on any host (with or without NKI).
  * S1 -- v1 slug is `dsa_sparse_attention.nki_v1` and matches operator
          prompt's requested cache identity.
  * S2 -- v1's `LSE_BASE_CONVENTION` matches v0's (`"natural"`), so
          any cross-verification consumer that gates on the constant
          sees the same value for both paths.
  * S3 -- `nki_runtime_available()` is a boolean and matches the
          `neuronxcc.nki` import result exactly (no fallback lying about
          being real).
  * S4 -- Fallback path shape check: on any host with torch installed,
          `dsa_sparse_attention_forward_nki_v1(force_fallback=True, ...)`
          returns tensors whose shapes match v0's contract exactly.
  * S5 -- `build_v1_cache_key(...)` differs from a v0-slug cache key
          for the same shape (silent-replay guard).
  * S6 -- Fallback path bit-matches v0 direct call: for the same
          inputs, `dsa_sparse_attention_forward_nki_v1(force_fallback=
          True, ...)` returns identical tensors to
          `dsa_sparse_attention_forward(...)` from v0 (this is the
          contract that lets a lane flip the slug and expect zero
          numeric drift when NKI isn't available yet).

S4 and S6 skip when torch is not importable (Windows author session
without torch is the operative case).

Run with:
    py -3 -m pytest -q kernels/tests/test_dsa_lightning_indexer_nki_v1_smoke.py

Absolute path: C:\\Users\\apumu\\research\\InfinityAI\\gemma4-trn2-handoff\\harness-v2\\staging\\reference-sweep-20260826T2150Z\\kernels\\tests\\test_dsa_lightning_indexer_nki_v1_smoke.py
"""
from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

# Make the sibling kernel module importable (matches v0 test convention).
HERE = pathlib.Path(__file__).resolve().parent
KERNEL_DIR = HERE.parent
sys.path.insert(0, str(KERNEL_DIR))


# ---------------------------------------------------------------------------
# Import-side smokes (host-independent)
# ---------------------------------------------------------------------------


def test_S0_v1_module_imports_cleanly():
    """v1 module imports without needing NKI or torch."""
    mod = importlib.import_module("dsa_lightning_indexer_nki_v1")
    assert mod is not None
    # Public API surface intact
    assert hasattr(mod, "dsa_sparse_attention_forward_nki_v1")
    assert hasattr(mod, "build_v1_cache_key")
    assert hasattr(mod, "nki_runtime_available")
    assert hasattr(mod, "KERNEL_SLUG_V1_NKI")
    assert hasattr(mod, "LSE_BASE_CONVENTION")


def test_S1_v1_slug_matches_operator_prompt():
    """Slug per operator prompt: `dsa_sparse_attention.nki_v1`."""
    from dsa_lightning_indexer_nki_v1 import KERNEL_SLUG_V1_NKI  # noqa: E402
    assert KERNEL_SLUG_V1_NKI == "dsa_sparse_attention.nki_v1"


def test_S2_v1_lse_base_matches_v0():
    """v1 LSE base convention MUST equal v0's or cross-verify breaks."""
    from dsa_lightning_indexer_nki_v1 import (  # noqa: E402
        LSE_BASE_CONVENTION as V1_BASE,
    )

    # v0 import is guarded because it pulls torch; skip if unavailable
    # so this test remains valuable on the Windows author host.
    torch = pytest.importorskip("torch", reason="torch not installed on this host")
    del torch  # only needed for the v0 import
    from dsa_lightning_indexer import (  # noqa: E402
        LSE_BASE_CONVENTION as V0_BASE,
    )
    assert V1_BASE == V0_BASE == "natural"


def test_S3_nki_runtime_flag_is_truthy_boolean():
    """`nki_runtime_available()` matches actual neuronxcc.nki import."""
    from dsa_lightning_indexer_nki_v1 import nki_runtime_available  # noqa: E402

    reported = nki_runtime_available()
    assert isinstance(reported, bool)

    try:
        import neuronxcc.nki  # noqa: F401
        actual_available = True
    except Exception:
        actual_available = False

    assert reported == actual_available


# ---------------------------------------------------------------------------
# Fallback-path shape/numerics smokes (torch-gated)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _torch_fixture():
    torch = pytest.importorskip("torch", reason="torch not installed on this host")
    return torch


def _tiny_case(torch, *, B=1, S_q=2, S_kv=8, H=2, D=8, topk=4):
    """Small-shape inputs suitable for a fallback-path shape check."""
    gen = torch.Generator().manual_seed(2026_08_27)
    Q = torch.randn(B, S_q, H, D, generator=gen, dtype=torch.bfloat16)
    K = torch.randn(B, S_kv, H, D, generator=gen, dtype=torch.bfloat16)
    V = torch.randn(B, S_kv, H, D, generator=gen, dtype=torch.bfloat16)
    # Deterministic index_topk_idx: first `topk` positions per query.
    idx = torch.arange(topk, dtype=torch.int32).view(1, 1, topk).expand(B, S_q, topk).contiguous()
    # position_ids per query: last positions so causality allows all `topk` picks
    position_ids = torch.full((B, S_q), S_kv - 1, dtype=torch.int64)
    key_lengths = torch.full((B,), S_kv, dtype=torch.int64)
    return Q, K, V, idx, position_ids, key_lengths


def test_S4_fallback_shape_matches_v0_contract(_torch_fixture):
    """Fallback returns tensors whose shapes match v0's contract."""
    torch = _torch_fixture
    from dsa_lightning_indexer_nki_v1 import (  # noqa: E402
        dsa_sparse_attention_forward_nki_v1,
    )

    Q, K, V, idx, pos, klen = _tiny_case(torch)
    B, S_q, H, D = Q.shape
    topk = int(idx.shape[-1])

    out = dsa_sparse_attention_forward_nki_v1(
        Q, K, V, idx,
        position_ids=pos,
        key_lengths=klen,
        topk=topk,
        force_fallback=True,
    )
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (B, S_q, H, D)
    assert out.dtype == Q.dtype

    # With return_lse=True it must be a 2-tuple whose second element is
    # [B, S_q, H] fp32 (natural-log LSE).
    out2, lse = dsa_sparse_attention_forward_nki_v1(
        Q, K, V, idx,
        position_ids=pos,
        key_lengths=klen,
        topk=topk,
        return_lse=True,
        force_fallback=True,
    )
    assert tuple(out2.shape) == (B, S_q, H, D)
    assert tuple(lse.shape) == (B, S_q, H)
    assert lse.dtype == torch.float32


def test_S5_v1_cache_key_differs_from_v0(_torch_fixture):
    """v1 slug must produce a distinct cache key from v0 for the same
    shape, so a slug flip is a fresh compile-cache line (top-5 rule #2)."""
    _torch_fixture  # ensure torch importable (v0 needs it)
    from dsa_lightning_indexer_nki_v1 import build_v1_cache_key  # noqa: E402
    from dsa_lightning_indexer import DsaKernelConfig  # noqa: E402

    v0_cfg = DsaKernelConfig(
        topk=2048,
        block_size=32,
        index_n_heads=64,
        index_head_dim=64,
        index_pool=1,
        causal=True,
        return_topk_for_indexshare=False,
    )
    v0_key = v0_cfg.cache_key()
    v1_key = build_v1_cache_key(
        topk=2048,
        block_size=32,
        index_n_heads=64,
        index_head_dim=64,
        index_pool=1,
        causal=True,
        return_topk_for_indexshare=False,
        return_lse=False,
    )
    assert v0_key != v1_key
    # v1 key MUST contain the operator-prompt slug for greppability.
    assert "dsa_sparse_attention.nki_v1" in v1_key

    # `return_lse=True` also gets its own key vs `return_lse=False`
    # (SGLang LSE fix analysis §7 action item #1).
    v1_key_lse = build_v1_cache_key(
        topk=2048,
        block_size=32,
        index_n_heads=64,
        index_head_dim=64,
        index_pool=1,
        causal=True,
        return_topk_for_indexshare=False,
        return_lse=True,
    )
    assert v1_key_lse != v1_key


def test_S6_fallback_matches_v0_direct(_torch_fixture):
    """Fallback path bit-matches a direct v0 call — proves the slug flip
    is safe on any NKI-less host."""
    torch = _torch_fixture
    from dsa_lightning_indexer_nki_v1 import (  # noqa: E402
        dsa_sparse_attention_forward_nki_v1,
    )
    from dsa_lightning_indexer import (  # noqa: E402
        dsa_sparse_attention_forward,
    )

    Q, K, V, idx, pos, klen = _tiny_case(torch)
    topk = int(idx.shape[-1])

    v0_out, v0_lse = dsa_sparse_attention_forward(
        Q, K, V, idx, pos, klen,
        topk=topk,
        return_lse=True,
    )
    v1_out, v1_lse = dsa_sparse_attention_forward_nki_v1(
        Q, K, V, idx,
        position_ids=pos,
        key_lengths=klen,
        topk=topk,
        return_lse=True,
        force_fallback=True,
    )
    assert torch.equal(v0_out, v1_out)
    # LSE is fp32; `torch.equal` is bitwise here (fallback delegates
    # to v0 with identical inputs, so bit-equality is expected).
    assert torch.equal(v0_lse, v1_lse)
