# SPDX-License-Identifier: Apache-2.0
"""DSv4-Flash DSA v2 wrapper-integration test (fleet-a-nki, 2026-08-28).

Purpose
-------
Prove that ``DSA_KERNEL_IMPL`` env-flip resolves to the correct kernel
slug on the DSv4-Flash wrapper, and that both the module-level view
(``get_emitted_kernel_slugs``) and the per-layer view
(``_LightningIndexerHead._emitted_dsa_slug``,
``_CSABlock._emitted_dsa_slug``) agree with what the compile driver
would advertise in the NEFF cache key.

DSv4 has NO KDA layers -- there is no KDA test to author. The DSA v2
kernel is baremetal-COMPILE-PASS per
``harness-v2/staging/reference-sweep-20260826T2150Z/kernels/
DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md`` §3C.

Note on runtime routing
-----------------------
The env flip on DSv4-Flash is a SLUG-ONLY advertisement: DSv4 does not
structurally invoke ``dsa_sparse_attention_forward`` (the indexer's
top-k is consumed by a block_bias mask feeding a dense softmax on the
extended KV axis). This test therefore verifies the ADVERTISED slug,
not a numerics swap. See
``vllm_neuron/model/dsv4_flash/kernel_dispatch.py`` for the
scope-note rationale.

Import strategy
---------------
Uses direct file-based imports (``importlib.util.spec_from_file_location``)
so the test collects on hosts without ``vllm`` (the existing dsv4
tests error at collection there). The dispatch and constructor code
under test does NOT depend on ``vllm``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest


_DSV4_DIR = Path(__file__).resolve().parent.parent


def _load_module(local_name: str, file_name: str):
    """File-import a single module without touching the package hierarchy."""
    full_name = f"_dsv4_dsa_integration.{local_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    parent = "_dsv4_dsa_integration"
    if parent not in sys.modules:
        parent_module = types.ModuleType(parent)
        parent_module.__path__ = [str(_DSV4_DIR)]
        sys.modules[parent] = parent_module
    spec = importlib.util.spec_from_file_location(
        full_name, str(_DSV4_DIR / file_name)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def _load_all():
    """Return (kernel_dispatch, config, neuron_wrapper) as file-imported modules.

    ``neuron_wrapper`` imports ``.kernel_dispatch`` and ``.config``
    with relative names; the ``_dsv4_dsa_integration`` synthetic parent
    package lets the two land under a shared namespace so the relative
    imports resolve cleanly.
    """
    kd = _load_module("kernel_dispatch", "kernel_dispatch.py")
    cfg = _load_module("config", "config.py")
    nw = _load_module("neuron_wrapper", "neuron_wrapper.py")
    return kd, cfg, nw


@contextmanager
def _env(name: str, value):
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


# ---------------------------------------------------------------------------
# Slug-resolution unit checks
# ---------------------------------------------------------------------------


def test_default_env_resolves_to_dsa_v0_golden_slug() -> None:
    kd, _cfg, _nw = _load_all()
    with _env(kd.DSA_KERNEL_IMPL_ENV, None):
        assert kd.resolve_dsa_impl_slug() == kd.DSA_CPU_GOLDEN_SLUG
        assert kd.get_emitted_kernel_slugs() == {
            "dsa": kd.DSA_CPU_GOLDEN_SLUG
        }


@pytest.mark.parametrize(
    "flip_value",
    ["nki_v2", "v2", "dsa_sparse_attention.nki_v2"],
)
def test_env_flip_resolves_to_dsa_v2_slug(flip_value: str) -> None:
    kd, _cfg, _nw = _load_all()
    with _env(kd.DSA_KERNEL_IMPL_ENV, flip_value):
        assert kd.resolve_dsa_impl_slug() == kd.DSA_NKI_V2_SLUG
        assert kd.get_emitted_kernel_slugs() == {"dsa": kd.DSA_NKI_V2_SLUG}


def test_unknown_dsa_slug_raises_not_silently_defaults() -> None:
    """Guard against typos landing on a stale NEFF cache line."""
    kd, _cfg, _nw = _load_all()
    with _env(kd.DSA_KERNEL_IMPL_ENV, "nki_v9_typo"):
        with pytest.raises(ValueError, match=r"not a known DSA slug"):
            kd.resolve_dsa_impl_slug()


def test_wrapper_reexports_get_emitted_kernel_slugs() -> None:
    """The compile driver imports from ``neuron_wrapper``, not from
    ``kernel_dispatch`` directly; the re-export must be present."""
    kd, _cfg, nw = _load_all()
    assert nw.get_emitted_kernel_slugs is kd.get_emitted_kernel_slugs
    assert nw.DSA_CPU_GOLDEN_SLUG == kd.DSA_CPU_GOLDEN_SLUG
    assert nw.DSA_NKI_V2_SLUG == kd.DSA_NKI_V2_SLUG


def test_dsv4_kernel_dispatch_has_no_kda_dispatch() -> None:
    """DSv4-Flash has no KDA layers; guard against copy-paste drift."""
    kd, _cfg, _nw = _load_all()
    assert not hasattr(kd, "resolve_kda_impl_slug")
    assert not hasattr(kd, "KDA_CPU_GOLDEN_SLUG")
    assert not hasattr(kd, "KDA_NKI_V3P2_SLUG")


def test_two_branches_emit_distinct_slugs() -> None:
    """Cache-identity safety: the two slugs must not collide."""
    kd, _cfg, _nw = _load_all()
    assert kd.DSA_CPU_GOLDEN_SLUG != kd.DSA_NKI_V2_SLUG, (
        "DSA CPU-golden and NKI v2 slugs collide -- a stale NEFF from "
        "one would silently serve the other."
    )


# ---------------------------------------------------------------------------
# Layer-level slug advertisement (constructor binds the env value)
# ---------------------------------------------------------------------------


def _build_indexer_head(nw, cfg_module):
    """CSA layer 2 is the first indexer-bearing layer in the DSv4 schedule."""
    src = cfg_module.DeepseekV4FlashInferenceConfig()
    return nw._LightningIndexerHead(src, layer_idx=2)


def _build_csa_block(nw, cfg_module):
    src = cfg_module.DeepseekV4FlashInferenceConfig()
    return nw._CSABlock(src, layer_idx=2)


def test_indexer_head_binds_default_slug_on_default_env() -> None:
    kd, cfg_module, nw = _load_all()
    with _env(kd.DSA_KERNEL_IMPL_ENV, None):
        head = _build_indexer_head(nw, cfg_module)
    assert head._emitted_dsa_slug == kd.DSA_CPU_GOLDEN_SLUG


def test_indexer_head_binds_v2_slug_on_env_flip() -> None:
    kd, cfg_module, nw = _load_all()
    with _env(kd.DSA_KERNEL_IMPL_ENV, "nki_v2"):
        head = _build_indexer_head(nw, cfg_module)
    assert head._emitted_dsa_slug == kd.DSA_NKI_V2_SLUG


def test_csa_block_binds_default_slug_on_default_env() -> None:
    kd, cfg_module, nw = _load_all()
    with _env(kd.DSA_KERNEL_IMPL_ENV, None):
        block = _build_csa_block(nw, cfg_module)
    assert block._emitted_dsa_slug == kd.DSA_CPU_GOLDEN_SLUG
    # And the child indexer head agrees (both were constructed under
    # the same env).
    assert block.indexer._emitted_dsa_slug == kd.DSA_CPU_GOLDEN_SLUG


def test_csa_block_binds_v2_slug_on_env_flip() -> None:
    kd, cfg_module, nw = _load_all()
    with _env(kd.DSA_KERNEL_IMPL_ENV, "nki_v2"):
        block = _build_csa_block(nw, cfg_module)
    assert block._emitted_dsa_slug == kd.DSA_NKI_V2_SLUG
    assert block.indexer._emitted_dsa_slug == kd.DSA_NKI_V2_SLUG


def test_get_emitted_kernel_slugs_reflects_wrapper_construction() -> None:
    """End-to-end: verify emitted, never requested.

    The module-level ``get_emitted_kernel_slugs()`` (what the compile
    driver reads) and the per-layer ``_emitted_dsa_slug`` (what the
    layer would advertise if introspected) agree with each other under
    both env branches.
    """
    kd, cfg_module, nw = _load_all()
    with _env(kd.DSA_KERNEL_IMPL_ENV, None):
        block = _build_csa_block(nw, cfg_module)
        emitted = nw.get_emitted_kernel_slugs()
        assert emitted["dsa"] == kd.DSA_CPU_GOLDEN_SLUG
        assert block._emitted_dsa_slug == emitted["dsa"]

    with _env(kd.DSA_KERNEL_IMPL_ENV, "nki_v2"):
        block2 = _build_csa_block(nw, cfg_module)
        emitted2 = nw.get_emitted_kernel_slugs()
        assert emitted2["dsa"] == kd.DSA_NKI_V2_SLUG
        assert block2._emitted_dsa_slug == emitted2["dsa"]


if __name__ == "__main__":
    # Direct-invocation entry point. The existing dsv4_flash tests do not
    # collect under pytest on hosts without ``vllm`` installed because
    # ``vllm_neuron/__init__.py`` unconditionally imports ``vllm``.
    # Running this file directly bypasses the package walk and exercises
    # every test function with an ergonomic pass/fail report. Prefer
    # this on the Windows author box; on the Trn2 / r7i / production
    # host where ``vllm`` is installed, pytest works normally.
    _failed = 0
    _passed = 0
    for _name in sorted(dir()):
        if not _name.startswith("test_"):
            continue
        _fn = globals()[_name]
        _marks = getattr(_fn, "pytestmark", ())
        _parametrised = any(
            getattr(m, "name", "") == "parametrize" for m in _marks
        )
        if _parametrised:
            _values = ["nki_v2", "v2", "dsa_sparse_attention.nki_v2"]
            for _v in _values:
                try:
                    _fn(_v)
                    _passed += 1
                    print(f"PASS {_name}[{_v}]")
                except Exception as _e:
                    _failed += 1
                    print(f"FAIL {_name}[{_v}]: {_e!r}")
        else:
            try:
                _fn()
                _passed += 1
                print(f"PASS {_name}")
            except Exception as _e:
                _failed += 1
                print(f"FAIL {_name}: {_e!r}")
    print("-" * 40)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    sys.exit(0 if _failed == 0 else 1)
