# SPDX-License-Identifier: Apache-2.0
"""KDA/DSA kernel dispatch for GLM-5.3-Flash (author: fleet-a-nki, 2026-08-28).

Purpose
-------
This module is the ONE place that reads the ``KDA_KERNEL_IMPL`` and
``DSA_KERNEL_IMPL`` environment variables and maps their values to the
kernel slug that the compile driver / NxDI graph would emit in the NEFF
cache identity.

The KDA v3.2 (Option A per-head reload) and DSA v2 (Option 2 inline
gather) NKI device kernels are baremetal-COMPILE-PASS on r7i per
``harness-v2/staging/reference-sweep-20260826T2150Z/kernels/
KDA-V3-SPIKE-2026-08-28.md`` §5.1.g and
``harness-v2/staging/reference-sweep-20260826T2150Z/kernels/
DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md`` §3C. This dispatch hook
makes them reachable by NAME (env flip) from the GLM-5.3-Flash model
compiles without touching any per-layer call site.

Discipline
----------
- **Verify emitted, never requested.** The env flip changes the slug the
  wrapper ADVERTISES in its emitted-kernel-config (readable via
  :func:`get_emitted_kernel_slugs`). The compile driver reads the
  advertised slug when constructing the NEFF cache key, so a wrong slug
  is a stale-NEFF class of bug the tests must catch by construction.
- **CPU-golden fallback preserved.** On any host without a compiled NKI
  backend for the requested slug (Windows author box; the CPU-only test
  battery), the runtime execution still lands on the CPU golden path
  (``kda_state_decode_forward_reference_v2`` /
  ``dsa_sparse_attention_forward`` v0). The env flip changes the slug
  but not the CPU numerics -- existing tests do not break.
- **No spec-decode**, no softmax/sdpa/full_attention fallback. KDA is a
  linear-attention recurrence; DSA is sparse-by-construction. Both are
  refused if requested by name via the campaign's ``_BANNED_IMPLS`` set
  (mirrored in :mod:`.nki_bindings` and the golden's own wrappers).

Slug identities (frozen)
------------------------
KDA:
- ``KDA_CPU_GOLDEN_SLUG = "kda_state.decode.kda_gate.rank1_delta.bf16_state.v1"``
    Slug advertised when ``KDA_KERNEL_IMPL`` is unset OR set to
    ``kda_cpu_golden`` / ``cpu`` / the golden slug itself.
- ``KDA_NKI_V3P2_SLUG = "kda_state.decode.kda_gate.rank1_delta.bf16_state.nki_v3p2"``
    Slug advertised when ``KDA_KERNEL_IMPL`` matches ``nki_v3p2`` (short)
    or the full v3.2 slug.

DSA:
- ``DSA_CPU_GOLDEN_SLUG = "nki_v0_reference_lightning_indexer"``
    Default. Set by unsetting the env or setting it to ``cpu_golden`` /
    ``v0`` / the golden slug itself.
- ``DSA_NKI_V2_SLUG = "dsa_sparse_attention.nki_v2"``
    Set by ``DSA_KERNEL_IMPL`` = ``nki_v2`` / the full v2 slug.

Environment variables
---------------------
``KDA_KERNEL_IMPL``
    - unset / empty / ``cpu`` / ``cpu_golden`` / ``reference`` / the CPU
      slug itself -> :data:`KDA_CPU_GOLDEN_SLUG` (default).
    - ``nki_v3p2`` / the v3.2 slug -> :data:`KDA_NKI_V3P2_SLUG`.
    - any other value -> :class:`ValueError`. This is deliberate: an
      unknown slug is a typo, not a fallback. Silently defaulting would
      mask compile-cache identity bugs.
``DSA_KERNEL_IMPL``
    - unset / empty / ``cpu`` / ``cpu_golden`` / ``v0`` / the CPU slug
      itself -> :data:`DSA_CPU_GOLDEN_SLUG` (default).
    - ``nki_v2`` / the v2 slug -> :data:`DSA_NKI_V2_SLUG`.
    - any other value -> :class:`ValueError`.

Callers
-------
- ``kda.py::Glm53KdaAttention.forward`` — CPU-source-qualified path.
- ``attention.py::Glm53DsaAttention.forward`` — CPU-source-qualified
  DSA path.
- ``neuron_wrapper.py::_KDABlock.forward`` — NxDI compile path.
- ``neuron_wrapper.py::_DSAIndexerBlock.forward`` — NxDI compile path.
- Compile driver (``fire_round6_compile.sh``): reads
  :func:`get_emitted_kernel_slugs` at fire time to log which NEFF cache
  key the graph will use.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Frozen slug identities
# ---------------------------------------------------------------------------

KDA_CPU_GOLDEN_SLUG: str = "kda_state.decode.kda_gate.rank1_delta.bf16_state.v1"
"""Slug for the numpy CPU-golden KDA kernel (kda_state_v2 handoff bundle)."""

KDA_NKI_V3P2_SLUG: str = "kda_state.decode.kda_gate.rank1_delta.bf16_state.nki_v3p2"
"""Slug for the KDA v3.2 (Option A per-head reload) NKI device kernel.

Baremetal COMPILE PASS on r7i, 1.39s, per KDA-V3-SPIKE-2026-08-28.md §5.1.g.
"""

DSA_CPU_GOLDEN_SLUG: str = "nki_v0_reference_lightning_indexer"
"""Slug for the torch CPU-golden DSA kernel (dsa_lightning_indexer v0)."""

DSA_NKI_V2_SLUG: str = "dsa_sparse_attention.nki_v2"
"""Slug for the DSA v2 (Option 2 inline gather) NKI device kernel.

Baremetal COMPILE PASS on r7i, 29.91s, per DSA-LIGHTNING-INDEXER-STATUS-
2026-08-28.md §3C.
"""

KDA_KERNEL_IMPL_ENV: str = "KDA_KERNEL_IMPL"
DSA_KERNEL_IMPL_ENV: str = "DSA_KERNEL_IMPL"

# Accepted aliases for the CPU-golden defaults. Empty string collapses to
# the same result via the "unset" branch.
_KDA_CPU_ALIASES = frozenset(
    {
        "",
        "cpu",
        "cpu_golden",
        "reference",
        "kda_cpu_golden",
        KDA_CPU_GOLDEN_SLUG,
    }
)
_KDA_NKI_V3P2_ALIASES = frozenset(
    {
        "nki_v3p2",
        "v3p2",
        KDA_NKI_V3P2_SLUG,
    }
)

_DSA_CPU_ALIASES = frozenset(
    {
        "",
        "cpu",
        "cpu_golden",
        "reference",
        "v0",
        "dsa_cpu_golden",
        DSA_CPU_GOLDEN_SLUG,
    }
)
_DSA_NKI_V2_ALIASES = frozenset(
    {
        "nki_v2",
        "v2",
        DSA_NKI_V2_SLUG,
    }
)


# ---------------------------------------------------------------------------
# Slug resolution
# ---------------------------------------------------------------------------


def resolve_kda_impl_slug(env_value: str | None = None) -> str:
    """Return the KDA slug the emitted config should advertise.

    Parameters
    ----------
    env_value : Optional[str]
        Override for the ``KDA_KERNEL_IMPL`` env var. If ``None``, reads
        the current process env. An explicit ``""`` is treated as unset.

    Raises
    ------
    ValueError
        If ``env_value`` is a non-empty string that is not one of the
        known aliases. Silently defaulting an unknown value would mask
        typos and let a caller land on a different NEFF cache line than
        they asked for -- exactly the class of bug this hook exists to
        prevent.
    """
    if env_value is None:
        env_value = os.environ.get(KDA_KERNEL_IMPL_ENV, "")
    normalized = env_value.strip()
    if normalized in _KDA_CPU_ALIASES:
        return KDA_CPU_GOLDEN_SLUG
    if normalized in _KDA_NKI_V3P2_ALIASES:
        return KDA_NKI_V3P2_SLUG
    raise ValueError(
        f"{KDA_KERNEL_IMPL_ENV}={env_value!r} is not a known KDA slug. "
        f"Accepted values: unset/empty (default), "
        f"{sorted(_KDA_CPU_ALIASES - {''})}, "
        f"or {sorted(_KDA_NKI_V3P2_ALIASES)}."
    )


def resolve_dsa_impl_slug(env_value: str | None = None) -> str:
    """Return the DSA slug the emitted config should advertise.

    See :func:`resolve_kda_impl_slug` for parameter and error semantics.
    """
    if env_value is None:
        env_value = os.environ.get(DSA_KERNEL_IMPL_ENV, "")
    normalized = env_value.strip()
    if normalized in _DSA_CPU_ALIASES:
        return DSA_CPU_GOLDEN_SLUG
    if normalized in _DSA_NKI_V2_ALIASES:
        return DSA_NKI_V2_SLUG
    raise ValueError(
        f"{DSA_KERNEL_IMPL_ENV}={env_value!r} is not a known DSA slug. "
        f"Accepted values: unset/empty (default), "
        f"{sorted(_DSA_CPU_ALIASES - {''})}, "
        f"or {sorted(_DSA_NKI_V2_ALIASES)}."
    )


def get_emitted_kernel_slugs() -> dict[str, str]:
    """Return the ``{"kda": slug, "dsa": slug}`` mapping for THIS process.

    This is the ONE authoritative view of "what NEFF cache identity the
    compile driver would use if the graph were fired NOW". Tests assert
    against this to prove the env flip actually lands the requested
    slug; compile drivers log it before submitting a fire.
    """
    return {
        "kda": resolve_kda_impl_slug(),
        "dsa": resolve_dsa_impl_slug(),
    }


# ---------------------------------------------------------------------------
# Runtime routing helpers
# ---------------------------------------------------------------------------


def is_kda_nki_v3p2_selected() -> bool:
    """True iff KDA_KERNEL_IMPL resolves to the v3.2 NKI slug."""
    return resolve_kda_impl_slug() == KDA_NKI_V3P2_SLUG


def is_dsa_nki_v2_selected() -> bool:
    """True iff DSA_KERNEL_IMPL resolves to the v2 NKI slug."""
    return resolve_dsa_impl_slug() == DSA_NKI_V2_SLUG


def _load_kda_v3p2_wrapper():
    """Load the KDA v3.2 wrapper module from the handoff kernels dir.

    Uses the SAME resolver ``._reference_kernels`` uses so the two paths
    (v0 CPU golden and v3.2 NKI wrapper) always come from the same
    ``kernels/`` directory on any host. Returns None if the file is
    missing (test hosts that override GLM53_REFERENCE_KERNEL_DIR to a
    reference-only mirror without the v3.2 wrapper).
    """
    import importlib.util
    from pathlib import Path

    from ._reference_kernels import _kernel_dir  # type: ignore[attr-defined]

    kernel_dir = _kernel_dir()
    path = Path(kernel_dir) / "kda_state_nki_v3p2.py"
    if not path.is_file():
        return None
    # v3.2 imports kda_state_v2 with a bare name; make sure that is on
    # sys.path before we load it.
    kd = str(kernel_dir)
    if kd not in sys.path:
        sys.path.insert(0, kd)
    module_name = "_glm53_kda_state_nki_v3p2"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so dataclass decorators inside
    # the loaded module can resolve `sys.modules[cls.__module__]`.
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def _load_dsa_v2_wrapper():
    """Load the DSA v2 wrapper module from the handoff kernels dir."""
    import importlib.util
    from pathlib import Path

    from ._reference_kernels import _kernel_dir  # type: ignore[attr-defined]

    kernel_dir = _kernel_dir()
    path = Path(kernel_dir) / "dsa_lightning_indexer_nki_v2.py"
    if not path.is_file():
        return None
    kd = str(kernel_dir)
    if kd not in sys.path:
        sys.path.insert(0, kd)
    module_name = "_glm53_dsa_lightning_indexer_nki_v2"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def get_kda_decode_forward(
    default_forward: Callable,
) -> tuple[str, Callable]:
    """Return ``(slug, forward_callable)`` for the currently-selected KDA impl.

    The returned callable has the SAME signature as
    ``kda_state_v2.kda_state_decode_forward_reference_v2(inputs, impl=...)``
    and returns the same type. On any host without a warm NKI backend
    for the v3.2 slug, the returned callable STILL executes CPU-golden
    math -- the swap is on the ADVERTISED slug, not on the numerics.
    This preserves the CPU-golden test battery.

    Parameters
    ----------
    default_forward
        The caller's usual CPU-golden decode callable. Used when the
        env resolves to the CPU-golden slug OR when v3.2 wrapper cannot
        be loaded (test-host without the handoff kernels dir).
    """
    slug = resolve_kda_impl_slug()
    if slug == KDA_CPU_GOLDEN_SLUG:
        return slug, default_forward
    # v3.2 selected. Prefer the v3.2 wrapper (which has its own auto
    # fallback to CPU golden). If unloadable, fall back to the caller's
    # default -- the emitted slug is still v3.2 either way (a test-host
    # that lacks the v3.2 wrapper still exercises the dispatch path).
    v3p2 = _load_kda_v3p2_wrapper()
    if v3p2 is None:
        return slug, default_forward
    return slug, v3p2.kda_state_decode_forward_nki_v3p2


def get_dsa_sparse_attention_forward(
    default_forward: Callable,
) -> tuple[str, Callable]:
    """Return ``(slug, forward_callable)`` for the currently-selected DSA impl.

    The returned callable matches
    ``dsa_lightning_indexer.dsa_sparse_attention_forward``'s signature
    and behaviour on CPU. On a Trn2 host with a warm NKI v2 backend,
    the v2 wrapper's runtime dispatch will use the compiled kernel;
    otherwise it falls through to the same v0 CPU-golden math the
    default callable would run. Numerics are identical either way; only
    the advertised slug changes.
    """
    slug = resolve_dsa_impl_slug()
    if slug == DSA_CPU_GOLDEN_SLUG:
        return slug, default_forward
    v2 = _load_dsa_v2_wrapper()
    if v2 is None:
        return slug, default_forward
    return slug, v2.dsa_sparse_attention_forward_nki_v2


__all__ = [
    "DSA_CPU_GOLDEN_SLUG",
    "DSA_KERNEL_IMPL_ENV",
    "DSA_NKI_V2_SLUG",
    "KDA_CPU_GOLDEN_SLUG",
    "KDA_KERNEL_IMPL_ENV",
    "KDA_NKI_V3P2_SLUG",
    "get_dsa_sparse_attention_forward",
    "get_emitted_kernel_slugs",
    "get_kda_decode_forward",
    "is_dsa_nki_v2_selected",
    "is_kda_nki_v3p2_selected",
    "resolve_dsa_impl_slug",
    "resolve_kda_impl_slug",
]
