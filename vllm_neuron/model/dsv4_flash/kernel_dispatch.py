# SPDX-License-Identifier: Apache-2.0
"""DSA kernel-slug env-flip dispatch for DSv4-Flash (fleet-a-nki, 2026-08-28).

Purpose
-------
Advertise the correct DSA NEFF cache identity in the DSv4-Flash CSA
lane's emitted config. The DSA v2 (Option 2 inline gather) NKI device
kernel is baremetal-COMPILE-PASS on r7i per
``harness-v2/staging/reference-sweep-20260826T2150Z/kernels/
DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md`` §3C. This module lets a
lane's compile driver flip between the CPU-golden slug (default) and
the v2 slug by env, without touching any per-layer call site.

Scope note vs GLM-5.3-Flash
---------------------------
DSv4-Flash does NOT structurally call
``dsa_lightning_indexer.dsa_sparse_attention_forward`` (the sparse
gather + softmax kernel). Its indexer path (`_LightningIndexerHead`)
computes per-query top-k indices via ReLU + sqrt scale + `torch.topk`
and the `_CSABlock` uses those indices to build an additive
``block_bias`` that MASKS a full-attention softmax on the extended KV
axis. That is the DeepSeek-V4 HF-parity contract; a sparse-gather
substitute would be a different model shape.

So on DSv4-Flash, the env flip is a SLUG-ONLY advertisement -- it does
NOT swap the runtime callable. The wrapper still exposes
``get_emitted_kernel_slugs()`` so the compile driver reads the same
authoritative identity as GLM-5.3-Flash reads. When a future revision
inlines a fused sparse-gather kernel, the env flip is already the
entrypoint.

DSv4 has NO KDA layers (its attention families are MQA + CSA + HCA +
SlidingOnly), so there is NO KDA dispatch here. Attempting to set
``KDA_KERNEL_IMPL`` while on DSv4-Flash has no effect and no error.

Environment variables
---------------------
``DSA_KERNEL_IMPL``
    - unset / empty / ``cpu`` / ``cpu_golden`` / ``v0`` / the CPU slug
      itself -> :data:`DSA_CPU_GOLDEN_SLUG` (default).
    - ``nki_v2`` / the v2 slug -> :data:`DSA_NKI_V2_SLUG`.
    - any other value -> :class:`ValueError`. Silent default on a typo
      would mask compile-cache identity bugs.

Callers
-------
- ``neuron_wrapper.py::_LightningIndexerHead.__init__`` -- record the
  slug the layer's config will advertise.
- ``neuron_wrapper.py::_CSABlock.__init__`` -- same.
- Compile driver -- read
  :func:`get_emitted_kernel_slugs` before firing to log which NEFF
  cache line the graph will use.

Anchors (absolute local paths)
------------------------------
- KDA v3.2 receipt (KDA does not apply here, listed for the cross-model
  campaign audit trail):
  ``C:\\Users\\apumu\\research\\InfinityAI\\gemma4-trn2-handoff\\
   harness-v2\\staging\\reference-sweep-20260826T2150Z\\kernels\\
   KDA-V3-SPIKE-2026-08-28.md``
- DSA v2 receipt:
  ``C:\\Users\\apumu\\research\\InfinityAI\\gemma4-trn2-handoff\\
   harness-v2\\staging\\reference-sweep-20260826T2150Z\\kernels\\
   DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md`` §3C.
- Parallel implementation:
  ``C:\\Users\\apumu\\research\\InfinityAI\\vllm-neuron-codex-alpha\\
   vllm_neuron\\model\\glm53_flash\\kernel_dispatch.py``
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Frozen slug identities (DSA only; DSv4 has no KDA)
# ---------------------------------------------------------------------------

DSA_CPU_GOLDEN_SLUG: str = "nki_v0_reference_lightning_indexer"
"""Slug for the torch CPU-golden DSA kernel (dsa_lightning_indexer v0)."""

DSA_NKI_V2_SLUG: str = "dsa_sparse_attention.nki_v2"
"""Slug for the DSA v2 (Option 2 inline gather) NKI device kernel.

Baremetal COMPILE PASS on r7i, 29.91s, per DSA-LIGHTNING-INDEXER-STATUS-
2026-08-28.md §3C.
"""

DSA_KERNEL_IMPL_ENV: str = "DSA_KERNEL_IMPL"

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


def resolve_dsa_impl_slug(env_value: str | None = None) -> str:
    """Return the DSA slug the emitted config should advertise.

    Raises
    ------
    ValueError
        If ``env_value`` is a non-empty string that is not one of the
        known aliases. Silently defaulting an unknown value would mask
        typos and land the compile driver on the wrong NEFF cache line.
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
    """Return the ``{"dsa": slug}`` mapping for THIS process.

    On DSv4-Flash this is the ONE authoritative view of "what NEFF
    cache identity the compile driver would use if the DSA layers were
    fired NOW". Tests assert against this to prove the env flip lands
    the requested slug; compile drivers log it before submitting a
    fire.

    Note: DSv4 has no KDA layers, so the returned dict does not carry
    a ``kda`` key. Callers cross-referencing the GLM-5.3-Flash /
    Kimi-K3 lanes' KDA slugs use those wrappers' own
    ``get_emitted_kernel_slugs()`` there instead.
    """
    return {
        "dsa": resolve_dsa_impl_slug(),
    }


def is_dsa_nki_v2_selected() -> bool:
    """True iff DSA_KERNEL_IMPL resolves to the v2 NKI slug."""
    return resolve_dsa_impl_slug() == DSA_NKI_V2_SLUG


__all__ = [
    "DSA_CPU_GOLDEN_SLUG",
    "DSA_KERNEL_IMPL_ENV",
    "DSA_NKI_V2_SLUG",
    "get_emitted_kernel_slugs",
    "is_dsa_nki_v2_selected",
    "resolve_dsa_impl_slug",
]
