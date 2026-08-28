# SPDX-License-Identifier: Apache-2.0
"""Registry and cache-identity surface for GLM-5.3-Flash.

`GLM53_SOURCE_CACHE_ABI` / `_GLM53_GRAPH_ID` pin the compile cache per the
COMPILE-FASTPATH.md convention: any change to (kernel slug set, MLA/DSA/KDA
geometry, layer count, IndexPool ratio) bumps a slug string and therefore the
cache key.  `get_models()` returns the NxDI-compatible wrapper class (defined
in `.neuron_wrapper`).  The wrapper is imported lazily inside `get_models` so
`.neuron_wrapper` can freely import the ABI constants back without triggering
a circular import at package load time.
"""

from __future__ import annotations

from .indexer import DSA_KERNEL_SLUG
from .kda import KDA_KERNEL_SLUG
from .moe import MOE_KERNEL_SLUG

GLM53_SOURCE_CACHE_ABI = (
    "glm53-flash-source-v1"
    f"|dsa={DSA_KERNEL_SLUG}"
    f"|kda={KDA_KERNEL_SLUG}"
    f"|moe={MOE_KERNEL_SLUG}"
    "|qk=256|nope=256|rope=0|index-kpool=4|layers=45"
)
_GLM53_GRAPH_ID = GLM53_SOURCE_CACHE_ABI


def get_models() -> list[tuple[str, type]]:
    # Deferred import breaks the neuron_wrapper <-> registry cycle: the
    # wrapper class carries the ABI on itself and pulls it from this module
    # at class-definition time.
    from .neuron_wrapper import NeuronGlm53FlashForCausalLM

    return [
        ("Glm5NextForConditionalGeneration", NeuronGlm53FlashForCausalLM),
    ]


__all__ = [
    "GLM53_SOURCE_CACHE_ABI",
    "_GLM53_GRAPH_ID",
    "get_models",
]
