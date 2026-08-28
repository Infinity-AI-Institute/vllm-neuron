# SPDX-License-Identifier: Apache-2.0
"""Registry and cache-identity surface for GLM-5.3-Flash."""

from __future__ import annotations

from .indexer import DSA_KERNEL_SLUG
from .kda import KDA_KERNEL_SLUG
from .model import NeuronGlm53FlashForCausalLM
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
    return [
        ("Glm5NextForConditionalGeneration", NeuronGlm53FlashForCausalLM),
    ]


__all__ = [
    "GLM53_SOURCE_CACHE_ABI",
    "_GLM53_GRAPH_ID",
    "get_models",
]
