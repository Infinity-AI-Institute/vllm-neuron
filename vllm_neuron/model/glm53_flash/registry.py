# SPDX-License-Identifier: Apache-2.0
"""Registry and cache-identity surface for GLM-5.3-Flash.

`GLM53_SOURCE_CACHE_ABI` / `_GLM53_GRAPH_ID` pin the compile cache per the
COMPILE-FASTPATH.md convention: any change to (kernel slug set, MLA/DSA/KDA
geometry, layer count, IndexPool ratio, wrapper-forward shape) bumps a slug
string and therefore the cache key.  `get_models()` returns the NxDI-compatible
wrapper class (defined in `.neuron_wrapper`).  The wrapper is imported lazily
inside `get_models` so `.neuron_wrapper` can freely import the ABI constants
back without triggering a circular import at package load time.

Round 2 (2026-08-28) bumps the slug: `_NeuronGlm53FlashModel` was replaced with
per-layer NxDI parallel primitives (ColumnParallelLinear / RowParallelLinear /
ParallelEmbedding) instead of the Round-1 single-Linear shell.  Any Round-1
compile artifact would deserialize into the wrong graph shape, so the slug
change forces the modular-compile flywheel to treat Round-2 artifacts as
cache-distinct.
"""

from __future__ import annotations

from .indexer import DSA_KERNEL_SLUG
from .kda import KDA_KERNEL_SLUG
from .moe import MOE_KERNEL_SLUG

# Round-1 slug:  "glm53-flash-source-v1"  (shell single-Linear per layer)
# Round-2 slug:  "glm53-flash-round2-nxdi-primitives-v1"
#   change set (relative to Round 1):
#     - per-layer KDA / All-NoPE MLA + DSA / MoE / Dense MLP blocks lowered to
#       ColumnParallelLinear + RowParallelLinear + ParallelEmbedding
#     - 45-layer dispatch: layers 0..2 dense, layers {3,7,...,43} DSA, others KDA
#     - shared-expert branch present in every MoE layer
#     - MoE routed-expert block declares NxDI blockwise-matmul workaround at
#       InferenceConfig init time
#   correctness bar deferred (Round 3): device NKI kernel bindings for KDA
#   state, DSA sparse-attn, blockwise MoE.  The Round-2 forward for those
#   kernel-dependent ops raises NotImplementedError until Round 3 binds NKI.
GLM53_SOURCE_CACHE_ABI = (
    "glm53-flash-round2-nxdi-primitives-v1"
    f"|dsa={DSA_KERNEL_SLUG}"
    f"|kda={KDA_KERNEL_SLUG}"
    f"|moe={MOE_KERNEL_SLUG}"
    "|qk=256|nope=256|rope=0|index-kpool=4|layers=45"
    "|hc-mult=4|routed-experts=288|top-k=8|shared-experts=1"
)
_GLM53_GRAPH_ID = "glm53-flash-nkiv0-refs-round2-v1"


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
