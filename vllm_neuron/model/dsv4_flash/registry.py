# SPDX-License-Identifier: Apache-2.0
"""Registry and cache-identity surface for DeepSeek-V4-Flash.

Same COMPILE-FASTPATH.md convention as GLM-5.3-Flash: any change to the
(kernel slug set, MQA/CSA/HCA geometry, layer count, indexer top-k,
wrapper-forward shape) bumps a slug string and therefore the compile
cache key.  Round-1 slug is a scaffold; the compile-fastpath will never
match a Round-1 artifact because the wrapper cannot init yet.
"""

from __future__ import annotations

DSV4_SOURCE_CACHE_ABI = (
    "dsv4-flash-round1-scaffold-v1"
    "|attn=mqa|kv=shared|head-dim=512|qk-rope-head=64"
    "|o-groups=8|o-lora=1024|q-lora=1024"
    "|layer-schedule=sliding|csa|hca"
    "|index-topk=512|index-n-heads=64|index-head-dim=128"
    "|routed-experts=256|top-k=6|shared-experts=1|hash-boot=3"
    "|hc-mult=4|sinkhorn=20|scoring=sqrtsoftplus"
    "|sliding-window=128|compress-csa=4|compress-hca=128"
    "|expert-dtype=fp4-ue8m0|non-expert-dtype=fp8-ue8m0"
)
_DSV4_GRAPH_ID = "dsv4-flash-scaffold-round1-v1"


def get_models() -> list[tuple[str, type]]:
    """Return the HF-architecture-id -> wrapper class binding.

    Deferred import breaks the neuron_wrapper <-> registry cycle.
    """
    from .neuron_wrapper import NeuronDeepseekV4FlashForCausalLM

    return [
        ("DeepseekV4ForCausalLM", NeuronDeepseekV4FlashForCausalLM),
    ]


__all__ = [
    "DSV4_SOURCE_CACHE_ABI",
    "_DSV4_GRAPH_ID",
    "get_models",
]
