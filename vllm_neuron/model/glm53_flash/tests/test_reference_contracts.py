from __future__ import annotations

from vllm_neuron.model.glm53_flash._reference_kernels import (
    load_glm52_moe,
    load_reference_kernel,
)
from vllm_neuron.model.glm53_flash.indexer import DSA_KERNEL_SLUG
from vllm_neuron.model.glm53_flash.kda import KDA_KERNEL_SLUG
from vllm_neuron.model.glm53_flash.model import NeuronGlm53FlashForCausalLM
from vllm_neuron.model.glm53_flash.registry import (
    _GLM53_GRAPH_ID,
    GLM53_SOURCE_CACHE_ABI,
    get_models,
)


def test_all_four_fleet_a_reference_files_import() -> None:
    dsa = load_reference_kernel("dsa")
    kda = load_reference_kernel("kda")
    moe = load_reference_kernel("moe")
    fp8 = load_reference_kernel("fp8")
    assert dsa.KERNEL_SLUG_V0_REFERENCE == DSA_KERNEL_SLUG
    assert kda.KDA_STATE_V2_KERNEL_SLUG == KDA_KERNEL_SLUG
    assert hasattr(moe, "MoEDispatchConfig")
    assert hasattr(fp8, "assert_indexer_multiplier_bounded")


def test_glm52_router_is_reused_from_its_source_file() -> None:
    reused = load_glm52_moe()
    assert reused.select_glm52_experts.__module__ == "_glm53_reused_glm52_moe"


def test_registry_and_graph_cache_identity_are_pinned() -> None:
    assert (
        dict(get_models())["Glm5NextForConditionalGeneration"]
        is NeuronGlm53FlashForCausalLM
    )
    assert _GLM53_GRAPH_ID == GLM53_SOURCE_CACHE_ABI
    assert DSA_KERNEL_SLUG in GLM53_SOURCE_CACHE_ABI
    assert KDA_KERNEL_SLUG in GLM53_SOURCE_CACHE_ABI
    assert "moe_dispatch.reference.v1" in GLM53_SOURCE_CACHE_ABI
