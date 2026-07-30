# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_neuron.model.glm52_moe_dsa.cache_layout import Glm52CacheLayout
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.dense_mlp import Glm52DenseMlp
from vllm_neuron.model.glm52_moe_dsa.expert_kernels import Glm52RoutedExperts
from vllm_neuron.model.glm52_moe_dsa.mla import Glm52MlaAttention
from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan
from vllm_neuron.model.glm52_moe_dsa.shared_expert import Glm52SharedExpert
from vllm_neuron.model.glm52_moe_dsa.static_fp8 import (
    NEURON_LEGACY_E4M3FN_QMAX240,
)


def _config(*, direct: bool) -> Glm52MoeDsaConfig:
    return Glm52MoeDsaConfig(
        shared_expert_dtype="bfloat16",
        static_fp8_weight_format=(
            NEURON_LEGACY_E4M3FN_QMAX240 if direct else None
        ),
    )


def _signature(module: torch.nn.Module) -> tuple[tuple[object, ...], ...]:
    parameters = (
        (
            "parameter",
            name,
            tuple(parameter.shape),
            parameter.dtype,
            parameter.requires_grad,
        )
        for name, parameter in module.named_parameters()
    )
    buffers = (
        ("buffer", name, tuple(buffer.shape), buffer.dtype)
        for name, buffer in module.named_buffers()
    )
    return tuple(sorted((*parameters, *buffers), key=lambda row: (row[0], row[1])))


def _components(config: Glm52MoeDsaConfig) -> tuple[torch.nn.Module, ...]:
    plan = RoutedExpertPlan(
        world_size=64,
        ep_degree=16,
        num_experts=config.n_routed_experts,
        expert_intermediate_size=config.moe_intermediate_size,
    )
    layout = Glm52CacheLayout.build(
        config,
        world_size=64,
        cache_dtype=torch.bfloat16,
    )
    return (
        Glm52MlaAttention(
            config,
            layer_idx=0,
            cache_layout=layout,
            world_size=64,
            static_fp8=True,
            device="meta",
        ),
        Glm52DenseMlp(
            config,
            world_size=64,
            global_rank=0,
            tp_group=type("Group", (), {"world_size": 64})(),
            static_fp8=True,
            device="meta",
        ),
        Glm52RoutedExperts(
            config,
            plan,
            global_rank=0,
            device="meta",
        ),
        Glm52SharedExpert(
            config,
            plan,
            global_rank=0,
            device="meta",
        ),
    )


def test_direct_encoding_changes_only_loader_data_not_graph_geometry() -> None:
    original = _components(_config(direct=False))
    direct = _components(_config(direct=True))

    assert tuple(map(_signature, original)) == tuple(map(_signature, direct))
    for original_module, direct_module in zip(original, direct, strict=True):
        assert type(original_module) is type(direct_module)


def test_direct_hybrid_shared_expert_remains_bf16_without_static_scales() -> None:
    shared = _components(_config(direct=True))[-1]

    assert shared.gate_proj.weight.dtype == torch.bfloat16
    assert shared.up_proj.weight.dtype == torch.bfloat16
    assert shared.down_proj.weight.dtype == torch.bfloat16
    assert not hasattr(shared.gate_proj, "weight_scale")
    assert shared.gate_up_input_scale is None
    assert shared.down_input_scale is None
