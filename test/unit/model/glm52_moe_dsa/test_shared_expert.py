# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan
from vllm_neuron.model.glm52_moe_dsa.shared_expert import Glm52SharedExpert
from vllm_neuron.utils.weight_loader import get_weight_loader


def _config() -> Glm52MoeDsaConfig:
    return Glm52MoeDsaConfig(
        vocab_size=16,
        hidden_size=4,
        num_hidden_layers=4,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=4,
        kv_lora_rank=2,
        qk_head_dim=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        index_n_heads=2,
        index_head_dim=2,
        index_topk=2,
        index_skip_topk_offset=1,
        index_topk_freq=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        first_k_dense_replace=1,
    )


def test_shared_expert_uses_routed_tp_subgroup() -> None:
    plan = RoutedExpertPlan(4, 2, 4, 8)
    module = Glm52SharedExpert(_config(), plan, global_rank=3)

    assert module.shared_tp_degree == 2
    assert module.shared_tp_rank == 1
    assert module.gate_proj.weight.shape == (4, 4)
    assert module.up_proj.weight.shape == (4, 4)
    assert module.down_proj.weight.shape == (4, 4)
    assert module.gate_proj.weight.dtype == torch.float8_e4m3fn
    assert module.gate_proj.weight_scale.shape == (128, 1)
    assert module.gate_up_input_scale.shape == (128, 1)
    assert get_weight_loader(module.gate_proj.weight).transform is not None
    assert get_weight_loader(module.down_proj.weight).transform is not None
    assert get_weight_loader(module.gate_up_input_scale).transform is not None


def test_shared_expert_rejects_prefill_unsafe_tp_degree() -> None:
    config = _config()
    config.moe_intermediate_size = 64
    plan = RoutedExpertPlan(32, 1, 4, 64)

    try:
        Glm52SharedExpert(config, plan, global_rank=0)
    except ValueError as error:
        assert "requires TP16 or smaller" in str(error)
    else:
        raise AssertionError("prefill-unsafe shared TP degree was accepted")
