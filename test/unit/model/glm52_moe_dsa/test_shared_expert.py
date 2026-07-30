# SPDX-License-Identifier: Apache-2.0

import torch
import types

from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan
from vllm_neuron.model.glm52_moe_dsa.shared_expert import Glm52SharedExpert
from vllm_neuron.utils.weight_loader import get_weight_loader


class _FakeSlice:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def get_shape(self) -> list[int]:
        return list(self.tensor.shape)

    def __getitem__(self, index):
        return self.tensor[index]


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


def test_hybrid_shared_expert_uses_bf16_weights_without_static_scales() -> None:
    config = _config()
    config.shared_expert_dtype = "bfloat16"
    plan = RoutedExpertPlan(4, 2, 4, 8)
    module = Glm52SharedExpert(config, plan, global_rank=3)

    assert module.static_fp8 is False
    assert module.gate_proj.weight.shape == (4, 4)
    assert module.up_proj.weight.shape == (4, 4)
    assert module.down_proj.weight.shape == (4, 4)
    assert module.gate_proj.weight.dtype == torch.bfloat16
    assert module.gate_up_input_scale is None
    assert module.down_input_scale is None
    assert not hasattr(module.gate_proj, "weight_scale")
    assert get_weight_loader(module.gate_proj.weight).transform is not None
    assert get_weight_loader(module.down_proj.weight).transform is not None


def test_hybrid_shared_expert_loader_uses_subgroup_rank_and_transposes() -> None:
    config = _config()
    config.shared_expert_dtype = "bfloat16"
    plan = RoutedExpertPlan(4, 2, 4, 8)
    module = Glm52SharedExpert(config, plan, global_rank=3)
    gate_source = torch.arange(8 * 4, dtype=torch.bfloat16).reshape(8, 4)
    down_source = torch.arange(4 * 8, dtype=torch.bfloat16).reshape(4, 8)

    gate = get_weight_loader(module.gate_proj.weight).load(
        [_FakeSlice(gate_source)],
        rank=3,
    )
    down = get_weight_loader(module.down_proj.weight).load(
        [_FakeSlice(down_source)],
        rank=3,
    )

    torch.testing.assert_close(gate, gate_source[4:8, :].T)
    torch.testing.assert_close(down, down_source[:, 4:8].T)


def test_hybrid_prefill_reuses_router_normalization() -> None:
    config = _config()
    config.shared_expert_dtype = "bfloat16"
    module = Glm52SharedExpert(
        config,
        RoutedExpertPlan(4, 2, 4, 8),
        global_rank=0,
    )
    module._run = types.MethodType(lambda _self, value: value, module)
    hidden = torch.full((2, 4), 9.0, dtype=torch.bfloat16)
    normalized = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    group = types.SimpleNamespace(world_size=1)

    output = module.forward_prefill(
        hidden,
        norm_weight=torch.ones(4, dtype=torch.bfloat16),
        eps=config.rms_norm_eps,
        tp_group=group,
        hidden_states_normalized=normalized,
    )

    torch.testing.assert_close(output, normalized)


def test_static_prefill_rejects_pre_normalized_input() -> None:
    config = _config()
    module = Glm52SharedExpert(
        config,
        RoutedExpertPlan(4, 2, 4, 8),
        global_rank=0,
    )

    try:
        module.forward_prefill(
            torch.ones(2, 4, dtype=torch.bfloat16),
            norm_weight=torch.ones(4, dtype=torch.bfloat16),
            eps=config.rms_norm_eps,
            tp_group=types.SimpleNamespace(world_size=1),
            hidden_states_normalized=torch.ones(2, 4, dtype=torch.bfloat16),
        )
    except ValueError as error:
        assert "fused RMSNorm" in str(error)
    else:
        raise AssertionError("static-FP8 prefill accepted pre-normalized input")


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
