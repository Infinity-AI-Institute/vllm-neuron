from __future__ import annotations

import pytest
import torch

from vllm_neuron.model.glm53_flash.config import (
    DSA_LAYER_INDICES,
    KDA_LAYER_INDICES,
    Glm53FlashInferenceConfig,
    Glm53LinearAttentionConfig,
)
from vllm_neuron.model.glm53_flash.mhc import Glm53MHC
from vllm_neuron.model.glm53_flash.model import NeuronGlm53FlashForCausalLM


def _tiny_config() -> Glm53FlashInferenceConfig:
    return Glm53FlashInferenceConfig(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=16,
        q_lora_rank=4,
        kv_lora_rank=4,
        v_head_dim=4,
        index_n_heads=1,
        index_head_dim=4,
        index_topk=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        torch_dtype=torch.float32,
        static_fp8=False,
        allow_reduced_shapes=True,
        linear_attn_config=Glm53LinearAttentionConfig(num_heads=2, head_dim=4),
    )


def test_one_step_dispatches_exactly_11_dsa_and_34_kda_layers() -> None:
    model = NeuronGlm53FlashForCausalLM(_tiny_config()).eval()
    stub_state_dict = {
        name: value.clone() for name, value in model.state_dict().items()
    }
    model.load_state_dict(stub_state_dict, strict=True)
    logits = model(torch.tensor([[1]], dtype=torch.long))
    assert logits.shape == (1, 1, 32)
    telemetry = model.get_telemetry()
    assert {i for i, count in telemetry["dsa_path_active"].items() if count} == set(
        DSA_LAYER_INDICES
    )
    assert {i for i, count in telemetry["kda_path_active"].items() if count} == set(
        KDA_LAYER_INDICES
    )
    assert all(telemetry["dsa_path_active"][i] == 1 for i in DSA_LAYER_INDICES)
    assert all(telemetry["kda_path_active"][i] == 1 for i in KDA_LAYER_INDICES)


def test_fp8_scale_load_rejects_ocp_448_range() -> None:
    model = NeuronGlm53FlashForCausalLM(_tiny_config()).eval()
    state = model.state_dict()
    scale_name = next(name for name in state if name.endswith("weight_scale"))
    state[scale_name] = torch.tensor(241.0, dtype=torch.float32)
    with pytest.raises(ValueError, match="240"):
        model.load_state_dict(state)


def test_fp8_scale_load_rejects_integer_dtype() -> None:
    model = NeuronGlm53FlashForCausalLM(_tiny_config()).eval()
    state = model.state_dict()
    scale_name = next(name for name in state if name.endswith("weight_scale"))
    state[scale_name] = torch.tensor(1, dtype=torch.int32)
    with pytest.raises(TypeError, match="float32, float16, or bfloat16"):
        model.load_state_dict(state)


def test_mhc_learned_scale_is_not_misclassified_as_an_fp8_scale() -> None:
    model = NeuronGlm53FlashForCausalLM(_tiny_config()).eval()
    state = model.state_dict()
    state["layers.0.hc_attn.scale"] = torch.tensor(
        [-100.0, 0.0, 500.0], dtype=torch.float32
    )
    model.load_state_dict(state, strict=True)


def test_mhc_sinkhorn_is_doubly_stochastic() -> None:
    config = _tiny_config()
    mhc = Glm53MHC(config)
    residual = torch.randn(2, 3, config.hc_mult, config.hidden_size)
    post_mix, comb_mix, layer_input = mhc.pre(residual)
    assert post_mix.shape == (2, 3, 4, 1)
    assert layer_input.shape == (2, 3, 8)
    assert torch.allclose(comb_mix.sum(dim=-1), torch.ones(2, 3, 4), atol=1e-4)
    assert torch.allclose(comb_mix.sum(dim=-2), torch.ones(2, 3, 4), atol=1e-4)


def test_autoregressive_calls_preserve_kda_and_dsa_state_without_teacher_forcing() -> (
    None
):
    torch.manual_seed(17)
    model = NeuronGlm53FlashForCausalLM(_tiny_config()).eval()
    prefill_logits = model(torch.tensor([[1, 2]], dtype=torch.long))
    greedy_token = prefill_logits[:, -1].argmax(dim=-1, keepdim=True)
    decode_logits = model(greedy_token)
    assert decode_logits.shape == (1, 1, 32)
    assert model._position_offset == 3
    for layer_idx in DSA_LAYER_INDICES:
        dsa = model.layers[layer_idx].self_attn.impl
        assert dsa._key_cache.shape[1] == 3
        assert dsa._value_cache.shape[1] == 3
        assert dsa._index_hidden_cache.shape[1] == 3
    for layer_idx in KDA_LAYER_INDICES:
        kda = model.layers[layer_idx].self_attn.impl
        assert kda._state_bf16 is not None
        assert kda._conv_state is not None
    assert model.reset_attention_state() == 45
    assert model._position_offset == 0
