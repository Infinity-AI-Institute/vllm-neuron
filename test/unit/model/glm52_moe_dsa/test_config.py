# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.static_fp8 import (
    NEURON_LEGACY_E4M3FN_QMAX240,
)


def test_frozen_glm52_geometry() -> None:
    config = Glm52MoeDsaConfig()

    assert config.hidden_size == 6_144
    assert config.num_hidden_layers == 78
    assert config.vocab_size == 154_880
    assert config.q_lora_rank == 2_048
    assert config.kv_lora_rank == 512
    assert config.qk_head_dim == 192 + 64
    assert config.n_routed_experts == 256
    assert config.num_experts_per_tok == 8
    assert config.routed_scaling_factor == 2.5
    assert config.full_indexer_layer_ids == (
        0,
        1,
        2,
        6,
        10,
        14,
        18,
        22,
        26,
        30,
        34,
        38,
        42,
        46,
        50,
        54,
        58,
        62,
        66,
        70,
        74,
    )
    assert len(config.shared_indexer_layer_ids) == 57
    assert config.mlp_layer_types[:3] == ("dense",) * 3
    assert config.mlp_layer_types[3:] == ("sparse",) * 75


def test_static_fp8_weight_format_does_not_change_graph_geometry() -> None:
    original = Glm52MoeDsaConfig()
    direct = Glm52MoeDsaConfig(
        static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
    )
    ignored = {"static_fp8_weight_format", "neuron_config"}

    assert {
        field: getattr(original, field)
        for field in original.__dataclass_fields__
        if field not in ignored
    } == {
        field: getattr(direct, field)
        for field in direct.__dataclass_fields__
        if field not in ignored
    }


def test_from_frozen_hf_shape(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["GlmMoeDsaForCausalLM"],
                "dtype": "bfloat16",
                "hidden_size": 6_144,
                "num_hidden_layers": 78,
            }
        ),
        encoding="utf-8",
    )

    config = Glm52MoeDsaConfig.from_configs(str(config_path), None)

    assert config.torch_dtype is torch.bfloat16
    assert config.full_indexer_layer_ids[-1] == 74


def test_from_configs_reads_explicit_direct_weight_format() -> None:
    config = Glm52MoeDsaConfig.from_configs(
        {
            "architectures": ["GlmMoeDsaForCausalLM"],
            "static_fp8_weight_format": NEURON_LEGACY_E4M3FN_QMAX240,
        },
        None,
    )

    assert config.static_fp8_weight_format == NEURON_LEGACY_E4M3FN_QMAX240


def test_rejects_wrong_router_semantics() -> None:
    with pytest.raises(ValueError, match="sigmoid/noaux_tc"):
        Glm52MoeDsaConfig(scoring_func="softmax")


def test_rejects_non_interleaved_rope() -> None:
    with pytest.raises(ValueError, match="interleaved"):
        Glm52MoeDsaConfig(indexer_rope_interleave=False)
