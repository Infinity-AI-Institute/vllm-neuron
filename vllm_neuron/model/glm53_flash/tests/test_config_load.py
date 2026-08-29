from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from vllm_neuron.model.glm53_flash.config import (
    DSA_LAYER_INDICES,
    KDA_LAYER_INDICES,
    Glm53FlashInferenceConfig,
)


def test_local_pretrained_config_has_dual_schedule_and_explicit_fp8_defaults(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "glm53"
    config_dir.mkdir()
    layer_types = [
        "deepseek_sparse_attention" if i in DSA_LAYER_INDICES else "linear_attention"
        for i in range(45)
    ]
    raw_config = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": 45,
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_head_dim": 256,
            "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 0,
            "v_head_dim": 256,
            "mla_use_nope": True,
            "index_kpool": 4,
            "index_kpool_always_select_tail": True,
            "index_kpool_compress": True,
            "layer_types": layer_types,
            "linear_attn_config": {
                "num_heads": 64,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "kda_layers": list(KDA_LAYER_INDICES),
                "full_attn_layers": list(DSA_LAYER_INDICES),
            },
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
            "first_k_dense_replace": 3,
            "vocab_size": 154880,
            "max_position_embeddings": 1048576,
            "eos_token_id": [154820, 154827, 154829],
            "dtype": "bfloat16",
        },
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    }
    (config_dir / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")
    config = Glm53FlashInferenceConfig.from_pretrained(config_dir)
    assert config.num_hidden_layers == 45
    assert (
        tuple(
            i
            for i, kind in enumerate(config.layer_types)
            if kind == "deepseek_sparse_attention"
        )
        == DSA_LAYER_INDICES
    )
    assert (
        tuple(
            i for i, kind in enumerate(config.layer_types) if kind == "linear_attention"
        )
        == KDA_LAYER_INDICES
    )
    assert config.linear_attn_config.full_attn_layers == DSA_LAYER_INDICES
    assert config.linear_attn_config.kda_layers == KDA_LAYER_INDICES
    assert config.qk_rope_head_dim == 0
    assert config.mla_use_nope is True
    assert config.index_kpool == 4
    assert config.quantization_config.quant_method == "fp8"
    assert config.quantization_config.activation_scheme == "dynamic"
    assert config.quantization_config.fmt == "e4m3"
    assert config.quantization_config.weight_block_size == (128, 128)
    assert config.static_fp8_weight_format == "neuron_legacy_e4m3fn_qmax240"
    for value in (
        config.fp8_weight_scale_default,
        config.fp8_activation_scale_default,
        config.key_cache_quant_multiplier,
        config.value_cache_quant_multiplier,
        config.indexer_cache_quant_multiplier,
    ):
        assert value is not None
        assert value <= 240.0


@pytest.mark.skipif(
    os.environ.get("GLM53_RUN_HF_CONFIG_TEST") != "1",
    reason="set GLM53_RUN_HF_CONFIG_TEST=1 for the live Hugging Face config gate",
)
def test_hugging_face_pretrained_config() -> None:
    pytest.importorskip("transformers")
    config = Glm53FlashInferenceConfig.from_pretrained("zai-org/GLM-5.3-Flash")
    assert config.num_hidden_layers == 45
    assert config.linear_attn_config.full_attn_layers == DSA_LAYER_INDICES
    assert config.linear_attn_config.kda_layers == KDA_LAYER_INDICES
    assert config.layer_types.count("deepseek_sparse_attention") == 11
    assert config.layer_types.count("linear_attention") == 34
    assert config.qk_head_dim == 256
    assert config.qk_nope_head_dim == 256
    assert config.qk_rope_head_dim == 0
    assert config.mla_use_nope is True
    assert config.index_kpool == 4
    assert config.index_kpool_always_select_tail is True
    assert config.index_kpool_compress is True
    assert config.vocab_size == 154880
    assert config.max_position_embeddings == 1048576
    assert config.eos_token_id == (154820, 154827, 154829)
    assert config.first_k_dense_replace == 3
    assert config.quantization_config.weight_block_size == (128, 128)
    assert config.torch_dtype in (torch.bfloat16, torch.float16, torch.float32)
