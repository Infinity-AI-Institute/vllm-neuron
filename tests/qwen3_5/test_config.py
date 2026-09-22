# SPDX-License-Identifier: Apache-2.0
"""Qwen3_5Config parses the nested dict transformers actually emits."""

import pytest
import torch

from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

transformers = pytest.importorskip("transformers")
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config as HFConfig,
)

# Qwen/Qwen3.5-9B at c202236235762e1c871ad0ccb60c8ee5ba337b9a (inf-lib's pinned architecture).
NINE_B_TEXT = {
    "vocab_size": 248320,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "rms_norm_eps": 1e-6,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 1e7,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
    },
}


def test_parses_transformers_nested_config():
    hf = HFConfig(text_config=NINE_B_TEXT, tie_word_embeddings=False)
    config = Qwen3_5Config.from_configs(hf.to_dict())
    assert config.hidden_size == 4096 and config.vocab_size == 248320
    assert config.rope_theta == 1e7 and config.rotary_dim == 64
    assert config.conv_dim == 8192
    assert config.full_attention_layers == [3, 7, 11, 15, 19, 23, 27, 31]
    assert len(config.linear_attention_layers) == 24
    assert config.tie_word_embeddings is False
    assert config.torch_dtype in (torch.bfloat16, torch.float32)


def test_defaults_are_the_9b():
    config = Qwen3_5Config()
    assert config.full_attention_layers == [3, 7, 11, 15, 19, 23, 27, 31]
    assert config.rotary_dim == 64 and config.conv_dim == 8192
