# SPDX-License-Identifier: Apache-2.0
from test.e2e.offline.multi_lora_inference import (
    OfflineCfg,
    multi_lora_offline_inference_config,
)

import pytest

# ---------- Test Tiny Llama Configs ---------- #
TINYLLAMA_CONFIGS = [
    # static multi-lora config
    OfflineCfg(
        name="tinyllama",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # LoRA adapter model IDs from HuggingFace
        lora_ckpt_dict={
            "lora_ckpt_dir": "/home/ubuntu/lora_adapters/TinyLlama-1.1B-Chat-v1.0/",
            "lora_ckpt_paths": {
                "lora_id_1": "barissglc/tinyllama-tarot-v1",
                "lora_id_2": "givyboy/TinyLlama-1.1B-Chat-v1.0-mental-health-conversational",
            },
        },
        max_loras=2,
        tp_degree=32,
        batch_size=2,
        random_adapter_ids=False,
    ),
    # dynamic multi-lora config
    OfflineCfg(
        name="tinyllama",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # LoRA adapter model IDs from HuggingFace
        lora_ckpt_dict={
            "lora_ckpt_dir": "/home/ubuntu/lora_adapters/TinyLlama-1.1B-Chat-v1.0/",
            "lora_ckpt_paths": {
                "lora_id_1": "barissglc/tinyllama-tarot-v1",
                "lora_id_2": "givyboy/TinyLlama-1.1B-Chat-v1.0-mental-health-conversational",
            },
            "lora_ckpt_paths_cpu": {
                "lora_id_1": "barissglc/tinyllama-tarot-v1",
                "lora_id_2": "givyboy/TinyLlama-1.1B-Chat-v1.0-mental-health-conversational",
                "lora_id_3": "barissglc/tinyllama-tarot-v1",
                "lora_id_4": "givyboy/TinyLlama-1.1B-Chat-v1.0-mental-health-conversational",
            },
        },
        max_loras=2,
        max_cpu_loras=4,
        tp_degree=32,
        batch_size=2,
        random_adapter_ids=False,
    ),
]


@pytest.mark.parametrize("config", TINYLLAMA_CONFIGS)
def test_multi_lora_tiny_offline_inference_config(config):
    return multi_lora_offline_inference_config(config)


def _test_multi_lora_tiny_quantize_offline_inference_config(config):
    config.base_model_quantized = True
    return multi_lora_offline_inference_config(config)
