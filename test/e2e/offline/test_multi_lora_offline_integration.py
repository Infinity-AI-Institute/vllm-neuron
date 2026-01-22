# SPDX-License-Identifier: Apache-2.0
from test.e2e.offline.multi_lora_inference import (
    OfflineCfg,
    multi_lora_offline_inference_config,
)

import pytest

# ---------- Test Config ---------- #
INTEGRATION_CONFIGS = [
    # static multi-lora config
    OfflineCfg(
        name="llama-3.1-8B-Instruct-static-lora",
        model="meta-llama/Llama-3.1-8B-Instruct",
        lora_ckpt_dict={
            "lora_ckpt_dir": "/home/ubuntu/lora_adapters/llama-3.1-8b-instruct/",
            "lora_ckpt_paths": {
                "lora_id_1": "Stefano-M/aixpa_amicifamiglia_short_prompt",
                "lora_id_2": "reissbaker/llama-3.1-8b-abliterated-lora",
                "lora_id_3": "GaetanMichelet/Llama-31-8B_task-2_180-samples_config-2",
                "lora_id_4": "Stefano-M/aixpa_amicifamiglia_short_prompt",
            },
        },
        max_loras=4,
        tp_degree=32,
        batch_size=2,
        fsx=True,
    ),
    # dynamic multi-lora config
    OfflineCfg(
        name="llama-3.1-8B-Instruct-dynamic-lora",
        model="meta-llama/Llama-3.1-8B-Instruct",
        lora_ckpt_dict={
            "lora_ckpt_dir": "/home/ubuntu/lora_adapters/llama-3.1-8b-instruct/",
            "lora_ckpt_paths": {
                "lora_id_1": "Stefano-M/aixpa_amicifamiglia_short_prompt",
                "lora_id_2": "reissbaker/llama-3.1-8b-abliterated-lora",
                "lora_id_3": "GaetanMichelet/Llama-31-8B_task-2_180-samples_config-2",
                "lora_id_4": "Stefano-M/aixpa_amicifamiglia_short_prompt",
            },
            "lora_ckpt_paths_cpu": {
                "lora_id_1": "Stefano-M/aixpa_amicifamiglia_short_prompt",
                "lora_id_2": "reissbaker/llama-3.1-8b-abliterated-lora",
                "lora_id_3": "GaetanMichelet/Llama-31-8B_task-2_180-samples_config-2",
                "lora_id_4": "Stefano-M/aixpa_amicifamiglia_short_prompt",
                "lora_id_5": "Stefano-M/aixpa_amicifamiglia_short_prompt",
                "lora_id_6": "reissbaker/llama-3.1-8b-abliterated-lora",
                "lora_id_7": "GaetanMichelet/Llama-31-8B_task-2_180-samples_config-2",
                "lora_id_8": "Stefano-M/aixpa_amicifamiglia_short_prompt",
            },
        },
        max_loras=4,
        max_cpu_loras=8,
        tp_degree=32,
        batch_size=4,
        fsx=True,
    ),
]


@pytest.mark.parametrize("config", INTEGRATION_CONFIGS)
def test_multi_lora_integrate_offline_inference_config(config):
    return multi_lora_offline_inference_config(config)
