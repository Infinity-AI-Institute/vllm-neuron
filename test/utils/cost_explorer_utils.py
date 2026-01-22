# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from test.utils.instance_type import get_instance_type

UTILS_DIR = Path(__file__).resolve().parent
CHAT_TMPL = UTILS_DIR / "server" / "prompt-template.jinja"

ACCURACY_MODELS = [
    "llama-3.1/llama-3.1-8b",
    "llama-3.3/llama-3.3-70b-instruct",
    "mistral/Mistral-7B-Instruct-v0.2",
    "qwen/qwen-3-14b",
]

PERF_MODELS = [
    "llama-3.1/llama-3.1-8b",
    "llama-3.3/llama-3.3-70b-instruct",
    "mistral/Mistral-7B-Instruct-v0.2",
    "qwen/qwen-3-14b",
]


def get_server_config():
    instance_type = get_instance_type()

    configs = {
        "trn1.2xlarge": {
            "batch_size": 1,
            "tp_degree": 2,
            "max_model_len": 256,
            "max_seq_len": 256,
            "override_neuron_config": {
                "max_context_length": 256,
                "max_new_tokens": 256,
                "is_continuous_batching": True,
                "ctx_batch_size": 1,
                "enable_bucketing": False,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
            },
            "mean_input_tokens": 256,
            "mean_output_tokens": 256,
        },
        "trn1.32xlarge": {
            "batch_size": 1,
            "tp_degree": 32,
            "max_model_len": 16384,
            "max_seq_len": 16384,
            "override_neuron_config": {
                "max_context_length": 14336,
                "max_new_tokens": 2048,
                "is_continuous_batching": True,
                "ctx_batch_size": 1,
                "enable_bucketing": False,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
            },
            "mean_input_tokens": 3072,
            "mean_output_tokens": 1024,
        },
        "trn2.48xlarge": {
            "batch_size": 1,
            "tp_degree": 64,
            "max_model_len": 16384,
            "max_seq_len": 16384,
            "override_neuron_config": {
                "max_context_length": 14336,
                "max_new_tokens": 2048,
                "is_continuous_batching": True,
                "ctx_batch_size": 1,
                "enable_bucketing": False,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
            },
            "mean_input_tokens": 3072,
            "mean_output_tokens": 1024,
        },
    }

    if instance_type not in configs:
        raise ValueError(
            f"Unsupported instance type: {instance_type}. "
            f"Supported types: {', '.join(configs.keys())}"
        )
    return configs[instance_type]


def get_perf_config():
    instance_type = get_instance_type()

    configs = {
        "trn1.2xlarge": {
            "input_size": 512,
            "output_size": 256,
        },
        "trn1.32xlarge": {
            "input_size": 3072,
            "output_size": 1024,
        },
        "trn2.48xlarge": {
            "input_size": 3072,
            "output_size": 1024,
        },
    }

    if instance_type not in configs:
        raise ValueError(
            f"Unsupported instance type: {instance_type}. "
            f"Supported types: {', '.join(configs.keys())}"
        )
    return configs[instance_type]


def get_accuracy_test_configs(models, chat_template_path):
    """Generate accuracy test configurations for all models."""
    base_config = get_server_config()
    configs = []

    for model in models:
        configs.append(
            {
                "name": f"acc-{model.split('/')[-1]}",
                "model": model,
                "model_path": model,
                "server_port": 8000,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "block_size": 32,
                "num_blocks_override": 1024,
                "custom_chat_template_path": str(chat_template_path),
                "disable_log_requests": True,
                "batch_size": base_config["batch_size"],
                "tp_degree": base_config["tp_degree"],
                "max_model_len": base_config["max_model_len"],
                "max_seq_len": base_config["max_seq_len"],
                "mean_input_tokens": base_config["mean_input_tokens"],
                "mean_output_tokens": base_config["mean_output_tokens"],
                "override_neuron_config": {
                    **base_config["override_neuron_config"],
                    "is_prefix_caching": False,
                    "is_block_kv_layout": False,
                },
            }
        )

    return configs


def get_perf_test_configs(models, chat_template_path):
    """Generate performance test configurations for all models."""
    server_config = get_server_config()
    perf_config = get_perf_config()
    configs = []

    for model in models:
        client_type = (
            "qwen3_moe_llm_perf"
            if "qwen" in model.lower()
            else "llm_perf_github_patched"
        )

        configs.append(
            {
                "model": model,
                "server": {
                    "name": f"perf-{model.split('/')[-1]}",
                    "model_path": model,
                    "server_port": 8000,
                    "disable_log_requests": True,
                    "custom_chat_template_path": str(chat_template_path),
                    "batch_size": server_config["batch_size"],
                    "tp_degree": server_config["tp_degree"],
                    "max_model_len": server_config["max_model_len"],
                    "max_seq_len": server_config["max_seq_len"],
                    "mean_input_tokens": server_config["mean_input_tokens"],
                    "mean_output_tokens": server_config["mean_output_tokens"],
                    "override_neuron_config": server_config["override_neuron_config"],
                },
                "perf": {
                    "client_type": client_type,
                    "max_concurrent_requests": 1,
                    "n_batches": 5,
                    "max_num_completed_requests": 10,
                    "stddev_input_tokens": 50,
                    "stddev_output_tokens": 50,
                    "timeout": 7200,
                    "sampling_mode": "greedy",
                    "input_size": perf_config["input_size"],
                    "output_size": perf_config["output_size"],
                },
            }
        )

    return configs


ACCURACY_TEST_CONFIGS = get_accuracy_test_configs(ACCURACY_MODELS, CHAT_TMPL)
PERF_TEST_CONFIGS = get_perf_test_configs(PERF_MODELS, CHAT_TMPL)
