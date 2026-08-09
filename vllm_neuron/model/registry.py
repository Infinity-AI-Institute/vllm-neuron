# SPDX-License-Identifier: Apache-2.0
import os

from .llama3 import LlamaForCausalLM
from .gpt_oss import GptOssForCausalLM
from .glm52_moe_dsa.factory import GlmMoeDsaForCausalLM
from .llama3 import Eagle3LlamaForCausalLM
from .qwen3_vl import Qwen3VLForConditionalGeneration


def get_models() -> list[tuple[str, type]]:
    """Return a list of available model classes.

    Returns:
        list[tuple[str, type]]: A list of tuples containing model names and their corresponding classes.
            Each tuple contains (model_name, model_class) where:
            - model_name (str): The string identifier for the model, compatible with Hugging Face transformers architecture
            - model_class (type): The actual model class implementation
    """
    models = [
        ("LlamaForCausalLM", LlamaForCausalLM),
        ("GptOssForCausalLM", GptOssForCausalLM),
        ("GlmMoeDsaForCausalLM", GlmMoeDsaForCausalLM),
        ("Eagle3LlamaForCausalLM", Eagle3LlamaForCausalLM),
        ("Qwen3VLForConditionalGeneration", Qwen3VLForConditionalGeneration),
    ]

    # SyntheticNeuronModel is a testing-only model that replaces real neural
    # network computation with deterministic KV cache fill/verify. Useful for
    # validating infrastructure (KV transfer, sharding, block management)
    # without requiring model weights or compilation.
    # Not for production inference — gated to avoid exposing to customers.
    if os.environ.get("VLLM_NEURON_SYNTHETIC_MODEL") == "1":
        from .synthetic import SyntheticNeuronModel

        models.append(("SyntheticNeuronModel", SyntheticNeuronModel))

    # Kimi K3's model class lives in neuronx_distributed_inference, which is
    # optional for vLLM-Neuron. Keep the rest of the registry available when
    # that package is not installed.
    try:
        from neuronx_distributed_inference.models.kimi_k3.serving.factory import (
            KimiK3ForCausalLM,
        )
    except ImportError:
        pass
    else:
        models.append(("KimiK3ForCausalLM", KimiK3ForCausalLM))

    return models
