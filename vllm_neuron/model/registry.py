# SPDX-License-Identifier: Apache-2.0
import os

from .llama3 import LlamaForCausalLM
from .gpt_oss import GptOssForCausalLM
from .llama3 import Eagle3LlamaForCausalLM
from .qwen3_vl import Qwen3VLForConditionalGeneration
from .gemma4 import Gemma4ForCausalLM
from .inkling import InklingForConditionalGeneration


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
        ("Eagle3LlamaForCausalLM", Eagle3LlamaForCausalLM),
        ("Qwen3VLForConditionalGeneration", Qwen3VLForConditionalGeneration),
        ("Gemma4ForCausalLM", Gemma4ForCausalLM),
        # The released 26B-A4B checkpoint nests the text config below the
        # conditional-generation config. Text-only Neuron serving uses the
        # same native text implementation for that outer architecture.
        ("Gemma4ForConditionalGeneration", Gemma4ForCausalLM),
        ("InklingForConditionalGeneration", InklingForConditionalGeneration),
        # The public checkpoint has a multimodal outer config.  Text-only
        # Neuron serving overrides the architecture to this alias so vLLM
        # does not initialize vision/audio preprocessing; the factory still
        # unwraps and validates the exact nested text_config.
        ("InklingForCausalLM", InklingForConditionalGeneration),
    ]

    # SyntheticNeuronModel is a testing-only model that replaces real neural
    # network computation with deterministic KV cache fill/verify. Useful for
    # validating infrastructure (KV transfer, sharding, block management)
    # without requiring model weights or compilation.
    # Not for production inference — gated to avoid exposing to customers.
    if os.environ.get("VLLM_NEURON_SYNTHETIC_MODEL") == "1":
        from .synthetic import SyntheticNeuronModel

        models.append(("SyntheticNeuronModel", SyntheticNeuronModel))

    return models
