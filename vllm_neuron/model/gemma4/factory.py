"""ModelRegistry factory for native Gemma 4 vLLM-Neuron onboarding."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .config import Gemma4Config


class Gemma4ForCausalLM(nn.Module):
    """Validate configuration and select the native Gemma 4 model."""

    def __init__(
        self,
        hf_config: PretrainedConfig | Gemma4Config,
        neuron_config: NeuronConfig | None = None,
    ):
        super().__init__()
        self._model = self._select_implementation(
            hf_config, neuron_config
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | Gemma4Config,
        neuron_config: NeuronConfig | None = None,
    ) -> nn.Module:
        """Return the runner-facing implementation directly."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig | Gemma4Config,
        neuron_config: NeuronConfig | None,
    ) -> nn.Module:
        config = Gemma4Config.from_configs(hf_config, neuron_config)
        cls._validate_config(config)
        from .model import Gemma4ForCausalLM as Model

        return Model(config)

    @staticmethod
    def _validate_config(config: Gemma4Config) -> None:
        if config.layer_types and (
            len(config.layer_types) != config.num_hidden_layers
        ):
            raise ValueError(
                "Gemma 4 layer_types must contain one entry per layer"
            )
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by "
                "num_key_value_heads"
            )
        if (
            config.num_attention_heads
            % config.num_global_key_value_heads
        ):
            raise ValueError(
                "num_attention_heads must be divisible by "
                "num_global_key_value_heads"
            )
        if not 1 <= config.top_k_experts <= config.num_experts:
            raise ValueError(
                "top_k_experts must be between 1 and num_experts"
            )
