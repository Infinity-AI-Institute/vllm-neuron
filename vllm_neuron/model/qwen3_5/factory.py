# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3.5 (text) on Neuron.

The published checkpoint's architecture is ``Qwen3_5ForConditionalGeneration``
(vision + text + MTP). This serves only the text decoder: the vision tower and
MTP head are not loaded.
"""

from torch import nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3_5ForCausalLM(nn.Module):
    """Extends nn.Module to satisfy vLLM's ModelRegistry requirements."""

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self.from_configs(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config)
        from .model import Qwen3_5ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @staticmethod
    def _validate_config(hf_config: PretrainedConfig) -> None:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError(
                "Qwen3.5 carries recurrent Gated DeltaNet state that the prefix "
                "cache does not hold. Serve with --no-enable-prefix-caching."
            )
