"""ModelRegistry factory for native Gemma 4 vLLM-Neuron onboarding."""

import torch.nn as nn

from .config import Gemma4Config


class Gemma4MoeForCausalLM(nn.Module):
    """Native vLLM-Neuron entry point.

    The implementation is intentionally kept behind this factory so model
    registration and platform/config validation are stable while attention,
    paged-KV writes, and MoE kernels are brought over from the reference port.
    """

    def __init__(self, config, neuron_config=None):
        super().__init__()
        self.config = Gemma4Config.from_configs(config, neuron_config)
        from .model import Gemma4MoeModel

        self.model = Gemma4MoeModel(self.config)

    @classmethod
    def from_configs(cls, config, neuron_config=None):
        return cls(config, neuron_config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
