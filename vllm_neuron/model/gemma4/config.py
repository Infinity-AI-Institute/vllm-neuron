"""Gemma 4 architecture configuration for the vLLM-Neuron model path."""

from dataclasses import dataclass, field
import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Gemma4Config:
    """Model parameters consumed by the native vLLM-Neuron implementation."""

    vocab_size: int = 262208
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 34
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 256
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_parameters: dict = field(default_factory=dict)
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config=None):
        raw = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
        # Gemma 4 stores text parameters below text_config.
        raw = raw.get("text_config", raw)
        fields = cls.__dataclass_fields__
        values = {key: value for key, value in raw.items() if key in fields}
        if values.get("torch_dtype") == "bfloat16":
            values["torch_dtype"] = torch.bfloat16
        values["neuron_config"] = neuron_config
        return cls(**values)
