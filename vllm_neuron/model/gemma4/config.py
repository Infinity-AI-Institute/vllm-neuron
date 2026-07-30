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
    global_head_dim: int = 512
    num_global_key_value_heads: int = 1
    query_pre_attn_scalar: float = 256.0
    attention_k_eq_v: bool = False
    layer_types: list[str] = field(default_factory=list)
    sliding_window: int | None = None
    num_kv_shared_layers: int = 0
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_parameters: dict = field(default_factory=dict)
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    def layer_is_global(self, layer_idx: int) -> bool:
        """Return whether a layer uses full/global attention."""
        if not self.layer_types:
            return True
        value = self.layer_types[layer_idx % len(self.layer_types)]
        return str(value).lower() in {"global", "full", "full_attention"}

    def attention_shape(self, layer_idx: int) -> tuple[int, int]:
        """Return the native (head_dim, KV-heads) pair for a layer."""
        if self.layer_is_global(layer_idx):
            return self.global_head_dim, self.num_global_key_value_heads
        return self.head_dim, self.num_key_value_heads

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config=None):
        if isinstance(hf_config, cls):
            hf_config.neuron_config = neuron_config
            return hf_config
        raw = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
        # Gemma 4 stores text parameters below text_config.
        raw = raw.get("text_config", raw)
        fields = cls.__dataclass_fields__
        values = {key: value for key, value in raw.items() if key in fields}
        if values.get("torch_dtype") == "bfloat16":
            values["torch_dtype"] = torch.bfloat16
        values["neuron_config"] = neuron_config
        return cls(**values)
