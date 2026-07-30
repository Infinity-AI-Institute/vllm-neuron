"""Gemma 4 architecture configuration for the vLLM-Neuron model path."""

from dataclasses import dataclass, field
import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Gemma4Config:
    """Model parameters consumed by the native vLLM-Neuron implementation."""

    vocab_size: int = 262144
    hidden_size: int = 2816
    intermediate_size: int = 2112
    moe_intermediate_size: int = 704
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    global_head_dim: int = 512
    num_global_key_value_heads: int = 2
    num_experts: int = 128
    top_k_experts: int = 8
    enable_moe_block: bool = True
    attention_k_eq_v: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    layer_types: list[str] = field(default_factory=list)
    sliding_window: int | None = 1024
    num_kv_shared_layers: int = 0
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_parameters: dict = field(default_factory=dict)
    hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int = 262144
    use_double_wide_mlp: bool = False
    final_logit_softcapping: float | None = 30.0
    tie_word_embeddings: bool = True
    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    def layer_is_global(self, layer_idx: int) -> bool:
        """Return whether a layer uses full/global attention."""
        if not self.layer_types:
            return True
        value = self.layer_types[layer_idx % len(self.layer_types)]
        return str(value).lower() in {"global", "full", "full_attention"}

    def layer_uses_shared_kv(self, layer_idx: int) -> bool:
        """Return whether a layer reuses K/V projected by an earlier layer."""
        first_shared = self.num_hidden_layers - self.num_kv_shared_layers
        return self.num_kv_shared_layers > 0 and layer_idx >= first_shared

    def layer_intermediate_size(self, layer_idx: int) -> int:
        """Return the dense MLP width for a layer."""
        if self.use_double_wide_mlp and self.layer_uses_shared_kv(layer_idx):
            return 2 * self.intermediate_size
        return self.intermediate_size

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
        # Transformers 5 serializes this field as ``dtype``. Older releases
        # used ``torch_dtype``.
        serialized_dtype = raw.get("dtype", values.get("torch_dtype"))
        if serialized_dtype in {"bfloat16", "bf16"}:
            values["torch_dtype"] = torch.bfloat16
        values["neuron_config"] = neuron_config
        return cls(**values)
