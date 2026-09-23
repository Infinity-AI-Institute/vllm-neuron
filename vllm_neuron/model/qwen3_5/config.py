# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 Config
==============

<-- MODEL-SPECIFIC: Qwen3.5 text decoder (the vision tower and MTP head are
not served here). Defaults are Qwen/Qwen3.5-9B at revision
c202236235762e1c871ad0ccb60c8ee5ba337b9a:

  32 layers: 24 Gated DeltaNet + 8 full attention (every 4th, "L L L A").
  Full attention: 16 Q / 4 KV heads, head_dim 256, rotary over 64 of 256
    channels (partial_rotary_factor 0.25), rope_theta 1e7, gated output.
  Gated DeltaNet: 16 key / 32 value heads, head dims 128, conv kernel 4.
  Untied lm_head [248320, 4096].

The checkpoint's architecture is ``Qwen3_5ForConditionalGeneration``; its
text fields live under ``text_config``.
"""

import json
from dataclasses import dataclass, field

import torch


@dataclass
class Qwen3_5Config:
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    layer_types: list[str] = field(default_factory=list)
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config (vllm_neuron.model.neuron_config.NeuronConfig)
    neuron_config: object | None = None

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = [
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must name every layer")
        if self.hidden_act != "silu":
            raise ValueError(f"Gated DeltaNet conv assumes SiLU, got {self.hidden_act}")

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def full_attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def linear_attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]

    @property
    def conv_dim(self) -> int:
        key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        return 2 * key_dim + self.linear_num_value_heads * self.linear_value_head_dim

    @classmethod
    def from_configs(cls, hf_config, neuron_config=None):
        """Build from an HF config (object, dict or JSON path)."""
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                raw = json.load(f)
        elif isinstance(hf_config, dict):
            raw = dict(hf_config)
        else:
            raw = hf_config.to_dict()
        text = raw.get("text_config", raw)
        if not isinstance(text, dict):
            text = text.to_dict()

        rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
        values = {
            "rope_theta": rope.get(
                "rope_theta", text.get("rope_theta", cls.rope_theta)
            ),
            "partial_rotary_factor": rope.get(
                "partial_rotary_factor",
                text.get("partial_rotary_factor", cls.partial_rotary_factor),
            ),
        }
        names = (
            {f for f in cls.__dataclass_fields__}
            - {"neuron_config", "torch_dtype"}
            - set(values)
        )
        values.update({k: text[k] for k in names if k in text})
        if "tie_word_embeddings" in raw:
            values["tie_word_embeddings"] = raw["tie_word_embeddings"]

        dtype = text.get("torch_dtype", raw.get("torch_dtype", "bfloat16"))
        values["torch_dtype"] = (
            getattr(torch, dtype) if isinstance(dtype, str) else dtype
        )
        values["neuron_config"] = neuron_config
        return cls(**values)
