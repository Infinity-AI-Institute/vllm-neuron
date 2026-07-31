"""Configuration boundary for the native Inkling-Small Neuron model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class InklingConfig:
    """Text-backbone parameters consumed by the Neuron implementation.

    The public checkpoint is a multimodal ``InklingConfig`` whose language
    model lives below ``text_config``.  Keeping a small native dataclass makes
    the compiler-facing contract explicit and independent of Transformers'
    release cadence.
    """

    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    hidden_size: int = 4096
    num_hidden_layers: int = 42
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    local_layer_ids: list[int] = field(default_factory=list)
    sliding_window_size: int = 512
    swa_num_attention_heads: int = 32
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 128
    rms_norm_eps: float = 1e-6
    use_embed_norm: bool = True
    use_sconv: bool = True
    sconv_kernel_size: int = 4
    dense_mlp_idx: int = 2
    dense_intermediate_size: int = 16384
    intermediate_size: int = 2048
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    n_shared_experts: int = 2
    shared_expert_sink: bool = True
    route_scale: float = 8.0
    use_gate_bias: bool = True
    gate_activation: str = "sigmoid"
    norm_after_topk: bool = True
    use_global_scale: bool = True
    log_scaling_n_floor: int | None = 128000
    log_scaling_alpha: float = 0.1
    logits_mup_width_multiplier: float | None = 16.0
    final_logit_softcapping: float | None = None
    max_position_embeddings: int = 1048576
    eos_token_id: int = 200006
    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    @property
    def padded_vocab_size(self) -> int:
        return self.vocab_size

    @property
    def conv_cache_width(self) -> int:
        """Elements stored per token for K/V/attention/MLP conv streams."""
        return 2 * self.head_dim + 2 * self.hidden_size

    @property
    def conv_cache_heads(self) -> int:
        if self.conv_cache_width % self.head_dim:
            raise ValueError(
                "Inkling conv streams must pack into head-sized cache rows"
            )
        return self.conv_cache_width // self.head_dim

    def layer_is_local(self, layer_idx: int) -> bool:
        return layer_idx in self.local_layer_ids

    def layer_is_dense(self, layer_idx: int) -> bool:
        return layer_idx < self.dense_mlp_idx

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict,
        neuron_config: NeuronConfig | None = None,
    ) -> InklingConfig:
        if isinstance(hf_config, cls):
            hf_config.neuron_config = neuron_config
            return hf_config
        raw = cls._raw_checkpoint_config(hf_config)
        outer = raw
        raw = raw.get("text_config", raw)
        fields = cls.__dataclass_fields__
        values = {key: value for key, value in raw.items() if key in fields}
        values["max_position_embeddings"] = int(
            raw.get(
                "model_max_length",
                raw.get("max_position_embeddings", cls.max_position_embeddings),
            )
        )
        values["eos_token_id"] = int(
            outer.get("eos_token_id", raw.get("eos_token_id", cls.eos_token_id))
        )
        serialized_dtype = raw.get("dtype", raw.get("torch_dtype"))
        if serialized_dtype in {"bfloat16", "bf16"}:
            values["torch_dtype"] = torch.bfloat16
        values["neuron_config"] = neuron_config
        return cls(**values)

    @staticmethod
    def _raw_checkpoint_config(
        hf_config: PretrainedConfig | dict,
    ) -> dict:
        """Return the checkpoint schema before Transformers normalizes it.

        Transformers' current ``InklingTextConfig`` translates the published
        checkpoint's ``dense_intermediate_size`` to ``intermediate_size`` and
        supplies a default ``moe_intermediate_size``. That loses the released
        sparse-expert width (2048) and would reshape real expert weights as if
        they used the dense width (16384). vLLM retains the checkpoint path in
        ``name_or_path``, so prefer its original config.json when available.
        """

        if isinstance(hf_config, dict):
            return dict(hf_config)
        source = getattr(hf_config, "name_or_path", None) or getattr(
            hf_config, "_name_or_path", None
        )
        if source:
            source_path = Path(source)
            config_path = (
                source_path / "config.json" if source_path.is_dir() else source_path
            )
            if config_path.is_file():
                with config_path.open(encoding="utf-8") as config_file:
                    return json.load(config_file)
        return hf_config.to_dict()
