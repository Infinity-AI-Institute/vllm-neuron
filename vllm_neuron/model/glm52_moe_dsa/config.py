# SPDX-License-Identifier: Apache-2.0
"""Configuration adapter for the frozen GLM-5.2 architecture."""

import json
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .static_fp8 import (
    OCP_E4M3FN_QMAX448,
    normalize_static_fp8_weight_format,
)


def _indexer_schedule(
    num_layers: int,
    skip_topk_offset: int,
    topk_frequency: int,
) -> tuple[str, ...]:
    """Build GLM's full/IndexShare layer schedule."""
    if skip_topk_offset < 0:
        raise ValueError("index_skip_topk_offset must be non-negative")
    if topk_frequency <= 0:
        raise ValueError("index_topk_freq must be positive")

    return tuple(
        "full"
        if layer_idx < skip_topk_offset
        or (layer_idx - skip_topk_offset + 1) % topk_frequency == 0
        else "shared"
        for layer_idx in range(num_layers)
    )


def _mlp_schedule(
    num_layers: int,
    first_k_dense_replace: int,
) -> tuple[str, ...]:
    if not 0 <= first_k_dense_replace <= num_layers:
        raise ValueError("first_k_dense_replace must be within the layer range")
    return tuple(
        "dense" if layer_idx < first_k_dense_replace else "sparse"
        for layer_idx in range(num_layers)
    )


@dataclass
class Glm52MoeDsaConfig:
    """Neuron-side representation of ``zai-org/GLM-5.2``.

    Defaults match revision ``b4734de4facf877f85769a911abafc5283eab3d9``.
    Reduced-shape values remain supported for CPU parity and kernel tests.
    """

    vocab_size: int = 154_880
    hidden_size: int = 6_144
    num_hidden_layers: int = 78
    intermediate_size: int = 12_288
    rms_norm_eps: float = 1e-5
    torch_dtype: torch.dtype = torch.bfloat16

    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    q_lora_rank: int = 2_048
    kv_lora_rank: int = 512
    qk_head_dim: int = 256
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    attention_bias: bool = False
    attention_dropout: float = 0.0

    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2_048
    index_skip_topk_offset: int = 3
    index_topk_freq: int = 4
    indexer_rope_interleave: bool = True
    indexer_types: tuple[str, ...] = field(default_factory=tuple)

    n_routed_experts: int = 256
    n_shared_experts: int = 1
    shared_expert_dtype: str = "fp8"
    static_fp8_weight_format: str = OCP_E4M3FN_QMAX448
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 2_048
    n_group: int = 1
    topk_group: int = 1
    first_k_dense_replace: int = 3
    moe_layer_freq: int = 1
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    moe_router_dtype: str = "float32"
    mlp_layer_types: tuple[str, ...] = field(default_factory=tuple)

    max_position_embeddings: int = 1_048_576
    rope_interleave: bool = True
    rope_parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "rope_theta": 8_000_000,
            "rope_type": "default",
        }
    )
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 1
    pad_token_id: int = 154_820
    eos_token_id: int | tuple[int, ...] = (154_820, 154_827, 154_829)

    neuron_config: NeuronConfig | None = None

    def __post_init__(self) -> None:
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.qk_head_dim != self.qk_nope_head_dim + self.qk_rope_head_dim:
            raise ValueError(
                "qk_head_dim must equal qk_nope_head_dim + qk_rope_head_dim"
            )
        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError(
                "GLM-5.2 currently requires equal query and KV head counts"
            )
        if self.n_routed_experts % self.num_experts_per_tok:
            raise ValueError("routed experts must be divisible by experts per token")
        if self.n_shared_experts != 1:
            raise ValueError(
                "the initial GLM-5.2 port requires exactly one shared expert"
            )
        if self.shared_expert_dtype not in ("fp8", "bfloat16"):
            raise ValueError("shared_expert_dtype must be either 'fp8' or 'bfloat16'")
        self.static_fp8_weight_format = normalize_static_fp8_weight_format(
            self.static_fp8_weight_format
        )
        if self.scoring_func != "sigmoid" or self.topk_method != "noaux_tc":
            raise ValueError("GLM-5.2 requires sigmoid/noaux_tc expert routing")
        if not self.norm_topk_prob:
            raise ValueError("GLM-5.2 requires normalized top-k routing weights")
        if self.moe_router_dtype != "float32":
            raise ValueError("GLM-5.2 router scores must be computed in FP32")
        if not self.rope_interleave or not self.indexer_rope_interleave:
            raise ValueError("GLM-5.2 requires interleaved main and indexer RoPE")
        if self.n_group != 1 or self.topk_group != 1:
            raise ValueError(
                "the initial GLM-5.2 port supports the frozen single router group"
            )

        if not self.indexer_types:
            self.indexer_types = _indexer_schedule(
                self.num_hidden_layers,
                self.index_skip_topk_offset,
                self.index_topk_freq,
            )
        else:
            self.indexer_types = tuple(self.indexer_types)
        if len(self.indexer_types) != self.num_hidden_layers:
            raise ValueError("indexer_types must contain one entry per backbone layer")
        if set(self.indexer_types) - {"full", "shared"}:
            raise ValueError("indexer_types entries must be 'full' or 'shared'")
        if self.indexer_types[0] != "full":
            raise ValueError("layer 0 must own a full indexer")

        if not self.mlp_layer_types:
            self.mlp_layer_types = _mlp_schedule(
                self.num_hidden_layers,
                self.first_k_dense_replace,
            )
        else:
            self.mlp_layer_types = tuple(self.mlp_layer_types)
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError(
                "mlp_layer_types must contain one entry per backbone layer"
            )
        expected_mlp_types = _mlp_schedule(
            self.num_hidden_layers,
            self.first_k_dense_replace,
        )
        if self.mlp_layer_types != expected_mlp_types:
            raise ValueError("mlp_layer_types does not match first_k_dense_replace")

    @property
    def full_indexer_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            layer_idx
            for layer_idx, indexer_type in enumerate(self.indexer_types)
            if indexer_type == "full"
        )

    @property
    def shared_indexer_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            layer_idx
            for layer_idx, indexer_type in enumerate(self.indexer_types)
            if indexer_type == "shared"
        )

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict[str, Any] | str,
        neuron_config: NeuronConfig | None,
    ) -> "Glm52MoeDsaConfig":
        if isinstance(hf_config, str):
            with open(hf_config, encoding="utf-8") as config_file:
                config_dict = json.load(config_file)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
        else:
            config_dict = dict(hf_config)

        architectures = config_dict.get("architectures", ())
        if architectures and "GlmMoeDsaForCausalLM" not in architectures:
            raise ValueError(
                "Expected checkpoint architecture GlmMoeDsaForCausalLM, "
                f"got {architectures!r}"
            )

        field_names = set(cls.__dataclass_fields__)
        filtered = {
            key: value for key, value in config_dict.items() if key in field_names
        }
        dtype = config_dict.get(
            "dtype",
            config_dict.get("torch_dtype", filtered.get("torch_dtype", "bfloat16")),
        )
        if isinstance(dtype, str):
            try:
                dtype = getattr(torch, dtype)
            except AttributeError as error:
                raise ValueError(f"Unsupported checkpoint dtype: {dtype}") from error
        filtered["torch_dtype"] = dtype
        filtered["neuron_config"] = neuron_config
        return cls(**filtered)
