# SPDX-License-Identifier: Apache-2.0
"""Explicit, inference-only configuration for GLM-5.3-Flash."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

DSA_LAYER_INDICES = tuple(range(3, 45, 4))
KDA_LAYER_INDICES = tuple(i for i in range(45) if i not in DSA_LAYER_INDICES)
GLM53_LAYER_TYPES = tuple(
    "deepseek_sparse_attention" if i in DSA_LAYER_INDICES else "linear_attention"
    for i in range(45)
)
GLM53_MLP_LAYER_TYPES = tuple("dense" if i < 3 else "sparse" for i in range(45))
FP8_SCALE_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _torch_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(value).removeprefix("torch.").lower()
    if key not in aliases:
        raise ValueError(f"unsupported torch dtype: {value!r}")
    return aliases[key]


def validate_fp8_scale(scale: torch.Tensor | float, name: str) -> torch.Tensor:
    """Enforce the native-E4M3 load contract for every scale-like field."""
    if scale is None:
        raise ValueError(f"{name} must have a non-None FP8 scale")
    value = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale)
    if value.dtype not in FP8_SCALE_DTYPES:
        raise TypeError(
            f"{name} must use float32, float16, or bfloat16; got {value.dtype}"
        )
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite and non-empty")
    if torch.any(value <= 0):
        raise ValueError(f"{name} must be strictly positive")
    maximum = torch.max(value.to(torch.float32)).item()
    if maximum > 240.0:
        raise ValueError(f"{name} exceeds native E4M3 max 240.0: {maximum}")
    return value


@dataclass
class Glm53LinearAttentionConfig:
    num_heads: int = 64
    head_dim: int = 128
    short_conv_kernel_size: int = 4
    gate_lower_bound: float = -5.0
    l2norm_eps: float = 1e-6
    kda_layers: tuple[int, ...] = KDA_LAYER_INDICES
    full_attn_layers: tuple[int, ...] = DSA_LAYER_INDICES


@dataclass
class Glm53QuantizationConfig:
    quant_method: str = "fp8"
    activation_scheme: str = "dynamic"
    fmt: str = "e4m3"
    weight_block_size: tuple[int, int] = (128, 128)


@dataclass
class Glm53FlashInferenceConfig:
    """GLM-5.3-Flash text-backbone config with no inferred numerics."""

    model_type: str = "glm5_next_text"
    architectures: tuple[str, ...] = ("Glm5NextForConditionalGeneration",)
    vocab_size: int = 154880
    hidden_size: int = 4096
    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    head_dim: int = 0
    intermediate_size: int = 12288
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_head_dim: int = 256
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 1048576
    mla_use_nope: bool = True
    tie_word_embeddings: bool = False
    pad_token_id: int = 154820
    eos_token_id: tuple[int, ...] = (154820, 154827, 154829)
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 4
    index_kpool_always_select_tail: bool = True
    index_kpool_compress: bool = True
    indexer_rope_interleave: bool = True
    indexer_types: tuple[str, ...] = ("full",) * 45
    index_share_for_mtp_iteration: bool = True
    n_routed_experts: int = 288
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    moe_intermediate_size: int = 2048
    routed_scaling_factor: float = 2.5
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    moe_router_dtype: str = "float32"
    router_aux_loss_coef: float = 0.001
    output_router_logits: bool = False
    swiglu_limit: float = 10.0
    first_k_dense_replace: int = 3
    mhc: bool = True
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    hc_post_alpha: float = 2.0
    num_nextn_predict_layers: int = 1
    use_cache: bool = True
    use_mtp: bool = False
    allow_reduced_shapes: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    static_fp8: bool = True
    static_fp8_weight_format: str = "neuron_legacy_e4m3fn_qmax240"
    fp8_weight_scale_default: float = 1.0
    fp8_activation_scale_default: float = 1.0
    key_cache_quant_multiplier: float = 1.0
    value_cache_quant_multiplier: float = 1.0
    indexer_cache_quant_multiplier: float = 1.0
    layer_types: tuple[str, ...] = GLM53_LAYER_TYPES
    mlp_layer_types: tuple[str, ...] = GLM53_MLP_LAYER_TYPES
    linear_attn_config: Glm53LinearAttentionConfig = field(
        default_factory=Glm53LinearAttentionConfig
    )
    quantization_config: Glm53QuantizationConfig = field(
        default_factory=Glm53QuantizationConfig
    )

    def __post_init__(self) -> None:
        self.torch_dtype = _torch_dtype(self.torch_dtype)
        if isinstance(self.linear_attn_config, Mapping):
            self.linear_attn_config = Glm53LinearAttentionConfig(
                **_known_values(Glm53LinearAttentionConfig, self.linear_attn_config)
            )
        if isinstance(self.quantization_config, Mapping):
            quant = dict(self.quantization_config)
            if "format" in quant and "fmt" not in quant:
                quant["fmt"] = quant["format"]
            self.quantization_config = Glm53QuantizationConfig(
                **_known_values(Glm53QuantizationConfig, quant)
            )
        self.quantization_config.weight_block_size = tuple(
            self.quantization_config.weight_block_size
        )
        self.architectures = tuple(self.architectures)
        self.eos_token_id = tuple(self.eos_token_id)
        self.indexer_types = tuple(self.indexer_types)
        self.layer_types = tuple(self.layer_types)
        self.mlp_layer_types = tuple(self.mlp_layer_types)
        self.linear_attn_config.kda_layers = tuple(self.linear_attn_config.kda_layers)
        self.linear_attn_config.full_attn_layers = tuple(
            self.linear_attn_config.full_attn_layers
        )
        self._validate_architecture()
        for name in (
            "fp8_weight_scale_default",
            "fp8_activation_scale_default",
            "key_cache_quant_multiplier",
            "value_cache_quant_multiplier",
            "indexer_cache_quant_multiplier",
        ):
            validate_fp8_scale(float(getattr(self, name)), name)

    def _validate_architecture(self) -> None:
        if self.num_hidden_layers != 45:
            raise ValueError("GLM-5.3-Flash requires exactly 45 text layers")
        if self.qk_head_dim != 256 or self.qk_nope_head_dim != 256:
            raise ValueError("GLM-5.3-Flash requires qk/nope head dim 256")
        if self.qk_rope_head_dim != 0:
            raise ValueError("GLM-5.3-Flash MLA is All-NoPE (rope head dim 0)")
        if self.layer_types != GLM53_LAYER_TYPES:
            raise ValueError("layer_types must encode DSA at [3,7,...,43]")
        if self.linear_attn_config.kda_layers != KDA_LAYER_INDICES:
            raise ValueError("linear_attn_config.kda_layers disagrees with layer_types")
        if self.linear_attn_config.full_attn_layers != DSA_LAYER_INDICES:
            raise ValueError(
                "linear_attn_config.full_attn_layers disagrees with layer_types"
            )
        if self.mlp_layer_types != GLM53_MLP_LAYER_TYPES:
            raise ValueError("layers 0-2 must be dense and layers 3-44 MoE")
        if self.index_kpool != 4 or not self.index_kpool_always_select_tail:
            raise ValueError("GLM-5.3-Flash requires IndexPool=4 and tail selection")
        if not self.index_kpool_compress:
            raise ValueError("GLM-5.3-Flash requires IndexPool compression")
        if not self.mla_use_nope:
            raise ValueError("GLM-5.3-Flash requires mla_use_nope=True")
        if not self.mhc or self.hc_mult != 4 or self.hc_sinkhorn_iters != 20:
            raise ValueError("GLM-5.3-Flash requires mHC x4 with 20 Sinkhorn steps")
        if not self.allow_reduced_shapes:
            frozen = {
                "vocab_size": 154880,
                "hidden_size": 4096,
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "intermediate_size": 12288,
                "q_lora_rank": 1536,
                "kv_lora_rank": 512,
                "v_head_dim": 256,
                "n_routed_experts": 288,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 2048,
            }
            mismatches = [
                f"{name}={getattr(self, name)!r}"
                for name, expected in frozen.items()
                if getattr(self, name) != expected
            ]
            if mismatches:
                raise ValueError(
                    "GLM-5.3-Flash frozen architecture mismatch: "
                    + ", ".join(mismatches)
                )
        if self.use_mtp:
            raise ValueError("MTP/speculative decode is outside this source port")

    @classmethod
    def from_configs(
        cls, config: Any, neuron_config: Any = None
    ) -> Glm53FlashInferenceConfig:
        del neuron_config
        outer = _as_mapping(config)
        raw = outer
        if "text_config" in outer:
            text = outer["text_config"]
            if not isinstance(text, Mapping):
                text = vars(text)
            raw = dict(text)
        aliases = {
            "num_experts": "n_routed_experts",
            "num_shared_experts": "n_shared_experts",
            "num_experts_per_token": "num_experts_per_tok",
            "linear_attention_config": "linear_attn_config",
            "dtype": "torch_dtype",
        }
        normalized = dict(raw)
        if "architectures" in outer:
            normalized["architectures"] = outer["architectures"]
        if "quantization_config" in outer:
            normalized["quantization_config"] = outer["quantization_config"]
        for source, target in aliases.items():
            if source in normalized and target not in normalized:
                normalized[target] = normalized[source]
        return cls(**_known_values(cls, normalized))

    @classmethod
    def from_pretrained(
        cls, model_id_or_path: str | Path, **kwargs: Any
    ) -> Glm53FlashInferenceConfig:
        path = Path(model_id_or_path)
        if path.is_dir():
            path = path / "config.json"
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            from transformers import PretrainedConfig

            raw, _ = PretrainedConfig.get_config_dict(str(model_id_or_path), **kwargs)
        return cls.from_configs(raw)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["torch_dtype"] = str(self.torch_dtype).removeprefix("torch.")
        return result


def _known_values(cls: type, values: Mapping[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(cls)}
    return {
        key: value
        for key, value in values.items()
        if key in names and value is not None
    }


def _as_mapping(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    return dict(vars(config))


__all__ = [
    "DSA_LAYER_INDICES",
    "GLM53_LAYER_TYPES",
    "KDA_LAYER_INDICES",
    "Glm53FlashInferenceConfig",
    "Glm53LinearAttentionConfig",
    "Glm53QuantizationConfig",
    "validate_fp8_scale",
]
