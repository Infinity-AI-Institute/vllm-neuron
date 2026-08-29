# SPDX-License-Identifier: Apache-2.0
"""Explicit, inference-only configuration for DeepSeek-V4-Flash.

Frozen from HF ``deepseek-ai/DeepSeek-V4-Flash-0731`` head SHA
``7872f01b1d1fe23eabc4c98b48bffcef5a386062``.  Every field below is either
copied verbatim from that snapshot's ``config.json`` or derived from the
transformers 5.15.1 ``deepseek_v4`` module docstrings — nothing is guessed.
Do NOT let a caller silently mutate architecture constants: the
``__post_init__`` validator refuses any deviation from the frozen set
(unless ``allow_reduced_shapes`` is set for a 1-layer compile smoke, mirror
of ``glm53_flash/config.py``).

Layer-schedule discipline: HF ships a ``compress_ratios`` array of length
``num_hidden_layers + num_nextn_predict_layers + 2`` (46 for V4-Flash).
This module lowers that to ``layer_types`` with values
{"sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"}
of length ``num_hidden_layers`` (43), dropping the leading and trailing
sliding-pad plus the MTP tail.  The mapping is exact and reversible.

FP4-UE8M0 discipline: unlike GLM-5.3-Flash (which reads reciprocal E4M3
block scales), V4-Flash's routed experts are FP4 with UE8M0 (unsigned
E8M0) block scales — the scale is an 8-bit exponent, not a float.
Dequant is ``w_bf16 = (w_fp32 * 2**exp).to(bf16)``.  The non-expert
weights are FP8 e4m3 with the same UE8M0 scale format — so this port
CANNOT reuse GLM-5.3-Flash's ``dequantize_block_fp8`` verbatim.
See :func:`validate_ue8m0_scale`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

# Frozen HF pin — refuse to convert any state_dict whose commit disagrees.
HF_REPO_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
HF_SNAPSHOT_SHA = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"

# Layer-type schedule derived from the checkpoint's ``compress_ratios``.
#   0   -> "sliding_attention" (window=128 only, no compressor)
#   4   -> "compressed_sparse_attention" (CSA: pool m=4 + Lightning Indexer top-k=512)
#   128 -> "heavily_compressed_attention" (HCA: pool m'=128, no indexer)
_COMPRESS_RATIOS_HF = (
    0,
    0,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    0,
    0,
    0,
)
# 43 hidden layers = 46-total - 2 leading sliding-pad - 3-of-3 trailing (of which 1 is MTP,
# the other 2 are sliding tail on the last hidden slots).  Preserves the HF order verbatim.
_DSV4_LAYER_TYPES = tuple(
    "sliding_attention"
    if ratio == 0
    else (
        "compressed_sparse_attention" if ratio == 4 else "heavily_compressed_attention"
    )
    for ratio in _COMPRESS_RATIOS_HF[:43]
)
# MLP layers: first 3 are hash_moe (bootstrap), all others are moe (top-k routed).
_DSV4_MLP_LAYER_TYPES = tuple("hash_moe" if idx < 3 else "moe" for idx in range(43))

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


def validate_ue8m0_scale(scale: torch.Tensor | int, name: str) -> torch.Tensor:
    """Enforce the MXFP4/FP8 UE8M0 block-scale contract.

    UE8M0 (a.k.a. ``float8_e8m0fnu``) is an unsigned 8-bit exponent.  Raw
    code X decodes to the multiplier ``2**(X - 127)`` when reinterpreted
    through the ``float8_e8m0fnu`` dtype (verified 2026-08-28 against the
    HF snapshot at ``deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b``: e.g.
    a routed-expert scale block with raw byte 120 casts to ``2**-7 ==
    0.0078125`` via ``.to(torch.float32)``).  Code 255 is reserved for
    NaN in the ``fnu`` variant and MUST be refused: it means the encoder
    could not represent that block.

    The dequantization multiplier is ``2**exp``, NEVER a raw-float
    multiplication.  A caller that reads the scale field as an fp32/fp16
    tensor and multiplies is silently off by an enormous factor.  This
    validator refuses:

      * a ``float8_e8m0fnu`` tensor that carries any NaN (raw byte 255);
      * any other floating-point dtype (fp16/bf16/fp32/e4m3/…) — those
        are never valid UE8M0 storage;
      * an integer tensor whose values are outside [0, 255].

    ``float8_e8m0fnu`` is treated as the canonical storage; integer
    dtypes (uint8/int8/int32/int64) are accepted only for legacy
    hand-crafted synthetic tensors and are interpreted with the *same*
    E8M0 bias convention (X → 2**(X-127)) — see
    :func:`ue8m0_scale_to_fp32_multiplier`.
    """
    if scale is None:
        raise ValueError(f"{name} must have a non-None UE8M0 scale tensor")
    value = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale)
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if value.dtype == torch.float8_e8m0fnu:
        # Only NaN codes (raw byte 255) are pathological; every other
        # code maps to a finite positive multiplier.  Cast to fp32 for the
        # NaN check — the raw-byte view would spuriously pass.
        as_fp32 = value.to(torch.float32)
        if torch.isnan(as_fp32).any():
            n_nan = int(torch.isnan(as_fp32).sum().item())
            raise ValueError(
                f"{name} carries {n_nan} NaN entries (raw byte 255) in a "
                "float8_e8m0fnu tensor; the encoder failed on those blocks "
                "and we refuse to broadcast NaN into every element they "
                "cover."
            )
        return value
    if value.dtype.is_floating_point:
        raise TypeError(
            f"{name} must be either a float8_e8m0fnu tensor (raw UE8M0 "
            f"storage) or an integer tensor holding UE8M0 exponent codes; "
            f"got dtype {value.dtype}"
        )
    ivalue = value.to(torch.int64)
    if int(ivalue.min().item()) < 0 or int(ivalue.max().item()) > 255:
        raise ValueError(
            f"{name} must hold UE8M0 exponents in [0, 255]; "
            f"got min={int(ivalue.min())}, max={int(ivalue.max())}"
        )
    return value


def ue8m0_scale_to_fp32_multiplier(scale: torch.Tensor) -> torch.Tensor:
    """Decode a UE8M0 block-scale tensor to its fp32 multiplier form.

    * ``float8_e8m0fnu`` storage: cast via ``.to(torch.float32)`` — PyTorch
      already applies the E8M0 bias (raw byte X → ``2**(X - 127)``) and
      converts NaN codes to fp32 NaN.  We rely on
      :func:`validate_ue8m0_scale` to have refused any NaN before we get
      here.
    * integer storage (uint8/int8/int32/int64): the caller writes the raw
      E8M0 code and expects the SAME bias — ``X → 2**(X - 127)``.
      Implemented via ``torch.ldexp(ones, X - 127)`` for exactness (a
      single integer shift of the fp32 exponent field, no rounding).
    """
    if scale.dtype == torch.float8_e8m0fnu:
        return scale.to(torch.float32)
    if scale.dtype.is_floating_point:
        raise TypeError(
            "ue8m0_scale_to_fp32_multiplier: refusing dtype "
            f"{scale.dtype} — only float8_e8m0fnu or integer dtypes carry "
            "raw E8M0 codes"
        )
    exp_biased = scale.to(torch.int32)
    ones = torch.ones_like(exp_biased, dtype=torch.float32)
    return torch.ldexp(ones, exp_biased - 127)


@dataclass
class DeepseekV4RopeScalingConfig:
    """YaRN scaling for the main-attention RoPE."""

    type: str = "yarn"
    factor: int = 16
    original_max_position_embeddings: int = 65536
    beta_fast: int = 32
    beta_slow: int = 1


@dataclass
class DeepseekV4QuantizationConfig:
    """FP8 e4m3 non-expert quantization descriptor.

    Routed experts use a separate FP4 encoding declared via
    ``expert_dtype='fp4'`` at the top-level config.  Both share the
    ``ue8m0`` block scale format.
    """

    quant_method: str = "fp8"
    activation_scheme: str = "dynamic"
    fmt: str = "e4m3"
    scale_fmt: str = "ue8m0"
    weight_block_size: tuple[int, int] = (128, 128)


@dataclass
class DeepseekV4FlashInferenceConfig:
    """DeepSeek-V4-Flash text-backbone config with no inferred numerics."""

    model_type: str = "deepseek_v4"
    architectures: tuple[str, ...] = ("DeepseekV4ForCausalLM",)
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1  # MQA / shared K=V
    head_dim: int = 512
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    partial_rotary_factor: float = 64 / 512  # derived; kept explicit for clarity
    o_groups: int = 8
    o_lora_rank: int = 1024
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    n_shared_experts: int = 1
    moe_intermediate_size: int = 2048
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    num_hash_layers: int = 3
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int | None = None
    num_nextn_predict_layers: int = (
        1  # MTP surface — DROPPED in wrapper (no spec-decode)
    )
    expert_dtype: str = "fp4"  # routed experts stored at FP4; non-experts at FP8
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    use_cache: bool = True
    allow_reduced_shapes: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    layer_types: tuple[str, ...] = _DSV4_LAYER_TYPES
    mlp_layer_types: tuple[str, ...] = _DSV4_MLP_LAYER_TYPES
    compress_ratios: tuple[int, ...] = _COMPRESS_RATIOS_HF
    rope_scaling: DeepseekV4RopeScalingConfig = field(
        default_factory=DeepseekV4RopeScalingConfig
    )
    quantization_config: DeepseekV4QuantizationConfig = field(
        default_factory=DeepseekV4QuantizationConfig
    )
    # DSpark fields are explicitly ignored (spec-decode is forbidden campaign-wide).
    hf_snapshot_sha: str = HF_SNAPSHOT_SHA

    def __post_init__(self) -> None:
        self.torch_dtype = _torch_dtype(self.torch_dtype)
        if isinstance(self.rope_scaling, Mapping):
            self.rope_scaling = DeepseekV4RopeScalingConfig(
                **_known_values(DeepseekV4RopeScalingConfig, self.rope_scaling)
            )
        if isinstance(self.quantization_config, Mapping):
            quant = dict(self.quantization_config)
            if "format" in quant and "fmt" not in quant:
                quant["fmt"] = quant["format"]
            self.quantization_config = DeepseekV4QuantizationConfig(
                **_known_values(DeepseekV4QuantizationConfig, quant)
            )
        self.quantization_config.weight_block_size = tuple(
            self.quantization_config.weight_block_size
        )
        self.architectures = tuple(self.architectures)
        self.layer_types = tuple(self.layer_types)
        self.mlp_layer_types = tuple(self.mlp_layer_types)
        self.compress_ratios = tuple(self.compress_ratios)
        self._validate_architecture()

    def _validate_architecture(self) -> None:
        if not self.allow_reduced_shapes:
            if self.num_hidden_layers != 43:
                raise ValueError("DeepSeek-V4-Flash requires exactly 43 hidden layers")
            if self.layer_types != _DSV4_LAYER_TYPES:
                raise ValueError(
                    "layer_types must match the frozen CSA/HCA/sliding schedule"
                )
            if self.mlp_layer_types != _DSV4_MLP_LAYER_TYPES:
                raise ValueError(
                    "mlp_layer_types must match 3-hash-MoE bootstrap + 40 routed MoE"
                )
        else:
            if len(self.layer_types) != self.num_hidden_layers:
                raise ValueError(
                    f"layer_types has {len(self.layer_types)} entries but "
                    f"num_hidden_layers={self.num_hidden_layers}"
                )
            if len(self.mlp_layer_types) != self.num_hidden_layers:
                raise ValueError(
                    f"mlp_layer_types has {len(self.mlp_layer_types)} entries "
                    f"but num_hidden_layers={self.num_hidden_layers}"
                )
        # Shape-independent invariants (hold in reduced-shape smoke too).
        if self.num_key_value_heads != 1:
            raise ValueError(
                "DeepSeek-V4-Flash uses shared K=V MQA (num_key_value_heads=1)"
            )
        if self.head_dim != 512:
            raise ValueError("DeepSeek-V4-Flash requires head_dim=512")
        if self.qk_rope_head_dim != 64:
            raise ValueError(
                "DeepSeek-V4-Flash requires partial RoPE with qk_rope_head_dim=64"
            )
        if self.o_groups != 8 or self.o_lora_rank != 1024:
            raise ValueError(
                "DeepSeek-V4-Flash grouped output projection requires "
                "o_groups=8 and o_lora_rank=1024"
            )
        if self.hc_mult != 4 or self.hc_sinkhorn_iters != 20:
            raise ValueError(
                "DeepSeek-V4-Flash requires mHC hc_mult=4 with 20 Sinkhorn iters"
            )
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError("DeepSeek-V4-Flash router uses sqrt(softplus(x)) scoring")
        if self.topk_method != "noaux_tc":
            raise ValueError(
                "DeepSeek-V4-Flash requires noaux_tc topk method (with "
                "e_score_correction_bias)"
            )
        if self.expert_dtype != "fp4":
            raise ValueError(
                "DeepSeek-V4-Flash routed experts are FP4; refusing to load "
                f"expert_dtype={self.expert_dtype!r}"
            )
        if self.quantization_config.scale_fmt != "ue8m0":
            raise ValueError("DeepSeek-V4-Flash quantization uses UE8M0 block scales")
        if not self.allow_reduced_shapes:
            frozen = {
                "vocab_size": 129280,
                "hidden_size": 4096,
                "num_attention_heads": 64,
                "num_key_value_heads": 1,
                "head_dim": 512,
                "q_lora_rank": 1024,
                "qk_rope_head_dim": 64,
                "o_groups": 8,
                "o_lora_rank": 1024,
                "n_routed_experts": 256,
                "num_experts_per_tok": 6,
                "moe_intermediate_size": 2048,
                "index_n_heads": 64,
                "index_head_dim": 128,
                "index_topk": 512,
                "sliding_window": 128,
            }
            mismatches = [
                f"{name}={getattr(self, name)!r}"
                for name, expected in frozen.items()
                if getattr(self, name) != expected
            ]
            if mismatches:
                raise ValueError(
                    "DeepSeek-V4-Flash frozen architecture mismatch: "
                    + ", ".join(mismatches)
                )

    @classmethod
    def from_configs(
        cls, config: Any, neuron_config: Any = None
    ) -> DeepseekV4FlashInferenceConfig:
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
            "dtype": "torch_dtype",
        }
        normalized = dict(raw)
        if "architectures" in outer:
            normalized["architectures"] = outer["architectures"]
        if "quantization_config" in outer:
            normalized["quantization_config"] = outer["quantization_config"]
        if "rope_scaling" in outer:
            normalized["rope_scaling"] = outer["rope_scaling"]
        for source, target in aliases.items():
            if source in normalized and target not in normalized:
                normalized[target] = normalized[source]
        # Preserve num_hash_layers alias if present as legacy key.
        return cls(**_known_values(cls, normalized))

    @classmethod
    def from_pretrained(
        cls, model_id_or_path: str | Path, **kwargs: Any
    ) -> DeepseekV4FlashInferenceConfig:
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
    "HF_REPO_ID",
    "HF_SNAPSHOT_SHA",
    "DeepseekV4FlashInferenceConfig",
    "DeepseekV4QuantizationConfig",
    "DeepseekV4RopeScalingConfig",
    "ue8m0_scale_to_fp32_multiplier",
    "validate_ue8m0_scale",
]
