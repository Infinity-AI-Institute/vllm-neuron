# SPDX-License-Identifier: Apache-2.0
"""Logical GLM-5.2 backbone tensor manifest and TP64 accounting."""

from collections.abc import Iterator
from dataclasses import dataclass
from math import prod
from typing import Literal

from .config import Glm52MoeDsaConfig

Placement = Literal["replicated", "tensor_parallel", "expert_parallel"]


@dataclass(frozen=True)
class WeightSpec:
    name: str
    shape: tuple[int, ...]
    placement: Placement

    @property
    def numel(self) -> int:
        return prod(self.shape)

    def local_numel(self, world_size: int) -> int:
        if self.placement == "replicated":
            return self.numel
        if self.numel % world_size:
            raise ValueError(
                f"{self.name} with {self.numel} elements is not divisible by "
                f"world_size={world_size}"
            )
        return self.numel // world_size


def iter_backbone_weight_specs(
    config: Glm52MoeDsaConfig,
) -> Iterator[WeightSpec]:
    """Yield the 78-layer MTP-off checkpoint contract.

    The frozen checkpoint also contains ``model.layers.78.*`` MTP tensors.
    They are deliberately outside this iterator because the first serving
    milestone disables MTP and must reject or explicitly ignore that prefix.
    """
    hidden = config.hidden_size
    yield WeightSpec(
        "model.embed_tokens.weight",
        (config.vocab_size, hidden),
        "tensor_parallel",
    )
    yield WeightSpec("model.norm.weight", (hidden,), "replicated")
    yield WeightSpec(
        "lm_head.weight",
        (config.vocab_size, hidden),
        "tensor_parallel",
    )

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        attention = f"{prefix}.self_attn"
        yield WeightSpec(
            f"{prefix}.input_layernorm.weight",
            (hidden,),
            "replicated",
        )
        yield WeightSpec(
            f"{prefix}.post_attention_layernorm.weight",
            (hidden,),
            "replicated",
        )
        yield WeightSpec(
            f"{attention}.q_a_proj.weight",
            (config.q_lora_rank, hidden),
            "replicated",
        )
        yield WeightSpec(
            f"{attention}.q_a_layernorm.weight",
            (config.q_lora_rank,),
            "replicated",
        )
        yield WeightSpec(
            f"{attention}.q_b_proj.weight",
            (config.num_attention_heads * config.qk_head_dim, config.q_lora_rank),
            "tensor_parallel",
        )
        yield WeightSpec(
            f"{attention}.kv_a_proj_with_mqa.weight",
            (config.kv_lora_rank + config.qk_rope_head_dim, hidden),
            "replicated",
        )
        yield WeightSpec(
            f"{attention}.kv_a_layernorm.weight",
            (config.kv_lora_rank,),
            "replicated",
        )
        yield WeightSpec(
            f"{attention}.kv_b_proj.weight",
            (
                config.num_attention_heads
                * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            ),
            "tensor_parallel",
        )
        yield WeightSpec(
            f"{attention}.o_proj.weight",
            (hidden, config.num_attention_heads * config.v_head_dim),
            "tensor_parallel",
        )

        if config.indexer_types[layer_idx] == "full":
            indexer = f"{attention}.indexer"
            yield WeightSpec(
                f"{indexer}.wq_b.weight",
                (
                    config.index_n_heads * config.index_head_dim,
                    config.q_lora_rank,
                ),
                "replicated",
            )
            yield WeightSpec(
                f"{indexer}.wk.weight",
                (config.index_head_dim, hidden),
                "replicated",
            )
            yield WeightSpec(
                f"{indexer}.k_norm.weight",
                (config.index_head_dim,),
                "replicated",
            )
            yield WeightSpec(
                f"{indexer}.k_norm.bias",
                (config.index_head_dim,),
                "replicated",
            )
            yield WeightSpec(
                f"{indexer}.weights_proj.weight",
                (config.index_n_heads, hidden),
                "replicated",
            )

        mlp = f"{prefix}.mlp"
        if config.mlp_layer_types[layer_idx] == "dense":
            yield WeightSpec(
                f"{mlp}.gate_proj.weight",
                (config.intermediate_size, hidden),
                "tensor_parallel",
            )
            yield WeightSpec(
                f"{mlp}.up_proj.weight",
                (config.intermediate_size, hidden),
                "tensor_parallel",
            )
            yield WeightSpec(
                f"{mlp}.down_proj.weight",
                (hidden, config.intermediate_size),
                "tensor_parallel",
            )
            continue

        yield WeightSpec(
            f"{mlp}.gate.weight",
            (config.n_routed_experts, hidden),
            "replicated",
        )
        yield WeightSpec(
            f"{mlp}.gate.e_score_correction_bias",
            (config.n_routed_experts,),
            "replicated",
        )
        yield WeightSpec(
            f"{mlp}.experts.gate_up_proj",
            (
                config.n_routed_experts,
                2 * config.moe_intermediate_size,
                hidden,
            ),
            "expert_parallel",
        )
        yield WeightSpec(
            f"{mlp}.experts.down_proj",
            (
                config.n_routed_experts,
                hidden,
                config.moe_intermediate_size,
            ),
            "expert_parallel",
        )
        shared_intermediate = (
            config.moe_intermediate_size * config.n_shared_experts
        )
        yield WeightSpec(
            f"{mlp}.shared_experts.gate_proj.weight",
            (shared_intermediate, hidden),
            "tensor_parallel",
        )
        yield WeightSpec(
            f"{mlp}.shared_experts.up_proj.weight",
            (shared_intermediate, hidden),
            "tensor_parallel",
        )
        yield WeightSpec(
            f"{mlp}.shared_experts.down_proj.weight",
            (hidden, shared_intermediate),
            "tensor_parallel",
        )


def estimate_local_weight_bytes(
    config: Glm52MoeDsaConfig,
    *,
    world_size: int,
    weight_bytes_per_element: int,
) -> int:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if weight_bytes_per_element <= 0:
        raise ValueError("weight bytes per element must be positive")
    return (
        sum(
            spec.local_numel(world_size)
            for spec in iter_backbone_weight_specs(config)
        )
        * weight_bytes_per_element
    )
