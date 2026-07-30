# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 backbone parameter manifest and rank-local HBM accounting."""

from collections.abc import Iterator
from dataclasses import dataclass
from math import prod
from typing import Literal

from .config import Glm52MoeDsaConfig

Placement = Literal[
    "replicated",
    "world_sharded",
    "expert_sharded",
    "expert_tp_sharded",
]
StorageDtype = Literal["fp8", "bf16", "fp32"]

_DTYPE_BYTES: dict[StorageDtype, int] = {
    "fp8": 1,
    "bf16": 2,
    "fp32": 4,
}
_STATIC_SCALE_ROWS = 128


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("alignment must be positive")
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class WeightSpec:
    """One logical parameter and its rank-local storage policy.

    ``shape`` is the unsharded logical shape unless ``placement`` is
    ``replicated``. ``shard_dim`` and ``shard_alignment`` describe padding in
    the materialized rank-local parameter. This is needed for the first three
    dense MLPs: 12,288 / TP64 is 192, while the static-FP8 kernel stores 256
    elements per rank.

    ``expert_sharded`` divides only over EP. ``expert_tp_sharded`` divides
    only over the TP subgroup inside each EP partition, so the resulting
    parameter is replicated once per EP partition.
    """

    name: str
    shape: tuple[int, ...]
    placement: Placement
    dtype: StorageDtype
    shard_dim: int | None = None
    shard_alignment: int = 1

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @property
    def bytes_per_element(self) -> int:
        return _DTYPE_BYTES[self.dtype]

    def _shard_degree(self, world_size: int, ep_degree: int) -> int:
        if world_size <= 0 or ep_degree <= 0:
            raise ValueError("world_size and ep_degree must be positive")
        if world_size % ep_degree:
            raise ValueError("world_size must be divisible by ep_degree")

        if self.placement == "replicated":
            return 1
        if self.placement == "world_sharded":
            return world_size
        if self.placement == "expert_sharded":
            return ep_degree
        if self.placement == "expert_tp_sharded":
            return world_size // ep_degree
        raise ValueError(f"unsupported placement {self.placement!r}")

    def local_numel(self, world_size: int, ep_degree: int = 1) -> int:
        shard_degree = self._shard_degree(world_size, ep_degree)
        if self.shard_dim is None:
            if self.numel % shard_degree:
                raise ValueError(
                    f"{self.name} with {self.numel} elements is not divisible "
                    f"by its placement group of size {shard_degree}"
                )
            return self.numel // shard_degree

        if not 0 <= self.shard_dim < len(self.shape):
            raise ValueError(f"{self.name} has an invalid shard dimension")
        shard_extent = self.shape[self.shard_dim]
        if shard_extent % shard_degree:
            raise ValueError(
                f"{self.name} shard extent {shard_extent} is not divisible by "
                f"its placement group of size {shard_degree}"
            )
        local_extent = _ceil_to_multiple(
            shard_extent // shard_degree,
            self.shard_alignment,
        )
        unsharded_numel = self.numel // shard_extent
        return unsharded_numel * local_extent

    def local_bytes(self, world_size: int, ep_degree: int = 1) -> int:
        return self.local_numel(world_size, ep_degree) * self.bytes_per_element


def _scale_spec(name: str, columns: int = 1) -> WeightSpec:
    return WeightSpec(
        name,
        (_STATIC_SCALE_ROWS, columns),
        "replicated",
        "fp32",
    )


def iter_backbone_weight_specs(
    config: Glm52MoeDsaConfig,
) -> Iterator[WeightSpec]:
    """Yield the materialized 78-layer, MTP-off parameter contract.

    The frozen checkpoint also contains ``model.layers.78.*`` MTP tensors.
    They are deliberately outside this iterator. Shapes for normal weights
    remain checkpoint-logical; expanded static-FP8 scale parameters use their
    actual rank-local kernel shapes.
    """

    hidden = config.hidden_size
    yield WeightSpec(
        "model.embed_tokens.weight",
        (config.vocab_size, hidden),
        "world_sharded",
        "bf16",
        shard_dim=0,
    )
    yield WeightSpec("model.norm.weight", (hidden,), "replicated", "bf16")
    yield WeightSpec(
        "lm_head.weight",
        (config.vocab_size, hidden),
        "world_sharded",
        "bf16",
        shard_dim=0,
    )

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        attention = f"{prefix}.self_attn"
        yield WeightSpec(
            f"{prefix}.input_layernorm.weight",
            (hidden,),
            "replicated",
            "bf16",
        )
        yield WeightSpec(
            f"{prefix}.post_attention_layernorm.weight",
            (hidden,),
            "replicated",
            "bf16",
        )
        yield WeightSpec(
            f"{attention}.q_a_proj.weight",
            (config.q_lora_rank, hidden),
            "replicated",
            "fp8",
        )
        yield WeightSpec(
            f"{attention}.q_a_layernorm.weight",
            (config.q_lora_rank,),
            "replicated",
            "bf16",
        )
        yield WeightSpec(
            f"{attention}.q_b_proj.weight",
            (config.num_attention_heads * config.qk_head_dim, config.q_lora_rank),
            "world_sharded",
            "fp8",
            shard_dim=0,
        )
        yield WeightSpec(
            f"{attention}.kv_a_proj_with_mqa.weight",
            (config.kv_lora_rank + config.qk_rope_head_dim, hidden),
            "replicated",
            "fp8",
        )
        yield WeightSpec(
            f"{attention}.kv_a_layernorm.weight",
            (config.kv_lora_rank,),
            "replicated",
            "bf16",
        )
        yield WeightSpec(
            f"{attention}.kv_b_proj.weight",
            (
                config.num_attention_heads
                * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            ),
            "world_sharded",
            "fp8",
            shard_dim=0,
        )
        yield WeightSpec(
            f"{attention}.o_proj.weight",
            (hidden, config.num_attention_heads * config.v_head_dim),
            "world_sharded",
            "fp8",
            shard_dim=1,
        )

        # The four column projections use the synthetic QKV ABI, which
        # broadcasts one checkpoint scalar into three kernel scale columns.
        for projection in (
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
        ):
            yield _scale_spec(f"{attention}.{projection}.weight_scale", columns=3)
            yield _scale_spec(f"{attention}.{projection}.input_scale")
        yield _scale_spec(f"{attention}.o_proj.weight_scale")
        yield _scale_spec(f"{attention}.o_proj.input_scale")

        if config.indexer_types[layer_idx] == "full":
            indexer = f"{attention}.indexer"
            yield WeightSpec(
                f"{indexer}.wq_b.weight",
                (
                    config.index_n_heads * config.index_head_dim,
                    config.q_lora_rank,
                ),
                "replicated",
                "bf16",
            )
            yield WeightSpec(
                f"{indexer}.wk.weight",
                (config.index_head_dim, hidden),
                "replicated",
                "bf16",
            )
            yield WeightSpec(
                f"{indexer}.k_norm.weight",
                (config.index_head_dim,),
                "replicated",
                "bf16",
            )
            yield WeightSpec(
                f"{indexer}.k_norm.bias",
                (config.index_head_dim,),
                "replicated",
                "bf16",
            )
            yield WeightSpec(
                f"{indexer}.weights_proj.weight",
                (config.index_n_heads, hidden),
                "replicated",
                "fp32",
            )

        mlp = f"{prefix}.mlp"
        if config.mlp_layer_types[layer_idx] == "dense":
            for projection in ("gate_proj", "up_proj"):
                yield WeightSpec(
                    f"{mlp}.{projection}.weight",
                    (config.intermediate_size, hidden),
                    "world_sharded",
                    "fp8",
                    shard_dim=0,
                    shard_alignment=128,
                )
            yield WeightSpec(
                f"{mlp}.down_proj.weight",
                (hidden, config.intermediate_size),
                "world_sharded",
                "fp8",
                shard_dim=1,
                shard_alignment=128,
            )
            for projection in ("gate_proj", "up_proj", "down_proj"):
                yield _scale_spec(f"{mlp}.{projection}.weight_scale")
            yield _scale_spec(f"{mlp}.gate_up_input_scale")
            yield _scale_spec(f"{mlp}.down_input_scale")
            continue

        yield WeightSpec(
            f"{mlp}.gate.weight",
            (config.n_routed_experts, hidden),
            "replicated",
            "fp32",
        )
        yield WeightSpec(
            f"{mlp}.gate.e_score_correction_bias",
            (config.n_routed_experts,),
            "replicated",
            "fp32",
        )
        yield WeightSpec(
            f"{mlp}.experts.gate_up_proj",
            (
                config.n_routed_experts,
                2 * config.moe_intermediate_size,
                hidden,
            ),
            "world_sharded",
            "fp8",
        )
        yield WeightSpec(
            f"{mlp}.experts.down_proj",
            (
                config.n_routed_experts,
                hidden,
                config.moe_intermediate_size,
            ),
            "world_sharded",
            "fp8",
        )
        yield WeightSpec(
            f"{mlp}.experts.gate_up_proj_scale",
            (
                config.n_routed_experts,
                2,
                config.moe_intermediate_size,
            ),
            "world_sharded",
            "fp32",
        )
        # Each expert's scalar down scale is expanded over the full hidden
        # dimension and therefore repeated on every expert-TP rank.
        yield WeightSpec(
            f"{mlp}.experts.down_proj_scale",
            (config.n_routed_experts, hidden),
            "expert_sharded",
            "fp32",
        )

        shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
        shared_dtype: StorageDtype = (
            "bf16" if config.shared_expert_dtype == "bfloat16" else "fp8"
        )
        for projection in ("gate_proj", "up_proj"):
            yield WeightSpec(
                f"{mlp}.shared_experts.{projection}.weight",
                (shared_intermediate, hidden),
                "expert_tp_sharded",
                shared_dtype,
                shard_dim=0,
            )
        yield WeightSpec(
            f"{mlp}.shared_experts.down_proj.weight",
            (hidden, shared_intermediate),
            "expert_tp_sharded",
            shared_dtype,
            shard_dim=1,
        )
        if config.shared_expert_dtype == "fp8":
            for projection in ("gate_proj", "up_proj", "down_proj"):
                yield _scale_spec(f"{mlp}.shared_experts.{projection}.weight_scale")
            yield _scale_spec(f"{mlp}.shared_experts.gate_up_input_scale")
            yield _scale_spec(f"{mlp}.shared_experts.down_input_scale")


def estimate_local_weight_bytes(
    config: Glm52MoeDsaConfig,
    *,
    world_size: int,
    ep_degree: int,
) -> int:
    """Return exact rank-local bytes for the static-FP8 materialization."""

    if world_size <= 0 or ep_degree <= 0:
        raise ValueError("world_size and ep_degree must be positive")
    if world_size % ep_degree:
        raise ValueError("world_size must be divisible by ep_degree")
    if config.n_routed_experts % ep_degree:
        raise ValueError("routed experts must be divisible by ep_degree")
    expert_tp_degree = world_size // ep_degree
    if config.moe_intermediate_size % expert_tp_degree:
        raise ValueError(
            "expert intermediate size must divide over the expert TP subgroup"
        )
    return sum(
        spec.local_bytes(world_size, ep_degree)
        for spec in iter_backbone_weight_specs(config)
    )
