# SPDX-License-Identifier: Apache-2.0
"""Checkpoint contract and routed-expert loaders for GLM-5.2.

The published ``GLM-5.2-FP8`` checkpoint uses 128x128 block-FP8 tensors.
Those tensors are not a valid input to the Trn2 static-FP8 kernels.  The
loaders in this module intentionally target the converted, per-projection
static-FP8 artifact described by the enablement plan.

Routed experts remain separate in the checkpoint and are fused only after
expert-parallel filtering and tensor-parallel sharding.  This keeps host
memory bounded and produces the exact layouts qualified on Trn2:

* gate/up weights: ``[E_local, H, 2, I_rank]``;
* down weights: ``[E_local, I_rank, H]``;
* gate/up row scales: ``[E_local, 2, I_rank]``; and
* down row scales: ``[E_local, H]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

from .config import Glm52MoeDsaConfig
from .parallelism import RoutedExpertPlan
from .weight_manifest import iter_backbone_weight_specs

if TYPE_CHECKING:
    from safetensors import PySafeSlice

CheckpointSource = str | list[str]

MTP_IGNORED_PREFIX = "model.layers.78."
STATIC_WEIGHT_SCALE_SUFFIX = ".weight_scale"
_OCP_E4M3_MAX = 448.0
_NEURON_LEGACY_E4M3_MAX = 240.0
_WEIGHT_DOWNSCALE = _NEURON_LEGACY_E4M3_MAX / _OCP_E4M3_MAX
_SCALE_COMPENSATION = _OCP_E4M3_MAX / _NEURON_LEGACY_E4M3_MAX


@dataclass(frozen=True)
class Glm52CheckpointContract:
    """Rank-local mapping for the MTP-disabled GLM-5.2 backbone."""

    mappings: dict[str, CheckpointSource]
    ignored_prefixes: tuple[str, ...] = (MTP_IGNORED_PREFIX,)

    @property
    def required_source_keys(self) -> frozenset[str]:
        keys: set[str] = set()
        for source in self.mappings.values():
            if isinstance(source, str):
                keys.add(source)
            else:
                keys.update(source)
        return frozenset(keys)

    def validate_source_keys(self, source_keys: set[str] | frozenset[str]) -> None:
        """Fail before loading if a mapped tensor is absent.

        Extra checkpoint tensors are permitted because conversion metadata,
        activation scales, and the deferred MTP layer are not model
        parameters.  MTP is the only model-layer prefix deliberately ignored.
        """

        missing = sorted(self.required_source_keys.difference(source_keys))
        if missing:
            preview = ", ".join(repr(key) for key in missing[:8])
            remainder = len(missing) - min(len(missing), 8)
            suffix = f" (+{remainder} more)" if remainder else ""
            raise KeyError(f"Missing GLM-5.2 checkpoint tensors: {preview}{suffix}")

    def ignored_source_keys(
        self, source_keys: set[str] | frozenset[str]
    ) -> frozenset[str]:
        return frozenset(
            key
            for key in source_keys
            if any(key.startswith(prefix) for prefix in self.ignored_prefixes)
        )


def _expert_weight_key(
    layer_idx: int,
    expert_idx: int,
    projection: str,
) -> str:
    return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{projection}.weight"


def _scale_key(weight_key: str) -> str:
    return weight_key.removesuffix(".weight") + STATIC_WEIGHT_SCALE_SUFFIX


def build_checkpoint_contract(
    config: Glm52MoeDsaConfig,
    plan: RoutedExpertPlan,
    *,
    global_rank: int,
) -> Glm52CheckpointContract:
    """Build the exact rank-local source mapping.

    Non-expert backbone tensors keep their Hugging Face names.  Sparse expert
    tensors are expanded to the local expert IDs, preserving gate/up order for
    each expert.  The source scale names follow the plugin's ModelOpt static
    FP8 convention and must contain one scalar per projection.
    """

    if config.num_hidden_layers >= 79:
        raise ValueError(
            "MTP-off serving requires fewer than 79 backbone layers so "
            f"{MTP_IGNORED_PREFIX!r} remains an unambiguous ignored prefix"
        )
    if plan.world_size <= global_rank or global_rank < 0:
        raise ValueError(f"global rank {global_rank} is outside the expert plan")
    if plan.num_experts != config.n_routed_experts:
        raise ValueError("expert plan and GLM config disagree on expert count")
    if plan.expert_intermediate_size != config.moe_intermediate_size:
        raise ValueError(
            "expert plan and GLM config disagree on expert intermediate size"
        )

    mappings: dict[str, CheckpointSource] = {}
    for spec in iter_backbone_weight_specs(config):
        if ".mlp.experts.gate_up_proj" in spec.name:
            continue
        if ".mlp.experts.down_proj" in spec.name:
            continue
        mappings[spec.name] = spec.name

    for layer_idx in range(config.num_hidden_layers):
        attention = f"model.layers.{layer_idx}.self_attn"
        for projection in (
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
        ):
            prefix = f"{attention}.{projection}"
            mappings[f"{prefix}.weight_scale"] = f"{prefix}.weight_scale"
            mappings[f"{prefix}.input_scale"] = f"{prefix}.input_scale"

    for layer_idx, layer_type in enumerate(config.mlp_layer_types):
        if layer_type != "dense":
            continue
        dense = f"model.layers.{layer_idx}.mlp"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            weight = f"{dense}.{projection}.weight"
            mappings[f"{weight}_scale"] = _scale_key(weight)
        mappings[f"{dense}.gate_up_input_scale"] = [
            f"{dense}.gate_proj.input_scale",
            f"{dense}.up_proj.input_scale",
        ]
        mappings[f"{dense}.down_input_scale"] = f"{dense}.down_proj.input_scale"

    local_experts = plan.local_expert_ids(global_rank)
    for layer_idx, layer_type in enumerate(config.mlp_layer_types):
        if layer_type != "sparse":
            continue
        prefix = f"model.layers.{layer_idx}.mlp.experts"
        gate_up_sources: list[str] = []
        gate_up_scale_sources: list[str] = []
        down_sources: list[str] = []
        down_scale_sources: list[str] = []
        for expert_idx in local_experts:
            gate = _expert_weight_key(layer_idx, expert_idx, "gate_proj")
            up = _expert_weight_key(layer_idx, expert_idx, "up_proj")
            down = _expert_weight_key(layer_idx, expert_idx, "down_proj")
            gate_up_sources.extend((gate, up))
            gate_up_scale_sources.extend((_scale_key(gate), _scale_key(up)))
            down_sources.append(down)
            down_scale_sources.append(_scale_key(down))

        mappings[f"{prefix}.gate_up_proj"] = gate_up_sources
        mappings[f"{prefix}.gate_up_proj_scale"] = gate_up_scale_sources
        mappings[f"{prefix}.down_proj"] = down_sources
        mappings[f"{prefix}.down_proj_scale"] = down_scale_sources

        shared = f"model.layers.{layer_idx}.mlp.shared_experts"
        if config.shared_expert_dtype == "fp8":
            for projection in ("gate_proj", "up_proj", "down_proj"):
                weight = f"{shared}.{projection}.weight"
                mappings[f"{weight}_scale"] = (
                    weight.removesuffix(".weight") + STATIC_WEIGHT_SCALE_SUFFIX
                )
            mappings[f"{shared}.gate_up_input_scale"] = [
                f"{shared}.gate_proj.input_scale",
                f"{shared}.up_proj.input_scale",
            ]
            mappings[f"{shared}.down_input_scale"] = f"{shared}.down_proj.input_scale"

    return Glm52CheckpointContract(mappings=mappings)


def _read_scalar(slice_obj: "PySafeSlice") -> torch.Tensor:
    shape = tuple(slice_obj.get_shape())
    raw = slice_obj[()] if not shape else slice_obj[:]
    scalar = raw.to(torch.float32).reshape(-1)
    if scalar.numel() != 1:
        raise ValueError(
            "GLM-5.2 Trn2 requires a converted per-projection static-FP8 "
            f"scale, got source shape {shape} with {scalar.numel()} values. "
            "The native 128x128 block-FP8 checkpoint must be converted first."
        )
    return scalar[0]


def _to_neuron_legacy_fp8(weight: torch.Tensor) -> torch.Tensor:
    """Convert ModelOpt OCP E4M3 values to the Trn2 kernel's 240 range."""

    return (
        (weight.to(torch.float32) * _WEIGHT_DOWNSCALE)
        .clamp(-_NEURON_LEGACY_E4M3_MAX, _NEURON_LEGACY_E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )


def _expert_tp_bounds(plan: RoutedExpertPlan, rank: int) -> tuple[int, int]:
    tp_rank = plan.expert_tp_rank(rank)
    start = tp_rank * plan.intermediate_per_rank
    return start, start + plan.intermediate_per_rank


def routed_gate_up_weight_loader(
    plan: RoutedExpertPlan,
) -> SafetensorsWeightLoader:
    """Fuse separate ``[I,H]`` gate/up tensors into the qualified ABI."""

    def transform(
        slices: list["PySafeSlice"],
        rank: int,
    ) -> torch.Tensor:
        expected = plan.experts_per_rank * 2
        if len(slices) != expected:
            raise ValueError(
                f"expected {expected} local gate/up tensors, got {len(slices)}"
            )
        start, end = _expert_tp_bounds(plan, rank)
        experts = []
        for offset in range(0, len(slices), 2):
            gate = slices[offset][start:end, :]
            up = slices[offset + 1][start:end, :]
            experts.append(torch.stack((gate.T, up.T), dim=1))
        return _to_neuron_legacy_fp8(torch.stack(experts, dim=0))

    return SafetensorsWeightLoader(transform=transform)


def routed_down_weight_loader(
    plan: RoutedExpertPlan,
) -> SafetensorsWeightLoader:
    """Shard separate ``[H,I]`` down tensors into ``[E,I_rank,H]``."""

    def transform(
        slices: list["PySafeSlice"],
        rank: int,
    ) -> torch.Tensor:
        if len(slices) != plan.experts_per_rank:
            raise ValueError(
                f"expected {plan.experts_per_rank} local down tensors, "
                f"got {len(slices)}"
            )
        start, end = _expert_tp_bounds(plan, rank)
        return _to_neuron_legacy_fp8(
            torch.stack(
                [slice_obj[:, start:end].T for slice_obj in slices],
                dim=0,
            )
        )

    return SafetensorsWeightLoader(transform=transform)


def routed_gate_up_scale_loader(
    plan: RoutedExpertPlan,
) -> SafetensorsWeightLoader:
    """Broadcast scalar gate/up dequant scales to kernel row scales."""

    def transform(
        slices: list["PySafeSlice"],
        rank: int,
    ) -> torch.Tensor:
        del rank
        expected = plan.experts_per_rank * 2
        if len(slices) != expected:
            raise ValueError(
                f"expected {expected} local gate/up scales, got {len(slices)}"
            )
        per_expert = []
        for offset in range(0, len(slices), 2):
            pair = (
                torch.stack(
                    (
                        _read_scalar(slices[offset]),
                        _read_scalar(slices[offset + 1]),
                    )
                )
                * _SCALE_COMPENSATION
            )
            per_expert.append(pair[:, None].expand(2, plan.intermediate_per_rank))
        return torch.stack(per_expert, dim=0).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def routed_down_scale_loader(
    plan: RoutedExpertPlan,
    *,
    hidden_size: int,
) -> SafetensorsWeightLoader:
    """Broadcast scalar down dequant scales to ``[E_local,H]``."""

    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")

    def transform(
        slices: list["PySafeSlice"],
        rank: int,
    ) -> torch.Tensor:
        del rank
        if len(slices) != plan.experts_per_rank:
            raise ValueError(
                f"expected {plan.experts_per_rank} local down scales, got {len(slices)}"
            )
        scalars = (
            torch.stack([_read_scalar(slice_obj) for slice_obj in slices])
            * _SCALE_COMPENSATION
        )
        return scalars[:, None].expand(plan.experts_per_rank, hidden_size).contiguous()

    return SafetensorsWeightLoader(transform=transform)
