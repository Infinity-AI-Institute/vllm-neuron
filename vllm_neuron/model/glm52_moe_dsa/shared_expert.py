# SPDX-License-Identifier: Apache-2.0
"""Hybrid-capable shared expert using the qualified subgroup topology."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
    sharding_weight_loader_with_padding,
)

from .checkpoint_mapping import (
    _read_scalar,
    _to_neuron_legacy_fp8,
)
from .config import Glm52MoeDsaConfig
from .parallelism import RoutedExpertPlan
from .static_fp8 import (
    OCP_E4M3FN_QMAX448,
    static_fp8_scale_multiplier,
)

if TYPE_CHECKING:
    from safetensors import PySafeSlice

_SCALE_ROWS = 128


def _static_fp8_sharding_loader(
    *,
    shard_dim: int,
    shard_size: int,
    num_shards: int,
    pad_dim: int | None = None,
    padded_size: int | None = None,
    weight_format: str = OCP_E4M3FN_QMAX448,
) -> SafetensorsWeightLoader:
    if pad_dim is None:
        base = sharding_weight_loader(
            shard_dim=shard_dim,
            shard_size=shard_size,
            num_shards=num_shards,
            is_storage_transposed=True,
        )
    else:
        if padded_size is None or padded_size < shard_size:
            raise ValueError("padded_size must be at least the shard size")
        base = sharding_weight_loader_with_padding(
            shard_dim=shard_dim,
            shard_size=shard_size,
            num_shards=num_shards,
            is_storage_transposed=True,
            pad_dim=pad_dim,
            padded_size=padded_size,
            unpadded_size=shard_size,
        )

    def transform(
        slices: list["PySafeSlice"],
        rank: int,
    ) -> torch.Tensor:
        return _to_neuron_legacy_fp8(
            base.load(slices, rank),
            weight_format,
        )

    return SafetensorsWeightLoader(transform=transform)


def _scalar_scale_loader(
    *,
    compensate_weight_range: bool,
    weight_format: str = OCP_E4M3FN_QMAX448,
) -> SafetensorsWeightLoader:
    def transform(
        slices: list["PySafeSlice"],
        rank: int,
    ) -> torch.Tensor:
        del rank
        if len(slices) != 1:
            raise ValueError(f"expected one scalar scale, got {len(slices)}")
        scalar = _read_scalar(slices[0])
        if compensate_weight_range:
            scalar = scalar * static_fp8_scale_multiplier(weight_format)
        return scalar.reshape(1, 1).expand(_SCALE_ROWS, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _consensus_input_scale_loader() -> SafetensorsWeightLoader:
    def transform(
        slices: list["PySafeSlice"],
        rank: int,
    ) -> torch.Tensor:
        del rank
        if len(slices) != 2:
            raise ValueError("expected gate and up input scales")
        gate = _read_scalar(slices[0])
        up = _read_scalar(slices[1])
        if not torch.equal(gate, up):
            raise ValueError(
                "shared-expert gate/up input scales must match for the fused "
                f"MLP kernel, got gate={gate.item()} and up={up.item()}"
            )
        return gate.reshape(1, 1).expand(_SCALE_ROWS, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


class _StaticFp8Projection(nn.Module):
    def __init__(
        self,
        shape: tuple[int, int],
        *,
        weight_loader: SafetensorsWeightLoader,
        weight_format: str,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                *shape,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(
                _SCALE_ROWS,
                1,
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )
        set_weight_loader(self.weight, weight_loader)
        set_weight_loader(
            self.weight_scale,
            _scalar_scale_loader(
                compensate_weight_range=True,
                weight_format=weight_format,
            ),
        )


class _Bf16Projection(nn.Module):
    def __init__(
        self,
        shape: tuple[int, int],
        *,
        weight_loader: SafetensorsWeightLoader,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(*shape, dtype=dtype, device=device),
            requires_grad=False,
        )
        set_weight_loader(self.weight, weight_loader)


class Glm52SharedExpert(nn.Module):
    """One shared expert replicated across EP partitions.

    The shared projection is sharded over the routed expert's TP subgroup,
    never the full TP64 world. Hardware qualification shows static-FP8
    prefill requires ``I/rank >= 128``; all supported EP8/16/32/64 plans use
    subgroup TP8/4/2/1 and satisfy that constraint.
    """

    def __init__(
        self,
        config: Glm52MoeDsaConfig,
        plan: RoutedExpertPlan,
        *,
        global_rank: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if config.n_shared_experts != 1:
            raise ValueError("GLM-5.2 requires exactly one shared expert")
        if plan.expert_intermediate_size != config.moe_intermediate_size:
            raise ValueError("shared and routed expert intermediate sizes must match")
        plan.local_expert_ids(global_rank)

        self.shared_tp_degree = plan.expert_tp_degree
        self.shared_tp_rank = plan.expert_tp_rank(global_rank)
        self.static_fp8 = config.shared_expert_dtype == "fp8"
        if self.static_fp8 and self.shared_tp_degree > 16:
            raise ValueError(
                "GLM-5.2 static-FP8 shared-expert prefill requires TP16 or "
                "smaller; shard it over the routed expert TP subgroup"
            )
        local_intermediate = plan.intermediate_per_rank
        if self.static_fp8:
            gate_up_loader = _static_fp8_sharding_loader(
                shard_dim=1,
                shard_size=local_intermediate,
                num_shards=self.shared_tp_degree,
                weight_format=config.static_fp8_weight_format,
            )
            down_loader = _static_fp8_sharding_loader(
                shard_dim=0,
                shard_size=local_intermediate,
                num_shards=self.shared_tp_degree,
                weight_format=config.static_fp8_weight_format,
            )
            projection_type = _StaticFp8Projection
            projection_kwargs = {
                "weight_format": config.static_fp8_weight_format,
            }
        else:
            gate_up_loader = sharding_weight_loader(
                shard_dim=1,
                shard_size=local_intermediate,
                num_shards=self.shared_tp_degree,
                is_storage_transposed=True,
            )
            down_loader = sharding_weight_loader(
                shard_dim=0,
                shard_size=local_intermediate,
                num_shards=self.shared_tp_degree,
                is_storage_transposed=True,
            )
            projection_type = _Bf16Projection
            projection_kwargs = {"dtype": config.torch_dtype}
        self.gate_proj = projection_type(
            (config.hidden_size, local_intermediate),
            weight_loader=gate_up_loader,
            device=device,
            **projection_kwargs,
        )
        self.up_proj = projection_type(
            (config.hidden_size, local_intermediate),
            weight_loader=gate_up_loader,
            device=device,
            **projection_kwargs,
        )
        self.down_proj = projection_type(
            (local_intermediate, config.hidden_size),
            weight_loader=down_loader,
            device=device,
            **projection_kwargs,
        )
        if self.static_fp8:
            self.gate_up_input_scale = nn.Parameter(
                torch.empty(
                    _SCALE_ROWS,
                    1,
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )
            self.down_input_scale = nn.Parameter(
                torch.empty(
                    _SCALE_ROWS,
                    1,
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )
            set_weight_loader(
                self.gate_up_input_scale,
                _consensus_input_scale_loader(),
            )
            set_weight_loader(
                self.down_input_scale,
                _scalar_scale_loader(compensate_weight_range=False),
            )
        else:
            self.gate_up_input_scale = None
            self.down_input_scale = None

    def _run(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from nkilib.core.utils.common_types import ActFnType, QuantizationType

        from vllm_neuron.functional.mlp import mlp

        if not self.static_fp8:
            return mlp(
                hidden_states,
                self.gate_proj.weight,
                self.up_proj.weight,
                self.down_proj.weight,
                act_fn=ActFnType.SiLU,
                quantization_type=QuantizationType.NONE,
                output_dtype="bfloat16",
            )
        return mlp(
            hidden_states,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
            act_fn=ActFnType.SiLU,
            quantization_type=QuantizationType.STATIC,
            gate_w_scale=self.gate_proj.weight_scale,
            up_w_scale=self.up_proj.weight_scale,
            down_w_scale=self.down_proj.weight_scale,
            gate_up_in_scale=self.gate_up_input_scale,
            down_in_scale=self.down_input_scale,
            output_dtype="bfloat16",
        )

    def forward_decode(self, hidden_states_normalized: torch.Tensor) -> torch.Tensor:
        """Run decode; TKG quantizes the BF16 normalized activation."""

        return self._run(hidden_states_normalized)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        *,
        norm_weight: torch.Tensor,
        eps: float,
        tp_group,
        hidden_states_normalized: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse norm+quant, gather subgroup SP, run MLP, then reduce-scatter."""

        if self.static_fp8:
            if hidden_states_normalized is not None:
                raise ValueError(
                    "static-FP8 shared prefill must use fused RMSNorm/quantization"
                )
            from nkilib.core.utils.common_types import QuantizationType

            from vllm_neuron.functional.rmsnorm_quant import rmsnorm_quant

            normalized = rmsnorm_quant(
                hidden_states,
                ln_w=norm_weight,
                input_dequant_scale=self.gate_up_input_scale,
                eps=eps,
                quantization_type=QuantizationType.STATIC,
            )
        elif hidden_states_normalized is not None:
            if hidden_states_normalized.shape != hidden_states.shape:
                raise ValueError(
                    "pre-normalized shared-expert input must match hidden states"
                )
            normalized = hidden_states_normalized
        else:
            input_dtype = hidden_states.dtype
            hidden_float = hidden_states.to(torch.float32)
            variance = hidden_float.pow(2).mean(dim=-1, keepdim=True)
            normalized = (
                hidden_float
                * torch.rsqrt(variance + eps)
                * norm_weight.to(torch.float32)
            ).to(input_dtype)
        if tp_group.world_size > 1:
            normalized = tp_group.all_gather(normalized, dim=0)
        output = self._run(normalized)
        if tp_group.world_size > 1:
            output = tp_group.reduce_scatter(output, dim=0)
        return output
