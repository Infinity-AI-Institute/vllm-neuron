# SPDX-License-Identifier: Apache-2.0
"""Static-FP8 shared expert using the hardware-qualified subgroup topology."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .checkpoint_mapping import (
    _SCALE_COMPENSATION,
    _read_scalar,
    _to_neuron_legacy_fp8,
)
from .config import Glm52MoeDsaConfig
from .parallelism import RoutedExpertPlan

if TYPE_CHECKING:
    from safetensors import PySafeSlice

_SCALE_ROWS = 128


def _static_fp8_sharding_loader(
    *,
    shard_dim: int,
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    base = sharding_weight_loader(
        shard_dim=shard_dim,
        shard_size=shard_size,
        num_shards=num_shards,
        is_storage_transposed=True,
    )

    def transform(
        slices: list["PySafeSlice"],
        rank: int,
    ) -> torch.Tensor:
        return _to_neuron_legacy_fp8(base.load(slices, rank))

    return SafetensorsWeightLoader(transform=transform)


def _scalar_scale_loader(
    *,
    compensate_weight_range: bool,
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
            scalar = scalar * _SCALE_COMPENSATION
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
            _scalar_scale_loader(compensate_weight_range=True),
        )


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
            raise ValueError(
                "shared and routed expert intermediate sizes must match"
            )
        plan.local_expert_ids(global_rank)

        self.shared_tp_degree = plan.expert_tp_degree
        self.shared_tp_rank = plan.expert_tp_rank(global_rank)
        if self.shared_tp_degree > 16:
            raise ValueError(
                "GLM-5.2 static-FP8 shared-expert prefill requires TP16 or "
                "smaller; shard it over the routed expert TP subgroup"
            )
        local_intermediate = plan.intermediate_per_rank
        gate_up_loader = _static_fp8_sharding_loader(
            shard_dim=1,
            shard_size=local_intermediate,
            num_shards=self.shared_tp_degree,
        )
        down_loader = _static_fp8_sharding_loader(
            shard_dim=0,
            shard_size=local_intermediate,
            num_shards=self.shared_tp_degree,
        )
        self.gate_proj = _StaticFp8Projection(
            (config.hidden_size, local_intermediate),
            weight_loader=gate_up_loader,
            device=device,
        )
        self.up_proj = _StaticFp8Projection(
            (config.hidden_size, local_intermediate),
            weight_loader=gate_up_loader,
            device=device,
        )
        self.down_proj = _StaticFp8Projection(
            (local_intermediate, config.hidden_size),
            weight_loader=down_loader,
            device=device,
        )
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

    def _run(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from nkilib.core.utils.common_types import ActFnType, QuantizationType

        from vllm_neuron.functional.mlp import mlp

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
    ) -> torch.Tensor:
        """Fuse RMSNorm+quantization before the static-FP8 CTE MLP."""

        from nkilib.core.utils.common_types import QuantizationType

        from vllm_neuron.functional.rmsnorm_quant import rmsnorm_quant

        hidden_quantized = rmsnorm_quant(
            hidden_states,
            ln_w=norm_weight,
            input_dequant_scale=self.gate_up_input_scale,
            eps=eps,
            quantization_type=QuantizationType.STATIC,
        )
        return self._run(hidden_quantized)
