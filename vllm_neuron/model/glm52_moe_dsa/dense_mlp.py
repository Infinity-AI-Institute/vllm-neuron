# SPDX-License-Identifier: Apache-2.0
"""Static-FP8 dense MLP for GLM-5.2 backbone layers 0-2."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_neuron.utils.weight_loader import set_weight_loader

from .config import Glm52MoeDsaConfig
from .shared_expert import (
    _SCALE_ROWS,
    _StaticFp8Projection,
    _consensus_input_scale_loader,
    _scalar_scale_loader,
    _static_fp8_sharding_loader,
)
from .sparse_mlp import glm52_rms_norm


class _DenseBf16Projection(nn.Module):
    def __init__(
        self,
        shape: tuple[int, int],
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(*shape, dtype=dtype, device=device),
            requires_grad=False,
        )


class Glm52DenseMlp(nn.Module):
    """GLM's first-three-layer SiLU MLP sharded over full tensor parallelism.

    Production uses converted static-FP8 weights. ``static_fp8=False`` exists
    only for CPU equation tests and reduced-shape golden comparisons.
    """

    def __init__(
        self,
        config: Glm52MoeDsaConfig,
        *,
        world_size: int,
        global_rank: int,
        tp_group=None,
        static_fp8: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= global_rank < world_size:
            raise ValueError("global_rank is outside the tensor-parallel world")
        if config.intermediate_size % world_size:
            raise ValueError("dense intermediate size must divide over TP")
        if tp_group is None and world_size > 1:
            from vllm.distributed.parallel_state import get_tp_group

            tp_group = get_tp_group()
        if tp_group is not None and tp_group.world_size != world_size:
            raise ValueError("TP group size does not match world_size")

        self.config = config
        self.world_size = world_size
        self.global_rank = global_rank
        self.tp_group = tp_group
        self.static_fp8 = static_fp8
        self.dtype = dtype or config.torch_dtype
        self.local_intermediate_size = config.intermediate_size // world_size
        self.kernel_intermediate_size = (
            (self.local_intermediate_size + 127) // 128
        ) * 128

        if static_fp8:
            gate_up_loader = _static_fp8_sharding_loader(
                shard_dim=1,
                shard_size=self.local_intermediate_size,
                num_shards=world_size,
                pad_dim=1,
                padded_size=self.kernel_intermediate_size,
                weight_format=config.static_fp8_weight_format,
            )
            down_loader = _static_fp8_sharding_loader(
                shard_dim=0,
                shard_size=self.local_intermediate_size,
                num_shards=world_size,
                pad_dim=0,
                padded_size=self.kernel_intermediate_size,
                weight_format=config.static_fp8_weight_format,
            )
            self.gate_proj = _StaticFp8Projection(
                (config.hidden_size, self.kernel_intermediate_size),
                weight_loader=gate_up_loader,
                weight_format=config.static_fp8_weight_format,
                device=device,
            )
            self.up_proj = _StaticFp8Projection(
                (config.hidden_size, self.kernel_intermediate_size),
                weight_loader=gate_up_loader,
                weight_format=config.static_fp8_weight_format,
                device=device,
            )
            self.down_proj = _StaticFp8Projection(
                (self.kernel_intermediate_size, config.hidden_size),
                weight_loader=down_loader,
                weight_format=config.static_fp8_weight_format,
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
        else:
            self.gate_proj = _DenseBf16Projection(
                (config.hidden_size, self.local_intermediate_size),
                dtype=self.dtype,
                device=device,
            )
            self.up_proj = _DenseBf16Projection(
                (config.hidden_size, self.local_intermediate_size),
                dtype=self.dtype,
                device=device,
            )
            self.down_proj = _DenseBf16Projection(
                (self.local_intermediate_size, config.hidden_size),
                dtype=self.dtype,
                device=device,
            )
            self.gate_up_input_scale = None
            self.down_input_scale = None

    def _run(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.static_fp8:
            gate = hidden_states.to(self.dtype) @ self.gate_proj.weight
            up = hidden_states.to(self.dtype) @ self.up_proj.weight
            activated = F.silu(gate.float()) * up.float()
            return activated.to(self.dtype) @ self.down_proj.weight

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

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        *,
        norm_weight: torch.Tensor,
    ) -> torch.Tensor:
        normalized = glm52_rms_norm(
            hidden_states,
            norm_weight,
            eps=self.config.rms_norm_eps,
        )
        output = self._run(normalized)
        if self.tp_group is not None and self.tp_group.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        *,
        norm_weight: torch.Tensor,
    ) -> torch.Tensor:
        if self.static_fp8:
            from nkilib.core.utils.common_types import QuantizationType

            from vllm_neuron.functional.rmsnorm_quant import rmsnorm_quant

            normalized = rmsnorm_quant(
                hidden_states,
                ln_w=norm_weight,
                input_dequant_scale=self.gate_up_input_scale,
                eps=self.config.rms_norm_eps,
                quantization_type=QuantizationType.STATIC,
            )
        else:
            normalized = glm52_rms_norm(
                hidden_states,
                norm_weight,
                eps=self.config.rms_norm_eps,
            )
        if self.tp_group is not None and self.tp_group.world_size > 1:
            normalized = self.tp_group.all_gather(normalized, dim=0)
        output = self._run(normalized)
        if self.tp_group is not None and self.tp_group.world_size > 1:
            output = self.tp_group.reduce_scatter(output, dim=0)
        return output
