# SPDX-License-Identifier: Apache-2.0
"""Executable GLM-5.2 routed-expert core for the qualified Trn2 kernels."""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm_neuron.utils.weight_loader import set_weight_loader

from .checkpoint_mapping import (
    routed_down_scale_loader,
    routed_down_weight_loader,
    routed_gate_up_scale_loader,
    routed_gate_up_weight_loader,
)
from .config import Glm52MoeDsaConfig
from .parallelism import RoutedExpertPlan


def dense_glm52_affinities(
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    num_experts: int,
) -> torch.Tensor:
    """Scatter GLM's normalized top-k weights into the kernel's dense ABI."""

    if expert_indices.shape != routing_weights.shape:
        raise ValueError("expert indices and routing weights must have equal shapes")
    if expert_indices.ndim != 2:
        raise ValueError("expert indices and routing weights must be rank-2")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if expert_indices.numel() and not torch.compiler.is_compiling():
        minimum = int(expert_indices.min())
        maximum = int(expert_indices.max())
        if minimum < 0 or maximum >= num_experts:
            raise ValueError("expert index is outside the configured expert range")

    affinities = torch.zeros(
        expert_indices.shape[0],
        num_experts,
        dtype=routing_weights.dtype,
        device=routing_weights.device,
    )
    return affinities.scatter(-1, expert_indices.to(torch.int64), routing_weights)


class Glm52RoutedExperts(nn.Module):
    """Rank-local static-FP8 expert weights and kernel dispatch.

    This module deliberately returns a rank-local partial result. The owning
    sparse MLP is responsible for TP/EP collectives and for adding the
    always-active shared expert exactly once.
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
        if plan.num_experts != config.n_routed_experts:
            raise ValueError("expert plan and config disagree on expert count")
        if plan.expert_intermediate_size != config.moe_intermediate_size:
            raise ValueError(
                "expert plan and config disagree on expert intermediate size"
            )
        plan.local_expert_ids(global_rank)

        self.plan = plan
        self.global_rank = global_rank
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.ep_rank = plan.ep_rank(global_rank)
        self.block_size = 256

        local_experts = plan.experts_per_rank
        local_intermediate = plan.intermediate_per_rank
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                local_experts,
                config.hidden_size,
                2,
                local_intermediate,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                local_experts,
                local_intermediate,
                config.hidden_size,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.gate_up_proj_scale = nn.Parameter(
            torch.empty(
                local_experts,
                2,
                local_intermediate,
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )
        self.down_proj_scale = nn.Parameter(
            torch.empty(
                local_experts,
                config.hidden_size,
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )

        set_weight_loader(
            self.gate_up_proj,
            routed_gate_up_weight_loader(plan),
        )
        set_weight_loader(
            self.down_proj,
            routed_down_weight_loader(plan),
        )
        set_weight_loader(
            self.gate_up_proj_scale,
            routed_gate_up_scale_loader(plan),
        )
        set_weight_loader(
            self.down_proj_scale,
            routed_down_scale_loader(plan, hidden_size=config.hidden_size),
        )

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run the qualified FP8 ``moe_tkg`` local partial."""

        from nkilib.core.utils.common_types import (
            ActFnType,
            ExpertAffinityScaleMode,
        )

        from vllm_neuron.functional.moe.moe_tkg import moe_tkg

        affinities = dense_glm52_affinities(
            expert_indices,
            routing_weights,
            num_experts=self.num_experts,
        )
        rank_id = torch.tensor(
            [[self.ep_rank]],
            dtype=torch.int32,
            device=hidden_states.device,
        )
        return moe_tkg(
            hidden_input=hidden_states,
            expert_gate_up_weights=self.gate_up_proj,
            expert_down_weights=self.down_proj,
            expert_affinities=affinities,
            expert_index=expert_indices.to(torch.int32),
            is_all_expert=True,
            rank_id=rank_id,
            expert_gate_up_weights_scale=self.gate_up_proj_scale,
            expert_down_weights_scale=self.down_proj_scale,
            mask_unselected_experts=True,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
            output_dtype=hidden_states.dtype,
        )

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        expert_affinities_masked: torch.Tensor,
        token_position_to_id: torch.Tensor,
        block_to_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Run the qualified scale-aware ``moe_cte(shard_on_i)`` local partial."""

        import nki.language as nl
        from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
        from nkilib.core.utils.common_types import (
            ActFnType,
            ExpertAffinityScaleMode,
        )

        from vllm_neuron.functional.moe.moe_cte import moe_cte

        local_experts = self.plan.experts_per_rank
        local_intermediate = self.plan.intermediate_per_rank
        return moe_cte(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=self.gate_up_proj,
            down_proj_weight=self.down_proj,
            token_position_to_id=token_position_to_id.to(torch.int32),
            block_to_expert=block_to_expert.to(torch.int32),
            block_size=self.block_size,
            implementation=MoECTEImplementation.shard_on_i,
            gate_up_proj_scale=self.gate_up_proj_scale.reshape(
                local_experts,
                1,
                2 * local_intermediate,
            ),
            down_proj_scale=self.down_proj_scale.unsqueeze(1),
            activation_function=ActFnType.SiLU,
            compute_dtype=nl.bfloat16,
            is_tensor_update_accumulating=True,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            # The blockwise mapper pads unused token slots with -1.  Tell the
            # Neuron kernel to skip those DMA reads instead of treating -1 as
            # an address, which otherwise triggers an out-of-bounds gather.
            skip_token=True,
        )
