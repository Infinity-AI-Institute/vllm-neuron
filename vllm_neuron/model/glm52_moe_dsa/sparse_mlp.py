# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 sparse MLP orchestration across routed and shared experts."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Glm52MoeDsaConfig
from .expert_kernels import Glm52RoutedExperts, dense_glm52_affinities
from .moe import Glm52ExpertRouter
from .parallelism import RoutedExpertPlan
from .shared_expert import Glm52SharedExpert


def glm52_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_float = hidden_states.to(torch.float32)
    variance = hidden_float.pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden_float * torch.rsqrt(variance + eps)
    return (normalized * weight.to(torch.float32)).to(input_dtype)


class Glm52SparseMlp(nn.Module):
    """External router plus qualified routed/shared expert paths.

    Routed partials combine across the full TP64 world. The shared expert is
    replicated per EP partition and combines only within that partition's
    expert-TP subgroup, preventing the shared branch from being multiplied by
    the EP degree.
    """

    def __init__(
        self,
        config: Glm52MoeDsaConfig,
        plan: RoutedExpertPlan,
        *,
        global_rank: int,
        tp_group=None,
        expert_tp_group=None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if tp_group is None or expert_tp_group is None:
            from vllm.distributed.parallel_state import get_tp_group
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_ep_tp_group,
            )

            tp_group = get_tp_group()
            expert_tp_group = get_neuron_ep_tp_group()
        if tp_group.world_size != plan.world_size:
            raise ValueError("full TP group size must match the expert plan world")
        if expert_tp_group.world_size != plan.expert_tp_degree:
            raise ValueError(
                "expert TP subgroup size must match the expert plan TP degree"
            )

        self.config = config
        self.plan = plan
        self.tp_group = tp_group
        self.expert_tp_group = expert_tp_group
        self.ep_rank = plan.ep_rank(global_rank)
        self.gate = Glm52ExpertRouter(
            config.hidden_size,
            config.n_routed_experts,
            config.num_experts_per_tok,
            config.routed_scaling_factor,
        )
        self.experts = Glm52RoutedExperts(
            config,
            plan,
            global_rank=global_rank,
            device=device,
        )
        self.shared_experts = Glm52SharedExpert(
            config,
            plan,
            global_rank=global_rank,
            device=device,
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
        expert_indices, routing_weights = self.gate(normalized)
        routed = self.experts.forward_decode(
            normalized,
            expert_indices,
            routing_weights,
        )
        if self.tp_group.world_size > 1:
            routed = self.tp_group.all_reduce(routed)

        shared = self.shared_experts.forward_decode(normalized)
        if self.expert_tp_group.world_size > 1:
            shared = self.expert_tp_group.all_reduce(shared)
        return routed + shared

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        *,
        norm_weight: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Route full-sequence tokens and return the local SP output."""

        from vllm_neuron.functional.moe.moe_blockwise import (
            build_blockwise_mapping,
        )

        normalized_local = glm52_rms_norm(
            hidden_states,
            norm_weight,
            eps=self.config.rms_norm_eps,
        )
        expert_indices, routing_weights = self.gate(normalized_local)
        affinities_local = dense_glm52_affinities(
            expert_indices,
            routing_weights,
            num_experts=self.config.n_routed_experts,
        ).to(normalized_local.dtype)

        padding_mask = None
        if positions is not None:
            last_real_idx = torch.argmax(positions)
            token_indices = torch.arange(
                positions.shape[0],
                device=positions.device,
            )
            padding_mask = token_indices <= last_real_idx

        normalized = normalized_local
        affinities = affinities_local
        if self.tp_group.world_size > 1:
            normalized = self.tp_group.all_gather(normalized, dim=0)
            affinities = self.tp_group.all_gather(affinities, dim=0)
            if padding_mask is not None:
                padding_mask = self.tp_group.all_gather(padding_mask, dim=0)

        first_expert = self.ep_rank * self.plan.experts_per_rank
        local_affinities = affinities[
            :,
            first_expert : first_expert + self.plan.experts_per_rank,
        ]
        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            _,
        ) = build_blockwise_mapping(
            expert_affinities=local_affinities,
            num_local_experts=self.plan.experts_per_rank,
            num_experts_per_token=self.config.num_experts_per_tok,
            block_size=self.experts.block_size,
            moe_group=self.expert_tp_group,
            tp_degree=self.plan.expert_tp_degree,
            padding_mask=padding_mask,
        )
        routed = self.experts.forward_prefill(
            normalized,
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
        )
        if self.tp_group.world_size > 1:
            routed = self.tp_group.reduce_scatter(routed, dim=0)

        shared = self.shared_experts.forward_prefill(
            hidden_states,
            norm_weight=norm_weight,
            eps=self.config.rms_norm_eps,
            tp_group=self.expert_tp_group,
        )
        return routed + shared
