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


def _prefill_padding_mask(
    slot_mapping: torch.Tensor,
    *,
    num_tokens: int,
) -> torch.Tensor:
    local_slot_mapping = slot_mapping.reshape(-1)
    if local_slot_mapping.numel() != num_tokens:
        raise ValueError(
            "slot_mapping must contain one entry per rank-local prefill token"
        )
    return local_slot_mapping >= 0


def _mask_padded_affinities(
    expert_affinities: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    if expert_affinities.ndim != 2:
        raise ValueError("expert affinities must be a two-dimensional tensor")
    flat_padding_mask = padding_mask.reshape(-1)
    if flat_padding_mask.numel() != expert_affinities.shape[0]:
        raise ValueError("padding mask must contain one entry per routed token")
    return expert_affinities.masked_fill(
        ~flat_padding_mask.to(torch.bool).unsqueeze(1),
        0,
    )


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
        static_fp8: bool = False,
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
            topk_backend="neuron" if static_fp8 else "torch",
            device=device,
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
        slot_mapping: torch.Tensor,
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

        # The scheduler marks every padded token with PAD_SLOT_ID (-1).
        # Position IDs are not authoritative here: prefill padding repeats the
        # final real position, so a fully padded SP rank can contain only that
        # repeated value and cannot infer that all of its tokens are invalid.
        padding_mask = _prefill_padding_mask(
            slot_mapping,
            num_tokens=hidden_states.shape[0],
        )

        normalized = normalized_local
        affinities = affinities_local
        if self.tp_group.world_size > 1:
            normalized = self.tp_group.all_gather(normalized, dim=0)
            affinities = self.tp_group.all_gather(affinities, dim=0)
            padding_mask = self.tp_group.all_gather(padding_mask, dim=0)

        first_expert = self.ep_rank * self.plan.experts_per_rank
        local_affinities = affinities[
            :,
            first_expert : first_expert + self.plan.experts_per_rank,
        ]
        # build_blockwise_mapping derives its token/expert membership from the
        # affinities it receives before applying its optional padding mask.
        # Zero invalid rows here so padded token IDs never enter that mapping.
        local_affinities = _mask_padded_affinities(
            local_affinities,
            padding_mask,
        )
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
            hidden_states_normalized=(
                normalized_local if not self.shared_experts.static_fp8 else None
            ),
        )
        return routed + shared
