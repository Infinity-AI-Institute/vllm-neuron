# SPDX-License-Identifier: Apache-2.0
"""Deterministic routed-expert ownership for hybrid EP+TP."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutedExpertPlan:
    world_size: int
    ep_degree: int
    num_experts: int
    expert_intermediate_size: int

    def __post_init__(self) -> None:
        if self.world_size <= 0 or self.ep_degree <= 0:
            raise ValueError("world_size and ep_degree must be positive")
        if self.world_size % self.ep_degree:
            raise ValueError("world_size must be divisible by ep_degree")
        if self.num_experts % self.ep_degree:
            raise ValueError("num_experts must be divisible by ep_degree")
        if self.expert_intermediate_size % self.expert_tp_degree:
            raise ValueError(
                "expert intermediate size must be divisible by expert TP degree"
            )

    @property
    def expert_tp_degree(self) -> int:
        return self.world_size // self.ep_degree

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.ep_degree

    @property
    def intermediate_per_rank(self) -> int:
        return self.expert_intermediate_size // self.expert_tp_degree

    def ep_rank(self, global_rank: int) -> int:
        self._validate_rank(global_rank)
        return global_rank // self.expert_tp_degree

    def expert_tp_rank(self, global_rank: int) -> int:
        self._validate_rank(global_rank)
        return global_rank % self.expert_tp_degree

    def local_expert_ids(self, global_rank: int) -> tuple[int, ...]:
        first_expert = self.ep_rank(global_rank) * self.experts_per_rank
        return tuple(range(first_expert, first_expert + self.experts_per_rank))

    def _validate_rank(self, global_rank: int) -> None:
        if not 0 <= global_rank < self.world_size:
            raise ValueError(f"global rank {global_rank} is outside the world")
