# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan


@pytest.mark.parametrize(
    ("ep_degree", "expert_tp", "experts_per_rank", "intermediate_per_rank"),
    (
        (8, 8, 32, 256),
        (16, 4, 16, 512),
        (32, 2, 8, 1_024),
        (64, 1, 4, 2_048),
    ),
)
def test_tp64_ep_ownership(
    ep_degree: int,
    expert_tp: int,
    experts_per_rank: int,
    intermediate_per_rank: int,
) -> None:
    plan = RoutedExpertPlan(
        world_size=64,
        ep_degree=ep_degree,
        num_experts=256,
        expert_intermediate_size=2_048,
    )

    assert plan.expert_tp_degree == expert_tp
    assert plan.experts_per_rank == experts_per_rank
    assert plan.intermediate_per_rank == intermediate_per_rank

    owners = {
        expert_id: set()
        for expert_id in range(plan.num_experts)
    }
    for rank in range(plan.world_size):
        for expert_id in plan.local_expert_ids(rank):
            owners[expert_id].add(rank)

    assert all(len(ranks) == expert_tp for ranks in owners.values())
    assert set().union(*owners.values()) == set(range(plan.world_size))


def test_tp_shards_of_one_ep_partition_are_disjoint() -> None:
    plan = RoutedExpertPlan(64, 16, 256, 2_048)

    assert plan.local_expert_ids(0) == plan.local_expert_ids(3)
    assert set(plan.local_expert_ids(0)).isdisjoint(plan.local_expert_ids(4))
    assert plan.expert_tp_rank(3) == 3
    assert plan.ep_rank(4) == 1
