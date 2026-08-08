# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_neuron.model.glm52_moe_dsa.parallelism import (
    ExpertPlanMeshMismatch,
    RoutedExpertPlan,
    assert_plan_matches_physical_mesh,
)
from vllm_neuron.parallel.neuron_parallel_state import _build_ep_group_ranks


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


# ---------------------------------------------------------------------------
# Plan-vs-PHYSICAL-MESH agreement.
#
# Everything above this line checks the ownership plan against itself, so it is
# green for any self-consistent plan -- including one that disagrees with the
# rank mesh the collectives actually run on. The tests below take their
# expectations from neuron_parallel_state._build_ep_group_ranks (the real mesh
# builder, imported, not reimplemented) so a mesh change cannot silently drift
# away from the plan.
#
# Mesh convention: rows are EP-TP groups, columns are EP groups. A rank's row
# index is its expert partition (== get_neuron_ep_rank()) and its column index
# is its position within that partition's TP subgroup
# (== get_neuron_ep_tp_group().rank_in_group).
# ---------------------------------------------------------------------------

TP64_EP_DEGREES = (
    pytest.param(
        8,
        marks=pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason=(
                "Known defect: at world_size=64, ep_degree=8 the row_size == 8 "
                "special case in _build_2d_mesh returns a non-contiguous "
                "Trainium2 mesh, which RoutedExpertPlan's arithmetic "
                "global_rank // expert_tp_degree cannot reproduce. 32 of 64 "
                "ranks get the wrong ep_rank and a different 32 get the wrong "
                "expert_tp_rank. Remove this xfail when RoutedExpertPlan is "
                "made mesh-aware; do not remove the test."
            ),
        ),
    ),
    16,  # as deployed
    32,
    64,
)


def _physical_mesh_indices(world_size: int, ep_degree: int) -> dict[int, tuple[int, int]]:
    """Map global rank -> (mesh row = ep_rank, mesh column = expert_tp_rank)."""

    ep_tp_group_ranks, _ = _build_ep_group_ranks(world_size, ep_degree)
    return {
        global_rank: (row_index, column_index)
        for row_index, row in enumerate(ep_tp_group_ranks)
        for column_index, global_rank in enumerate(row)
    }


@pytest.mark.parametrize("ep_degree", TP64_EP_DEGREES)
def test_plan_agrees_with_physical_rank_mesh(ep_degree: int) -> None:
    """Every rank's arithmetic ownership must equal its mesh position."""

    plan = RoutedExpertPlan(
        world_size=64,
        ep_degree=ep_degree,
        num_experts=256,
        expert_intermediate_size=2_048,
    )
    mesh = _physical_mesh_indices(plan.world_size, plan.ep_degree)

    assert sorted(mesh) == list(range(plan.world_size))

    disagreements = [
        (
            global_rank,
            (plan.ep_rank(global_rank), plan.expert_tp_rank(global_rank)),
            mesh[global_rank],
        )
        for global_rank in range(plan.world_size)
        if (plan.ep_rank(global_rank), plan.expert_tp_rank(global_rank))
        != mesh[global_rank]
    ]

    assert not disagreements, (
        f"EP{ep_degree}: {len(disagreements)} of {plan.world_size} ranks have a "
        "plan (ep_rank, expert_tp_rank) that differs from their physical mesh "
        f"(row, column). First five: {disagreements[:5]}"
    )


@pytest.mark.parametrize("ep_degree", TP64_EP_DEGREES)
def test_every_ep_tp_group_shares_one_expert_window(ep_degree: int) -> None:
    """The invariant build_blockwise_mapping actually depends on.

    Glm52SparseMlp hands the mesh-derived EP-TP group to build_blockwise_mapping
    as `moe_group`. That function shards mapping construction by
    `moe_group.rank_in_group` and stitches the shards with all_gather(dim=0) and
    an all_reduce(MAX) standing in for a disjoint union -- both of which are only
    valid if every rank in the group sliced the SAME expert window out of the
    world-gathered affinities. The window comes from the arithmetic
    `plan.ep_rank(...) * plan.experts_per_rank`, so this asserts that arithmetic
    is constant across each physical mesh row.
    """

    plan = RoutedExpertPlan(
        world_size=64,
        ep_degree=ep_degree,
        num_experts=256,
        expert_intermediate_size=2_048,
    )
    ep_tp_group_ranks, _ = _build_ep_group_ranks(plan.world_size, plan.ep_degree)

    split_rows = {
        row_index: sorted(
            {plan.ep_rank(r) * plan.experts_per_rank for r in row}
        )
        for row_index, row in enumerate(ep_tp_group_ranks)
        if len({plan.ep_rank(r) for r in row}) != 1
    }

    assert not split_rows, (
        f"EP{ep_degree}: {len(split_rows)} of {len(ep_tp_group_ranks)} EP-TP "
        "groups contain more than one expert window, so build_blockwise_mapping "
        "would stitch its mapping from mismatched halves on every rank in those "
        f"groups. Rows -> first_expert values: {split_rows}"
    )


def test_mesh_guard_names_both_values_and_explains_the_stakes() -> None:
    """The guard must be actionable enough that nobody just deletes it."""

    plan = RoutedExpertPlan(64, 8, 256, 2_048)
    mesh = _physical_mesh_indices(64, 8)
    mesh_ep_rank, mesh_expert_tp_rank = mesh[4]

    # Rank 4 sits in mesh row 1 but 4 // 8 == 0.
    assert plan.ep_rank(4) == 0
    assert mesh_ep_rank == 1

    with pytest.raises(ExpertPlanMeshMismatch) as excinfo:
        assert_plan_matches_physical_mesh(
            plan,
            4,
            mesh_ep_rank=mesh_ep_rank,
            mesh_expert_tp_rank=mesh_expert_tp_rank,
        )

    message = str(excinfo.value)
    assert "plan.ep_rank(4)" in message
    assert "get_neuron_ep_rank()" in message
    assert "build_blockwise_mapping" in message
    assert "MUST NOT DELETE" in message


def test_mesh_guard_is_a_tautology_on_the_deployed_ep16_path() -> None:
    """EP16 is what we serve; the guard must cost it nothing."""

    plan = RoutedExpertPlan(64, 16, 256, 2_048)
    mesh = _physical_mesh_indices(64, 16)

    for global_rank in range(plan.world_size):
        mesh_ep_rank, mesh_expert_tp_rank = mesh[global_rank]
        assert_plan_matches_physical_mesh(
            plan,
            global_rank,
            mesh_ep_rank=mesh_ep_rank,
            mesh_expert_tp_rank=mesh_expert_tp_rank,
        )
