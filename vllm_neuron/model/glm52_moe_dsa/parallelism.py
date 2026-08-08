# SPDX-License-Identifier: Apache-2.0
"""Deterministic routed-expert ownership for hybrid EP+TP."""

from dataclasses import dataclass


class ExpertPlanMeshMismatch(RuntimeError):
    """The arithmetic ownership plan disagrees with the physical Neuron mesh."""


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


def physical_mesh_ep_ranks() -> tuple[int, int] | None:
    """Return ``(ep_rank, expert_tp_rank)`` as the *physical* Neuron mesh sees them.

    Returns ``None`` when the Neuron EP process groups have not been initialized
    (single-process construction: unit tests, checkpoint conversion, probes).
    In that case there is no physical mesh to disagree with.

    ``neuron_parallel_state._build_2d_mesh`` lays the world out as a 2-D mesh
    whose *rows* become the EP-TP groups and whose *columns* become the EP
    groups.  A rank's mesh row index therefore **is** its expert partition, and
    its column index is its position inside that partition's TP subgroup.  Those
    two indices are exactly ``get_neuron_ep_group().rank_in_group`` (which is
    what ``get_neuron_ep_rank()`` returns) and
    ``get_neuron_ep_tp_group().rank_in_group``.

    The mesh is **not** always the naive contiguous one: on a 64-rank Trainium2
    with ``row_size == 8`` (i.e. ``ep_degree == 8``) ``_build_2d_mesh`` returns a
    hard-coded topology-aware mesh whose rows are non-contiguous, e.g. row 0 is
    ``[0, 1, 2, 3, 12, 13, 14, 15]``.  ``RoutedExpertPlan`` derives ownership
    arithmetically instead, which reproduces only the contiguous ``else`` branch.
    """

    try:
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_ep_group,
            get_neuron_ep_tp_group,
        )

        ep_group = get_neuron_ep_group()
        ep_tp_group = get_neuron_ep_tp_group()
    except (ImportError, AssertionError):
        return None
    return ep_group.rank_in_group, ep_tp_group.rank_in_group


def assert_plan_matches_physical_mesh(
    plan: RoutedExpertPlan,
    global_rank: int,
    *,
    mesh_ep_rank: int,
    mesh_expert_tp_rank: int,
) -> None:
    """Fail fast when routed-expert ownership disagrees with the physical mesh.

    ``plan`` decides *which experts' weights this rank owns*; the mesh decides
    *which ranks collectively rebuild one expert's mapping*.  Both are live on
    the prefill routed path and nothing else ties them together, so this is the
    only place the two conventions are ever compared.

    Raises:
        ExpertPlanMeshMismatch: if either index disagrees.
    """

    plan_ep_rank = plan.ep_rank(global_rank)
    plan_expert_tp_rank = plan.expert_tp_rank(global_rank)
    if plan_ep_rank == mesh_ep_rank and plan_expert_tp_rank == mesh_expert_tp_rank:
        return

    first_expert = plan_ep_rank * plan.experts_per_rank
    mesh_first_expert = mesh_ep_rank * plan.experts_per_rank
    raise ExpertPlanMeshMismatch(
        "GLM-5.2 routed-expert ownership disagrees with the physical Neuron mesh "
        f"on global rank {global_rank}.\n"
        f"  plan.ep_rank({global_rank})                       = {plan_ep_rank}   "
        f"(arithmetic: global_rank // expert_tp_degree)\n"
        f"  get_neuron_ep_rank()                       = {mesh_ep_rank}   "
        "(physical: this rank's row in the Neuron 2-D mesh)\n"
        f"  plan.expert_tp_rank({global_rank})                = "
        f"{plan_expert_tp_rank}   (arithmetic: global_rank % expert_tp_degree)\n"
        f"  get_neuron_ep_tp_group().rank_in_group      = {mesh_expert_tp_rank}   "
        "(physical: this rank's column in that row)\n"
        f"  world_size={plan.world_size} ep_degree={plan.ep_degree} "
        f"expert_tp_degree={plan.expert_tp_degree} "
        f"experts_per_rank={plan.experts_per_rank}\n"
        f"  This rank would load experts [{first_expert}, "
        f"{first_expert + plan.experts_per_rank}) but the mesh places it in the "
        f"group that rebuilds experts [{mesh_first_expert}, "
        f"{mesh_first_expert + plan.experts_per_rank}).\n"
        "\n"
        "WHY THIS IS FATAL, AND WHY YOU MUST NOT DELETE THIS CHECK TO GET PAST "
        "STARTUP:\n"
        "  Glm52SparseMlp.forward_prefill slices its local expert window out of "
        "the\n"
        "  world-gathered affinities using the ARITHMETIC ep_rank "
        "(`first_expert`), but\n"
        "  it hands the MESH-DERIVED EP-TP group to build_blockwise_mapping as "
        "`moe_group`.\n"
        "  build_blockwise_mapping sub-shards mapping construction by "
        "`moe_group.rank_in_group`\n"
        "  and stitches the shards back with all_gather(dim=0) plus an "
        "all_reduce(MAX) that\n"
        "  stands in for a disjoint union.  Both steps assume every rank in the "
        "group passed\n"
        "  the SAME expert window.  When the two conventions disagree, a group "
        "holds two\n"
        "  different windows and the stitched mapping sends tokens to the wrong "
        "experts --\n"
        "  for every rank in that group, not just the mismatched ones.\n"
        "\n"
        "  There is no runtime signal for this. The expert-index range checks in "
        "expert_kernels.py\n"
        "  and attention.py are gated on `not torch.compiler.is_compiling()` and "
        "are therefore\n"
        "  absent from the traced NEFF, and cache_ops.py clamps out-of-range "
        "positions instead of\n"
        "  raising. The failure mode is plausible-looking but wrong output, not "
        "a crash. That is\n"
        "  precisely why this must be a startup failure.\n"
        "\n"
        "KNOWN TRIGGER: ep_degree=8 on a 64-rank Trainium2. "
        "neuron_parallel_state._build_2d_mesh\n"
        "  applies a topology-aware non-contiguous mesh when "
        "`not VLLM_NEURON_SWITCH_CC and total == 64\n"
        "  and row_size == 8`; row_size is world_size // ep_degree, so EP8 is "
        "the only reachable\n"
        "  point where it fires. EP16 (as deployed), EP32 and EP64 all use the "
        "contiguous branch\n"
        "  and agree with the arithmetic plan on every rank.\n"
        "\n"
        "FIX, DO NOT SUPPRESS: run at ep_degree=16 (the deployed, unaffected "
        "configuration), or\n"
        "  make RoutedExpertPlan consult the mesh -- i.e. take ep_rank from "
        "get_neuron_ep_rank()\n"
        "  and expert_tp_rank from get_neuron_ep_tp_group().rank_in_group, the "
        "convention AWS's\n"
        "  own gpt_oss reference models already use. That change reassigns "
        "rank->weight ownership\n"
        "  and needs accelerator re-verification including a fresh weight load."
    )
