# SPDX-License-Identifier: Apache-2.0
import importlib.util
import operator
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.fx import Graph, GraphModule

ROOT = Path(__file__).parents[3]
MODULE_NAME = "_rank_parameterized_capture_test_target"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME,
    ROOT / "vllm_neuron/compile/rank_parameterized_capture.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = MODULE
SPEC.loader.exec_module(MODULE)

EXPERIMENT_ENV = MODULE.EXPERIMENT_ENV
K3RankContract = MODULE.K3RankContract
OneCaptureBackend = MODULE.OneCaptureBackend
FxGraphHandoffArtifact = MODULE.FxGraphHandoffArtifact
RankBinding = MODULE.RankBinding
RankParameterizedPlan = MODULE.RankParameterizedPlan
RankReuseRejected = MODULE.RankReuseRejected
graph_semantic_digest = MODULE.graph_semantic_digest


AUDIT = "sha256:rank-arithmetic-on-tensor-dataflow"


class RankAwareModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4, dtype=torch.float32))

    def forward(self, value: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
        return value * self.weight + rank.to(torch.float32)


def k3_values(rank: int) -> dict[str, int]:
    return K3RankContract().expected(rank)


def binding_for(
    plan: RankParameterizedPlan,
    rank: int,
    *,
    weight: torch.Tensor | None = None,
    candidate_graph: GraphModule | None = None,
    semantic_values: dict[str, int] | None = None,
    replica_groups: dict[str, tuple[int, ...]] | None = None,
) -> RankBinding:
    inputs = list(_capture_inputs)
    inputs[0] = (
        torch.full_like(inputs[0], float(rank + 1), requires_grad=True)
        if weight is None
        else weight
    )
    inputs[-1] = torch.tensor(rank, dtype=torch.int32)
    return RankBinding(
        rank=rank,
        example_inputs=tuple(inputs),
        semantic_values=semantic_values or k3_values(rank),
        replica_groups=replica_groups or {},
        candidate_graph=candidate_graph,
        source_audit_digest=AUDIT,
    )


@pytest.fixture(autouse=True)
def enable_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXPERIMENT_ENV, "1")
    torch.compiler.reset()
    yield
    torch.compiler.reset()


_capture_inputs: tuple[torch.Tensor, ...]


def capture_plan() -> tuple[OneCaptureBackend, RankParameterizedPlan]:
    global _capture_inputs
    backend = OneCaptureBackend(
        # Dynamo lifts the parameter before the two user inputs.
        rank_input_index=2,
        replica_groups={},
        source_audit_digest=AUDIT,
    )
    model = RankAwareModel()
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    compiled(torch.ones(4), torch.tensor(0, dtype=torch.int32))
    assert backend.plan is not None
    _capture_inputs = (
        model.weight.detach().clone().requires_grad_(True),
        torch.ones(4),
        torch.tensor(0, dtype=torch.int32),
    )
    # Use the exact backend input ABI, including requires_grad and alias state.
    return backend, backend.plan


def equivalent_graph() -> GraphModule:
    graph = Graph()
    weight = graph.placeholder("weight")
    value = graph.placeholder("value")
    rank = graph.placeholder("rank")
    mul = graph.call_function(operator.mul, (value, weight))
    cast = graph.call_method("to", (rank, torch.float32))
    out = graph.call_function(operator.add, (mul, cast))
    graph.output(out)
    return GraphModule(nn.Module(), graph)


def changed_graph() -> GraphModule:
    graph = Graph()
    weight = graph.placeholder("weight")
    value = graph.placeholder("value")
    rank = graph.placeholder("rank")
    mul = graph.call_function(operator.mul, (value, weight))
    cast = graph.call_method("to", (rank, torch.float32))
    out = graph.call_function(operator.sub, (mul, cast))
    graph.output(out)
    return GraphModule(nn.Module(), graph)


def test_one_capture_lowers_64_rank_bindings_with_distinct_payloads() -> None:
    backend, plan = capture_plan()
    lowered: list[int] = []

    def lowerer(gm, inputs, binding):
        del gm, inputs
        lowered.append(binding.rank)
        return binding.rank

    results = plan.lower_all([binding_for(plan, rank) for rank in range(64)], lowerer)

    assert backend.capture_count == 1
    assert results == list(range(64))
    assert lowered == list(range(64))
    assert k3_values(47)["real_attention_heads"] == 2
    assert k3_values(48)["real_attention_heads"] == 0
    assert k3_values(63)["padded_attention_heads"] == 2


def test_default_off_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EXPERIMENT_ENV)
    with pytest.raises(RankReuseRejected, match="required"):
        MODULE.require_experiment_enabled()


def test_rank_formula_rejects_wrong_14x_expert_offset() -> None:
    _, plan = capture_plan()
    values = k3_values(7)
    values["expert_start"] += 14
    with pytest.raises(RankReuseRejected, match="semantic contract mismatch"):
        plan.validate(binding_for(plan, 7, semantic_values=values))


def test_padded_head_boundary_is_payload_only_but_shape_change_rejects() -> None:
    _, plan = capture_plan()
    plan.validate(binding_for(plan, 47))
    plan.validate(binding_for(plan, 48))

    inputs = list(binding_for(plan, 48).example_inputs)
    inputs[0] = torch.ones(5, dtype=inputs[0].dtype, requires_grad=True)
    changed = RankBinding(
        rank=48,
        example_inputs=tuple(inputs),
        semantic_values=k3_values(48),
        replica_groups={},
        source_audit_digest=AUDIT,
    )
    with pytest.raises(RankReuseRejected, match="input ABI changed"):
        plan.validate(changed)


def test_changed_graph_semantics_reject() -> None:
    _, plan = capture_plan()
    # A separately supplied candidate graph is an optional qualification
    # check. It must match the one captured by Dynamo exactly.
    with pytest.raises(RankReuseRejected, match="graph semantics changed"):
        plan.validate(binding_for(plan, 1, candidate_graph=changed_graph()))


def test_python_scalar_rank_rejects() -> None:
    _, plan = capture_plan()
    inputs = list(binding_for(plan, 1).example_inputs)
    inputs[-1] = 1
    binding = RankBinding(
        rank=1,
        example_inputs=tuple(inputs),
        semantic_values=k3_values(1),
        replica_groups={},
        source_audit_digest=AUDIT,
    )
    with pytest.raises(RankReuseRejected, match="rank input must be a tensor"):
        plan.validate(binding)


def test_replica_group_change_rejects() -> None:
    _, base = capture_plan()
    plan = RankParameterizedPlan(
        **{
            **base.__dict__,
            "replica_groups": {"world": tuple(range(64))},
        }
    )
    with pytest.raises(RankReuseRejected, match="replica groups changed"):
        plan.validate(binding_for(plan, 1, replica_groups={"world": tuple(range(63))}))


def test_alias_change_rejects() -> None:
    _, plan = capture_plan()
    binding = binding_for(plan, 1)
    inputs = list(binding.example_inputs)
    # Give input 1 a view into input 0 while preserving its tensor metadata.
    inputs[1] = inputs[0].detach()
    changed = RankBinding(
        rank=1,
        example_inputs=tuple(inputs),
        semantic_values=k3_values(1),
        replica_groups={},
        source_audit_digest=AUDIT,
    )
    with pytest.raises(RankReuseRejected, match="alias partition changed"):
        plan.validate(changed)


def test_fx_handoff_binds_distinct_rank_weight_payloads() -> None:
    _, plan = capture_plan()
    artifact = FxGraphHandoffArtifact.from_plan(plan)
    graph, inputs = artifact.bind(binding_for(plan, 17))

    assert isinstance(graph, GraphModule)
    assert inputs[0][0].item() == 18
    assert inputs[-1].item() == 17


def test_fx_handoff_rejects_tampered_graph_payload() -> None:
    _, plan = capture_plan()
    artifact = FxGraphHandoffArtifact.from_plan(plan)
    changed = replace(
        artifact,
        graph_pickle=artifact.graph_pickle + b"tampered",
    )
    with pytest.raises(RankReuseRejected, match="payload digest changed"):
        changed.restore_graph()


def test_cross_process_nki_handoff_rejects() -> None:
    _, plan = capture_plan()
    artifact = FxGraphHandoffArtifact.from_plan(
        replace(plan, nki_signatures=("nki-semantic-id",))
    )
    with pytest.raises(RankReuseRejected, match="kernel and constant-argument"):
        artifact.restore_graph(consumer_pid=artifact.producer_pid + 1)


def test_cross_process_collective_handoff_rejects() -> None:
    _, plan = capture_plan()
    artifact = FxGraphHandoffArtifact.from_plan(
        replace(plan, replica_groups={"tp:0": tuple(range(64))})
    )
    with pytest.raises(RankReuseRejected, match="process-group names"):
        artifact.restore_graph(consumer_pid=artifact.producer_pid + 1)


def test_process_independent_fx_handoff_restores() -> None:
    _, plan = capture_plan()
    artifact = FxGraphHandoffArtifact.from_plan(plan)
    graph = artifact.restore_graph(consumer_pid=artifact.producer_pid + 1)
    assert graph_semantic_digest(graph) == plan.graph_digest
