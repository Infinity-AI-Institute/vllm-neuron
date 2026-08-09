# SPDX-License-Identifier: Apache-2.0
"""Fail-closed prototype for reusing one Dynamo/FX capture across ranks.

This module intentionally has no integration with the production capture
backend.  It qualifies the narrow boundary between Dynamo capture and
rank-specific lowering, and stays disabled unless explicitly enabled.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

EXPERIMENT_ENV = "VLLM_NEURON_EXPERIMENTAL_RANK_PARAMETERIZED_CAPTURE"


class RankReuseRejected(RuntimeError):
    """Raised when a candidate rank cannot safely reuse the captured FX graph."""


def require_experiment_enabled(environ: Mapping[str, str] | None = None) -> None:
    """Require an exact opt-in; ambiguous truthy values remain disabled."""

    value = (os.environ if environ is None else environ).get(EXPERIMENT_ENV, "0")
    if value != "1":
        raise RankReuseRejected(
            f"{EXPERIMENT_ENV}=1 is required; rank-parameterized capture is "
            "an unqualified experiment"
        )


def _target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", type(target).__module__)
    qualname = getattr(target, "__qualname__", None)
    if qualname is None:
        qualname = getattr(target, "__name__", type(target).__qualname__)
    return f"{module}.{qualname}:{target}"


def _resolve_attr(root: Any, target: str) -> Any:
    value = root
    for atom in target.split("."):
        value = getattr(value, atom)
    return value


def _canonical(value: Any, node_ids: Mapping[torch.fx.Node, int]) -> Any:
    if isinstance(value, torch.fx.Node):
        return {"node": node_ids[value]}
    if isinstance(value, torch.Tensor):
        # Tensor literals are not expected in node arguments, but retaining a
        # metadata identity here prevents silently treating them as scalars.
        return {"tensor_literal": TensorABI.from_value(value).to_json()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item, node_ids) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical(item, node_ids)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, slice):
        return {
            "slice": [
                _canonical(value.start, node_ids),
                _canonical(value.stop, node_ids),
                _canonical(value.step, node_ids),
            ]
        }
    if isinstance(value, torch.dtype):
        return {"dtype": str(value)}
    if isinstance(value, torch.device):
        return {"device": str(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def graph_semantic_payload(gm: torch.fx.GraphModule) -> dict[str, Any]:
    """Return a name-insensitive semantic description of an FX graph.

    Tensor-valued ``get_attr`` nodes are rejected.  Rank-local model tensors
    must be lifted placeholders so rank payloads can vary without becoming HLO
    constants.
    """

    nodes = list(gm.graph.nodes)
    node_ids = {node: index for index, node in enumerate(nodes)}
    result: list[dict[str, Any]] = []
    for node in nodes:
        target = (
            f"<{node.op}>"
            if node.op in ("placeholder", "output")
            else _target_name(node.target)
        )
        entry = {
            "op": node.op,
            "target": target,
            "args": _canonical(node.args, node_ids),
            "kwargs": _canonical(node.kwargs, node_ids),
        }
        if node.op == "get_attr":
            constant = _resolve_attr(gm, str(node.target))
            if isinstance(constant, torch.Tensor):
                raise RankReuseRejected(
                    f"tensor-valued get_attr {node.target!s} is rank-reuse unsafe; "
                    "lift it to an FX placeholder"
                )
            entry["constant"] = _canonical(constant, node_ids)
        result.append(entry)
    return {"nodes": result}


def graph_semantic_digest(gm: torch.fx.GraphModule) -> str:
    payload = json.dumps(
        graph_semantic_payload(gm), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TensorABI:
    shape: tuple[int, ...]
    dtype: str
    stride: tuple[int, ...]
    layout: str
    device_type: str
    requires_grad: bool

    @classmethod
    def from_value(cls, value: torch.Tensor) -> TensorABI:
        return cls(
            shape=tuple(int(dim) for dim in value.shape),
            dtype=str(value.dtype),
            stride=tuple(int(dim) for dim in value.stride()),
            layout=str(value.layout),
            device_type=value.device.type,
            requires_grad=bool(value.requires_grad),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "dtype": self.dtype,
            "stride": self.stride,
            "layout": self.layout,
            "device_type": self.device_type,
            "requires_grad": self.requires_grad,
        }


@dataclass(frozen=True)
class ValueABI:
    tensor: TensorABI | None
    scalar_type: str | None
    scalar_value: str | None

    @classmethod
    def from_value(cls, value: Any, *, rank_input: bool = False) -> ValueABI:
        if isinstance(value, torch.Tensor):
            return cls(TensorABI.from_value(value), None, None)
        # A rank value is only accepted as a tensor.  Every other Python value
        # is frozen because Dynamo may have specialized control flow on it.
        if rank_input:
            raise RankReuseRejected("rank input must be a tensor, not a Python scalar")
        return cls(
            None,
            f"{type(value).__module__}.{type(value).__qualname__}",
            repr(value),
        )


def _storage_identity(value: torch.Tensor) -> tuple[int, int] | None:
    try:
        return (int(value.untyped_storage()._cdata), int(value.storage_offset()))
    except (AttributeError, RuntimeError, TypeError):
        return None


def alias_partition(values: Sequence[Any]) -> tuple[tuple[int, int] | None, ...]:
    """Canonicalize tensor storage aliases without retaining storage IDs."""

    aliases: dict[int, int] = {}
    result: list[tuple[int, int] | None] = []
    for value in values:
        if not isinstance(value, torch.Tensor):
            result.append(None)
            continue
        identity = _storage_identity(value)
        if identity is None:
            # Unknown alias state is unsafe: accepting it would miss K/V cache
            # or recurrent-state alias changes.
            raise RankReuseRejected("cannot determine tensor input alias identity")
        storage_id, offset = identity
        alias_id = aliases.setdefault(storage_id, len(aliases))
        result.append((alias_id, offset))
    return tuple(result)


def _collective_group_names(gm: torch.fx.GraphModule) -> tuple[str, ...]:
    groups: list[str] = []
    for node in gm.graph.nodes:
        target = _target_name(node.target)
        if "collective" not in target and "_c10d_functional" not in target:
            continue
        strings = [item for item in node.args if isinstance(item, str)]
        strings.extend(item for item in node.kwargs.values() if isinstance(item, str))
        if strings:
            group_name = strings[-1]
            if group_name not in groups:
                groups.append(group_name)
    return tuple(groups)


def _nki_node_signatures(gm: torch.fx.GraphModule) -> tuple[str, ...]:
    signatures: list[str] = []
    node_ids = {node: index for index, node in enumerate(gm.graph.nodes)}
    for node in gm.graph.nodes:
        if "nki_kernel_wrapper" not in _target_name(node.target):
            continue
        payload = {
            "target": _target_name(node.target),
            "grid": _canonical(node.kwargs.get("grid"), node_ids),
            "arg_names": _canonical(node.kwargs.get("arg_names"), node_ids),
            "constant_args_key": _canonical(
                node.kwargs.get("constant_args_key"), node_ids
            ),
            # The integrated 342a388 cache normalizes registry indices, but a
            # reuse plan remains process-local and verifies the complete NKI
            # call signature before lowering.
            "kernel_idx": _canonical(node.kwargs.get("kernel_idx"), node_ids),
            "backend_config_sha256": hashlib.sha256(
                str(node.kwargs.get("backend_config", "")).encode()
            ).hexdigest(),
        }
        signatures.append(
            hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
    return tuple(signatures)


@dataclass(frozen=True)
class K3RankContract:
    world_size: int = 64
    experts_per_rank: int = 14
    local_vocab_size: int = 2560
    real_attention_heads: int = 96
    padded_attention_heads: int = 128

    @property
    def digest(self) -> str:
        return hashlib.sha256(repr(self).encode()).hexdigest()

    def expected(self, rank: int) -> dict[str, int]:
        if not 0 <= rank < self.world_size:
            raise RankReuseRejected(f"rank {rank} is outside [0, {self.world_size})")
        heads_per_rank = self.padded_attention_heads // self.world_size
        real_local_heads = max(
            0,
            min(heads_per_rank, self.real_attention_heads - rank * heads_per_rank),
        )
        expert_start = rank * self.experts_per_rank
        vocab_start = rank * self.local_vocab_size
        return {
            "ep_rank": rank,
            "expert_start": expert_start,
            "expert_end": expert_start + self.experts_per_rank,
            "expert_count": self.experts_per_rank,
            "vocab_start": vocab_start,
            "vocab_end": vocab_start + self.local_vocab_size,
            "local_vocab_size": self.local_vocab_size,
            "real_attention_heads": real_local_heads,
            "padded_attention_heads": heads_per_rank - real_local_heads,
        }

    def validate(self, rank: int, values: Mapping[str, int]) -> None:
        expected = self.expected(rank)
        missing = sorted(set(expected) - set(values))
        if missing:
            raise RankReuseRejected(f"rank {rank} semantic fields missing: {missing}")
        mismatches = {
            key: (expected_value, values[key])
            for key, expected_value in expected.items()
            if values[key] != expected_value
        }
        if mismatches:
            raise RankReuseRejected(
                f"rank {rank} semantic contract mismatch: {mismatches}"
            )


@dataclass(frozen=True)
class RankBinding:
    rank: int
    example_inputs: tuple[Any, ...]
    semantic_values: Mapping[str, int]
    replica_groups: Mapping[str, tuple[int, ...]]
    candidate_graph: torch.fx.GraphModule | None = None
    source_audit_digest: str = ""


@dataclass(frozen=True)
class RankParameterizedPlan:
    graph_module: torch.fx.GraphModule
    graph_digest: str
    input_abi: tuple[ValueABI, ...]
    input_aliases: tuple[tuple[int, int] | None, ...]
    rank_input_index: int
    replica_groups: Mapping[str, tuple[int, ...]]
    nki_signatures: tuple[str, ...]
    source_audit_digest: str
    contract: K3RankContract
    owner_pid: int

    @classmethod
    def build(
        cls,
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
        *,
        rank_input_index: int,
        replica_groups: Mapping[str, Sequence[int]],
        source_audit_digest: str,
        contract: K3RankContract | None = None,
    ) -> RankParameterizedPlan:
        require_experiment_enabled()
        if not source_audit_digest:
            raise RankReuseRejected(
                "a non-empty source audit digest is required to attest that every "
                "rank-sensitive Python value was moved onto tensor dataflow"
            )
        values = tuple(example_inputs)
        index = (
            rank_input_index
            if rank_input_index >= 0
            else len(values) + rank_input_index
        )
        if not 0 <= index < len(values):
            raise RankReuseRejected("rank input index is outside example_inputs")
        rank_value = values[index]
        if not isinstance(rank_value, torch.Tensor):
            raise RankReuseRejected("rank input must be a tensor")
        if rank_value.numel() != 1 or rank_value.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise RankReuseRejected("rank input must be a one-element integer tensor")

        placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
        if len(placeholders) != len(values):
            raise RankReuseRejected(
                "FX placeholder count does not match the flattened input count"
            )
        if not placeholders[index].users:
            raise RankReuseRejected("rank tensor placeholder has no FX dataflow users")

        observed_groups = _collective_group_names(gm)
        normalized_groups = {
            name: tuple(int(rank) for rank in ranks)
            for name, ranks in replica_groups.items()
        }
        missing_groups = sorted(set(observed_groups) - set(normalized_groups))
        if missing_groups:
            raise RankReuseRejected(
                f"collective replica groups are unresolved: {missing_groups}"
            )

        return cls(
            graph_module=gm,
            graph_digest=graph_semantic_digest(gm),
            input_abi=tuple(
                ValueABI.from_value(value, rank_input=(offset == index))
                for offset, value in enumerate(values)
            ),
            input_aliases=alias_partition(values),
            rank_input_index=index,
            replica_groups=normalized_groups,
            nki_signatures=_nki_node_signatures(gm),
            source_audit_digest=source_audit_digest,
            contract=contract or K3RankContract(),
            owner_pid=os.getpid(),
        )

    def validate(self, binding: RankBinding) -> None:
        require_experiment_enabled()
        if os.getpid() != self.owner_pid:
            raise RankReuseRejected(
                "cross-process FX/NKI registry reuse is not qualified; lower in the "
                "capture process or implement an explicit registry-safe transport"
            )
        if binding.source_audit_digest != self.source_audit_digest:
            raise RankReuseRejected("rank-sensitive source audit digest changed")
        self.contract.validate(binding.rank, binding.semantic_values)
        values = tuple(binding.example_inputs)
        if len(values) != len(self.input_abi):
            raise RankReuseRejected("candidate input count changed")
        candidate_abi = tuple(
            ValueABI.from_value(value, rank_input=(offset == self.rank_input_index))
            for offset, value in enumerate(values)
        )
        if candidate_abi != self.input_abi:
            raise RankReuseRejected("candidate tensor/scalar input ABI changed")
        if alias_partition(values) != self.input_aliases:
            raise RankReuseRejected("candidate input alias partition changed")

        rank_value = values[self.rank_input_index]
        try:
            materialized_rank = int(rank_value.detach().cpu().reshape(()).item())
        except (RuntimeError, TypeError):
            materialized_rank = None
        if materialized_rank is not None and materialized_rank != binding.rank:
            raise RankReuseRejected(
                f"rank tensor contains {materialized_rank}, binding says {binding.rank}"
            )

        normalized_groups = {
            name: tuple(int(rank) for rank in ranks)
            for name, ranks in binding.replica_groups.items()
        }
        if normalized_groups != self.replica_groups:
            raise RankReuseRejected("collective replica groups changed")

        if binding.candidate_graph is not None:
            if graph_semantic_digest(binding.candidate_graph) != self.graph_digest:
                raise RankReuseRejected("candidate FX graph semantics changed")
            if _nki_node_signatures(binding.candidate_graph) != self.nki_signatures:
                raise RankReuseRejected("candidate NKI call semantics changed")

    def lower_all(
        self,
        bindings: Sequence[RankBinding],
        lowerer: Callable[[torch.fx.GraphModule, tuple[Any, ...], RankBinding], Any],
    ) -> list[Any]:
        """Validate each rank, clone the mutable FX graph, and lower sequentially."""

        results: list[Any] = []
        seen: set[int] = set()
        for binding in bindings:
            if binding.rank in seen:
                raise RankReuseRejected(f"duplicate rank binding {binding.rank}")
            self.validate(binding)
            seen.add(binding.rank)
            results.append(
                lowerer(
                    copy.deepcopy(self.graph_module), binding.example_inputs, binding
                )
            )
        return results


@dataclass(frozen=True)
class FxGraphHandoffArtifact:
    """Serialized FX plus the minimum fail-closed lowering contract.

    The pickle is a trusted-local artifact, not an interchange or security
    boundary.  NKI and collective graphs are intentionally process-affine:
    their Python registries/process groups are not contained in GraphModule.
    """

    schema_version: int
    producer_pid: int
    graph_pickle: bytes
    graph_pickle_sha256: str
    graph_digest: str
    input_abi: tuple[ValueABI, ...]
    input_aliases: tuple[tuple[int, int] | None, ...]
    rank_input_index: int
    replica_groups: Mapping[str, tuple[int, ...]]
    nki_signatures: tuple[str, ...]
    source_audit_digest: str
    contract: K3RankContract

    @classmethod
    def from_plan(cls, plan: RankParameterizedPlan) -> FxGraphHandoffArtifact:
        require_experiment_enabled()
        try:
            graph_pickle = pickle.dumps(
                plan.graph_module,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except (AttributeError, pickle.PicklingError, TypeError) as error:
            raise RankReuseRejected(
                f"FX GraphModule is not serializable: {type(error).__name__}: {error}"
            ) from error
        return cls(
            schema_version=1,
            producer_pid=plan.owner_pid,
            graph_pickle=graph_pickle,
            graph_pickle_sha256=hashlib.sha256(graph_pickle).hexdigest(),
            graph_digest=plan.graph_digest,
            input_abi=plan.input_abi,
            input_aliases=plan.input_aliases,
            rank_input_index=plan.rank_input_index,
            replica_groups=dict(plan.replica_groups),
            nki_signatures=plan.nki_signatures,
            source_audit_digest=plan.source_audit_digest,
            contract=plan.contract,
        )

    def _validate_transport(self, consumer_pid: int) -> None:
        if self.schema_version != 1:
            raise RankReuseRejected(
                f"unsupported FX handoff schema {self.schema_version}"
            )
        if hashlib.sha256(self.graph_pickle).hexdigest() != self.graph_pickle_sha256:
            raise RankReuseRejected("FX handoff graph payload digest changed")
        if consumer_pid == self.producer_pid:
            return
        blockers: list[str] = []
        if self.nki_signatures:
            blockers.append(
                "NKI HOP nodes depend on the producer's process-global kernel and "
                "constant-argument registries"
            )
        if self.replica_groups:
            blockers.append(
                "collective nodes depend on producer-local process-group names and "
                "registrations"
            )
        if blockers:
            raise RankReuseRejected(
                "cross-process FX handoff is unsafe: " + "; ".join(blockers)
            )

    def restore_graph(
        self,
        *,
        consumer_pid: int | None = None,
    ) -> torch.fx.GraphModule:
        require_experiment_enabled()
        resolved_pid = os.getpid() if consumer_pid is None else consumer_pid
        self._validate_transport(resolved_pid)
        try:
            gm = pickle.loads(self.graph_pickle)
        except Exception as error:
            raise RankReuseRejected(
                f"FX GraphModule could not be restored: {type(error).__name__}: {error}"
            ) from error
        if not isinstance(gm, torch.fx.GraphModule):
            raise RankReuseRejected("FX handoff did not restore a GraphModule")
        if graph_semantic_digest(gm) != self.graph_digest:
            raise RankReuseRejected("restored FX graph semantics changed")
        if _nki_node_signatures(gm) != self.nki_signatures:
            raise RankReuseRejected("restored NKI call semantics changed")
        if set(_collective_group_names(gm)) - set(self.replica_groups):
            raise RankReuseRejected("restored collective group is unresolved")
        return gm

    def bind(
        self,
        binding: RankBinding,
    ) -> tuple[torch.fx.GraphModule, tuple[Any, ...]]:
        """Restore and validate rank inputs immediately before FX-to-HLO."""

        gm = self.restore_graph()
        plan = RankParameterizedPlan(
            graph_module=gm,
            graph_digest=self.graph_digest,
            input_abi=self.input_abi,
            input_aliases=self.input_aliases,
            rank_input_index=self.rank_input_index,
            replica_groups=self.replica_groups,
            nki_signatures=self.nki_signatures,
            source_audit_digest=self.source_audit_digest,
            contract=self.contract,
            owner_pid=os.getpid(),
        )
        plan.validate(binding)
        return gm, tuple(binding.example_inputs)


class OneCaptureBackend:
    """Torch backend that records exactly one FX graph for CPU qualification."""

    def __init__(
        self,
        *,
        rank_input_index: int,
        replica_groups: Mapping[str, Sequence[int]],
        source_audit_digest: str,
        contract: K3RankContract | None = None,
    ) -> None:
        self.rank_input_index = rank_input_index
        self.replica_groups = replica_groups
        self.source_audit_digest = source_audit_digest
        self.contract = contract
        self.capture_count = 0
        self.plan: RankParameterizedPlan | None = None

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[Any]
    ) -> Callable[..., Any]:
        self.capture_count += 1
        if self.capture_count != 1:
            raise RankReuseRejected("Dynamo attempted more than one FX capture")
        self.plan = RankParameterizedPlan.build(
            gm,
            example_inputs,
            rank_input_index=self.rank_input_index,
            replica_groups=self.replica_groups,
            source_audit_digest=self.source_audit_digest,
            contract=self.contract,
        )
        return gm.forward
