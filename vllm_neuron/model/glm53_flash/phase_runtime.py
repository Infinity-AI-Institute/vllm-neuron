"""CPU-testable adapter for the independently serialized GLM phases.

The retained TKG and CTE models expose the same resident ``NxDModel`` API,
but their generated ``initialize`` bodies use different SDK loaders. TKG
uses ``torch.classes.neuron.LayoutTransformation`` and CTE uses
``torch.ops.neuron._parallel_load``. This adapter calls only the public
serialized-model seam; the resident SDK performs the phase-local operation.

It does not import Torch, construct a device, or copy state between models.
It returns a host-verifiable state contract and refuses to pair phases whose
initialized KV-state key schemas differ.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PHASE_RUNTIME_SCHEMA = "glm53-phase-runtime-adapter-v1"
_EXPECTED_LOADER = {
    "tkg": "torch.classes.neuron.LayoutTransformation.forward(checkpoint, False)",
    "cte": "torch.ops.neuron._parallel_load(checkpoint)",
}


class Glm53PhaseRuntimeError(ValueError):
    """The resident phase models cannot form one exact state contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Glm53PhaseRuntimeError(message)


def _state_key_signature(state: Any) -> tuple[str, ...]:
    """Return the deterministic key schema without materializing tensor data."""

    _require(
        isinstance(state, Sequence) and not isinstance(state, (str, bytes)),
        "resident NxD state must be a sequence of rank dictionaries",
    )
    keys: list[str] = []
    for rank, row in enumerate(state):
        _require(
            isinstance(row, Mapping),
            f"resident NxD state rank {rank} is not a mapping",
        )
        for key in row:
            _require(
                isinstance(key, str),
                f"resident NxD state key at rank {rank} is not text",
            )
            keys.append(key)
    _require(keys, "resident NxD state is empty")
    _require(
        len(keys) == len(set(keys)),
        "resident NxD state contains duplicate keys",
    )
    return tuple(sorted(keys))


def _signature_sha256(keys: tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(keys) + "\n").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Glm53PhaseState:
    """Non-owning view of the state produced by one resident phase model."""

    phase: str
    loader_api: str
    rank_count: int
    state_keys: tuple[str, ...]
    state_keys_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "loader_api": self.loader_api,
            "rank_count": self.rank_count,
            "state_key_count": len(self.state_keys),
            "state_keys_sha256": self.state_keys_sha256,
        }


def _nxd_model(traced_model: Any) -> Any:
    model = getattr(traced_model, "nxd_model", None)
    _require(model is not None, "traced model lacks resident nxd_model")
    return model


def initialize_phase(
    traced_model: Any,
    phase: str,
    sharded_checkpoint: Any,
    *,
    use_saved_weights: bool = False,
) -> Glm53PhaseState:
    """Initialize one serialized phase through its resident NxD API."""

    _require(phase in _EXPECTED_LOADER, f"unsupported GLM phase: {phase}")
    nxd_model = _nxd_model(traced_model)
    method_name = "initialize_with_saved_weights" if use_saved_weights else "initialize"
    method = getattr(nxd_model, method_name, None)
    _require(callable(method), f"resident NxD model lacks {method_name}()")
    if use_saved_weights:
        method()
    else:
        method(sharded_checkpoint)
    state = getattr(nxd_model, "state", None)
    keys = _state_key_signature(state)
    return Glm53PhaseState(
        phase=phase,
        loader_api=_EXPECTED_LOADER[phase],
        rank_count=len(state),
        state_keys=keys,
        state_keys_sha256=_signature_sha256(keys),
    )


@dataclass(frozen=True)
class Glm53PairedPhaseRuntime:
    """Initialized CTE/TKG handles and their shared KV-state contract."""

    cte: Glm53PhaseState
    tkg: Glm53PhaseState

    @classmethod
    def initialize(
        cls,
        *,
        cte_model: Any,
        tkg_model: Any,
        cte_checkpoint: Any,
        tkg_checkpoint: Any,
    ) -> Glm53PairedPhaseRuntime:
        cte = initialize_phase(cte_model, "cte", cte_checkpoint)
        tkg = initialize_phase(tkg_model, "tkg", tkg_checkpoint)
        _require(
            cte.rank_count == tkg.rank_count,
            "CTE/TKG resident rank count differs",
        )
        _require(
            cte.state_keys == tkg.state_keys,
            "CTE/TKG resident KV-state keys differ",
        )
        return cls(cte=cte, tkg=tkg)

    def assert_continuation_state(self, state: Any) -> None:
        """Fail closed unless a device runner presents the retained schema."""

        keys = _state_key_signature(state)
        _require(
            keys == self.cte.state_keys,
            "continuation state schema differs from CTE/TKG",
        )

    def handoff_cte_outputs(
        self,
        runtime_owner: Any,
        outputs: Any,
        *,
        state_offset: int = 1,
    ) -> dict[str, Any]:
        """Copy CTE state through the wrapper's resident NxDI hook.

        The GLM wrapper already implements the supported CPU-simulation seam,
        ``_copy_past_key_values``.  This method makes the paired runner use
        that exact hook and verifies the destination phase-local cache layouts
        after the copy.  It deliberately does not assign ``nxd_model.state``:
        that object is the SDK's loader state, not the hybrid KDA/DSA runtime
        cache.  On device, input/output aliases perform this handoff in-place.
        """

        _require(
            type(state_offset) is int and state_offset >= 0,
            "state_offset must be a non-negative integer",
        )
        _require(
            isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)),
            "CTE outputs must be a logits-plus-state sequence",
        )
        copier = getattr(runtime_owner, "_copy_past_key_values", None)
        _require(
            callable(copier),
            "paired runtime owner lacks the resident _copy_past_key_values hook",
        )

        targets: list[tuple[str, list[Any]]] = []
        for phase_name in ("context_encoding_model", "token_generation_model"):
            phase_model = getattr(runtime_owner, phase_name, None)
            inner = getattr(phase_model, "model", None)
            cache = getattr(inner, "past_key_values", None)
            if cache is not None:
                targets.append((phase_name, list(cache)))
        _require(targets, "paired runtime owner exposes no phase-local KV cache")
        expected_count = len(targets[0][1])
        _require(expected_count > 0, "phase-local KV cache is empty")
        _require(
            all(len(cache) == expected_count for _, cache in targets),
            "CTE/TKG phase-local KV cache lengths differ",
        )
        new_state = list(outputs[state_offset:])
        _require(
            len(new_state) == expected_count,
            "CTE output state count does not match the phase-local KV cache",
        )
        for index, value in enumerate(new_state):
            _require(value is not None, f"CTE output state {index} is None")
            for phase_name, cache in targets:
                target = cache[index]
                _require(
                    getattr(target, "shape", None) == getattr(value, "shape", None),
                    f"{phase_name} state {index} shape differs from CTE output",
                )
                _require(
                    getattr(target, "dtype", None) == getattr(value, "dtype", None),
                    f"{phase_name} state {index} dtype differs from CTE output",
                )

        # The wrapper owns the actual SDK-compatible assignment and output
        # slicing (including captured-tensor and sampler offsets).
        copier(outputs)
        for index, value in enumerate(new_state):
            for phase_name, cache in targets:
                target = cache[index]
                target_data = getattr(target, "data", target)
                data_ptr = getattr(target_data, "data_ptr", None)
                value_ptr = getattr(value, "data_ptr", None)
                if callable(data_ptr) and callable(value_ptr):
                    _require(
                        data_ptr() == value_ptr(),
                        f"{phase_name} state {index} was not transferred by the resident hook",
                    )
        return {
            "state_offset": state_offset,
            "state_count": expected_count,
            "destination_phases": [name for name, _ in targets],
            "layout_verified": True,
            "continuation_state_transferred": True,
            "runtime_permitted": False,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": PHASE_RUNTIME_SCHEMA,
            "state_contract": {
                "rank_count": self.cte.rank_count,
                "state_key_count": len(self.cte.state_keys),
                "state_keys_sha256": self.cte.state_keys_sha256,
                "equal_cte_tkg": self.cte.state_keys == self.tkg.state_keys,
            },
            "cte": self.cte.to_mapping(),
            "tkg": self.tkg.to_mapping(),
            "claims": {
                "device_initialized": False,
                "continuation_state_transferred": False,
                "runtime_permitted": False,
                "correctness_40_of_40": False,
                "performance": False,
            },
        }


__all__ = [
    "PHASE_RUNTIME_SCHEMA",
    "Glm53PairedPhaseRuntime",
    "Glm53PhaseRuntimeError",
    "Glm53PhaseState",
    "initialize_phase",
]
