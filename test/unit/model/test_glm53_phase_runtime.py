from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "vllm_neuron/model/glm53_flash/phase_runtime.py"
SPEC = importlib.util.spec_from_file_location("glm53_phase_runtime", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _ResidentNxD:
    def __init__(self, state):
        self.state = state
        self.calls: list[tuple[str, object]] = []

    def initialize(self, checkpoint):
        self.calls.append(("initialize", checkpoint))

    def initialize_with_saved_weights(self):
        self.calls.append(("initialize_with_saved_weights", None))


class _Traced:
    def __init__(self, state):
        self.nxd_model = _ResidentNxD(state)


def _state():
    return [{"past_key_values.0": object()}, {"past_key_values.1": object()}]


def test_paired_adapter_calls_resident_api_and_binds_state_schema():
    cte = _Traced(_state())
    tkg = _Traced(_state())
    pair = MODULE.Glm53PairedPhaseRuntime.initialize(
        cte_model=cte,
        tkg_model=tkg,
        cte_checkpoint="cte-rank-checkpoint",
        tkg_checkpoint="tkg-rank-checkpoint",
    )
    assert cte.nxd_model.calls == [("initialize", "cte-rank-checkpoint")]
    assert tkg.nxd_model.calls == [("initialize", "tkg-rank-checkpoint")]
    assert pair.cte.loader_api == "torch.ops.neuron._parallel_load(checkpoint)"
    assert "LayoutTransformation" in pair.tkg.loader_api
    assert pair.cte.state_keys == pair.tkg.state_keys
    pair.assert_continuation_state(_state())
    assert pair.to_mapping()["claims"]["runtime_permitted"] is False


def test_saved_weights_path_is_explicit_and_does_not_accept_checkpoint():
    model = _Traced(_state())
    state = MODULE.initialize_phase(model, "tkg", None, use_saved_weights=True)
    assert model.nxd_model.calls == [("initialize_with_saved_weights", None)]
    assert state.rank_count == 2


def test_pair_rejects_kv_state_key_drift_before_runtime():
    cte = _Traced(_state())
    tkg = _Traced([{"past_key_values.0": object()}, {"past_key_values.2": object()}])
    with pytest.raises(MODULE.Glm53PhaseRuntimeError, match="KV-state keys"):
        MODULE.Glm53PairedPhaseRuntime.initialize(
            cte_model=cte,
            tkg_model=tkg,
            cte_checkpoint=object(),
            tkg_checkpoint=object(),
        )


def test_adapter_rejects_bad_continuation_state_and_missing_api():
    model = _Traced(_state())
    model.nxd_model.initialize = None
    with pytest.raises(MODULE.Glm53PhaseRuntimeError, match="lacks initialize"):
        MODULE.initialize_phase(model, "cte", object())

    cte = _Traced(_state())
    tkg = _Traced(_state())
    pair = MODULE.Glm53PairedPhaseRuntime.initialize(
        cte_model=cte,
        tkg_model=tkg,
        cte_checkpoint=object(),
        tkg_checkpoint=object(),
    )
    with pytest.raises(MODULE.Glm53PhaseRuntimeError, match="continuation state"):
        pair.assert_continuation_state([{"past_key_values.99": object()}])
