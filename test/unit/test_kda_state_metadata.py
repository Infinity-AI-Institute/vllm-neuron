from types import SimpleNamespace

import torch

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner


class _Assignment:
    def __init__(self, slots, reset_mask):
        self.slots = slots
        self.reset_mask = reset_mask


class _SlotTable:
    def __init__(self):
        self.assign_calls = []
        self.committed = []

    def assign(self, req_ids, *, batch_size, device):
        self.assign_calls.append((tuple(req_ids), batch_size, device))
        return _Assignment(
            torch.tensor([3, 9], dtype=torch.int64, device=device),
            torch.tensor([True, False], dtype=torch.bool, device=device),
        )

    def commit(self, assignment):
        self.committed.append(assignment)


def _runner():
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner._kda_slot_table = _SlotTable()
    runner._kda_state_assignment = None
    runner.input_batch = SimpleNamespace(req_ids=["a"])
    return runner


def test_synthetic_kda_metadata_has_stable_inputs_without_allocator_mutation():
    runner = _runner()
    metadata = {"layer": {}}

    result = runner._add_kda_state_metadata(
        metadata,
        num_rows=2,
        device=torch.device("cpu"),
        synthetic=True,
    )

    assert result["layer"]["state_slot_mapping"].tolist() == [0, 1]
    assert result["layer"]["state_reset_mask"].tolist() == [False, False]
    assert runner._kda_slot_table.assign_calls == []


def test_runtime_kda_metadata_uses_request_identity_and_commits_afterward():
    runner = _runner()
    metadata = {"layer": {}}

    result = runner._add_kda_state_metadata(
        metadata,
        num_rows=2,
        device=torch.device("cpu"),
        synthetic=False,
    )

    assert result["layer"]["state_slot_mapping"].tolist() == [3, 9]
    assert result["layer"]["state_reset_mask"].tolist() == [True, False]
    assignment = runner._kda_state_assignment
    runner._commit_kda_state_slots()
    assert runner._kda_slot_table.committed == [assignment]
    assert runner._kda_state_assignment is None
