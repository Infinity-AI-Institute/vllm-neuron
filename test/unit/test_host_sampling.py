from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import (
    AsyncNeuronModelRunnerOutput,
    NeuronModelRunner,
    _copy_sampling_metadata_to_cpu,
)


def _sampling_metadata(**overrides):
    values = {
        "temperature": torch.tensor([0.7]),
        "top_p": torch.tensor([0.9]),
        "top_k": torch.tensor([64]),
        "prompt_token_ids": torch.tensor([[1, 2]]),
        "frequency_penalties": torch.tensor([0.0]),
        "presence_penalties": torch.tensor([0.0]),
        "repetition_penalties": torch.tensor([1.0]),
        "allowed_token_ids_mask": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_copy_sampling_metadata_to_cpu_is_non_mutating():
    metadata = _sampling_metadata()

    copied = _copy_sampling_metadata_to_cpu(metadata)

    assert copied is not metadata
    for field, value in vars(copied).items():
        if torch.is_tensor(value):
            assert value.device.type == "cpu"
    assert metadata.temperature is not None


def test_host_sampler_materializes_logits_on_cpu():
    class PendingDeviceLogits:
        def __init__(self):
            self.cpu_calls = 0

        def cpu(self):
            self.cpu_calls += 1
            return torch.ones(1, 8)

    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(sampling_metadata=_sampling_metadata())
    runner.use_async_scheduling = False
    runner.on_device_sampling = False
    runner._sampling_output_validator = lambda _: (_ for _ in ()).throw(
        AssertionError("host sampling must not call the token-ID validator")
    )
    captured = {}

    def sampler(**kwargs):
        captured.update(kwargs)
        return "sampled"

    runner.sampler = sampler
    logits = PendingDeviceLogits()

    result = runner._sample(logits)

    assert result == "sampled"
    assert logits.cpu_calls == 1
    assert captured["logits"].device.type == "cpu"
    assert captured["sampling_metadata"].temperature.device.type == "cpu"


class _StrictSamplingOutputModel:
    def __init__(self, vocab_size=12):
        self.vocab_size = vocab_size
        self.calls = []

    def validate_sampling_output(self, token_ids):
        self.calls.append(token_ids)
        if token_ids.device.type != "cpu":
            raise ValueError("sampling output must be on CPU")
        if token_ids.dtype != torch.int32:
            raise TypeError("sampling output must use int32")
        if token_ids.ndim != 1:
            raise ValueError("sampling output must be one-dimensional")
        if torch.any((token_ids < 0) | (token_ids >= self.vocab_size)):
            raise RuntimeError("sampling output contains an out-of-range token ID")


def _on_device_runner(model=None):
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.on_device_sampling = True
    runner.use_async_scheduling = False
    runner._on_device_logits = None
    runner._sampling_output_validator = getattr(model, "validate_sampling_output", None)
    runner.model = model
    runner.input_batch = SimpleNamespace(
        vocab_size=12,
        sampling_metadata=SimpleNamespace(max_num_logprobs=0),
    )
    runner.drafter = None
    return runner


def test_sync_on_device_sampling_validates_raw_cpu_ids_before_list_conversion():
    model = _StrictSamplingOutputModel()
    runner = _on_device_runner(model)
    # load_model() captures the bound hook before torch.compile replaces this
    # field with an OptimizedModule wrapper.
    runner.model = SimpleNamespace()

    result = runner._sample(torch.tensor([0, 11], dtype=torch.int32))

    assert result.sampled_token_ids == [[0], [11]]
    assert len(model.calls) == 1
    assert model.calls[0].device.type == "cpu"
    assert model.calls[0].shape == (2,)
    assert model.calls[0].dtype == torch.int32


def test_on_device_sampling_falls_back_for_models_without_validation_hook():
    runner = _on_device_runner(SimpleNamespace())

    result = runner._sample(torch.tensor([2, 7], dtype=torch.int32))

    assert result.sampled_token_ids == [[2], [7]]


def _async_output(token_ids, model):
    runner = _on_device_runner(model)
    updates = []
    runner._update_batch_state_with_samples = lambda sampled, snapshot_req_ids=None: (
        updates.append((sampled, snapshot_req_ids))
    )
    output = SimpleNamespace(
        sampled_token_ids=token_ids,
        req_ids=[f"req-{i}" for i in range(token_ids.shape[0])],
    )
    return AsyncNeuronModelRunnerOutput(output, runner), output, updates


def test_async_on_device_sampling_validates_before_state_and_output_mutation():
    model = _StrictSamplingOutputModel()
    async_output, output, updates = _async_output(
        torch.tensor([0, 11], dtype=torch.int32), model
    )

    materialized = async_output.get_output()

    assert materialized is output
    assert output.sampled_token_ids == [[0], [11]]
    assert updates == [([[0], [11]], ["req-0", "req-1"])]
    assert len(model.calls) == 1


@pytest.mark.parametrize("bad_id", [-1, 12])
def test_async_on_device_sampling_rejects_invalid_id_before_mutation(bad_id):
    model = _StrictSamplingOutputModel()
    raw = torch.tensor([bad_id], dtype=torch.int32)
    async_output, output, updates = _async_output(raw, model)

    with pytest.raises(RuntimeError, match="out-of-range"):
        async_output.get_output()

    assert output.sampled_token_ids is raw
    assert updates == []


def test_async_on_device_sampling_preserves_shape_for_validation():
    model = _StrictSamplingOutputModel()
    raw = torch.tensor([[1]], dtype=torch.int32)
    async_output, output, updates = _async_output(raw, model)

    with pytest.raises(ValueError, match="one-dimensional"):
        async_output.get_output()

    assert model.calls[0].shape == (1, 1)
    assert output.sampled_token_ids is raw
    assert updates == []


def test_async_on_device_sampling_preserves_dtype_for_validation():
    model = _StrictSamplingOutputModel()
    raw = torch.tensor([1], dtype=torch.int64)
    async_output, output, updates = _async_output(raw, model)

    with pytest.raises(TypeError, match="int32"):
        async_output.get_output()

    assert model.calls[0].dtype == torch.int64
    assert output.sampled_token_ids is raw
    assert updates == []


def test_validated_async_output_is_materialized_before_scheduler_state_update():
    model = _StrictSamplingOutputModel()
    async_output, output, updates = _async_output(
        torch.tensor([3], dtype=torch.int32), model
    )
    runner = async_output._model_runner
    runner.use_async_scheduling = True
    runner._host_validated_async_scheduling_enabled = True
    runner.async_execution_buffer = {"async_output": async_output}
    events = []

    original_update = runner._update_batch_state_with_samples

    def record_sample_update(sampled, snapshot_req_ids=None):
        events.append("sample_validated")
        return original_update(sampled, snapshot_req_ids)

    runner._update_batch_state_with_samples = record_sample_update
    runner._update_states = lambda scheduler_output: events.append("scheduler_state")

    runner._update_states_after_async_sampling_validation(object())

    assert events == ["sample_validated", "scheduler_state"]
    assert output.sampled_token_ids == [[3]]
    assert updates == [([[3]], ["req-0"])]
    assert runner._host_validation_barrier_steps == 1


@pytest.mark.parametrize("bad_id", [-1, 12])
def test_invalid_async_output_blocks_scheduler_and_slot_lifecycle(bad_id):
    model = _StrictSamplingOutputModel()
    raw = torch.tensor([bad_id], dtype=torch.int32)
    async_output, output, updates = _async_output(raw, model)
    runner = async_output._model_runner
    runner.use_async_scheduling = True
    runner._host_validated_async_scheduling_enabled = True
    runner.async_execution_buffer = {"async_output": async_output}
    state_updates = []
    runner._update_states = lambda scheduler_output: state_updates.append(
        scheduler_output
    )

    with pytest.raises(RuntimeError, match="out-of-range"):
        runner._update_states_after_async_sampling_validation(object())

    assert state_updates == []
    assert output.sampled_token_ids is raw
    assert updates == []
    assert runner._host_validation_barrier_steps == 1


def test_async_models_without_validator_keep_unmaterialized_future_path():
    raw = torch.tensor([4], dtype=torch.int32)
    async_output, output, updates = _async_output(raw, SimpleNamespace())
    runner = async_output._model_runner
    runner.use_async_scheduling = True
    runner.async_execution_buffer = {"async_output": async_output}
    state_updates = []
    runner._update_states = lambda scheduler_output: state_updates.append(
        scheduler_output
    )
    marker = object()

    runner._update_states_after_async_sampling_validation(marker)

    assert state_updates == [marker]
    assert output.sampled_token_ids is raw
    assert updates == []
    assert not hasattr(runner, "_host_validation_barrier_steps")


def test_validated_async_scheduling_is_off_by_default(monkeypatch):
    runner = _on_device_runner(_StrictSamplingOutputModel())
    runner.use_async_scheduling = True
    monkeypatch.delenv(
        "VLLM_NEURON_EXPERIMENTAL_HOST_VALIDATED_ASYNC_SCHEDULING",
        raising=False,
    )

    with pytest.raises(RuntimeError, match="off-by-default experiment"):
        runner._configure_host_validated_async_scheduling()

    assert runner._host_validated_async_scheduling_enabled is False


def test_validated_async_scheduling_requires_explicit_opt_in(monkeypatch):
    runner = _on_device_runner(_StrictSamplingOutputModel())
    runner.use_async_scheduling = True
    monkeypatch.setenv(
        "VLLM_NEURON_EXPERIMENTAL_HOST_VALIDATED_ASYNC_SCHEDULING", "1"
    )

    runner._configure_host_validated_async_scheduling()

    assert runner._host_validated_async_scheduling_enabled is True


def test_validated_async_state_update_refuses_missing_opt_in():
    runner = _on_device_runner(_StrictSamplingOutputModel())
    runner.use_async_scheduling = True
    runner.async_execution_buffer = {}
    state_updates = []
    runner._update_states = lambda scheduler_output: state_updates.append(
        scheduler_output
    )

    with pytest.raises(RuntimeError, match="without its explicit experiment opt-in"):
        runner._update_states_after_async_sampling_validation(object())

    assert state_updates == []
