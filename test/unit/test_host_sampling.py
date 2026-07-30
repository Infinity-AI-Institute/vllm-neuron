from types import SimpleNamespace

import torch

from vllm_neuron.vllm.worker.neuron_model_runner import (
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
