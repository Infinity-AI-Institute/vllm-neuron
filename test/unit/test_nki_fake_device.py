# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_neuron.nki.nki_hop import _fake_output_device


def test_fake_output_prefers_runtime_device_over_meta_weights():
    weights = torch.empty((2, 2), device="meta")
    runtime_input = torch.empty((2, 2), device="cpu")

    assert _fake_output_device((weights, runtime_input)) == torch.device("cpu")


def test_fake_output_stays_meta_when_every_tensor_is_meta():
    assert _fake_output_device((torch.empty(1, device="meta"),)) == torch.device(
        "meta"
    )


def test_fake_output_requires_a_tensor_argument():
    with pytest.raises(ValueError, match="No tensor arguments"):
        _fake_output_device((1, "constant"))
