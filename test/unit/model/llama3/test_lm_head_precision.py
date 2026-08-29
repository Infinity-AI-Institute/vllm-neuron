# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict

import pytest
import torch

from vllm_neuron.model.llama3.lm_head_precision import (
    prepare_lm_head_input,
    require_lm_head_output_dtype,
    require_lm_head_weight_dtype,
    resolve_lm_head_dtype,
)
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.cpl import ColumnParallelLinear


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, torch.bfloat16), ("bfloat16", torch.bfloat16), ("float32", torch.float32)],
)
def test_resolve_lm_head_dtype(requested, expected):
    assert resolve_lm_head_dtype(torch.bfloat16, requested) == expected


@pytest.mark.parametrize(
    "value", ["float16", "fp32", "bf16", True, 32, {"dtype": "float32"}]
)
def test_neuron_config_rejects_noncanonical_lm_head_dtype(value):
    with pytest.raises(ValueError, match="lm_head_dtype must be"):
        NeuronConfig.from_dict({"lm_head_dtype": value})


@pytest.mark.parametrize(
    "key", ["lm_head_output_dtype", "logits_dtype", "lm_head_dytpe"]
)
def test_neuron_config_rejects_precision_aliases_and_typos(key):
    with pytest.raises(ValueError, match="Unknown lm_head/logits precision"):
        NeuronConfig.from_dict({key: "float32"})


def test_lm_head_dtype_survives_emit_and_reload():
    config = NeuronConfig(lm_head_dtype="float32", on_device_sampling_config=None)
    emitted = asdict(config)
    assert emitted["lm_head_dtype"] == "float32"
    assert NeuronConfig.from_dict(emitted).lm_head_dtype == "float32"


@pytest.mark.parametrize("requested", ["bfloat16", "float32"])
def test_real_column_parallel_linear_obeys_compute_and_output_boundary(requested):
    dtype = resolve_lm_head_dtype(torch.bfloat16, requested)
    head = ColumnParallelLinear(4, 4, bias=False, dtype=dtype)
    hidden = prepare_lm_head_input(torch.ones((1, 4), dtype=torch.bfloat16), dtype)
    logits = require_lm_head_output_dtype(head(hidden), dtype)

    assert hidden.dtype == dtype
    assert head.weight.dtype == dtype
    assert logits.dtype == dtype


def test_posthoc_float_cast_does_not_satisfy_boundary():
    # This reproduces the relevant failure shape: the bf16 output has already
    # tied, so casting it to float32 cannot restore the fp32 winner.
    bf16_logits = torch.tensor([17.5, 17.5], dtype=torch.bfloat16)
    fp32_logits = torch.tensor([17.5001, 17.5002], dtype=torch.float32)
    assert bf16_logits.float().argmax().item() == 0
    assert fp32_logits.argmax().item() == 1

    with pytest.raises(RuntimeError, match="output dtype contract violated"):
        require_lm_head_output_dtype(bf16_logits, torch.float32)


def test_output_guard_rejects_backend_dtype_drift():
    with pytest.raises(RuntimeError, match="expected torch.float32"):
        require_lm_head_output_dtype(
            torch.zeros((1, 4), dtype=torch.bfloat16), torch.float32
        )


def test_weight_guard_rejects_checkpoint_assignment_dtype_drift():
    with pytest.raises(RuntimeError, match="after checkpoint load"):
        require_lm_head_weight_dtype(
            torch.zeros((4, 4), dtype=torch.bfloat16), torch.float32
        )
