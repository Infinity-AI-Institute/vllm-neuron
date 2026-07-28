# SPDX-License-Identifier: Apache-2.0
"""Compile/numeric probe for one exact GLM-5.2 TP64 MLA decode layer."""

import argparse
import json
import statistics
import time

import torch

import vllm_neuron  # noqa: F401
from examples.vllm_neuron.glm52.mla_projection_static_fp8_probe import (
    _dequantize_activation,
    _make_probe,
)
from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.indexer import Glm52IndexShareState

_CONTEXT = 32_768
_BLOCK_SIZE = 32
_FP8_MAX = 240.0


class MlaDecodeProbe(torch.nn.Module):
    def __init__(self, attention) -> None:
        super().__init__()
        self.attention = attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        position_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        metadata = {
            "layers.3.self_attn": {
                "slot_mapping": slot_mapping,
                "block_size": _BLOCK_SIZE,
                "block_table_tensor": block_table,
            }
        }
        output, _ = self.attention.forward_paged_decode(
            hidden_states,
            (cos, sin),
            position_ids,
            metadata,
            key_cache=key_cache,
            value_cache=value_cache,
            previous_index_state=Glm52IndexShareState(
                topk_indices=topk_indices,
                source_layer_idx=2,
            ),
        )
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dtype",
        choices=("bf16", "fp8"),
        default="bf16",
    )
    args = parser.parse_args()

    projection_probe, projection_inputs, projection_expected = _make_probe()
    attention = projection_probe.attention
    config = Glm52MoeDsaConfig()
    hidden, cos, sin = projection_inputs

    key_start = config.q_lora_rank + config.qk_head_dim
    value_start = key_start + config.qk_head_dim
    output_start = value_start + config.v_head_dim
    expected_key = projection_expected[:, key_start:value_start]
    expected_value = projection_expected[:, value_start:output_start]
    expected_output = projection_expected[:, output_start:]

    key_template = expected_key.reshape(1, 1, 1, config.qk_head_dim)
    value_template = expected_value.reshape(1, 1, 1, config.v_head_dim)
    if args.cache_dtype == "fp8":
        key_multiplier = (
            _FP8_MAX / expected_key.abs().amax().to(torch.float32)
        ).clamp_max(torch.finfo(torch.float32).max)
        value_multiplier = (
            _FP8_MAX / expected_value.abs().amax().to(torch.float32)
        ).clamp_max(torch.finfo(torch.float32).max)
        attention.set_cache_quant_multipliers(
            key=float(key_multiplier),
            value=float(value_multiplier),
        )
        key_template = (
            (key_template.float() * key_multiplier)
            .clamp(-_FP8_MAX, _FP8_MAX)
            .to(torch.float8_e4m3fn)
        )
        value_template = (
            (value_template.float() * value_multiplier)
            .clamp(-_FP8_MAX, _FP8_MAX)
            .to(torch.float8_e4m3fn)
        )
        value_dequant = value_template.float() / value_multiplier
        o_input_scale = attention.o_proj.input_scale[0, 0].detach().cpu()
        o_weight_scale = attention.o_proj.weight_scale[0, 0].detach().cpu()
        expected_output = (
            _dequantize_activation(
                value_dequant.reshape(1, config.v_head_dim),
                o_input_scale,
            )
            @ (attention.o_proj.weight.detach().cpu().float() * o_weight_scale)
        ).to(torch.bfloat16)

    key_cache = key_template.expand(
        _CONTEXT // _BLOCK_SIZE,
        1,
        _BLOCK_SIZE,
        config.qk_head_dim,
    ).clone()
    value_cache = value_template.expand(
        _CONTEXT // _BLOCK_SIZE,
        1,
        _BLOCK_SIZE,
        config.v_head_dim,
    ).clone()
    position_ids = torch.tensor([_CONTEXT - 1], dtype=torch.long)
    slot_mapping = position_ids.clone()
    block_table = torch.arange(
        _CONTEXT // _BLOCK_SIZE,
        dtype=torch.int32,
    ).reshape(1, -1)
    topk_indices = torch.arange(
        _CONTEXT - config.index_topk,
        _CONTEXT,
        dtype=torch.int32,
    ).reshape(1, config.index_topk)

    model = MlaDecodeProbe(attention)
    inputs = (
        hidden,
        cos,
        sin,
        key_cache,
        value_cache,
        position_ids,
        slot_mapping,
        block_table,
        topk_indices,
    )
    device = torch.device("neuron:0")
    model = model.to(device)
    device_inputs = tuple(tensor.to(device) for tensor in inputs)

    started = time.perf_counter()
    compiled = torch.compile(model, backend=get_compile_backend_name())
    output = compiled(*device_inputs).to("cpu")
    elapsed = time.perf_counter() - started
    max_abs_error = (output.float() - expected_output.float()).abs().max().item()
    torch.testing.assert_close(
        output.float(),
        expected_output.float(),
        atol=1.0,
        rtol=0.15,
    )

    hot_seconds = []
    for _ in range(5):
        started = time.perf_counter()
        compiled(*device_inputs).to("cpu")
        hot_seconds.append(time.perf_counter() - started)

    print(
        json.dumps(
            {
                "status": "passed",
                "backend": get_compile_backend_name(),
                "precision": f"static_fp8_weights_{args.cache_dtype}_cache",
                "context": _CONTEXT,
                "block_size": _BLOCK_SIZE,
                "selected_keys": config.index_topk,
                "compile_and_first_run_seconds": elapsed,
                "hot_run_seconds_min": min(hot_seconds),
                "hot_run_seconds_median": statistics.median(hot_seconds),
                "max_abs_error": max_abs_error,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
