# SPDX-License-Identifier: Apache-2.0
"""Compile/numeric probe for one full-index GLM-5.2 TP64 decode layer."""

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

_CONTEXT = 32_768
_BLOCK_SIZE = 32
_FP8_MAX = 240.0
_FULL_INDEX_LAYER = 2
_SHARED_INDEX_LAYER = 3


class MlaFullIndexDecodeProbe(torch.nn.Module):
    def __init__(self, full_attention, shared_attention) -> None:
        super().__init__()
        self.full_attention = full_attention
        self.shared_attention = shared_attention

    def forward(
        self,
        full_hidden_states: torch.Tensor,
        shared_hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        full_key_cache: torch.Tensor,
        full_value_cache: torch.Tensor,
        shared_key_cache: torch.Tensor,
        shared_value_cache: torch.Tensor,
        indexer_cache: torch.Tensor,
        position_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        metadata = {
            "layers.2.self_attn": {
                "slot_mapping": slot_mapping,
                "block_size": _BLOCK_SIZE,
                "block_table_tensor": block_table,
            },
            "layers.3.self_attn": {
                "slot_mapping": slot_mapping,
                "block_size": _BLOCK_SIZE,
                "block_table_tensor": block_table,
            },
            "glm52.indexer_cache.1": {
                "slot_mapping": slot_mapping,
                "block_size": _BLOCK_SIZE,
                "block_table_tensor": block_table,
            },
        }
        full_output, full_state = self.full_attention.forward_paged_decode(
            full_hidden_states,
            (cos, sin),
            position_ids,
            metadata,
            key_cache=full_key_cache,
            value_cache=full_value_cache,
            indexer_cache=indexer_cache,
            previous_index_state=None,
        )
        shared_output, shared_state = self.shared_attention.forward_paged_decode(
            shared_hidden_states,
            (cos, sin),
            position_ids,
            metadata,
            key_cache=shared_key_cache,
            value_cache=shared_value_cache,
            previous_index_state=full_state,
        )
        return (
            full_output,
            shared_output,
            full_state.topk_indices,
            shared_state.topk_indices,
            full_key_cache,
            full_value_cache,
            shared_key_cache,
            shared_value_cache,
            indexer_cache,
        )


def _prepare_main_caches(
    attention,
    config: Glm52MoeDsaConfig,
    projection_expected: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    key_start = config.q_lora_rank + config.qk_head_dim
    value_start = key_start + config.qk_head_dim
    output_start = value_start + config.v_head_dim
    expected_key = projection_expected[:, key_start:value_start]
    expected_value = projection_expected[:, value_start:output_start]

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
        (
            expected_key.reshape(1, 1, 1, config.qk_head_dim).float()
            * key_multiplier
        )
        .clamp(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
    )
    value_template = (
        (
            expected_value.reshape(1, 1, 1, config.v_head_dim).float()
            * value_multiplier
        )
        .clamp(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
    )
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
    key_cache[-1, 0, -1].zero_()
    value_cache[-1, 0, -1].zero_()

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
    return (
        key_cache,
        value_cache,
        expected_output,
        expected_key.reshape(config.qk_head_dim),
        expected_value.reshape(config.v_head_dim),
        key_multiplier,
        value_multiplier,
    )


def _prepare_indexer(
    attention,
    config: Glm52MoeDsaConfig,
    hidden_states: torch.Tensor,
    projection_expected: torch.Tensor,
    *,
    cache_dtype: torch.dtype,
) -> torch.Tensor:
    indexer = attention.indexer
    if indexer is None:
        raise RuntimeError("probe must use a full-index layer")

    with torch.no_grad():
        q_resid = projection_expected[:, : config.q_lora_rank].float()
        q_resid_scale = q_resid.square().sum().clamp_min(1e-12)
        indexer.wq_b.weight.zero_()
        for head in range(config.index_n_heads):
            indexer.wq_b.weight[
                head * config.index_head_dim,
                :,
            ] = (q_resid / q_resid_scale).to(indexer.wq_b.weight.dtype)
        indexer.wk.weight.zero_()
        indexer.k_norm.weight.zero_()
        indexer.k_norm.bias.zero_()
        indexer.k_norm.bias[0] = 1
        hidden_fp32 = hidden_states.float()
        hidden_scale = hidden_fp32.square().sum().clamp_min(1e-12)
        indexer.weights_proj.weight.zero_()
        indexer.weights_proj.weight[:] = hidden_fp32 / hidden_scale

    indexer_cache = torch.zeros(
        _CONTEXT // _BLOCK_SIZE,
        1,
        _BLOCK_SIZE,
        config.index_head_dim,
        dtype=cache_dtype,
    )
    if cache_dtype == torch.float8_e4m3fn:
        indexer.set_cache_quant_multiplier(_FP8_MAX)
    return indexer_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-cache-dtype",
        choices=("bf16", "fp8"),
        default="fp8",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")

    full_probe, full_inputs, full_expected = _make_probe(
        layer_idx=_FULL_INDEX_LAYER,
    )
    shared_probe, shared_inputs, shared_expected = _make_probe(
        layer_idx=_SHARED_INDEX_LAYER,
    )
    full_attention = full_probe.attention
    shared_attention = shared_probe.attention
    config = Glm52MoeDsaConfig()
    full_hidden, cos, sin = full_inputs
    shared_hidden, shared_cos, shared_sin = shared_inputs
    torch.testing.assert_close(cos, shared_cos)
    torch.testing.assert_close(sin, shared_sin)
    (
        full_key_cache,
        full_value_cache,
        full_expected_output,
        full_expected_key_slot,
        full_expected_value_slot,
        full_key_multiplier,
        full_value_multiplier,
    ) = _prepare_main_caches(
        full_attention,
        config,
        full_expected,
    )
    (
        shared_key_cache,
        shared_value_cache,
        shared_expected_output,
        shared_expected_key_slot,
        shared_expected_value_slot,
        shared_key_multiplier,
        shared_value_multiplier,
    ) = _prepare_main_caches(
        shared_attention,
        config,
        shared_expected,
    )
    index_cache_dtype = (
        torch.bfloat16
        if args.index_cache_dtype == "bf16"
        else torch.float8_e4m3fn
    )
    indexer_cache = _prepare_indexer(
        full_attention,
        config,
        full_hidden,
        full_expected,
        cache_dtype=index_cache_dtype,
    )

    position_ids = torch.tensor([_CONTEXT - 1], dtype=torch.long)
    slot_mapping = position_ids.clone()
    block_table = torch.arange(
        _CONTEXT // _BLOCK_SIZE,
        dtype=torch.int32,
    ).reshape(1, -1)
    model = MlaFullIndexDecodeProbe(full_attention, shared_attention)
    inputs = (
        full_hidden,
        shared_hidden,
        cos,
        sin,
        full_key_cache,
        full_value_cache,
        shared_key_cache,
        shared_value_cache,
        indexer_cache,
        position_ids,
        slot_mapping,
        block_table,
    )
    device = torch.device("neuron:0")
    model = model.to(device)
    device_inputs = tuple(tensor.to(device) for tensor in inputs)

    started = time.perf_counter()
    compiled = torch.compile(model, backend=get_compile_backend_name())
    result = compiled(*device_inputs)
    (
        full_output,
        shared_output,
        full_topk,
        shared_topk,
        device_full_key_cache,
        device_full_value_cache,
        device_shared_key_cache,
        device_shared_value_cache,
        device_indexer_cache,
    ) = result
    full_output = full_output.to("cpu")
    shared_output = shared_output.to("cpu")
    full_topk = full_topk.to("cpu")
    shared_topk = shared_topk.to("cpu")
    elapsed = time.perf_counter() - started
    full_max_abs_error = (
        full_output.float() - full_expected_output.float()
    ).abs().max().item()
    shared_max_abs_error = (
        shared_output.float() - shared_expected_output.float()
    ).abs().max().item()
    torch.testing.assert_close(
        full_output.float(),
        full_expected_output.float(),
        atol=1.0,
        rtol=0.15,
    )
    torch.testing.assert_close(
        shared_output.float(),
        shared_expected_output.float(),
        atol=1.0,
        rtol=0.15,
    )
    torch.testing.assert_close(full_topk, shared_topk)
    if full_topk.shape != (1, config.index_topk):
        raise AssertionError(f"unexpected top-k shape: {full_topk.shape}")
    if int(full_topk.min()) < 0 or int(full_topk.max()) >= _CONTEXT:
        raise AssertionError("indexer selected a position outside the sequence")
    if torch.unique(full_topk).numel() != config.index_topk:
        raise AssertionError("indexer top-k contains duplicate positions")
    if not torch.any(full_topk == _CONTEXT - 1):
        raise AssertionError("full indexer did not select the current position")

    torch.testing.assert_close(
        device_full_key_cache[-1, 0, -1].to("cpu").float()
        / full_key_multiplier,
        full_expected_key_slot.float(),
        atol=0.15,
        rtol=0.15,
    )
    torch.testing.assert_close(
        device_full_value_cache[-1, 0, -1].to("cpu").float()
        / full_value_multiplier,
        full_expected_value_slot.float(),
        atol=0.15,
        rtol=0.15,
    )
    torch.testing.assert_close(
        device_shared_key_cache[-1, 0, -1].to("cpu").float()
        / shared_key_multiplier,
        shared_expected_key_slot.float(),
        atol=0.15,
        rtol=0.15,
    )
    torch.testing.assert_close(
        device_shared_value_cache[-1, 0, -1].to("cpu").float()
        / shared_value_multiplier,
        shared_expected_value_slot.float(),
        atol=0.15,
        rtol=0.15,
    )
    expected_index_value = _FP8_MAX if args.index_cache_dtype == "fp8" else 1
    actual_index_slot = device_indexer_cache[-1, 0, -1].to("cpu").float()
    torch.testing.assert_close(
        actual_index_slot[0],
        torch.tensor(expected_index_value, dtype=torch.float32),
    )
    torch.testing.assert_close(
        actual_index_slot[1:],
        torch.zeros(config.index_head_dim - 1),
    )

    hot_seconds = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        hot_result = compiled(*device_inputs)
        hot_result[0].to("cpu")
        hot_result[2].to("cpu")
        hot_seconds.append(time.perf_counter() - started)

    print(
        json.dumps(
            {
                "status": "passed",
                "backend": get_compile_backend_name(),
                "precision": (
                    "static_fp8_weights_fp8_main_cache_"
                    f"{args.index_cache_dtype}_index_cache"
                ),
                "layers": [_FULL_INDEX_LAYER, _SHARED_INDEX_LAYER],
                "context": _CONTEXT,
                "block_size": _BLOCK_SIZE,
                "selected_keys": config.index_topk,
                "compile_and_first_run_seconds": elapsed,
                "hot_run_seconds_min": min(hot_seconds),
                "hot_run_seconds_median": statistics.median(hot_seconds),
                "full_max_abs_error": full_max_abs_error,
                "shared_max_abs_error": shared_max_abs_error,
                "minimum_index": int(full_topk.min()),
                "maximum_index": int(full_topk.max()),
                "indexshare_exact": bool(torch.equal(full_topk, shared_topk)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
