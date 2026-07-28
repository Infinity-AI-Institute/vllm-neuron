# SPDX-License-Identifier: Apache-2.0
"""Compile/numeric probe for exact GLM-5.2 expanded sparse MLA decode."""

import argparse
import json
import statistics
import time

import torch

import vllm_neuron  # noqa: F401
from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.model.glm52_moe_dsa.attention import (
    glm52_paged_sparse_attention,
    glm52_sparse_attention,
)
from vllm_neuron.model.glm52_moe_dsa.cache_ops import write_paged_cache
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig


class SparseAttentionDenseProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scaling = Glm52MoeDsaConfig().qk_head_dim**-0.5

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return glm52_sparse_attention(
            query,
            key_cache,
            value_cache,
            topk_indices,
            position_ids=position_ids,
            scaling=self.scaling,
        )


class SparseAttentionPagedProbe(SparseAttentionDenseProbe):
    block_size = 32

    def forward(
        self,
        query: torch.Tensor,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        position_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        write_paged_cache(
            key_cache,
            new_key,
            slot_mapping,
            block_size=self.block_size,
        )
        write_paged_cache(
            value_cache,
            new_value,
            slot_mapping,
            block_size=self.block_size,
        )
        return glm52_paged_sparse_attention(
            query,
            key_cache,
            value_cache,
            topk_indices,
            block_table,
            block_size=self.block_size,
            position_ids=position_ids,
            scaling=self.scaling,
        )


def _dense_inputs(
    config: Glm52MoeDsaConfig,
    context: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    query = torch.zeros(
        1,
        1,
        1,
        config.qk_head_dim,
        dtype=torch.bfloat16,
    )
    key_cache = torch.zeros(
        1,
        context,
        1,
        config.qk_head_dim,
        dtype=torch.bfloat16,
    )
    value_cache = torch.zeros(
        1,
        context,
        1,
        config.v_head_dim,
        dtype=torch.bfloat16,
    )
    value_cache[..., 0] = 16
    value_cache[:, -config.index_topk :, :, 0] = 2
    topk_indices = torch.arange(
        context - config.index_topk,
        context,
        dtype=torch.int32,
    ).reshape(1, 1, config.index_topk)
    position_ids = torch.tensor([[context - 1]], dtype=torch.long)
    expected = torch.zeros_like(query)
    expected[..., 0] = 2
    return (
        query,
        key_cache,
        value_cache,
        topk_indices,
        position_ids,
    ), expected


def _paged_inputs(
    config: Glm52MoeDsaConfig,
    context: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    block_size = SparseAttentionPagedProbe.block_size
    if context % block_size:
        raise ValueError(f"paged context must divide by block size {block_size}")

    query = torch.zeros(
        1,
        1,
        1,
        config.qk_head_dim,
        dtype=torch.bfloat16,
    )
    new_key = torch.zeros(
        1,
        1,
        config.qk_head_dim,
        dtype=torch.bfloat16,
    )
    new_value = torch.zeros(
        1,
        1,
        config.v_head_dim,
        dtype=torch.bfloat16,
    )
    new_value[..., 0] = config.index_topk
    key_cache = torch.zeros(
        context // block_size,
        1,
        block_size,
        config.qk_head_dim,
        dtype=torch.bfloat16,
    )
    value_cache = torch.zeros_like(key_cache)
    value_cache[..., 0] = 16
    value_cache.reshape(context, 1, config.v_head_dim)[
        -config.index_topk :, :, 0
    ] = 0
    topk_indices = torch.arange(
        context - config.index_topk,
        context,
        dtype=torch.int32,
    ).reshape(1, 1, config.index_topk)
    position_ids = torch.tensor([[context - 1]], dtype=torch.long)
    slot_mapping = torch.tensor([context - 1], dtype=torch.long)
    block_table = torch.arange(
        context // block_size,
        dtype=torch.int32,
    ).reshape(1, -1)
    expected = torch.zeros_like(query)
    expected[..., 0] = 1
    return (
        query,
        new_key,
        new_value,
        key_cache,
        value_cache,
        topk_indices,
        position_ids,
        slot_mapping,
        block_table,
    ), expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context",
        type=int,
        choices=(4_096, 8_192, 32_768),
        default=4_096,
    )
    parser.add_argument(
        "--mode",
        choices=("dense", "paged"),
        default="dense",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")

    config = Glm52MoeDsaConfig()
    if args.mode == "dense":
        model = SparseAttentionDenseProbe()
        inputs, expected = _dense_inputs(config, args.context)
    else:
        model = SparseAttentionPagedProbe()
        inputs, expected = _paged_inputs(config, args.context)

    device = torch.device("neuron:0")
    model = model.to(device)
    device_inputs = tuple(tensor.to(device) for tensor in inputs)

    started = time.perf_counter()
    compiled = torch.compile(model, backend=get_compile_backend_name())
    output = compiled(*device_inputs).to("cpu")
    elapsed = time.perf_counter() - started
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)

    hot_seconds = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        hot_output = compiled(*device_inputs).to("cpu")
        hot_seconds.append(time.perf_counter() - started)
    torch.testing.assert_close(hot_output, expected, atol=1e-2, rtol=1e-2)

    print(
        json.dumps(
            {
                "status": "passed",
                "backend": get_compile_backend_name(),
                "context": args.context,
                "mode": args.mode,
                "qk_head_dim": config.qk_head_dim,
                "value_head_dim": config.v_head_dim,
                "selected_keys": config.index_topk,
                "compile_and_first_run_seconds": elapsed,
                "hot_run_seconds_min": min(hot_seconds),
                "hot_run_seconds_median": statistics.median(hot_seconds),
                "repetitions": args.repetitions,
                "output_first": float(output[..., 0].item()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
