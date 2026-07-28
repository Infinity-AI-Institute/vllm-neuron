# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_neuron.model.glm52_moe_dsa.cache_ops import (
    gather_paged_cache,
    write_paged_cache,
)


def test_paged_cache_round_trip_respects_block_table_order() -> None:
    cache = torch.zeros(3, 1, 2, 2, dtype=torch.bfloat16)
    values = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
            [[7.0, 8.0]],
        ],
        dtype=torch.bfloat16,
    )
    write_paged_cache(
        cache,
        values,
        torch.tensor([4, 5, 0, 1]),
        block_size=2,
    )

    gathered = gather_paged_cache(
        cache,
        torch.tensor([[2], [0]]),
        block_size=2,
        output_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(gathered[0], values[:2])
    torch.testing.assert_close(gathered[1], values[2:])


def test_fp8_paged_cache_uses_explicit_quantization_multiplier() -> None:
    cache = torch.zeros(1, 1, 2, 2, dtype=torch.float8_e4m3fn)
    values = torch.tensor(
        [[[0.5, -1.0]], [[2.0, -0.25]]],
        dtype=torch.bfloat16,
    )
    multiplier = torch.tensor([16.0])
    write_paged_cache(
        cache,
        values,
        torch.tensor([0, 1]),
        block_size=2,
        quant_multiplier=multiplier,
    )
    gathered = gather_paged_cache(
        cache,
        torch.tensor([[0]]),
        block_size=2,
        output_dtype=torch.bfloat16,
        quant_multiplier=multiplier,
    )

    torch.testing.assert_close(gathered.squeeze(0), values, atol=0.02, rtol=0.02)
