# SPDX-License-Identifier: Apache-2.0
"""Compile/numeric probe for the exact GLM-5.2 DSA decode indexer."""

import argparse
import json
import statistics
import time

import torch

import vllm_neuron  # noqa: F401
from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.model.glm52_moe_dsa.cache_layout import IndexerCacheBinding
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.indexer import Glm52FullIndexer


class IndexerDenseProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = Glm52MoeDsaConfig()
        self.indexer = Glm52FullIndexer(
            config,
            layer_idx=0,
            cache_binding=IndexerCacheBinding(
                layer_idx=0,
                cache_name="glm52.indexer_cache.0",
                cache_slot=0,
            ),
            dtype=torch.bfloat16,
            topk_backend="neuron",
        )
        with torch.no_grad():
            self.indexer.wq_b.weight.zero_()
            self.indexer.wq_b.weight[
                torch.arange(config.index_n_heads) * config.index_head_dim,
                0,
            ] = 1
            self.indexer.weights_proj.weight.zero_()
            self.indexer.weights_proj.weight[:, 0] = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.indexer.forward_dense(
            hidden_states,
            q_resid,
            cos,
            sin,
            key_cache=key_cache,
            position_ids=position_ids,
        )


class IndexerPagedProbe(IndexerDenseProbe):
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        position_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        metadata = {
            "glm52.indexer_cache.0": {
                "slot_mapping": slot_mapping,
                "block_size": 128,
                "block_table_tensor": block_table,
            }
        }
        return self.indexer.forward_paged(
            hidden_states,
            q_resid,
            cos,
            sin,
            position_ids=position_ids,
            attn_metadata=metadata,
            key_cache=key_cache,
        )


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
    hidden_states = torch.zeros(
        1,
        1,
        config.hidden_size,
        dtype=torch.bfloat16,
    )
    hidden_states[..., 0] = 1
    q_resid = torch.zeros(
        1,
        1,
        config.q_lora_rank,
        dtype=torch.bfloat16,
    )
    q_resid[..., 0] = 1
    cos_shape = (
        (1, 1, config.qk_rope_head_dim)
        if args.mode == "dense"
        else (1, config.qk_rope_head_dim)
    )
    cos = torch.ones(cos_shape, dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    if args.mode == "dense":
        key_cache = torch.zeros(
            1,
            args.context,
            config.index_head_dim,
            dtype=torch.bfloat16,
        )
        key_cache[0, -config.index_topk :, 0] = 1
        position_ids = torch.tensor([[args.context - 1]], dtype=torch.long)
        extra_inputs: tuple[torch.Tensor, ...] = ()
    else:
        if args.context % 128:
            raise ValueError("paged context must divide by block size 128")
        key_cache = torch.zeros(
            args.context // 128,
            1,
            128,
            config.index_head_dim,
            dtype=torch.bfloat16,
        )
        key_cache.reshape(-1, config.index_head_dim)[
            -config.index_topk :, 0
        ] = 1
        position_ids = torch.tensor([args.context - 1], dtype=torch.long)
        extra_inputs = (
            torch.tensor([args.context - 1], dtype=torch.long),
            torch.arange(args.context // 128, dtype=torch.int32).reshape(1, -1),
        )
    expected = torch.arange(
        args.context - config.index_topk,
        args.context,
        dtype=torch.int32,
    ).reshape(1, 1, config.index_topk)

    device = torch.device("neuron:0")
    model = (
        IndexerDenseProbe()
        if args.mode == "dense"
        else IndexerPagedProbe()
    )
    if args.mode == "paged":
        with torch.no_grad():
            model.indexer.wk.weight.zero_()
            model.indexer.k_norm.weight.zero_()
            model.indexer.k_norm.bias.zero_()
            model.indexer.k_norm.bias[0] = 1
    model = model.to(device)
    inputs = (
        (
            hidden_states
            if args.mode == "dense"
            else hidden_states.reshape(1, config.hidden_size)
        ).to(device),
        (
            q_resid
            if args.mode == "dense"
            else q_resid.reshape(1, config.q_lora_rank)
        ).to(device),
        cos.to(device),
        sin.to(device),
        key_cache.to(device),
        position_ids.to(device),
        *(tensor.to(device) for tensor in extra_inputs),
    )

    started = time.perf_counter()
    compiled = torch.compile(model, backend=get_compile_backend_name())
    output = compiled(*inputs).to("cpu")
    elapsed = time.perf_counter() - started
    output_sorted = output.sort(dim=-1).values
    torch.testing.assert_close(
        output_sorted.reshape(1, 1, config.index_topk),
        expected,
    )

    hot_seconds = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        compiled(*inputs).to("cpu")
        hot_seconds.append(time.perf_counter() - started)

    print(
        json.dumps(
            {
                "status": "passed",
                "backend": get_compile_backend_name(),
                "context": args.context,
                "mode": args.mode,
                "index_heads": config.index_n_heads,
                "index_head_dim": config.index_head_dim,
                "top_k": config.index_topk,
                "compile_and_first_run_seconds": elapsed,
                "hot_run_seconds_min": min(hot_seconds),
                "hot_run_seconds_median": statistics.median(hot_seconds),
                "repetitions": args.repetitions,
                "minimum_index": int(output.min()),
                "maximum_index": int(output.max()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
