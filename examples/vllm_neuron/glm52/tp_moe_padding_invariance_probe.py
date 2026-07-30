# SPDX-License-Identifier: Apache-2.0
"""Qualify padded GLM-5.2 routed MoE across one four-rank Trn2 TP group."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.distributed.parallel_state as vllm_parallel_state
import vllm_neuron  # noqa: F401
from moe_static_fp8_probe import _make_cte_inputs, _shape_for_ep
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_model_parallel_group,
    init_world_group,
)
from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.functional.moe.moe_blockwise import build_blockwise_mapping


class TpMoePaddingInvariantProbe(torch.nn.Module):
    def __init__(
        self,
        kernel: torch.nn.Module,
        tp_group,
        *,
        local_experts: int,
        top_k: int,
        real_tokens: int,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.tp_group = tp_group
        self.local_experts = local_experts
        self.top_k = top_k
        self.real_tokens = real_tokens

    def _run(
        self,
        hidden: torch.Tensor,
        affinities: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_affinities = affinities * padding_mask.to(affinities.dtype).unsqueeze(1)
        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            _,
        ) = build_blockwise_mapping(
            expert_affinities=masked_affinities,
            num_local_experts=self.local_experts,
            num_experts_per_token=self.top_k,
            block_size=self.kernel.block_size,
            moe_group=self.tp_group,
            tp_degree=self.tp_group.world_size,
            padding_mask=padding_mask,
        )
        partial = self.kernel(
            hidden,
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
        )
        return self.tp_group.reduce_scatter(partial, dim=0)

    def forward(
        self,
        hidden: torch.Tensor,
        affinities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = torch.arange(hidden.shape[0], device=hidden.device)
        all_valid = torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        padding_mask = token_ids < self.real_tokens
        return (
            self._run(hidden, affinities, all_valid),
            self._run(hidden, affinities, padding_mask),
        )


def _initialize_groups(rank: int, world_size: int, master_port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NEURON_RT_NUM_CORES"] = "1"
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(rank)

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{master_port}",
        world_size=world_size,
        rank=rank,
    )
    vllm_parallel_state._WORLD = init_world_group(
        list(range(world_size)),
        rank,
        "gloo",
    )
    vllm_parallel_state._NODE_COUNT = 1
    vllm_parallel_state._TP = init_model_parallel_group(
        [list(range(world_size))],
        rank,
        "gloo",
        use_message_queue_broadcaster=False,
        group_name="tp",
    )
    return get_tp_group()


def _worker(
    rank: int,
    world_size: int,
    master_port: int,
    ep_degree: int,
    tokens: int,
    real_tokens: int,
    use_fp8: bool,
    routing: str,
) -> None:
    tp_group = _initialize_groups(rank, world_size, master_port)
    try:
        local_experts, _ = _shape_for_ep(ep_degree)
        kernel, inputs, expected = _make_cte_inputs(
            ep_degree,
            tokens,
            use_fp8,
            MoECTEImplementation.shard_on_i,
        )
        hidden, flattened_affinities, _, _ = inputs
        affinities = flattened_affinities.view(tokens, local_experts)
        if routing == "rotating":
            active_experts = min(8, local_experts)
            token_ids = torch.arange(tokens).unsqueeze(1)
            expert_offsets = torch.arange(active_experts).unsqueeze(0)
            selected_experts = (token_ids + expert_offsets) % local_experts
            affinities.zero_()
            affinities.scatter_(
                1,
                selected_experts,
                2.5 / active_experts,
            )
            kernel.gate_up_weight[1:].copy_(
                kernel.gate_up_weight[0].unsqueeze(0).expand_as(
                    kernel.gate_up_weight[1:]
                )
            )
            kernel.down_weight[1:].copy_(
                kernel.down_weight[0].unsqueeze(0).expand_as(
                    kernel.down_weight[1:]
                )
            )
            if kernel.gate_up_scale is not None:
                kernel.gate_up_scale[1:].copy_(
                    kernel.gate_up_scale[0].unsqueeze(0).expand_as(
                        kernel.gate_up_scale[1:]
                    )
                )
                kernel.down_scale[1:].copy_(
                    kernel.down_scale[0].unsqueeze(0).expand_as(
                        kernel.down_scale[1:]
                    )
                )
            expected = expected * active_experts
        model = TpMoePaddingInvariantProbe(
            kernel,
            tp_group,
            local_experts=local_experts,
            top_k=8,
            real_tokens=real_tokens,
        )

        device = torch.device("neuron:0")
        compiled = torch.compile(
            model.to(device),
            backend=get_compile_backend_name(),
        )
        all_valid_output, padded_output = compiled(
            hidden.to(device),
            affinities.to(device),
        )
        all_valid_output = all_valid_output.to("cpu").float()
        padded_output = padded_output.to("cpu").float()

        local_tokens = tokens // world_size
        local_start = rank * local_tokens
        local_real = max(0, min(local_tokens, real_tokens - local_start))
        if local_real:
            real = slice(0, local_real)
            max_abs_delta = float(
                (all_valid_output[real] - padded_output[real]).abs().max().item()
            )
            expected_local = expected[
                local_start : local_start + local_real
            ].float() * world_size
            atol = (0.75 if use_fp8 else 0.15) * world_size
            rtol = 0.08 if use_fp8 else 0.02
            torch.testing.assert_close(
                padded_output[real],
                expected_local,
                atol=atol,
                rtol=rtol,
            )
            if max_abs_delta != 0.0:
                raise AssertionError(
                    "real-token output changed between padded and all-valid "
                    f"mappings: max |delta|={max_abs_delta}"
                )
        else:
            max_abs_delta = 0.0

        padded_local_start = max(0, real_tokens - local_start)
        padded_max_abs = float(padded_output[padded_local_start:].abs().max().item())
        if padded_max_abs != 0.0:
            raise AssertionError(
                f"padded routed-MoE output must be zero, observed {padded_max_abs}"
            )

        print(
            json.dumps(
                {
                    "status": "passed",
                    "rank": rank,
                    "world_size": world_size,
                    "ep_degree": ep_degree,
                    "tokens": tokens,
                    "real_tokens": real_tokens,
                    "local_real_tokens": local_real,
                    "routing": routing,
                    "dtype": "fp8" if use_fp8 else "bf16",
                    "real_padded_vs_all_valid_max_abs_delta": max_abs_delta,
                    "padded_output_max_abs": padded_max_abs,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        cpu_sync = torch.ones(1, dtype=torch.int32)
        dist.all_reduce(cpu_sync, group=tp_group.cpu_group)
        if int(cpu_sync.item()) != world_size:
            raise AssertionError("CPU cleanup synchronization did not include all ranks")
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--ep-degree", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--real-tokens", type=int, default=192)
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="fp8")
    parser.add_argument(
        "--routing",
        choices=("uniform", "rotating"),
        default="uniform",
    )
    parser.add_argument("--master-port", type=int, default=29581)
    args = parser.parse_args()
    if args.world_size != 4:
        raise ValueError("this bounded probe is defined for one four-core Trn2 chip")
    if args.tokens % args.world_size:
        raise ValueError("tokens must divide evenly across the TP probe ranks")
    if not 0 < args.real_tokens < args.tokens:
        raise ValueError("real-tokens must be between 1 and tokens-1")

    mp.spawn(
        _worker,
        args=(
            args.world_size,
            args.master_port,
            args.ep_degree,
            args.tokens,
            args.real_tokens,
            args.dtype == "fp8",
            args.routing,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
