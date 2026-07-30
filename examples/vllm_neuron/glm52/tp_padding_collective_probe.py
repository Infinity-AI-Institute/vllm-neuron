# SPDX-License-Identifier: Apache-2.0
"""Verify GLM-5.2 prefill padding-mask gathers across one Trainium2 chip."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm_neuron  # noqa: F401
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
    get_tp_group,
    init_model_parallel_group,
    init_world_group,
)
import vllm.distributed.parallel_state as vllm_parallel_state
from vllm_neuron.envs import get_compile_backend_name


class TpPaddingCollectiveProbe(torch.nn.Module):
    def __init__(self, tp_group) -> None:
        super().__init__()
        self.tp_group = tp_group

    def forward(
        self,
        local_bool_mask: torch.Tensor,
        local_int_mask: torch.Tensor,
        local_slot_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.tp_group.all_gather(local_bool_mask, dim=0),
            self.tp_group.all_gather(local_int_mask, dim=0),
            self.tp_group.all_gather(local_slot_mapping, dim=0),
        )


def _worker(
    rank: int,
    world_size: int,
    master_port: int,
    local_tokens: int,
    real_tokens: int,
) -> None:
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
    # vLLM's generic node-count helper performs a barrier without an explicit
    # CPU device.  In this minimal process it runs before the Neuron
    # PrivateUse1 runtime hook is registered.  Initialize the same world
    # GroupCoordinator directly and record the known single-node topology.
    vllm_parallel_state._WORLD = init_world_group(
        list(range(world_size)),
        rank,
        "gloo",
    )
    vllm_parallel_state._NODE_COUNT = 1
    # The production TP GroupCoordinator uses a message-queue broadcaster for
    # scheduler metadata.  This tensor-only probe needs only its device
    # communicator; omitting the broadcaster avoids another unrelated CPU
    # topology barrier before the PrivateUse1 runtime hook is installed.
    vllm_parallel_state._TP = init_model_parallel_group(
        [list(range(world_size))],
        rank,
        "gloo",
        use_message_queue_broadcaster=False,
        group_name="tp",
    )
    try:
        tp_group = get_tp_group()
        global_tokens = world_size * local_tokens
        global_slot_mapping = torch.cat(
            (
                torch.arange(real_tokens, dtype=torch.int64),
                torch.full(
                    (global_tokens - real_tokens,),
                    -1,
                    dtype=torch.int64,
                ),
            )
        )
        start = rank * local_tokens
        local_slot_mapping = global_slot_mapping[start : start + local_tokens]
        local_bool_mask = local_slot_mapping >= 0
        local_int_mask = local_bool_mask.to(torch.int32)

        device = torch.device("neuron:0")
        model = TpPaddingCollectiveProbe(tp_group).to(device)
        compiled = torch.compile(model, backend=get_compile_backend_name())
        gathered_bool, gathered_int, gathered_slots = compiled(
            local_bool_mask.to(device),
            local_int_mask.to(device),
            local_slot_mapping.to(device),
        )
        gathered_bool = gathered_bool.to("cpu")
        gathered_int = gathered_int.to("cpu")
        gathered_slots = gathered_slots.to("cpu")

        expected_bool = global_slot_mapping >= 0
        torch.testing.assert_close(gathered_bool, expected_bool)
        torch.testing.assert_close(gathered_int, expected_bool.to(torch.int32))
        torch.testing.assert_close(gathered_slots, global_slot_mapping)
        torch.testing.assert_close(gathered_bool, gathered_int.to(torch.bool))

        print(
            json.dumps(
                {
                    "status": "passed",
                    "rank": rank,
                    "world_size": world_size,
                    "local_tokens": local_tokens,
                    "real_tokens": real_tokens,
                    "bool_dtype": str(gathered_bool.dtype),
                    "bool_vs_int_equal": bool(
                        torch.equal(gathered_bool, gathered_int.to(torch.bool))
                    ),
                    "slot_order_exact": bool(
                        torch.equal(gathered_slots, global_slot_mapping)
                    ),
                    "real_count": int(gathered_bool.sum().item()),
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
    parser.add_argument("--local-tokens", type=int, default=32)
    parser.add_argument("--real-tokens", type=int, default=70)
    parser.add_argument("--master-port", type=int, default=29571)
    args = parser.parse_args()
    if args.world_size != 4:
        raise ValueError("this bounded probe is defined for one four-core Trn2 chip")
    global_tokens = args.world_size * args.local_tokens
    if not 0 < args.real_tokens < global_tokens:
        raise ValueError("real-tokens must be between 1 and the global token count")

    mp.spawn(
        _worker,
        args=(
            args.world_size,
            args.master_port,
            args.local_tokens,
            args.real_tokens,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
