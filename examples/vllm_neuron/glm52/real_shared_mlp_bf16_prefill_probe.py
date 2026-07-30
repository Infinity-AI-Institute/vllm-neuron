# SPDX-License-Identifier: Apache-2.0
"""Probe a real BF16 layer-3 GLM-5.2 shared expert on one Trn2 chip."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from nkilib.core.utils.common_types import ActFnType, QuantizationType
from safetensors import safe_open

import vllm.distributed.parallel_state as vllm_parallel_state
import vllm_neuron  # noqa: F401
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_model_parallel_group,
    init_world_group,
)
from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.functional.mlp import mlp


_LAYER = 3
_WORLD_SIZE = 4
_TOKENS = 2048
_HIDDEN_SIZE = 6144
_INTERMEDIATE_SIZE = 2048


def _load_rank_weights(
    checkpoint: Path,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start = rank * (_INTERMEDIATE_SIZE // _WORLD_SIZE)
    end = start + (_INTERMEDIATE_SIZE // _WORLD_SIZE)
    prefix = f"model.layers.{_LAYER}.mlp.shared_experts"
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        gate = source.get_slice(f"{prefix}.gate_proj.weight")[start:end, :].T
        up = source.get_slice(f"{prefix}.up_proj.weight")[start:end, :].T
        down = source.get_slice(f"{prefix}.down_proj.weight")[:, start:end].T
    return (
        gate.contiguous(),
        up.contiguous(),
        down.contiguous(),
    )


class RealBf16SharedMlp(torch.nn.Module):
    def __init__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
        tp_group,
    ) -> None:
        super().__init__()
        self.register_buffer("gate", gate)
        self.register_buffer("up", up)
        self.register_buffer("down", down)
        self.tp_group = tp_group

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        partial = mlp(
            hidden,
            self.gate,
            self.up,
            self.down,
            act_fn=ActFnType.SiLU,
            quantization_type=QuantizationType.NONE,
            output_dtype="bfloat16",
        )
        return self.tp_group.reduce_scatter(partial, dim=0)


def _initialize_groups(rank: int, master_port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(_WORLD_SIZE)
    os.environ["NEURON_RT_NUM_CORES"] = "1"
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(rank)
    torch.set_num_threads(12)

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{master_port}",
        world_size=_WORLD_SIZE,
        rank=rank,
    )
    vllm_parallel_state._WORLD = init_world_group(
        list(range(_WORLD_SIZE)),
        rank,
        "gloo",
    )
    vllm_parallel_state._NODE_COUNT = 1
    vllm_parallel_state._TP = init_model_parallel_group(
        [list(range(_WORLD_SIZE))],
        rank,
        "gloo",
        use_message_queue_broadcaster=False,
        group_name="tp",
    )
    return get_tp_group()


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.float()
    expected = expected.float()
    delta = actual - expected
    per_token_cosine = F.cosine_similarity(actual, expected, dim=-1)
    return {
        "cosine": float(
            F.cosine_similarity(actual.reshape(-1), expected.reshape(-1), dim=0)
        ),
        "relative_l2": float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(expected).clamp_min(
                torch.finfo(torch.float32).tiny
            )
        ),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "per_token_cosine_min": float(per_token_cosine.min()),
        "per_token_cosine_mean": float(per_token_cosine.mean()),
    }


def _worker(
    rank: int,
    master_port: int,
    checkpoint_raw: str,
    reference_raw: str,
) -> None:
    tp_group = _initialize_groups(rank, master_port)
    try:
        checkpoint = Path(checkpoint_raw)
        reference = torch.load(
            reference_raw,
            map_location="cpu",
            weights_only=True,
        )
        real_hidden = reference["mlp_input"].reshape(-1, _HIDDEN_SIZE)
        real_tokens = int(reference["prompt_length"])
        if real_hidden.shape != (real_tokens, _HIDDEN_SIZE):
            raise ValueError("reference MLP input shape is invalid")
        gate, up, down = _load_rank_weights(checkpoint, rank)

        cpu_partial = (
            F.silu(real_hidden.float() @ gate.float())
            * (real_hidden.float() @ up.float())
        ) @ down.float()
        dist.reduce(cpu_partial, dst=0, group=tp_group.cpu_group)

        hidden = torch.zeros(
            _TOKENS,
            _HIDDEN_SIZE,
            dtype=torch.bfloat16,
        )
        hidden[:real_tokens].copy_(real_hidden)
        model = RealBf16SharedMlp(gate, up, down, tp_group)
        device = torch.device("neuron:0")
        compiled = torch.compile(
            model.to(device),
            backend=get_compile_backend_name(),
        )
        actual = compiled(hidden.to(device)).cpu()

        local_tokens = _TOKENS // _WORLD_SIZE
        local_real = real_tokens if rank == 0 else 0
        padded_max_abs = float(actual[local_real:].float().abs().max())
        if padded_max_abs != 0.0:
            raise AssertionError(
                f"rank {rank} padded output is nonzero: {padded_max_abs}"
            )
        result: dict[str, object] = {
            "status": "passed",
            "rank": rank,
            "world_size": _WORLD_SIZE,
            "layer": _LAYER,
            "tokens": _TOKENS,
            "real_tokens": real_tokens,
            "local_tokens": local_tokens,
            "local_real_tokens": local_real,
            "dtype": "bfloat16",
            "padded_output_max_abs": padded_max_abs,
        }
        probe_passed = torch.ones(1, dtype=torch.int32)
        if rank == 0:
            actual_real = actual[:real_tokens]
            bf16_reference = reference["shared_output"].reshape(
                real_tokens,
                _HIDDEN_SIZE,
            )
            kernel_vs_cpu = _metrics(actual_real, cpu_partial)
            kernel_vs_reference = _metrics(actual_real, bf16_reference)
            cpu_vs_reference = _metrics(cpu_partial, bf16_reference)
            result.update(
                {
                    "kernel_vs_cpu": kernel_vs_cpu,
                    "kernel_vs_bf16_reference": kernel_vs_reference,
                    "cpu_vs_bf16_reference": cpu_vs_reference,
                }
            )
            if kernel_vs_reference["cosine"] < 0.99:
                result["status"] = "failed"
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status"] != "passed":
            probe_passed.zero_()
        dist.broadcast(probe_passed, src=0, group=tp_group.cpu_group)
        if int(probe_passed.item()) != 1:
            raise AssertionError(
                "BF16 shared-expert kernel is below 0.99 cosine against "
                "the pinned BF16 layer reference"
            )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--master-port", type=int, default=29587)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    if not 1 <= args.master_port <= 65535:
        raise ValueError("--master-port must be a valid TCP port")
    mp.spawn(
        _worker,
        args=(
            args.master_port,
            str(args.checkpoint.resolve()),
            str(args.reference.resolve()),
        ),
        nprocs=_WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()
