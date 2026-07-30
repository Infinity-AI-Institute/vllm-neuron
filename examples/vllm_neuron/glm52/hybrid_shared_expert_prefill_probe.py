# SPDX-License-Identifier: Apache-2.0
"""Qualify the production BF16-shared prefill path on one Trn2 chip."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
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
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan
from vllm_neuron.model.glm52_moe_dsa.shared_expert import Glm52SharedExpert

_LAYER = 3
_WORLD_SIZE = 4
_GLOBAL_WORLD_SIZE = 64
_EP_DEGREE = 16
_PREFILL_TOKENS = 2_048
_LOCAL_TOKENS = _PREFILL_TOKENS // _GLOBAL_WORLD_SIZE
_HIDDEN_SIZE = 6_144
_INTERMEDIATE_SIZE = 2_048
_GROUP_OFFSETS = (0, _WORLD_SIZE)


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
    return gate.contiguous(), up.contiguous(), down.contiguous()


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


class _HybridSharedPrefill(torch.nn.Module):
    def __init__(
        self,
        shared: Glm52SharedExpert,
        tp_group,
        *,
        eps: float,
    ) -> None:
        super().__init__()
        self.shared = shared
        self.tp_group = tp_group
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.shared.forward_prefill(
            hidden,
            norm_weight=torch.ones(
                _HIDDEN_SIZE,
                dtype=hidden.dtype,
                device=hidden.device,
            ),
            eps=self.eps,
            tp_group=self.tp_group,
            hidden_states_normalized=hidden,
        )


def _metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    delta = actual_float - expected_float
    return {
        "cosine": float(
            F.cosine_similarity(
                actual_float.reshape(-1),
                expected_float.reshape(-1),
                dim=0,
            )
        ),
        "relative_l2": float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(expected_float).clamp_min(
                torch.finfo(torch.float32).tiny
            )
        ),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
    }


def _worker(
    rank: int,
    master_port: int,
    weights_raw: str,
    reference_raw: str,
) -> None:
    tp_group = _initialize_groups(rank, master_port)
    try:
        reference = torch.load(
            reference_raw,
            map_location="cpu",
            weights_only=True,
        )
        prompt_length = int(reference["prompt_length"])
        if prompt_length != 192:
            raise ValueError(f"expected the 192-token gate prompt, got {prompt_length}")
        hidden = torch.zeros(
            _PREFILL_TOKENS,
            _HIDDEN_SIZE,
            dtype=torch.bfloat16,
        )
        hidden[:prompt_length].copy_(
            reference["mlp_input"].reshape(prompt_length, _HIDDEN_SIZE)
        )
        expected = torch.zeros_like(hidden)
        expected[:prompt_length].copy_(
            reference["shared_output"].reshape(prompt_length, _HIDDEN_SIZE)
        )

        config = Glm52MoeDsaConfig(shared_expert_dtype="bfloat16")
        plan = RoutedExpertPlan(
            world_size=_GLOBAL_WORLD_SIZE,
            ep_degree=_EP_DEGREE,
            num_experts=config.n_routed_experts,
            expert_intermediate_size=config.moe_intermediate_size,
        )
        gate, up, down = _load_rank_weights(Path(weights_raw), rank)
        shared = Glm52SharedExpert(
            config,
            plan,
            global_rank=rank,
        )
        with torch.no_grad():
            shared.gate_proj.weight.copy_(gate)
            shared.up_proj.weight.copy_(up)
            shared.down_proj.weight.copy_(down)
        module = _HybridSharedPrefill(
            shared,
            tp_group,
            eps=config.rms_norm_eps,
        )
        device = torch.device("neuron:0")
        compiled = torch.compile(
            module.to(device),
            backend=get_compile_backend_name(),
        )

        rows: list[dict[str, object]] = []
        passed = True
        for group_offset in _GROUP_OFFSETS:
            global_rank = group_offset + rank
            start = global_rank * _LOCAL_TOKENS
            end = start + _LOCAL_TOKENS
            actual = compiled(hidden[start:end].to(device)).cpu()
            expected_local = expected[start:end]
            real_tokens = max(0, min(end, prompt_length) - start)
            padded = actual[real_tokens:]
            padded_max_abs = (
                float(padded.float().abs().max()) if padded.numel() else 0.0
            )
            row: dict[str, object] = {
                "group_offset": group_offset,
                "rank": rank,
                "global_rank": global_rank,
                "token_range": [start, end],
                "real_tokens": real_tokens,
                "padded_output_max_abs": padded_max_abs,
            }
            if real_tokens:
                row["metrics"] = _metrics(
                    actual[:real_tokens],
                    expected_local[:real_tokens],
                )
                passed &= row["metrics"]["cosine"] >= 0.99
            passed &= padded_max_abs == 0.0
            rows.append(row)

        result = {
            "status": "passed" if passed else "failed",
            "rank": rank,
            "shared_expert_dtype": config.shared_expert_dtype,
            "world_size": _GLOBAL_WORLD_SIZE,
            "ep_degree": _EP_DEGREE,
            "expert_tp_degree": _WORLD_SIZE,
            "prefill_tokens": _PREFILL_TOKENS,
            "local_tokens": _LOCAL_TOKENS,
            "rows": rows,
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        all_passed = torch.tensor(int(passed), dtype=torch.int32)
        dist.all_reduce(
            all_passed,
            op=dist.ReduceOp.MIN,
            group=tp_group.cpu_group,
        )
        if int(all_passed.item()) != 1:
            raise AssertionError("hybrid shared-expert production-path probe failed")
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--master-port", type=int, default=29589)
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    if not 1 <= args.master_port <= 65_535:
        raise ValueError("--master-port must be a valid TCP port")
    mp.spawn(
        _worker,
        args=(
            args.master_port,
            str(args.weights.resolve()),
            str(args.reference.resolve()),
        ),
        nprocs=_WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()
