# SPDX-License-Identifier: Apache-2.0
"""Validate BF16 shared-expert checkpoint sharding without accelerator devices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan
from vllm_neuron.model.glm52_moe_dsa.shared_expert import Glm52SharedExpert
from vllm_neuron.utils.weight_loader import get_weight_loader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=(3, 40, 77))
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=(0, 1, 2, 3, 4, 63),
    )
    return parser.parse_args()


def _digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint16)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _load_index(checkpoint: Path) -> dict[str, str]:
    with (checkpoint / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint index has no weight_map")
    return {str(key): str(value) for key, value in weight_map.items()}


def _source_key(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.shared_experts.{projection}.weight"


def _load_projection(
    checkpoint: Path,
    weight_map: dict[str, str],
    key: str,
    parameter: torch.nn.Parameter,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    shard = checkpoint / weight_map[key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        source_slice = handle.get_slice(key)
        loaded = get_weight_loader(parameter).load([source_slice], rank)
        source = handle.get_tensor(key)
    return loaded, source


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    weight_map = _load_index(checkpoint)
    config = Glm52MoeDsaConfig(shared_expert_dtype="bfloat16")
    plan = RoutedExpertPlan(
        world_size=64,
        ep_degree=16,
        num_experts=config.n_routed_experts,
        expert_intermediate_size=config.moe_intermediate_size,
    )
    rows: list[dict[str, object]] = []

    for layer in args.layers:
        for rank in args.ranks:
            module = Glm52SharedExpert(
                config,
                plan,
                global_rank=rank,
                device="cpu",
            )
            local_rank = rank % plan.expert_tp_degree
            start = local_rank * plan.intermediate_per_rank
            end = start + plan.intermediate_per_rank
            for projection in ("gate_proj", "up_proj", "down_proj"):
                key = _source_key(layer, projection)
                parameter = getattr(module, projection).weight
                loaded, source = _load_projection(
                    checkpoint,
                    weight_map,
                    key,
                    parameter,
                    rank,
                )
                if projection == "down_proj":
                    expected = source[:, start:end].T.contiguous()
                else:
                    expected = source[start:end, :].T.contiguous()
                if loaded.shape != parameter.shape:
                    raise ValueError(
                        f"{key} rank {rank}: loaded shape {tuple(loaded.shape)} "
                        f"!= parameter shape {tuple(parameter.shape)}"
                    )
                if loaded.dtype != torch.bfloat16:
                    raise ValueError(f"{key} rank {rank}: loaded dtype {loaded.dtype}")
                if not torch.equal(loaded, expected):
                    raise ValueError(
                        f"{key} rank {rank}: loader output differs from the "
                        "expected subgroup shard"
                    )
                rows.append(
                    {
                        "layer": layer,
                        "rank": rank,
                        "shared_tp_rank": local_rank,
                        "projection": projection,
                        "shape": list(loaded.shape),
                        "sha256": _digest(loaded),
                    }
                )

    print(
        json.dumps(
            {
                "status": "passed",
                "checkpoint": str(checkpoint),
                "layers": args.layers,
                "ranks": args.ranks,
                "projection_checks": len(rows),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
