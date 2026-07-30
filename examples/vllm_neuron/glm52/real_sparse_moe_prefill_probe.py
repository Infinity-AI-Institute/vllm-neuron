# SPDX-License-Identifier: Apache-2.0
"""Probe layer-3 routed-MoE prefill with real GLM-5.2 weights and activations."""

from __future__ import annotations

import argparse
import contextlib
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
from vllm_neuron.functional.moe.moe_blockwise import build_blockwise_mapping
from vllm_neuron.model.glm52_moe_dsa.checkpoint_mapping import (
    build_checkpoint_contract,
)
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.expert_kernels import (
    Glm52RoutedExperts,
    dense_glm52_affinities,
)
from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan
from vllm_neuron.utils.weight_loader import get_weight_loader


_LAYER = 3
_WORLD_SIZE = 4
_PRODUCTION_WORLD_SIZE = 64
_EP_DEGREE = 16
_TOKENS = 2048


def _read_index(checkpoint_dir: Path) -> dict[str, str]:
    return json.loads(
        (checkpoint_dir / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )["weight_map"]


def _load_expert_parameters(
    experts: Glm52RoutedExperts,
    *,
    checkpoint_dir: Path,
    weight_map: dict[str, str],
    config: Glm52MoeDsaConfig,
    plan: RoutedExpertPlan,
    global_rank: int,
) -> None:
    contract = build_checkpoint_contract(
        config,
        plan,
        global_rank=global_rank,
    )
    prefix = f"model.layers.{_LAYER}.mlp.experts"
    for name, parameter in experts.named_parameters():
        model_key = f"{prefix}.{name}"
        sources = contract.mappings[model_key]
        source_keys = [sources] if isinstance(sources, str) else sources
        with contextlib.ExitStack() as stack:
            slices = []
            for source_key in source_keys:
                handle = stack.enter_context(
                    safe_open(
                        checkpoint_dir / weight_map[source_key],
                        framework="pt",
                        device="cpu",
                    )
                )
                slices.append(handle.get_slice(source_key))
            loaded = get_weight_loader(parameter).load(slices, global_rank)
        if loaded.shape != parameter.shape or loaded.dtype != parameter.dtype:
            raise ValueError(
                f"loader contract mismatch for {model_key}: "
                f"loaded={loaded.shape}/{loaded.dtype}, "
                f"parameter={parameter.shape}/{parameter.dtype}"
            )
        with torch.no_grad():
            parameter.copy_(loaded)


def _static_cpu_partial(
    experts: Glm52RoutedExperts,
    hidden: torch.Tensor,
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Emulate the rank-local legacy-FP8 expert shard on CPU."""

    hidden_float = hidden.float()
    output = torch.zeros_like(hidden_float)
    gate_up = (
        experts.gate_up_proj.float()
        * experts.gate_up_proj_scale[:, None, :, :]
    )
    down = (
        experts.down_proj.float()
        * experts.down_proj_scale[:, None, :]
    )
    for local_expert in range(experts.plan.experts_per_rank):
        route_positions = torch.nonzero(
            expert_indices == local_expert,
            as_tuple=False,
        )
        if not route_positions.numel():
            continue
        token_ids = route_positions[:, 0]
        route_slots = route_positions[:, 1]
        selected = hidden_float[token_ids]
        projected = selected @ gate_up[local_expert].reshape(
            experts.hidden_size,
            -1,
        )
        gate, up = projected.chunk(2, dim=-1)
        partial = (F.silu(gate) * up) @ down[local_expert]
        partial *= routing_weights[token_ids, route_slots, None].float()
        output.index_add_(0, token_ids, partial)
    return output


class RealSparseMoePrefillProbe(torch.nn.Module):
    def __init__(
        self,
        experts: Glm52RoutedExperts,
        tp_group,
        *,
        real_tokens: int,
    ) -> None:
        super().__init__()
        self.experts = experts
        self.tp_group = tp_group
        self.real_tokens = real_tokens

    def forward(
        self,
        hidden: torch.Tensor,
        local_affinities: torch.Tensor,
    ) -> torch.Tensor:
        token_ids = torch.arange(hidden.shape[0], device=hidden.device)
        padding_mask = token_ids < self.real_tokens
        local_affinities = local_affinities * padding_mask.to(
            local_affinities.dtype
        ).unsqueeze(1)
        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            _,
        ) = build_blockwise_mapping(
            expert_affinities=local_affinities,
            num_local_experts=self.experts.plan.experts_per_rank,
            num_experts_per_token=self.experts.top_k,
            block_size=self.experts.block_size,
            moe_group=self.tp_group,
            tp_degree=self.tp_group.world_size,
            padding_mask=padding_mask,
        )
        partial = self.experts.forward_prefill(
            hidden,
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
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
    flat_actual = actual.reshape(-1)
    flat_expected = expected.reshape(-1)
    cosine = float(
        F.cosine_similarity(flat_actual, flat_expected, dim=0).item()
    )
    relative_l2 = float(
        torch.linalg.vector_norm(delta)
        / torch.linalg.vector_norm(expected).clamp_min(
            torch.finfo(torch.float32).tiny
        )
    )
    per_token_cosine = F.cosine_similarity(actual, expected, dim=-1)
    return {
        "cosine": cosine,
        "relative_l2": relative_l2,
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "per_token_cosine_min": float(per_token_cosine.min()),
        "per_token_cosine_mean": float(per_token_cosine.mean()),
    }


def _worker(
    rank: int,
    master_port: int,
    checkpoint_dir_raw: str,
    reference_raw: str,
) -> None:
    tp_group = _initialize_groups(rank, master_port)
    try:
        checkpoint_dir = Path(checkpoint_dir_raw)
        reference = torch.load(
            reference_raw,
            map_location="cpu",
            weights_only=True,
        )
        real_hidden = reference["mlp_input"].reshape(
            -1,
            reference["mlp_input"].shape[-1],
        )
        real_tokens = int(reference["prompt_length"])
        if real_hidden.shape[0] != real_tokens:
            raise ValueError("reference MLP input does not match prompt length")
        global_indices = reference["expert_indices"].to(torch.int64)
        routing_weights = reference["routing_weights"].float()

        config = Glm52MoeDsaConfig.from_configs(
            str(checkpoint_dir / "config.json"),
            neuron_config=None,
        )
        plan = RoutedExpertPlan(
            world_size=_PRODUCTION_WORLD_SIZE,
            ep_degree=_EP_DEGREE,
            num_experts=config.n_routed_experts,
            expert_intermediate_size=config.moe_intermediate_size,
        )
        experts = Glm52RoutedExperts(
            config,
            plan,
            global_rank=rank,
        )
        _load_expert_parameters(
            experts,
            checkpoint_dir=checkpoint_dir,
            weight_map=_read_index(checkpoint_dir),
            config=config,
            plan=plan,
            global_rank=rank,
        )

        ep0_mask = global_indices < plan.experts_per_rank
        local_indices = torch.where(
            ep0_mask,
            global_indices,
            torch.zeros_like(global_indices),
        )
        local_weights = torch.where(
            ep0_mask,
            routing_weights,
            torch.zeros_like(routing_weights),
        )
        cpu_partial = _static_cpu_partial(
            experts,
            real_hidden,
            local_indices,
            local_weights,
        )
        dist.reduce(cpu_partial, dst=0, group=tp_group.cpu_group)

        hidden = torch.zeros(
            _TOKENS,
            config.hidden_size,
            dtype=torch.bfloat16,
        )
        hidden[:real_tokens].copy_(real_hidden)
        full_affinities = dense_glm52_affinities(
            global_indices,
            routing_weights,
            num_experts=config.n_routed_experts,
        ).to(torch.bfloat16)
        local_affinities = torch.zeros(
            _TOKENS,
            plan.experts_per_rank,
            dtype=torch.bfloat16,
        )
        local_affinities[:real_tokens].copy_(
            full_affinities[:, : plan.experts_per_rank]
        )

        model = RealSparseMoePrefillProbe(
            experts,
            tp_group,
            real_tokens=real_tokens,
        )
        device = torch.device("neuron:0")
        compiled = torch.compile(
            model.to(device),
            backend=get_compile_backend_name(),
        )
        actual = compiled(
            hidden.to(device),
            local_affinities.to(device),
        ).cpu()

        local_tokens = _TOKENS // _WORLD_SIZE
        local_real = real_tokens if rank == 0 else 0
        padded_start = local_real
        padded_max_abs = float(actual[padded_start:].float().abs().max())
        if padded_max_abs != 0.0:
            raise AssertionError(
                f"rank {rank} padded output is nonzero: {padded_max_abs}"
            )

        result: dict[str, object] = {
            "status": "passed",
            "rank": rank,
            "production_global_rank": rank,
            "world_size": _WORLD_SIZE,
            "production_world_size": _PRODUCTION_WORLD_SIZE,
            "ep_degree": _EP_DEGREE,
            "layer": _LAYER,
            "tokens": _TOKENS,
            "real_tokens": real_tokens,
            "local_tokens": local_tokens,
            "local_real_tokens": local_real,
            "padded_output_max_abs": padded_max_abs,
        }
        probe_passed = torch.ones(1, dtype=torch.int32)
        if rank == 0:
            actual_real = actual[:real_tokens]
            bf16_reference = reference["routed_ep0_output"].reshape(
                real_tokens,
                config.hidden_size,
            )
            kernel_vs_static = _metrics(actual_real, cpu_partial)
            static_vs_bf16 = _metrics(cpu_partial, bf16_reference)
            kernel_vs_bf16 = _metrics(actual_real, bf16_reference)
            result.update(
                {
                    "ep0_route_count": int(ep0_mask.sum()),
                    "kernel_vs_static": kernel_vs_static,
                    "static_vs_bf16": static_vs_bf16,
                    "kernel_vs_bf16": kernel_vs_bf16,
                }
            )
            if kernel_vs_static["cosine"] < 0.99:
                result["status"] = "failed"
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status"] != "passed":
            probe_passed.zero_()
        dist.broadcast(probe_passed, src=0, group=tp_group.cpu_group)
        if int(probe_passed.item()) != 1:
            raise AssertionError(
                "real routed-MoE kernel is below 0.99 cosine against "
                "the effective static-FP8 CPU reference"
            )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--master-port", type=int, default=29584)
    args = parser.parse_args()
    if not args.checkpoint_dir.is_dir():
        raise FileNotFoundError(args.checkpoint_dir)
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    if not 1 <= args.master_port <= 65535:
        raise ValueError("--master-port must be a valid TCP port")
    mp.spawn(
        _worker,
        args=(
            args.master_port,
            str(args.checkpoint_dir.resolve()),
            str(args.reference.resolve()),
        ),
        nprocs=_WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()
