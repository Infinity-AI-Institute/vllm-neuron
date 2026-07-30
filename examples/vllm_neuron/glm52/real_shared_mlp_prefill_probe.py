# SPDX-License-Identifier: Apache-2.0
"""Probe layer-3 shared-expert prefill with real GLM-5.2 weights and activations."""

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
from vllm_neuron.model.glm52_moe_dsa.checkpoint_mapping import (
    build_checkpoint_contract,
)
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan
from vllm_neuron.model.glm52_moe_dsa.shared_expert import Glm52SharedExpert
from vllm_neuron.utils.weight_loader import get_weight_loader


_LAYER = 3
_WORLD_SIZE = 4
_PRODUCTION_WORLD_SIZE = 64
_EP_DEGREE = 16
_TOKENS = 2048
_FP8_MAX = 240.0


def _read_index(checkpoint_dir: Path) -> dict[str, str]:
    return json.loads(
        (checkpoint_dir / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )["weight_map"]


def _load_shared_parameters(
    shared: Glm52SharedExpert,
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
    prefix = f"model.layers.{_LAYER}.mlp.shared_experts"
    for name, parameter in shared.named_parameters():
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


def _quantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    scaled = tensor.float() / scale
    saturated = int((scaled.abs() > _FP8_MAX).sum())
    return (
        scaled.clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn),
        saturated,
    )


def _static_cpu_partial(
    shared: Glm52SharedExpert,
    hidden_bf16: torch.Tensor,
    hidden_quantized: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    gate_input_scale = shared.gate_up_input_scale[0, 0].float()
    down_input_scale = shared.down_input_scale[0, 0].float()
    hidden = hidden_quantized.float() * gate_input_scale
    gate_weight = (
        shared.gate_proj.weight.float()
        * shared.gate_proj.weight_scale[0, 0].float()
    )
    up_weight = (
        shared.up_proj.weight.float()
        * shared.up_proj.weight_scale[0, 0].float()
    )
    down_weight = (
        shared.down_proj.weight.float()
        * shared.down_proj.weight_scale[0, 0].float()
    )
    intermediate = F.silu(hidden @ gate_weight) * (hidden @ up_weight)
    intermediate_quantized, saturated = _quantize(
        intermediate,
        down_input_scale,
    )
    output = (intermediate_quantized.float() * down_input_scale) @ down_weight
    unquantized_intermediate = (
        F.silu(hidden_bf16.float() @ gate_weight)
        * (hidden_bf16.float() @ up_weight)
    )
    weight_only_output = unquantized_intermediate @ down_weight
    return output, weight_only_output, saturated


class RealSharedMlpPrefillProbe(torch.nn.Module):
    def __init__(self, shared: Glm52SharedExpert, tp_group) -> None:
        super().__init__()
        self.shared = shared
        self.tp_group = tp_group

    def forward(self, hidden_quantized: torch.Tensor) -> torch.Tensor:
        partial = self.shared._run(hidden_quantized)
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
    cosine = float(
        F.cosine_similarity(actual.reshape(-1), expected.reshape(-1), dim=0)
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
        shared = Glm52SharedExpert(
            config,
            plan,
            global_rank=rank,
        )
        _load_shared_parameters(
            shared,
            checkpoint_dir=checkpoint_dir,
            weight_map=_read_index(checkpoint_dir),
            config=config,
            plan=plan,
            global_rank=rank,
        )

        input_scale = shared.gate_up_input_scale[0, 0].float()
        real_hidden_quantized, input_saturated = _quantize(
            real_hidden,
            input_scale,
        )
        (
            cpu_partial,
            weight_only_partial,
            intermediate_saturated,
        ) = _static_cpu_partial(
            shared,
            real_hidden,
            real_hidden_quantized,
        )
        dist.reduce(cpu_partial, dst=0, group=tp_group.cpu_group)
        dist.reduce(
            weight_only_partial,
            dst=0,
            group=tp_group.cpu_group,
        )
        intermediate_saturation = torch.tensor(
            [intermediate_saturated],
            dtype=torch.int64,
        )
        dist.reduce(
            intermediate_saturation,
            dst=0,
            group=tp_group.cpu_group,
        )

        hidden_quantized = torch.zeros(
            _TOKENS,
            config.hidden_size,
            dtype=torch.float8_e4m3fn,
        )
        hidden_quantized[:real_tokens].copy_(real_hidden_quantized)
        model = RealSharedMlpPrefillProbe(shared, tp_group)
        device = torch.device("neuron:0")
        compiled = torch.compile(
            model.to(device),
            backend=get_compile_backend_name(),
        )
        actual = compiled(hidden_quantized.to(device)).cpu()

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
            bf16_reference = reference["shared_output"].reshape(
                real_tokens,
                config.hidden_size,
            )
            kernel_vs_static = _metrics(actual_real, cpu_partial)
            static_vs_bf16 = _metrics(cpu_partial, bf16_reference)
            weight_only_vs_bf16 = _metrics(
                weight_only_partial,
                bf16_reference,
            )
            activation_quantization_effect = _metrics(
                cpu_partial,
                weight_only_partial,
            )
            kernel_vs_bf16 = _metrics(actual_real, bf16_reference)
            result.update(
                {
                    "input_scale": float(input_scale),
                    "input_absmax": float(real_hidden.float().abs().max()),
                    "input_quantized_saturated_values": input_saturated,
                    "input_value_count": int(real_hidden.numel()),
                    "intermediate_quantized_saturated_values": int(
                        intermediate_saturation.item()
                    ),
                    "kernel_vs_static": kernel_vs_static,
                    "static_vs_bf16": static_vs_bf16,
                    "weight_only_vs_bf16": weight_only_vs_bf16,
                    "activation_quantization_effect": (
                        activation_quantization_effect
                    ),
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
                "real shared-expert kernel is below 0.99 cosine against "
                "the effective static-FP8 CPU reference"
            )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--master-port", type=int, default=29585)
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
