# SPDX-License-Identifier: Apache-2.0
"""Run one real layer-0 GLM-5.2 MLA column projection on Trn2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

import vllm_neuron  # noqa: F401
from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.model.glm52_moe_dsa.cache_layout import Glm52CacheLayout
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.mla import Glm52MlaAttention
from vllm_neuron.model.glm52_moe_dsa.sparse_mlp import glm52_rms_norm

OCP_MAX = 448.0
NEURON_MAX = 240.0
SCALE_ROWS = 128
PROJECTIONS = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
)


def read_tensor(
    checkpoint_dir: Path,
    weight_map: dict[str, str],
    key: str,
) -> torch.Tensor:
    with safe_open(
        checkpoint_dir / weight_map[key],
        framework="pt",
        device="cpu",
    ) as shard:
        return shard.get_tensor(key)


def load_column(
    module,
    checkpoint_dir: Path,
    weight_map: dict[str, str],
    prefix: str,
    *,
    rank: int,
) -> None:
    source = read_tensor(checkpoint_dir, weight_map, f"{prefix}.weight")
    local_output = module.weight.shape[1]
    if source.shape[0] != local_output:
        source = source[rank * local_output : (rank + 1) * local_output]
    weight = (
        (source.float() * (NEURON_MAX / OCP_MAX))
        .clamp(-NEURON_MAX, NEURON_MAX)
        .to(torch.float8_e4m3fn)
        .T.contiguous()
    )
    weight_scale = (
        read_tensor(checkpoint_dir, weight_map, f"{prefix}.weight_scale").float()
        * (OCP_MAX / NEURON_MAX)
    )
    input_scale = read_tensor(
        checkpoint_dir,
        weight_map,
        f"{prefix}.input_scale",
    ).float()
    with torch.no_grad():
        module.weight.copy_(weight)
        module.weight_scale.copy_(
            weight_scale.reshape(1, 1).expand(SCALE_ROWS, 3)
        )
        module.input_scale.copy_(
            input_scale.reshape(1, 1).expand(SCALE_ROWS, 1)
        )


def static_linear(hidden: torch.Tensor, module) -> torch.Tensor:
    input_scale = module.input_scale[0, 0].float()
    weight_scale = module.weight_scale[0, 0].float()
    quantized = (
        (hidden.float() / input_scale)
        .clamp(-NEURON_MAX, NEURON_MAX)
        .to(torch.float8_e4m3fn)
        .float()
    )
    return (
        (quantized * input_scale) @ (module.weight.float() * weight_scale)
    ).to(torch.bfloat16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--projection", choices=PROJECTIONS, required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--real-tokens", type=int, default=512)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", default="neuron:0")
    args = parser.parse_args()
    if args.tokens <= 0 or not 0 < args.real_tokens <= args.tokens:
        parser.error("require 0 < --real-tokens <= --tokens")
    if not 0 <= args.rank < 64:
        parser.error("--rank must be between 0 and 63")
    return args


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    config = Glm52MoeDsaConfig()
    weight_map = json.loads(
        (checkpoint_dir / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )["weight_map"]
    oracle = torch.load(args.oracle.resolve(), map_location="cpu", weights_only=True)
    input_ids = oracle["input_ids"][0, : args.real_tokens].long()
    embedding = read_tensor(
        checkpoint_dir,
        weight_map,
        "model.embed_tokens.weight",
    )
    input_norm = read_tensor(
        checkpoint_dir,
        weight_map,
        "model.layers.0.input_layernorm.weight",
    )
    hidden = F.embedding(input_ids, embedding)
    normalized = glm52_rms_norm(
        hidden,
        input_norm,
        eps=config.rms_norm_eps,
    )
    padded = torch.zeros(args.tokens, config.hidden_size, dtype=torch.bfloat16)
    padded[: args.real_tokens].copy_(normalized)

    attention = Glm52MlaAttention(
        config,
        layer_idx=0,
        cache_layout=Glm52CacheLayout.build(
            config,
            world_size=64,
            cache_dtype=torch.bfloat16,
        ),
        world_size=64,
        static_fp8=True,
    )
    layer_prefix = "model.layers.0.self_attn"
    for name in PROJECTIONS:
        load_column(
            getattr(attention, name),
            checkpoint_dir,
            weight_map,
            f"{layer_prefix}.{name}",
            rank=args.rank,
        )
    with torch.no_grad():
        attention.q_a_layernorm.copy_(
            read_tensor(
                checkpoint_dir,
                weight_map,
                f"{layer_prefix}.q_a_layernorm.weight",
            )
        )
        attention.kv_a_layernorm.copy_(
            read_tensor(
                checkpoint_dir,
                weight_map,
                f"{layer_prefix}.kv_a_layernorm.weight",
            )
        )

    q_a = static_linear(padded, attention.q_a_proj)
    q_resid = glm52_rms_norm(
        q_a,
        attention.q_a_layernorm,
        eps=config.rms_norm_eps,
    )
    compressed = static_linear(padded, attention.kv_a_proj_with_mqa)
    kv_pass = compressed[:, : config.kv_lora_rank]
    k_pass = glm52_rms_norm(
        kv_pass,
        attention.kv_a_layernorm,
        eps=config.rms_norm_eps,
    )
    inputs = {
        "q_a_proj": padded,
        "q_b_proj": q_resid,
        "kv_a_proj_with_mqa": padded,
        "kv_b_proj": k_pass,
    }
    module = getattr(attention, args.projection)
    projection_input = inputs[args.projection]
    expected = static_linear(projection_input, module)

    device = torch.device(args.device)
    module = module.to(device)
    started = time.perf_counter()
    compiled = torch.compile(module, backend=get_compile_backend_name())
    actual = compiled(projection_input.to(device)).cpu()
    elapsed = time.perf_counter() - started

    actual = actual[: args.real_tokens].float()
    expected = expected[: args.real_tokens].float()
    delta = actual - expected
    relative_l2 = float(
        torch.linalg.vector_norm(delta)
        / torch.linalg.vector_norm(expected).clamp_min(torch.finfo(torch.float32).tiny)
    )
    result = {
        "status": "measured",
        "projection": args.projection,
        "rank": args.rank,
        "tokens": args.tokens,
        "real_tokens": args.real_tokens,
        "compile_and_first_run_seconds": elapsed,
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(torch.sqrt(delta.square().mean())),
        "relative_l2_error": relative_l2,
        "expected_absmax": float(expected.abs().max()),
        "actual_absmax": float(actual.abs().max()),
    }
    if relative_l2 > 0.08 or result["max_abs_error"] > 1.0:
        print(json.dumps(result, indent=2, sort_keys=True))
        raise AssertionError(f"projection parity failed: {result}")
    result["status"] = "passed"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
