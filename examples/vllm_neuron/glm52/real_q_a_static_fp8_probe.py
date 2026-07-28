# SPDX-License-Identifier: Apache-2.0
"""Run layer-0 q_a with real GLM-5.2 weights and calibration inputs on Trn2."""

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
from vllm_neuron.model.glm52_moe_dsa.mla import _ColumnProjection
from vllm_neuron.model.glm52_moe_dsa.sparse_mlp import glm52_rms_norm


OCP_MAX = 448.0
NEURON_MAX = 240.0
SCALE_ROWS = 128


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--real-tokens", type=int, default=512)
    parser.add_argument("--device", default="neuron:0")
    args = parser.parse_args()
    if args.tokens <= 0 or not 0 < args.real_tokens <= args.tokens:
        parser.error("require 0 < --real-tokens <= --tokens")
    return args


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
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
    norm_weight = read_tensor(
        checkpoint_dir,
        weight_map,
        "model.layers.0.input_layernorm.weight",
    )
    hidden = F.embedding(input_ids, embedding)
    normalized = glm52_rms_norm(
        hidden,
        norm_weight,
        eps=float(config["rms_norm_eps"]),
    )
    padded = torch.zeros(
        args.tokens,
        int(config["hidden_size"]),
        dtype=torch.bfloat16,
    )
    padded[: args.real_tokens].copy_(normalized)

    prefix = "model.layers.0.self_attn.q_a_proj"
    ocp_weight = read_tensor(
        checkpoint_dir,
        weight_map,
        f"{prefix}.weight",
    )
    ocp_weight_scale = read_tensor(
        checkpoint_dir,
        weight_map,
        f"{prefix}.weight_scale",
    ).float()
    input_scale = read_tensor(
        checkpoint_dir,
        weight_map,
        f"{prefix}.input_scale",
    ).float()
    legacy_weight = (
        (ocp_weight.float() * (NEURON_MAX / OCP_MAX))
        .clamp(-NEURON_MAX, NEURON_MAX)
        .to(torch.float8_e4m3fn)
        .T.contiguous()
    )
    legacy_weight_scale = ocp_weight_scale * (OCP_MAX / NEURON_MAX)
    quantized_input = (
        (padded.float() / input_scale)
        .clamp(-NEURON_MAX, NEURON_MAX)
        .to(torch.float8_e4m3fn)
        .float()
    )
    expected = (
        (quantized_input * input_scale)
        @ (legacy_weight.float() * legacy_weight_scale)
    ).to(torch.bfloat16)

    projection = _ColumnProjection(
        int(config["hidden_size"]),
        int(config["q_lora_rank"]),
        shard_output=False,
        world_size=64,
        static_fp8=True,
        dtype=torch.bfloat16,
        device=None,
    )
    with torch.no_grad():
        projection.weight.copy_(legacy_weight)
        projection.weight_scale.copy_(
            legacy_weight_scale.reshape(1, 1).expand(SCALE_ROWS, 3)
        )
        projection.input_scale.copy_(
            input_scale.reshape(1, 1).expand(SCALE_ROWS, 1)
        )

    device = torch.device(args.device)
    projection = projection.to(device)
    device_input = padded.to(device)
    started = time.perf_counter()
    compiled = torch.compile(projection, backend=get_compile_backend_name())
    actual = compiled(device_input).cpu()
    compile_and_first_run = time.perf_counter() - started

    delta = actual[: args.real_tokens].float() - expected[: args.real_tokens].float()
    reference = expected[: args.real_tokens].float()
    relative_l2 = float(
        torch.linalg.vector_norm(delta)
        / torch.linalg.vector_norm(reference).clamp_min(torch.finfo(torch.float32).tiny)
    )
    sample_delta = delta[-1]
    result = {
        "status": "measured",
        "backend": get_compile_backend_name(),
        "projection": prefix,
        "tokens": args.tokens,
        "real_tokens": args.real_tokens,
        "input_scale": float(input_scale),
        "ocp_weight_scale": float(ocp_weight_scale),
        "legacy_weight_scale": float(legacy_weight_scale),
        "compile_and_first_run_seconds": compile_and_first_run,
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(torch.sqrt(delta.square().mean())),
        "relative_l2_error": relative_l2,
        "sample_position": args.real_tokens - 1,
        "sample_max_abs_error": float(sample_delta.abs().max()),
        "sample_mean_abs_error": float(sample_delta.abs().mean()),
        "expected_absmax": float(reference.abs().max()),
        "actual_absmax": float(actual[: args.real_tokens].float().abs().max()),
    }
    if relative_l2 > 0.08 or result["max_abs_error"] > 1.0:
        print(json.dumps(result, indent=2, sort_keys=True))
        raise AssertionError(f"real q_a projection parity failed: {result}")
    result["status"] = "passed"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
