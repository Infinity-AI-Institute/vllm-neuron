# SPDX-License-Identifier: Apache-2.0
"""Real-weight numeric probe for every rank-0 GLM-5.2 MLA projection."""

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
from vllm_neuron.model.glm52_moe_dsa.mla import (
    Glm52MlaAttention,
    apply_glm52_interleaved_rope,
)
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


def legacy_weight(source: torch.Tensor) -> torch.Tensor:
    return (
        (source.float() * (NEURON_MAX / OCP_MAX))
        .clamp(-NEURON_MAX, NEURON_MAX)
        .to(torch.float8_e4m3fn)
    )


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
    loaded = legacy_weight(source).T.contiguous()
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
        module.weight.copy_(loaded)
        module.weight_scale.copy_(
            weight_scale.reshape(1, 1).expand(SCALE_ROWS, 3)
        )
        module.input_scale.copy_(
            input_scale.reshape(1, 1).expand(SCALE_ROWS, 1)
        )


def load_output(
    module,
    checkpoint_dir: Path,
    weight_map: dict[str, str],
    prefix: str,
    *,
    rank: int,
) -> None:
    source = read_tensor(checkpoint_dir, weight_map, f"{prefix}.weight")
    local_input = module.weight.shape[0]
    source = source[:, rank * local_input : (rank + 1) * local_input]
    loaded = legacy_weight(source).T.contiguous()
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
        module.weight.copy_(loaded)
        module.weight_scale.copy_(
            weight_scale.reshape(1, 1).expand(SCALE_ROWS, 1)
        )
        module.input_scale.copy_(
            input_scale.reshape(1, 1).expand(SCALE_ROWS, 1)
        )


def static_linear(hidden: torch.Tensor, module) -> torch.Tensor:
    input_scale = module.input_scale[0, 0].float()
    weight_scale = module.weight_scale[0, 0].float()
    dequantized_input = (
        (hidden.float() / input_scale)
        .clamp(-NEURON_MAX, NEURON_MAX)
        .to(torch.float8_e4m3fn)
        .float()
        * input_scale
    )
    return (
        dequantized_input @ (module.weight.float() * weight_scale)
    ).to(torch.bfloat16)


class ProjectionProbe(torch.nn.Module):
    def __init__(self, attention: Glm52MlaAttention) -> None:
        super().__init__()
        self.attention = attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.attention.project(hidden_states, cos, sin)
        output = self.attention.o_proj(projected.value.reshape(hidden_states.shape[0], -1))
        return torch.cat(
            (
                projected.q_resid.reshape(hidden_states.shape[0], -1),
                projected.query.reshape(hidden_states.shape[0], -1),
                projected.key.reshape(hidden_states.shape[0], -1),
                projected.value.reshape(hidden_states.shape[0], -1),
                output.reshape(hidden_states.shape[0], -1),
            ),
            dim=-1,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--real-tokens", type=int, default=512)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", default="neuron:0")
    parser.add_argument("--save-tensors", type=Path)
    args = parser.parse_args()
    if args.tokens <= 0 or not 0 < args.real_tokens <= args.tokens:
        parser.error("require 0 < --real-tokens <= --tokens")
    if not 0 <= args.rank < 64:
        parser.error("--rank must be between 0 and 63")
    return args


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    raw_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    config = Glm52MoeDsaConfig()
    for field_name in (
        "hidden_size",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_head_dim",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    ):
        if getattr(config, field_name) != int(raw_config[field_name]):
            raise ValueError(f"checkpoint disagrees with {field_name}")
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

    layout = Glm52CacheLayout.build(
        config,
        world_size=64,
        cache_dtype=torch.bfloat16,
    )
    attention = Glm52MlaAttention(
        config,
        layer_idx=0,
        cache_layout=layout,
        world_size=64,
        static_fp8=True,
    )
    layer_prefix = "model.layers.0.self_attn"
    for name in (
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
    ):
        load_column(
            getattr(attention, name),
            checkpoint_dir,
            weight_map,
            f"{layer_prefix}.{name}",
            rank=args.rank,
        )
    load_output(
        attention.o_proj,
        checkpoint_dir,
        weight_map,
        f"{layer_prefix}.o_proj",
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

    cos = torch.ones(
        args.tokens,
        config.qk_rope_head_dim,
        dtype=torch.bfloat16,
    )
    sin = torch.zeros_like(cos)
    q_a = static_linear(padded, attention.q_a_proj)
    q_resid = glm52_rms_norm(
        q_a,
        attention.q_a_layernorm,
        eps=config.rms_norm_eps,
    )
    query = static_linear(q_resid, attention.q_b_proj).reshape(
        args.tokens,
        1,
        config.qk_head_dim,
    )
    compressed = static_linear(padded, attention.kv_a_proj_with_mqa)
    kv_pass, k_rot = torch.split(
        compressed,
        [config.kv_lora_rank, config.qk_rope_head_dim],
        dim=-1,
    )
    k_pass = glm52_rms_norm(
        kv_pass,
        attention.kv_a_layernorm,
        eps=config.rms_norm_eps,
    )
    expanded = static_linear(k_pass, attention.kv_b_proj).reshape(
        args.tokens,
        1,
        config.qk_nope_head_dim + config.v_head_dim,
    )
    k_nope, value = torch.split(
        expanded,
        [config.qk_nope_head_dim, config.v_head_dim],
        dim=-1,
    )
    q_pass, q_rot = torch.split(
        query,
        [config.qk_nope_head_dim, config.qk_rope_head_dim],
        dim=-1,
    )
    q_rot, k_rot = apply_glm52_interleaved_rope(
        q_rot,
        k_rot.unsqueeze(-2),
        cos,
        sin,
        unsqueeze_dim=-2,
    )
    query = torch.cat((q_pass, q_rot), dim=-1)
    key = torch.cat((k_nope, k_rot.expand(-1, 1, -1)), dim=-1)
    output = static_linear(value.reshape(args.tokens, -1), attention.o_proj)
    expected = torch.cat(
        (
            q_resid.reshape(args.tokens, -1),
            query.reshape(args.tokens, -1),
            key.reshape(args.tokens, -1),
            value.reshape(args.tokens, -1),
            output.reshape(args.tokens, -1),
        ),
        dim=-1,
    )

    device = torch.device(args.device)
    model = ProjectionProbe(attention).to(device)
    started = time.perf_counter()
    compiled = torch.compile(model, backend=get_compile_backend_name())
    actual = compiled(padded.to(device), cos.to(device), sin.to(device)).cpu()
    compile_and_first_run = time.perf_counter() - started

    actual = actual[: args.real_tokens].float()
    expected = expected[: args.real_tokens].float()
    if args.save_tensors is not None:
        args.save_tensors.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actual": actual,
                "expected": expected,
            },
            args.save_tensors,
        )
        args.save_tensors.chmod(0o644)
    delta = actual - expected
    relative_l2 = float(
        torch.linalg.vector_norm(delta)
        / torch.linalg.vector_norm(expected).clamp_min(torch.finfo(torch.float32).tiny)
    )
    segments = {}
    offset = 0
    for name, width in (
        ("q_resid", config.q_lora_rank),
        ("query_nope", config.qk_nope_head_dim),
        ("query_rope", config.qk_rope_head_dim),
        ("key_nope", config.qk_nope_head_dim),
        ("key_rope", config.qk_rope_head_dim),
        ("value", config.v_head_dim),
        ("o_proj", config.hidden_size),
    ):
        segment_actual = actual[:, offset : offset + width]
        segment_expected = expected[:, offset : offset + width]
        segment_delta = segment_actual - segment_expected
        segment_reference_norm = torch.linalg.vector_norm(segment_expected)
        segments[name] = {
            "max_abs_error": float(segment_delta.abs().max()),
            "mean_abs_error": float(segment_delta.abs().mean()),
            "rmse": float(torch.sqrt(segment_delta.square().mean())),
            "relative_l2_error": float(
                torch.linalg.vector_norm(segment_delta)
                / segment_reference_norm.clamp_min(torch.finfo(torch.float32).tiny)
            ),
            "expected_absmax": float(segment_expected.abs().max()),
            "actual_absmax": float(segment_actual.abs().max()),
            "sample_max_abs_error": float(segment_delta[-1].abs().max()),
        }
        offset += width
    if offset != actual.shape[-1]:
        raise AssertionError(f"segment widths {offset} != output width {actual.shape[-1]}")

    result = {
        "status": "measured",
        "backend": get_compile_backend_name(),
        "rank": args.rank,
        "tokens": args.tokens,
        "real_tokens": args.real_tokens,
        "compile_and_first_run_seconds": compile_and_first_run,
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(torch.sqrt(delta.square().mean())),
        "relative_l2_error": relative_l2,
        "sample_max_abs_error": float(delta[-1].abs().max()),
        "sample_mean_abs_error": float(delta[-1].abs().mean()),
        "expected_absmax": float(expected.abs().max()),
        "actual_absmax": float(actual.abs().max()),
        "segments": segments,
    }
    if relative_l2 > 0.08 or result["max_abs_error"] > 1.0:
        print(json.dumps(result, indent=2, sort_keys=True))
        raise AssertionError(f"real MLA projection parity failed: {result}")
    result["status"] = "passed"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
