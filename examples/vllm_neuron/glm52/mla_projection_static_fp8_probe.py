# SPDX-License-Identifier: Apache-2.0
"""Compile/numeric probe for GLM-5.2's exact TP64 MLA projections."""

import json
import statistics
import time

import torch

import vllm_neuron  # noqa: F401
from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.model.glm52_moe_dsa.cache_layout import Glm52CacheLayout
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.mla import Glm52MlaAttention
from vllm_neuron.model.glm52_moe_dsa.sparse_mlp import glm52_rms_norm

_FP8_MAX = 240.0
_SCALE_ROWS = 128


def _quantize_static(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (
        tensor.abs().amax().to(torch.float32) / _FP8_MAX
    ).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (tensor.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(
        torch.float8_e4m3fn
    )
    return quantized, scale


def _dequantize_activation(
    tensor: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return (
        (tensor.float() / scale)
        .clamp(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
        .float()
        * scale
    )


def _rows(scale: torch.Tensor, columns: int = 1) -> torch.Tensor:
    return scale.reshape(1, 1).expand(_SCALE_ROWS, columns).contiguous()


def _prepare_projection(
    projection,
    input_ref: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    weight_ref = (
        torch.randn(
            projection.weight.shape,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    )
    weight, weight_scale = _quantize_static(weight_ref)
    _, input_scale = _quantize_static(input_ref)
    with torch.no_grad():
        projection.weight.copy_(weight)
        projection.weight_scale.copy_(
            _rows(weight_scale, projection.weight_scale.shape[1])
        )
        projection.input_scale.copy_(_rows(input_scale))
    input_dequant = _dequantize_activation(input_ref, input_scale)
    return (
        input_dequant @ (weight.float() * weight_scale)
    ).to(torch.bfloat16)


class MlaProjectionProbe(torch.nn.Module):
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
        output = self.attention.o_proj(projected.value.reshape(1, -1))
        return torch.cat(
            (
                projected.q_resid.reshape(1, -1),
                projected.query.reshape(1, -1),
                projected.key.reshape(1, -1),
                projected.value.reshape(1, -1),
                output.reshape(1, -1),
            ),
            dim=-1,
        )


def _make_probe(
    *,
    layer_idx: int = 3,
) -> tuple[MlaProjectionProbe, tuple[torch.Tensor, ...], torch.Tensor]:
    config = Glm52MoeDsaConfig()
    layout = Glm52CacheLayout.build(
        config,
        world_size=64,
        cache_dtype=torch.bfloat16,
    )
    attention = Glm52MlaAttention(
        config,
        layer_idx=layer_idx,
        cache_layout=layout,
        world_size=64,
        static_fp8=True,
    )
    generator = torch.Generator().manual_seed(5_252)
    hidden = torch.randn(
        1,
        config.hidden_size,
        dtype=torch.bfloat16,
        generator=generator,
    )
    cos = torch.ones(
        1,
        config.qk_rope_head_dim,
        dtype=torch.bfloat16,
    )
    sin = torch.zeros_like(cos)
    with torch.no_grad():
        attention.q_a_layernorm.fill_(1)
        attention.kv_a_layernorm.fill_(1)

    q_a = _prepare_projection(
        attention.q_a_proj,
        hidden,
        generator=generator,
    )
    q_resid = glm52_rms_norm(
        q_a,
        attention.q_a_layernorm,
        eps=config.rms_norm_eps,
    )
    query = _prepare_projection(
        attention.q_b_proj,
        q_resid,
        generator=generator,
    ).reshape(1, 1, config.qk_head_dim)

    compressed = _prepare_projection(
        attention.kv_a_proj_with_mqa,
        hidden,
        generator=generator,
    )
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
    expanded = _prepare_projection(
        attention.kv_b_proj,
        k_pass,
        generator=generator,
    ).reshape(
        1,
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
    q_rot = torch.cat((q_rot[..., 0::2], q_rot[..., 1::2]), dim=-1)
    k_rot = torch.cat((k_rot[..., 0::2], k_rot[..., 1::2]), dim=-1)
    query = torch.cat((q_pass, q_rot), dim=-1)
    key = torch.cat((k_nope, k_rot.unsqueeze(1)), dim=-1)

    output = _prepare_projection(
        attention.o_proj,
        value.reshape(1, config.v_head_dim),
        generator=generator,
    )
    expected = torch.cat(
        (
            q_resid.reshape(1, -1),
            query.reshape(1, -1),
            key.reshape(1, -1),
            value.reshape(1, -1),
            output.reshape(1, -1),
        ),
        dim=-1,
    )
    return MlaProjectionProbe(attention), (hidden, cos, sin), expected


def main() -> None:
    model, inputs, expected = _make_probe()
    config = Glm52MoeDsaConfig()
    device = torch.device("neuron:0")
    model = model.to(device)
    device_inputs = tuple(tensor.to(device) for tensor in inputs)

    started = time.perf_counter()
    compiled = torch.compile(model, backend=get_compile_backend_name())
    output = compiled(*device_inputs).to("cpu")
    elapsed = time.perf_counter() - started
    max_abs_error = (output.float() - expected.float()).abs().max().item()
    torch.testing.assert_close(
        output.float(),
        expected.float(),
        atol=1.0,
        rtol=0.12,
    )

    hot_seconds = []
    for _ in range(5):
        started = time.perf_counter()
        compiled(*device_inputs).to("cpu")
        hot_seconds.append(time.perf_counter() - started)

    print(
        json.dumps(
            {
                "status": "passed",
                "backend": get_compile_backend_name(),
                "precision": "static_fp8",
                "hidden_size": config.hidden_size,
                "q_lora_rank": config.q_lora_rank,
                "kv_lora_rank": config.kv_lora_rank,
                "local_qk_dim": config.qk_head_dim,
                "local_value_dim": config.v_head_dim,
                "compile_and_first_run_seconds": elapsed,
                "hot_run_seconds_min": min(hot_seconds),
                "hot_run_seconds_median": statistics.median(hot_seconds),
                "max_abs_error": max_abs_error,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
