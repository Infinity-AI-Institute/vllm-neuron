# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.dense_mlp import Glm52DenseMlp
from vllm_neuron.model.glm52_moe_dsa.static_fp8 import (
    NEURON_LEGACY_E4M3FN_QMAX240,
)
from vllm_neuron.utils.weight_loader import get_weight_loader


def _config() -> Glm52MoeDsaConfig:
    return Glm52MoeDsaConfig(
        hidden_size=4,
        num_hidden_layers=4,
        intermediate_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_head_dim=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        first_k_dense_replace=3,
    )


def test_dense_mlp_decode_matches_frozen_silu_equation() -> None:
    config = _config()
    module = Glm52DenseMlp(
        config,
        world_size=1,
        global_rank=0,
        static_fp8=False,
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(52)
    with torch.no_grad():
        module.gate_proj.weight.copy_(
            torch.randn(module.gate_proj.weight.shape, generator=generator)
        )
        module.up_proj.weight.copy_(
            torch.randn(module.up_proj.weight.shape, generator=generator)
        )
        module.down_proj.weight.copy_(
            torch.randn(module.down_proj.weight.shape, generator=generator)
        )

    hidden = torch.randn(3, config.hidden_size, generator=generator)
    norm_weight = torch.randn(config.hidden_size, generator=generator)
    normalized = hidden.float() * torch.rsqrt(
        hidden.float().square().mean(-1, keepdim=True) + config.rms_norm_eps
    )
    normalized = normalized * norm_weight
    expected = (
        F.silu(normalized @ module.gate_proj.weight)
        * (normalized @ module.up_proj.weight)
    ) @ module.down_proj.weight

    actual = module.forward_decode(hidden, norm_weight=norm_weight)

    torch.testing.assert_close(actual, expected)


def test_exact_dense_static_fp8_rank_shapes() -> None:
    config = Glm52MoeDsaConfig()
    module = Glm52DenseMlp(
        config,
        world_size=64,
        global_rank=0,
        tp_group=type("Group", (), {"world_size": 64})(),
        static_fp8=True,
        device="meta",
    )

    assert module.local_intermediate_size == 192
    assert module.kernel_intermediate_size == 256
    assert module.gate_proj.weight.shape == (6_144, 256)
    assert module.up_proj.weight.shape == (6_144, 256)
    assert module.down_proj.weight.shape == (256, 6_144)
    assert module.gate_proj.weight_scale.shape == (128, 1)
    assert module.gate_up_input_scale.shape == (128, 1)


class _TensorSlice:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def get_shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)

    def __getitem__(self, item) -> torch.Tensor:
        return self.tensor[item]


def test_dense_static_fp8_loader_zero_pads_kernel_only() -> None:
    module = Glm52DenseMlp(
        _config(),
        world_size=1,
        global_rank=0,
        static_fp8=True,
    )
    gate_checkpoint = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    gate_checkpoint = gate_checkpoint.to(torch.float8_e4m3fn)
    down_checkpoint = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    down_checkpoint = down_checkpoint.to(torch.float8_e4m3fn)

    gate = get_weight_loader(module.gate_proj.weight).load(
        [_TensorSlice(gate_checkpoint)],
        0,
    )
    down = get_weight_loader(module.down_proj.weight).load(
        [_TensorSlice(down_checkpoint)],
        0,
    )

    assert gate.shape == (4, 128)
    assert down.shape == (128, 4)
    assert torch.count_nonzero(gate[:, 8:].float()) == 0
    assert torch.count_nonzero(down[8:, :].float()) == 0


def test_dense_direct_legacy_loader_preserves_layout_bytes_and_scale() -> None:
    config = _config()
    config.static_fp8_weight_format = NEURON_LEGACY_E4M3FN_QMAX240
    module = Glm52DenseMlp(
        config,
        world_size=1,
        global_rank=0,
        static_fp8=True,
    )
    checkpoint = torch.tensor(
        [
            [-240.0, -128.0, -1.0, 0.0],
            [1.0, 16.0, 128.0, 240.0],
            [-224.0, -112.0, -0.5, 0.5],
            [2.0, 32.0, 112.0, 224.0],
            [-208.0, -96.0, -0.25, 0.25],
            [4.0, 48.0, 96.0, 208.0],
            [-192.0, -80.0, -2.0, 8.0],
            [12.0, 64.0, 80.0, 192.0],
        ],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)

    loaded = get_weight_loader(module.gate_proj.weight).load(
        [_TensorSlice(checkpoint)],
        0,
    )
    loaded_scale = get_weight_loader(module.gate_proj.weight_scale).load(
        [_TensorSlice(torch.tensor(0.125, dtype=torch.float32))],
        0,
    )

    assert torch.equal(
        loaded[:, :8].contiguous().view(torch.uint8),
        checkpoint.T.contiguous().view(torch.uint8),
    )
    torch.testing.assert_close(loaded_scale, torch.full((128, 1), 0.125))
