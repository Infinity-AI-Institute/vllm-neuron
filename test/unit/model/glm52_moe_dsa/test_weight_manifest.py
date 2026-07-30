# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.weight_manifest import (
    estimate_local_weight_bytes,
    iter_backbone_weight_specs,
)


def _specs():
    return {spec.name: spec for spec in iter_backbone_weight_specs(Glm52MoeDsaConfig())}


def test_frozen_attention_and_expert_shapes() -> None:
    specs = _specs()

    assert specs["model.layers.0.self_attn.q_b_proj.weight"].shape == (
        16_384,
        2_048,
    )
    assert specs["model.layers.0.self_attn.kv_b_proj.weight"].shape == (
        28_672,
        512,
    )
    assert specs["model.layers.0.self_attn.indexer.wq_b.weight"].shape == (
        4_096,
        2_048,
    )
    assert "model.layers.3.self_attn.indexer.wq_b.weight" not in specs
    assert specs["model.layers.3.mlp.experts.gate_up_proj"].shape == (
        256,
        4_096,
        6_144,
    )
    assert specs["model.layers.3.mlp.experts.down_proj"].shape == (
        256,
        6_144,
        2_048,
    )
    assert all(not name.startswith("model.layers.78.") for name in specs)


def test_manifest_tracks_mixed_dtypes_and_expanded_scales() -> None:
    specs = _specs()

    assert specs["model.embed_tokens.weight"].dtype == "bf16"
    assert specs["lm_head.weight"].dtype == "bf16"
    assert specs["model.layers.0.self_attn.q_a_proj.weight"].dtype == "fp8"
    assert specs["model.layers.3.mlp.gate.weight"].dtype == "fp32"
    assert specs["model.layers.3.mlp.gate.e_score_correction_bias"].dtype == "fp32"
    assert specs["model.layers.0.self_attn.q_a_proj.weight_scale"].shape == (128, 3)
    assert specs["model.layers.3.mlp.experts.gate_up_proj_scale"].dtype == "fp32"
    assert specs["model.layers.3.mlp.shared_experts.gate_up_input_scale"].shape == (
        128,
        1,
    )


def test_dense_tp64_accounts_for_128_element_kernel_padding() -> None:
    specs = _specs()
    gate = specs["model.layers.0.mlp.gate_proj.weight"]
    down = specs["model.layers.0.mlp.down_proj.weight"]

    # Logical I/TP is 12,288/64 = 192; the kernel parameter pads it to 256.
    assert gate.local_numel(world_size=64, ep_degree=16) == 6_144 * 256
    assert down.local_numel(world_size=64, ep_degree=16) == 256 * 6_144
    assert gate.local_numel(64, 16) - gate.numel // 64 == 6_144 * 64


def test_ep16_expanded_scale_storage_matches_kernel_parameters() -> None:
    specs = _specs()
    routed_gate_up = specs["model.layers.3.mlp.experts.gate_up_proj_scale"]
    routed_down = specs["model.layers.3.mlp.experts.down_proj_scale"]

    assert routed_gate_up.local_numel(64, 16) == 16 * 2 * 512
    # Down scales span the full hidden dimension and repeat on all four
    # expert-TP ranks in each EP16 partition.
    assert routed_down.placement == "expert_sharded"
    assert routed_down.local_numel(64, 16) == 16 * 6_144
    assert (
        sum(spec.local_bytes(64, 16) for spec in specs.values() if "scale" in spec.name)
        == 35_324_928
    )


@pytest.mark.parametrize(
    ("ep_degree", "shared_weight_local_numel"),
    [
        (8, 2_048 * 6_144 // 8),
        (16, 2_048 * 6_144 // 4),
        (32, 2_048 * 6_144 // 2),
        (64, 2_048 * 6_144),
    ],
)
def test_shared_expert_is_sharded_only_over_expert_tp_subgroup(
    ep_degree: int,
    shared_weight_local_numel: int,
) -> None:
    shared_gate = _specs()["model.layers.3.mlp.shared_experts.gate_proj.weight"]

    assert shared_gate.placement == "expert_tp_sharded"
    assert shared_gate.local_numel(64, ep_degree) == shared_weight_local_numel


def test_hybrid_manifest_uses_bf16_shared_weights_and_no_shared_scales() -> None:
    config = Glm52MoeDsaConfig(shared_expert_dtype="bfloat16")
    specs = {spec.name: spec for spec in iter_backbone_weight_specs(config)}
    shared = "model.layers.3.mlp.shared_experts"

    assert specs[f"{shared}.gate_proj.weight"].dtype == "bf16"
    assert specs[f"{shared}.up_proj.weight"].dtype == "bf16"
    assert specs[f"{shared}.down_proj.weight"].dtype == "bf16"
    assert f"{shared}.gate_proj.weight_scale" not in specs
    assert f"{shared}.gate_up_input_scale" not in specs

    static_bytes = estimate_local_weight_bytes(
        Glm52MoeDsaConfig(),
        world_size=64,
        ep_degree=16,
    )
    hybrid_bytes = estimate_local_weight_bytes(
        config,
        world_size=64,
        ep_degree=16,
    )
    assert hybrid_bytes == 15_164_077_056
    assert hybrid_bytes - static_bytes == 707_596_800


@pytest.mark.parametrize(
    ("ep_degree", "expected_bytes"),
    [
        (8, 14_132_077_056),
        (16, 14_456_480_256),
        (32, 15_149_523_456),
        (64, 16_557_728_256),
    ],
)
def test_tp64_static_fp8_exact_bytes(
    ep_degree: int,
    expected_bytes: int,
) -> None:
    assert (
        estimate_local_weight_bytes(
            Glm52MoeDsaConfig(),
            world_size=64,
            ep_degree=ep_degree,
        )
        == expected_bytes
    )


def test_invalid_expert_topology_is_rejected() -> None:
    with pytest.raises(ValueError, match="world_size must be divisible"):
        estimate_local_weight_bytes(
            Glm52MoeDsaConfig(),
            world_size=64,
            ep_degree=10,
        )
