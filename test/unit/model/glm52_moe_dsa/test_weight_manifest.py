# SPDX-License-Identifier: Apache-2.0

from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.weight_manifest import (
    estimate_local_weight_bytes,
    iter_backbone_weight_specs,
)


def test_frozen_attention_and_expert_shapes() -> None:
    config = Glm52MoeDsaConfig()
    specs = {
        spec.name: spec
        for spec in iter_backbone_weight_specs(config)
    }

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


def test_tp64_static_fp8_accounting_is_bounded() -> None:
    local_bytes = estimate_local_weight_bytes(
        Glm52MoeDsaConfig(),
        world_size=64,
        weight_bytes_per_element=1,
    )

    # Replicated MLA low-rank projections, indexers, norms, and routers make
    # checkpoint_bytes / 64 an invalid per-rank estimate.
    assert local_bytes == 13_164_143_616
    assert 12 * 1024**3 < local_bytes < 13 * 1024**3
