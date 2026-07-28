# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 MoE/DSA model components."""

from .attention import (
    apply_glm52_interleaved_rope,
    glm52_index_scores,
    glm52_index_topk,
    glm52_paged_sparse_attention,
    glm52_sparse_attention,
)
from .checkpoint_mapping import (
    Glm52CheckpointContract,
    build_checkpoint_contract,
    routed_down_scale_loader,
    routed_down_weight_loader,
    routed_gate_up_scale_loader,
    routed_gate_up_weight_loader,
)
from .cache_layout import Glm52CacheLayout, IndexerCacheBinding
from .cache_ops import (
    gather_paged_cache,
    gather_selected_paged_cache,
    write_paged_cache,
)
from .config import Glm52MoeDsaConfig
from .dense_mlp import Glm52DenseMlp
from .expert_kernels import Glm52RoutedExperts, dense_glm52_affinities
from .moe import Glm52ExpertRouter, select_glm52_experts
from .indexer import (
    Glm52FullIndexer,
    Glm52IndexerProjection,
    Glm52IndexShareState,
    advance_index_share_state,
)
from .mla import Glm52MlaAttention, Glm52MlaProjection
from .parallelism import RoutedExpertPlan
from .shared_expert import Glm52SharedExpert
from .sparse_mlp import Glm52SparseMlp, glm52_rms_norm
from .weight_manifest import (
    WeightSpec,
    estimate_local_weight_bytes,
    iter_backbone_weight_specs,
)

__all__ = [
    "Glm52ExpertRouter",
    "Glm52FullIndexer",
    "Glm52CheckpointContract",
    "Glm52CacheLayout",
    "Glm52IndexerProjection",
    "Glm52IndexShareState",
    "Glm52MlaAttention",
    "Glm52MlaProjection",
    "Glm52MoeDsaConfig",
    "Glm52DenseMlp",
    "Glm52RoutedExperts",
    "Glm52SharedExpert",
    "Glm52SparseMlp",
    "IndexerCacheBinding",
    "RoutedExpertPlan",
    "WeightSpec",
    "apply_glm52_interleaved_rope",
    "advance_index_share_state",
    "build_checkpoint_contract",
    "dense_glm52_affinities",
    "glm52_index_scores",
    "glm52_index_topk",
    "glm52_paged_sparse_attention",
    "glm52_sparse_attention",
    "glm52_rms_norm",
    "estimate_local_weight_bytes",
    "gather_paged_cache",
    "gather_selected_paged_cache",
    "iter_backbone_weight_specs",
    "routed_down_scale_loader",
    "routed_down_weight_loader",
    "routed_gate_up_scale_loader",
    "routed_gate_up_weight_loader",
    "select_glm52_experts",
    "write_paged_cache",
]
