# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 MoE/DSA model components."""

from .attention import (
    apply_glm52_interleaved_rope,
    glm52_index_scores,
    glm52_index_topk,
)
from .config import Glm52MoeDsaConfig
from .moe import Glm52ExpertRouter, select_glm52_experts
from .parallelism import RoutedExpertPlan
from .weight_manifest import (
    WeightSpec,
    estimate_local_weight_bytes,
    iter_backbone_weight_specs,
)

__all__ = [
    "Glm52ExpertRouter",
    "Glm52MoeDsaConfig",
    "RoutedExpertPlan",
    "WeightSpec",
    "apply_glm52_interleaved_rope",
    "glm52_index_scores",
    "glm52_index_topk",
    "estimate_local_weight_bytes",
    "iter_backbone_weight_specs",
    "select_glm52_experts",
]
