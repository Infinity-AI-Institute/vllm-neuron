# SPDX-License-Identifier: Apache-2.0
"""Thin GLM-5.3 wrapper over the qualified GLM-5.2 MoE layout."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._reference_kernels import load_glm52_moe, load_reference_kernel
from .config import Glm53FlashInferenceConfig
from .dense_mlp import Glm53DenseMlp

MOE_KERNEL_SLUG = "moe_dispatch.reference.v1"
_GLM52_MOE = load_glm52_moe()
Glm52ExpertRouter = _GLM52_MOE.Glm52ExpertRouter
select_glm52_experts = _GLM52_MOE.select_glm52_experts


class Glm53SparseMlp(nn.Module):
    """CPU qualification path; device dispatch uses the imported Fleet-A ABI."""

    def __init__(self, config: Glm53FlashInferenceConfig) -> None:
        super().__init__()
        # Import, rather than copy, the session-qualified dispatch definitions.
        load_reference_kernel("moe")
        self.config = config
        self.gate_router = Glm52ExpertRouter(
            config.hidden_size,
            config.n_routed_experts,
            config.num_experts_per_tok,
            config.routed_scaling_factor,
            topk_backend="torch",
        )
        self.gate = nn.Parameter(
            torch.empty(
                config.n_routed_experts,
                config.hidden_size,
                config.moe_intermediate_size,
                dtype=config.torch_dtype,
            )
        )
        self.up = nn.Parameter(torch.empty_like(self.gate))
        self.down = nn.Parameter(
            torch.empty(
                config.n_routed_experts,
                config.moe_intermediate_size,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        )
        nn.init.normal_(self.gate, std=config.hidden_size**-0.5)
        nn.init.normal_(self.up, std=config.hidden_size**-0.5)
        nn.init.normal_(self.down, std=config.moe_intermediate_size**-0.5)
        self.shared_expert = _SharedExpert(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        tokens = hidden_states.reshape(-1, shape[-1])
        selected, weights = self.gate_router(tokens)
        routed = torch.zeros_like(tokens)
        limit = self.config.swiglu_limit
        for choice in range(self.config.num_experts_per_tok):
            expert_ids = selected[:, choice]
            for expert_id in expert_ids.unique().tolist():
                mask = expert_ids == expert_id
                local = tokens[mask]
                gate = (local @ self.gate[expert_id]).clamp(max=limit)
                up = (local @ self.up[expert_id]).clamp(-limit, limit)
                result = (F.silu(gate) * up) @ self.down[expert_id]
                routed[mask] += result * weights[mask, choice : choice + 1].to(
                    result.dtype
                )
        return (routed + self.shared_expert(tokens)).view(shape)


class _SharedExpert(Glm53DenseMlp):
    def __init__(self, config: Glm53FlashInferenceConfig) -> None:
        nn.Module.__init__(self)
        self.limit = config.swiglu_limit
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.moe_intermediate_size,
            bias=False,
            dtype=config.torch_dtype,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.moe_intermediate_size,
            bias=False,
            dtype=config.torch_dtype,
        )
        self.down_proj = nn.Linear(
            config.moe_intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
        )


__all__ = [
    "MOE_KERNEL_SLUG",
    "Glm52ExpertRouter",
    "Glm53SparseMlp",
    "select_glm52_experts",
]
