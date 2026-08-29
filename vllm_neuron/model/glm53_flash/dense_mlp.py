# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2-compatible dense MLP with GLM-5.3 clamped SwiGLU."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from .config import Glm53FlashInferenceConfig

if TYPE_CHECKING:
    from ..glm52_moe_dsa.dense_mlp import Glm52DenseMlp


class Glm53DenseMlp(nn.Module):
    def __init__(self, config: Glm53FlashInferenceConfig) -> None:
        super().__init__()
        self.limit = config.swiglu_limit
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            dtype=config.torch_dtype,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            dtype=config.torch_dtype,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states).clamp(max=self.limit)
        up = self.up_proj(hidden_states).clamp(-self.limit, self.limit)
        return self.down_proj(F.silu(gate) * up)


def __getattr__(name: str):
    # Preserve the requested 5.2 re-export without importing its Neuron-only
    # dependencies on the CPU reference path.
    if name == "Glm52DenseMlp":
        from ..glm52_moe_dsa.dense_mlp import Glm52DenseMlp

        return Glm52DenseMlp
    raise AttributeError(name)


__all__ = ["Glm52DenseMlp", "Glm53DenseMlp"]
