# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 router semantics and MoE building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def select_glm52_experts(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    routed_scaling_factor: float,
    topk_backend: str = "torch",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts using GLM's selection-only correction bias.

    The learned correction bias changes which experts win top-k, but it must
    never contribute to the routing weights. The weights are gathered from
    the original sigmoid scores, L1-normalized, and then scaled.
    """
    if gate_weight.ndim != 2:
        raise ValueError("gate_weight must have shape [experts, hidden]")
    if hidden_states.shape[-1] != gate_weight.shape[-1]:
        raise ValueError("hidden size does not match gate_weight")
    if correction_bias.shape != (gate_weight.shape[0],):
        raise ValueError("correction_bias must have one value per expert")
    if not 0 < top_k <= gate_weight.shape[0]:
        raise ValueError("top_k must be within the expert count")
    if topk_backend not in ("torch", "neuron"):
        raise ValueError("topk_backend must be 'torch' or 'neuron'")

    router_logits = F.linear(
        hidden_states.to(torch.float32),
        gate_weight.to(torch.float32),
    )
    router_scores = torch.sigmoid(router_logits)
    selection_scores = router_scores + correction_bias.to(torch.float32)
    if topk_backend == "neuron":
        # GLM has 256 experts and selects 8 per token. Lowering torch.topk for
        # this shape emits an unsupported HLO sort on Trn2, while the rotational
        # NKI top-k kernel is qualified for it. Keep the import lazy so CPU-only
        # router tests do not require NKI.
        from vllm_neuron.functional.topk import topk as neuron_topk

        _, expert_indices = neuron_topk(
            selection_scores,
            k=top_k,
            dim=-1,
            gather_dim=-1,
        )
    else:
        expert_indices = torch.topk(
            selection_scores,
            k=top_k,
            dim=-1,
            sorted=False,
        ).indices

    routing_weights = torch.gather(router_scores, -1, expert_indices)
    routing_weights = routing_weights / routing_weights.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(torch.float32).tiny)
    routing_weights = routing_weights * routed_scaling_factor
    return expert_indices, routing_weights


class Glm52ExpertRouter(nn.Module):
    """FP32 GLM router kept external to the routed-expert kernel."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
        topk_backend: str = "torch",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if topk_backend not in ("torch", "neuron"):
            raise ValueError("topk_backend must be 'torch' or 'neuron'")
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_backend = topk_backend
        self.weight = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32, device=device),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return select_glm52_experts(
            hidden_states,
            self.weight,
            self.e_score_correction_bias,
            top_k=self.top_k,
            routed_scaling_factor=self.routed_scaling_factor,
            topk_backend=self.topk_backend,
        )
