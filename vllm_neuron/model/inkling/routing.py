"""Exact Inkling routed- and shared-expert selection.

Inkling differs from common sigmoid MoEs in two important ways:

* the correction bias affects routed-expert selection, but not mixture weight;
* the six routed and two always-on shared experts are normalized together in
  log-sigmoid space.

Keeping this seam independent of the expert kernel makes it directly
comparable with the Transformers reference and prevents an approximate generic
router from silently changing tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_neuron.functional.topk import topk as neuron_topk


def inkling_route_from_logits(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    num_routed_experts: int,
    num_shared_experts: int,
    top_k: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dense routed affinities, routed IDs, and shared gammas."""
    routed_logits = router_logits[..., :num_routed_experts]
    shared_logits = router_logits[
        ..., num_routed_experts : num_routed_experts + num_shared_experts
    ]
    selection_scores = torch.sigmoid(routed_logits) + correction_bias
    # A raw torch.topk lowers to an HLO sort, which is not supported on Trn2.
    # The shared functional dispatches to the rotational NKI TopK kernel on
    # Neuron and retains torch.topk as the CPU/reference fallback.  Inkling
    # cannot use the generic fused router because its correction bias is added
    # *after* sigmoid and affects selection only.
    _, routed_ids = neuron_topk(
        selection_scores,
        k=top_k,
        dim=-1,
        gather_dim=-1,
        process_group=None,
    )

    selected_logits = torch.gather(routed_logits, -1, routed_ids)
    active_logits = torch.cat((selected_logits, shared_logits), dim=-1)
    log_weights = F.logsigmoid(active_logits)
    weights = torch.exp(
        log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
    )
    weights = weights * route_scale * global_scale.float()

    routed_weights = weights[..., :top_k]
    shared_gammas = weights[..., top_k:]
    routed_affinities = torch.zeros_like(
        routed_logits, dtype=routed_weights.dtype
    ).scatter(-1, routed_ids, routed_weights)
    return routed_affinities, routed_ids.to(torch.int32), shared_gammas


def inkling_route(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    num_routed_experts: int,
    num_shared_experts: int,
    top_k: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project in FP32 and apply the checkpoint-native routing rule."""
    router_logits = hidden_states.float() @ router_weight.float().T
    affinities, indices, shared = inkling_route_from_logits(
        router_logits,
        correction_bias.float(),
        global_scale.float(),
        num_routed_experts=num_routed_experts,
        num_shared_experts=num_shared_experts,
        top_k=top_k,
        route_scale=route_scale,
    )
    return router_logits, affinities, indices, shared
