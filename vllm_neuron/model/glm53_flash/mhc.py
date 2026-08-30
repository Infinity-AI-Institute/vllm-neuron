# SPDX-License-Identifier: Apache-2.0
"""Torch reference for GLM-5.3 manifold-constrained hyper-connections."""

from __future__ import annotations

import torch
from torch import nn

from .config import Glm53FlashInferenceConfig


class Glm53MHC(nn.Module):
    """One 24-row pre-mixer and its residual-stream post composition."""

    def __init__(self, config: Glm53FlashInferenceConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.rms_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.post_alpha = config.hc_post_alpha
        mix_rows = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix_rows, self.hc_mult * self.hidden_size))
        self.base = nn.Parameter(torch.zeros(mix_rows, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        nn.init.normal_(self.fn, std=(self.hc_mult * self.hidden_size) ** -0.5)

    def pre(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual.shape[-2:] != (self.hc_mult, self.hidden_size):
            raise ValueError("mHC residual has an incorrect stream/hidden shape")
        outer_shape = residual.shape[:-2]
        flat = residual.reshape(-1, self.hc_mult, self.hidden_size)
        tokens = flat.shape[0]
        x = flat.flatten(1).to(torch.float32)
        mixes = x @ self.fn.to(torch.float32).t()
        variance = x.square().mean(dim=-1, keepdim=True)
        mixes = mixes * torch.rsqrt(variance + self.rms_eps)

        pre_logits = (
            mixes[:, : self.hc_mult] * self.scale[0] + self.base[: self.hc_mult]
        )
        pre_mix = torch.sigmoid(pre_logits) + self.hc_eps

        post_logits = (
            mixes[:, self.hc_mult : 2 * self.hc_mult] * self.scale[1]
            + self.base[self.hc_mult : 2 * self.hc_mult]
        )
        post_mix = torch.sigmoid(post_logits) * self.post_alpha

        comb_logits = mixes[:, 2 * self.hc_mult :].view(
            tokens, self.hc_mult, self.hc_mult
        )
        comb_logits = comb_logits * self.scale[2] + self.base[2 * self.hc_mult :].view(
            1, self.hc_mult, self.hc_mult
        )
        comb_mix = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + self.hc_eps)

        layer_input = torch.sum(
            pre_mix.unsqueeze(-1) * flat.to(torch.float32), dim=1
        ).to(residual.dtype)
        return (
            post_mix.view(*outer_shape, self.hc_mult, 1),
            comb_mix.view(*outer_shape, self.hc_mult, self.hc_mult),
            layer_input.view(*outer_shape, self.hidden_size),
        )

    @staticmethod
    def post(
        layer_output: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
    ) -> torch.Tensor:
        mixed_residual = torch.einsum(
            "...ij,...ih->...jh",
            comb_mix.to(torch.float32),
            residual.to(torch.float32),
        )
        post_term = post_mix.to(torch.float32) * layer_output.unsqueeze(-2).to(
            torch.float32
        )
        return (mixed_residual + post_term).to(residual.dtype)


__all__ = ["Glm53MHC"]
