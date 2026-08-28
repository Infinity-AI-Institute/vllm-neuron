# SPDX-License-Identifier: Apache-2.0
"""Source-qualified GLM-5.3-Flash causal language model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .attention import Glm53FlashAttention
from .config import Glm53FlashInferenceConfig, validate_fp8_scale
from .dense_mlp import Glm53DenseMlp
from .mhc import Glm53MHC
from .mla import rms_norm
from .moe import Glm53SparseMlp
from .telemetry import Glm53FlashTelemetry


class Glm53FlashDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Glm53FlashInferenceConfig,
        *,
        layer_idx: int,
        telemetry: Glm53FlashTelemetry,
    ) -> None:
        super().__init__()
        self.input_norm = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.torch_dtype)
        )
        self.post_attention_norm = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.torch_dtype)
        )
        self.eps = config.rms_norm_eps
        self.self_attn = Glm53FlashAttention(
            config, layer_idx=layer_idx, telemetry=telemetry
        )
        self.hc_attn = Glm53MHC(config)
        self.hc_mlp = Glm53MHC(config)
        self.mlp = (
            Glm53DenseMlp(config)
            if config.mlp_layer_types[layer_idx] == "dense"
            else Glm53SparseMlp(config)
        )

    def forward(
        self, residual_streams: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        post_mix, comb_mix, hidden_states = self.hc_attn.pre(residual_streams)
        normalized = rms_norm(hidden_states, self.input_norm, self.eps)
        attention_output = self.self_attn(normalized, position_ids)
        residual_streams = self.hc_attn.post(
            attention_output, residual_streams, post_mix, comb_mix
        )

        post_mix, comb_mix, hidden_states = self.hc_mlp.pre(residual_streams)
        normalized = rms_norm(hidden_states, self.post_attention_norm, self.eps)
        mlp_output = self.mlp(normalized)
        return self.hc_mlp.post(mlp_output, residual_streams, post_mix, comb_mix)


class NeuronGlm53FlashForCausalLM(nn.Module):
    """Text-only, MTP-off GLM-5.3-Flash source reference."""

    def __init__(self, config: Glm53FlashInferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.telemetry = Glm53FlashTelemetry()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=config.torch_dtype
        )
        self.layers = nn.ModuleList(
            Glm53FlashDecoderLayer(config, layer_idx=layer, telemetry=self.telemetry)
            for layer in range(config.num_hidden_layers)
        )
        self.final_norm = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.torch_dtype)
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
        )
        self._position_offset = 0

    @classmethod
    def from_configs(
        cls, hf_config: Any, neuron_config: Any = None
    ) -> NeuronGlm53FlashForCausalLM:
        return cls(Glm53FlashInferenceConfig.from_configs(hf_config, neuron_config))

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_decode_metadata: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if spec_decode_metadata is not None:
            raise NotImplementedError("MTP/speculative decode is outside this port")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(0)
        batch, length, _ = hidden_states.shape
        if positions is None:
            positions = torch.arange(
                self._position_offset,
                self._position_offset + length,
                device=hidden_states.device,
            ).expand(batch, length)
        elif positions.ndim == 1:
            positions = positions.expand(batch, -1)
        self._position_offset = max(
            self._position_offset,
            int(positions.max().item()) + 1,
        )
        # mHC starts with four embedding copies, stays wide through every
        # decoder sublayer, then collapses to an unweighted mean at the head.
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, 1, self.config.hc_mult, 1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        hidden_states = hidden_states.mean(dim=-2)
        hidden_states = rms_norm(
            hidden_states, self.final_norm, self.config.rms_norm_eps
        )
        return self.lm_head(hidden_states)

    def reset_kda_state(self, reset_mask: torch.Tensor | None = None) -> int:
        return sum(
            layer.self_attn.reset_state(reset_mask)
            for layer in self.layers
            if layer.self_attn.path == "kda"
        )

    def reset_attention_state(self, reset_mask: torch.Tensor | None = None) -> int:
        count = sum(layer.self_attn.reset_state(reset_mask) for layer in self.layers)
        self._position_offset = 0
        return count

    def reset_telemetry(self) -> None:
        self.telemetry.reset()

    def get_telemetry(self) -> dict[str, dict[int, int]]:
        return self.telemetry.snapshot()

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        for name, value in state_dict.items():
            if _is_fp8_scale_name(name):
                validate_fp8_scale(value, name)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ):
        raise NotImplementedError(
            "GLM-5.3 checkpoint sharding remains operator-gated; use load_state_dict "
            "for the CPU source qualification path"
        )


def _is_fp8_scale_name(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return (
        leaf.endswith(("weight_scale", "activation_scale", "input_scale"))
        or "quant_multiplier" in leaf
    )


__all__ = ["Glm53FlashDecoderLayer", "NeuronGlm53FlashForCausalLM"]
