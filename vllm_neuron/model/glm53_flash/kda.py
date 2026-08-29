# SPDX-License-Identifier: Apache-2.0
"""KDA v2 stateful linear attention using the canonical CPU golden."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ._reference_kernels import load_reference_kernel
from .config import Glm53FlashInferenceConfig, validate_fp8_scale
from .kernel_dispatch import (
    KDA_CPU_GOLDEN_SLUG,
    KDA_NKI_V3P2_SLUG,
    get_kda_decode_forward,
    resolve_kda_impl_slug,
)
from .mla import rms_norm

KDA_KERNEL_SLUG = KDA_CPU_GOLDEN_SLUG


class _Glm53RmsNormGated(nn.Module):
    def __init__(self, head_dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return rms_norm(hidden_states, self.weight, self.eps) * torch.sigmoid(
            gate.to(torch.float32)
        ).to(hidden_states.dtype)


class Glm53KdaAttention(nn.Module):
    """Short-conv front end plus persistent bf16 KDA state."""

    def __init__(self, config: Glm53FlashInferenceConfig, *, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        linear = config.linear_attn_config
        self.num_heads = linear.num_heads
        self.head_dim = linear.head_dim
        self.qkv_dim = linear.num_heads * linear.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size, self.qkv_dim, bias=False, dtype=config.torch_dtype
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.qkv_dim, bias=False, dtype=config.torch_dtype
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.qkv_dim, bias=False, dtype=config.torch_dtype
        )
        channels = 3 * self.qkv_dim
        self.conv1d = nn.Conv1d(
            channels,
            channels,
            kernel_size=linear.short_conv_kernel_size,
            groups=channels,
            bias=False,
            dtype=config.torch_dtype,
        )
        self.f_a_proj = nn.Linear(
            config.hidden_size, self.head_dim, bias=False, dtype=config.torch_dtype
        )
        self.f_b_proj = nn.Linear(
            self.head_dim, self.qkv_dim, bias=False, dtype=config.torch_dtype
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.qkv_dim, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.b_proj = nn.Linear(
            config.hidden_size, self.num_heads, bias=False, dtype=config.torch_dtype
        )
        self.g_a_proj = nn.Linear(
            config.hidden_size, self.head_dim, bias=False, dtype=config.torch_dtype
        )
        self.g_b_proj = nn.Linear(
            self.head_dim, self.qkv_dim, bias=False, dtype=config.torch_dtype
        )
        self.o_norm = _Glm53RmsNormGated(self.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(
            self.qkv_dim, config.hidden_size, bias=False, dtype=config.torch_dtype
        )
        self.register_buffer(
            "weight_scale",
            torch.tensor(config.fp8_weight_scale_default, dtype=torch.float32),
        )
        self.register_buffer(
            "activation_scale",
            torch.tensor(config.fp8_activation_scale_default, dtype=torch.float32),
        )
        self._state_bf16: np.ndarray | None = None
        self._conv_state: torch.Tensor | None = None

    def _apply_short_conv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _, _ = query.shape
        widths = (query.shape[-1], key.shape[-1], value.shape[-1])
        joined = torch.cat((query, key, value), dim=-1).flatten(-2).transpose(1, 2)
        history = self.conv1d.kernel_size[0] - 1
        if self._conv_state is None or self._conv_state.shape[:2] != joined.shape[:2]:
            self._conv_state = torch.zeros(
                joined.shape[0],
                joined.shape[1],
                history,
                dtype=joined.dtype,
                device=joined.device,
            )
        conv_input = torch.cat((self._conv_state, joined), dim=-1)
        self._conv_state = conv_input[..., -history:].detach().clone()
        joined = (
            F.silu(self.conv1d(conv_input))
            .transpose(1, 2)
            .view(batch, length, self.num_heads, sum(widths))
        )
        return torch.split(joined, widths, dim=-1)

    def reset_state(self, reset_mask: torch.Tensor | None = None) -> int:
        if self._state_bf16 is None and self._conv_state is None:
            return 0
        if reset_mask is None:
            batch = (
                self._state_bf16.shape[0]
                if self._state_bf16 is not None
                else self._conv_state.shape[0]
            )
            reset_mask = torch.ones(batch, dtype=torch.bool)
        mask = reset_mask.detach().cpu().numpy().astype(np.bool_)
        if self._state_bf16 is not None:
            golden = load_reference_kernel("kda")
            self._state_bf16 = golden.kda_state_reset_v2(self._state_bf16, mask)
        if self._conv_state is not None:
            torch_mask = reset_mask.to(device=self._conv_state.device, dtype=torch.bool)
            self._conv_state[torch_mask] = 0
        return int(mask.sum())

    def forward(
        self, hidden_states: torch.Tensor, *, impl: str = "reference"
    ) -> torch.Tensor:
        if hidden_states.device.type != "cpu":
            raise NotImplementedError("source qualification uses the CPU KDA v2 golden")
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, length, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch, length, self.num_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch, length, self.num_heads, self.head_dim
        )
        query, key, value = self._apply_short_conv(query, key, value)
        g_raw = self.f_b_proj(self.f_a_proj(hidden_states)).view(
            batch, length, self.num_heads, self.head_dim
        )
        beta_raw = self.b_proj(hidden_states)
        if self._state_bf16 is None or self._state_bf16.shape[0] != batch:
            self._state_bf16 = np.zeros(
                (batch, self.num_heads, self.head_dim, self.head_dim),
                dtype=np.float32,
            )
        golden = load_reference_kernel("kda")
        params = golden.KdaLayerParams(
            a_log=self.A_log.detach().to(torch.float32).numpy(),
            g_bias=self.dt_bias.detach()
            .to(torch.float32)
            .view(self.num_heads, self.head_dim)
            .numpy(),
            lower_bound=self.config.linear_attn_config.gate_lower_bound,
            l2norm_eps=self.config.linear_attn_config.l2norm_eps,
        )
        inputs = golden.KdaDecodeInputsV2(
            query=query.detach().to(torch.float32).numpy(),
            key=key.detach().to(torch.float32).numpy(),
            value=value.detach().to(torch.float32).numpy(),
            g_raw=g_raw.detach().to(torch.float32).numpy(),
            beta_raw=beta_raw.detach().to(torch.float32).numpy(),
            state_bf16=self._state_bf16,
            params=params,
        )
        if length == 1:
            # Dispatch through the KDA slug resolver: env `KDA_KERNEL_IMPL`
            # picks between the CPU golden (default) and the v3.2 NKI wrapper.
            # Both callables have the same signature/return type; on CPU
            # hosts the v3.2 wrapper's own `impl="auto"` falls through to
            # the CPU golden, so the numerics are bit-identical and the
            # existing tests do not break. The emitted slug reflects the
            # env choice for the compile driver's NEFF-cache-key.
            emitted_slug, kda_forward = get_kda_decode_forward(
                golden.kda_state_decode_forward_v2
            )
            self._last_emitted_kda_slug = emitted_slug
            outputs = kda_forward(inputs, impl=impl)
            output_np, self._state_bf16 = outputs.y, outputs.state_bf16
        else:
            output_np, self._state_bf16 = golden.kda_state_prefill_forward_reference_v2(
                inputs.query,
                inputs.key,
                inputs.value,
                inputs.g_raw,
                inputs.beta_raw,
                inputs.state_bf16,
                inputs.params,
            )
        output = torch.from_numpy(output_np).to(hidden_states.dtype)
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view_as(output)
        output = self.o_norm(output, gate)
        return self.o_proj(output.flatten(-2))

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        validate_fp8_scale(self.weight_scale, "weight_scale")
        validate_fp8_scale(self.activation_scale, "activation_scale")


def emitted_kda_kernel_slug() -> str:
    """The KDA slug this process would advertise in the NEFF cache key.

    Reads ``KDA_KERNEL_IMPL`` and returns the matching slug. See
    :mod:`.kernel_dispatch` for the accepted alias set.
    """
    return resolve_kda_impl_slug()


__all__ = [
    "KDA_CPU_GOLDEN_SLUG",
    "KDA_KERNEL_SLUG",
    "KDA_NKI_V3P2_SLUG",
    "Glm53KdaAttention",
    "emitted_kda_kernel_slug",
]
