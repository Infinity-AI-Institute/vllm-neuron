"""CPU numerical oracle for native Gemma 4 onboarding.

This module mirrors the Gemma 4 text checkpoint's mathematical structure.  It
does not provide serving kernels or a substitute for Neuron execution.  Its
purpose is to establish small, deterministic correctness seams before the same
components are expressed with vLLM-Neuron functional operators.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Gemma4Config
from .weights import Gemma4WeightMapper


class Gemma4RMSNorm(nn.Module):
    """Gemma 4 RMSNorm using a learned scale initialized to one."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype=torch.bfloat16,
        with_scale: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        # Gemma 4 deliberately uses pow rather than rsqrt.  The distinction is
        # observable after Neuron lowering, so preserve it in the oracle.
        x = x * torch.pow(x.pow(2).mean(dim=-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            x = x * self.weight.float()
        return x.to(input_dtype)


class Gemma4ValueNorm(Gemma4RMSNorm):
    """Scale-free RMS normalization applied to attention values."""

    def __init__(
        self, head_dim: int, eps: float = 1e-6, dtype=torch.bfloat16
    ):
        del dtype
        super().__init__(head_dim, eps=eps, dtype=None, with_scale=False)


class Gemma4RotaryEmbedding(nn.Module):
    """Generate Gemma 4 default or proportional rotary factors."""

    def __init__(
        self, head_dim: int, theta: float, rotary_dim: int | None = None
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim or head_dim
        active = 1.0 / (
            theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )
        inactive_pairs = (head_dim - self.rotary_dim) // 2
        if inactive_pairs:
            active = torch.cat((active, torch.zeros(inactive_pairs)))
        self.register_buffer("inv_freq", active, persistent=False)

    def forward(
        self, position_ids: torch.Tensor, dtype=torch.bfloat16
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = position_ids.float().reshape(-1)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return x * cos[:, None, :] + _rotate_half(x) * sin[:, None, :]


class Gemma4PagedKVCache:
    """Small CPU cache with the same per-layer native shape contract."""

    def __init__(
        self,
        num_slots: int,
        num_kv_heads: int,
        head_dim: int,
        dtype=torch.bfloat16,
    ):
        self.key = torch.zeros(
            num_slots, num_kv_heads, head_dim, dtype=dtype
        )
        self.value = torch.zeros_like(self.key)

    @property
    def shape(self):
        return self.key.shape

    def write(
        self,
        slot_mapping: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        slots = slot_mapping.to(device="cpu", dtype=torch.long).flatten()
        expected = (slots.numel(), *self.key.shape[1:])
        if key.shape != value.shape:
            raise ValueError(
                f"key/value shape mismatch: {key.shape} vs {value.shape}"
            )
        if tuple(key.shape) != expected:
            raise ValueError(
                f"cache write expects {expected}, got {tuple(key.shape)}"
            )
        if slots.numel() and (
            slots.min() < 0 or slots.max() >= self.key.shape[0]
        ):
            raise IndexError(
                f"slot_mapping exceeds cache size {self.key.shape[0]}"
            )
        self.key.index_copy_(0, slots, key.to(self.key.dtype))
        self.value.index_copy_(0, slots, value.to(self.value.dtype))

    def read(
        self, slot_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = slot_mapping.to(device="cpu", dtype=torch.long).flatten()
        if slots.numel() and (
            slots.min() < 0 or slots.max() >= self.key.shape[0]
        ):
            raise IndexError(
                f"slot_mapping exceeds cache size {self.key.shape[0]}"
            )
        return (
            self.key.index_select(0, slots),
            self.value.index_select(0, slots),
        )


class Gemma4ReferenceAttention(nn.Module):
    """Ordinary PyTorch attention used to validate Neuron attention seams."""

    def __init__(
        self,
        head_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        sliding_window: int | None = None,
    ):
        super().__init__()
        if num_query_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        self.head_dim = head_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        # Gemma 4 passes scaling=1.0 to attention.
        self.scale = 1.0

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: Gemma4PagedKVCache | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Inputs are [tokens, heads, head_dim].
        if cache is not None:
            if slot_mapping is None:
                raise ValueError(
                    "slot_mapping is required when using a KV cache"
                )
            cache.write(slot_mapping, key, value)
            key, value = cache.read(slot_mapping)
        repeat = self.num_query_heads // self.num_kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        scores = (
            torch.einsum(
                "thd,shd->hts", query.float(), key.float()
            )
            * self.scale
        )
        query_positions = torch.arange(query.shape[0])[:, None]
        key_positions = torch.arange(key.shape[0])[None, :]
        allowed = key_positions <= query_positions
        if self.sliding_window is not None:
            allowed &= key_positions > query_positions - self.sliding_window
        scores = scores.masked_fill(
            ~allowed[None, :, :], torch.finfo(scores.dtype).min
        )
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        return torch.einsum(
            "hts,shd->thd", probs, value.float()
        ).to(query.dtype)


class Gemma4Linear(nn.Module):
    """Linear layer carrying its vLLM-Neuron checkpoint-loader policy."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        name: str,
        tp_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        role = Gemma4WeightMapper.loader_kind(name)
        sharded_size = output_size if role == "column" else input_size
        if role in {"column", "row"} and sharded_size % tp_size:
            raise ValueError(
                f"{name} dimension {sharded_size} is not divisible by TP "
                f"size {tp_size}"
            )
        if role == "column":
            shape = (output_size // tp_size, input_size)
        elif role == "row":
            shape = (output_size, input_size // tp_size)
        else:
            shape = (output_size, input_size)
        self.weight = nn.Parameter(torch.empty(*shape, dtype=dtype))
        loader = Gemma4WeightMapper.make_loader(
            name,
            shard_size=(
                shape[0]
                if role == "column"
                else shape[1]
                if role == "row"
                else 0
            ),
            tp_size=tp_size,
        )
        setattr(self.weight, "weight_loader", loader)
        nn.init.normal_(self.weight, std=input_size**-0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.to(self.weight.dtype), self.weight)


class Gemma4ReferenceAttentionBlock(nn.Module):
    """Q/K/V projections, norms, RoPE, cache, and output projection."""

    def __init__(
        self,
        config: Gemma4Config,
        layer_idx: int,
        tp_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_query_heads = config.num_attention_heads
        self.head_dim, self.num_kv_heads = config.attention_shape(layer_idx)
        prefix = f"model.layers.{layer_idx}.self_attn"
        self.q_proj = Gemma4Linear(
            config.hidden_size,
            self.num_query_heads * self.head_dim,
            f"{prefix}.q_proj.weight",
            tp_size,
            config.torch_dtype,
        )
        self.k_proj = Gemma4Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            f"{prefix}.k_proj.weight",
            tp_size,
            config.torch_dtype,
        )
        self.v_proj = (
            None
            if config.layer_is_global(layer_idx)
            and config.attention_k_eq_v
            else Gemma4Linear(
                config.hidden_size,
                self.num_kv_heads * self.head_dim,
                f"{prefix}.v_proj.weight",
                tp_size,
                config.torch_dtype,
            )
        )
        self.o_proj = Gemma4Linear(
            self.num_query_heads * self.head_dim,
            config.hidden_size,
            f"{prefix}.o_proj.weight",
            tp_size,
            config.torch_dtype,
        )
        self.q_norm = Gemma4RMSNorm(
            self.head_dim, config.rms_norm_eps, config.torch_dtype
        )
        self.k_norm = Gemma4RMSNorm(
            self.head_dim, config.rms_norm_eps, config.torch_dtype
        )
        self.v_norm = Gemma4ValueNorm(
            self.head_dim, config.rms_norm_eps, config.torch_dtype
        )
        rope = config.rope_parameters.get(
            "full_attention"
            if config.layer_is_global(layer_idx)
            else "sliding_attention",
            {},
        )
        rotary_factor = float(rope.get("partial_rotary_factor", 1.0))
        self.rotary = Gemma4RotaryEmbedding(
            self.head_dim,
            float(
                rope.get(
                    "rope_theta",
                    1_000_000.0
                    if config.layer_is_global(layer_idx)
                    else 10_000.0,
                )
            ),
            int(self.head_dim * rotary_factor),
        )
        self.attention = Gemma4ReferenceAttention(
            self.head_dim,
            self.num_query_heads,
            self.num_kv_heads,
            None
            if config.layer_is_global(layer_idx)
            else config.sliding_window,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cache: Gemma4PagedKVCache | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = hidden_states.reshape(-1, self.hidden_size)
        if position_ids is None:
            position_ids = torch.arange(tokens.shape[0], device=tokens.device)
        cos, sin = self.rotary(position_ids, tokens.dtype)
        query = self.q_proj(tokens).reshape(
            -1, self.num_query_heads, self.head_dim
        )
        key = self.k_proj(tokens).reshape(
            -1, self.num_kv_heads, self.head_dim
        )
        value = (
            key
            if self.v_proj is None
            else self.v_proj(tokens).reshape(
                -1, self.num_kv_heads, self.head_dim
            )
        )
        query = _apply_rotary(self.q_norm(query), cos, sin)
        key = _apply_rotary(self.k_norm(key), cos, sin)
        value = self.v_norm(value)
        attended = self.attention(
            query, key, value, cache, slot_mapping
        )
        return self.o_proj(
            attended.reshape(-1, self.num_query_heads * self.head_dim)
        ).reshape_as(hidden_states)


class Gemma4ReferenceMLP(nn.Module):
    """Dense gated MLP that runs in parallel with Gemma 4's expert block."""

    def __init__(
        self, config: Gemma4Config, layer_idx: int, tp_size: int = 1
    ):
        super().__init__()
        width = config.layer_intermediate_size(layer_idx)
        prefix = f"model.layers.{layer_idx}.mlp"
        self.gate_proj = Gemma4Linear(
            config.hidden_size,
            width,
            f"{prefix}.gate_proj.weight",
            tp_size,
            config.torch_dtype,
        )
        self.up_proj = Gemma4Linear(
            config.hidden_size,
            width,
            f"{prefix}.up_proj.weight",
            tp_size,
            config.torch_dtype,
        )
        self.down_proj = Gemma4Linear(
            width,
            config.hidden_size,
            f"{prefix}.down_proj.weight",
            tp_size,
            config.torch_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(hidden_states), approximate="tanh")
            * self.up_proj(hidden_states)
        )


class Gemma4ReferenceRouter(nn.Module):
    """Gemma 4 RMS-normalized, scaled top-k router."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.top_k = config.top_k_experts
        self.num_experts = config.num_experts
        self.root_hidden_size = config.hidden_size**-0.5
        prefix = f"model.layers.{layer_idx}.router"
        self.norm = Gemma4RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            with_scale=False,
        )
        self.proj = Gemma4Linear(
            config.hidden_size,
            config.num_experts,
            f"{prefix}.proj.weight",
            dtype=config.torch_dtype,
        )
        self.scale = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.torch_dtype)
        )
        self.per_expert_scale = nn.Parameter(
            torch.ones(config.num_experts, dtype=config.torch_dtype)
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        routed = (
            self.norm(hidden_states)
            * self.scale
            * self.root_hidden_size
        )
        scores = self.proj(routed)
        probabilities = torch.softmax(scores.float(), dim=-1)
        weights, indices = torch.topk(
            probabilities, self.top_k, dim=-1
        )
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.per_expert_scale[indices].float()
        return probabilities, weights, indices


class Gemma4ReferenceExperts(nn.Module):
    """Stacked expert tensors matching the checkpoint storage layout."""

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                2 * config.moe_intermediate_size,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
                dtype=config.torch_dtype,
            )
        )
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        for expert_id in range(self.num_experts):
            token_rows, choices = torch.where(
                top_k_indices == expert_id
            )
            if token_rows.numel() == 0:
                continue
            selected = hidden_states.index_select(0, token_rows)
            gate, up = F.linear(
                selected, self.gate_up_proj[expert_id]
            ).chunk(2, dim=-1)
            expert_output = F.linear(
                F.gelu(gate, approximate="tanh") * up,
                self.down_proj[expert_id],
            )
            expert_output = expert_output * top_k_weights[
                token_rows, choices
            ].to(expert_output.dtype).unsqueeze(-1)
            output.index_add_(
                0, token_rows, expert_output.to(output.dtype)
            )
        return output


class Gemma4ReferenceMoE(nn.Module):
    """Composable router-plus-experts oracle used by focused unit tests."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
    ):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        config = Gemma4Config(
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k_experts=top_k,
            torch_dtype=torch.float32,
        )
        self.router = Gemma4ReferenceRouter(config, layer_idx=0)
        self.experts = Gemma4ReferenceExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])
        _, weights, indices = self.router(flat)
        return self.experts(flat, indices, weights).reshape(original_shape)


class Gemma4ReferenceDecoderLayer(nn.Module):
    """Exact dense-plus-MoE Gemma 4 decoder composition."""

    def __init__(
        self, config: Gemma4Config, layer_idx: int, tp_size: int = 1
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Gemma4ReferenceAttentionBlock(
            config, layer_idx, tp_size
        )
        self.mlp = Gemma4ReferenceMLP(config, layer_idx, tp_size)
        self.input_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            self.router = Gemma4ReferenceRouter(config, layer_idx)
            self.experts = Gemma4ReferenceExperts(config)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(
                config.hidden_size,
                config.rms_norm_eps,
                config.torch_dtype,
            )
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(
                config.hidden_size,
                config.rms_norm_eps,
                config.torch_dtype,
            )
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(
                config.hidden_size,
                config.rms_norm_eps,
                config.torch_dtype,
            )
        self.layer_scalar = nn.Parameter(
            torch.ones(1, dtype=config.torch_dtype), requires_grad=False
        )
        if config.hidden_size_per_layer_input:
            raise NotImplementedError(
                "The CPU oracle does not implement per-layer embeddings yet"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cache: Gemma4PagedKVCache | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_ids, cache, slot_mapping
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        dense = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        if self.enable_moe_block:
            dense = self.post_feedforward_layernorm_1(dense)
            flat_residual = residual.reshape(-1, residual.shape[-1])
            _, weights, indices = self.router(flat_residual)
            expert_input = self.pre_feedforward_layernorm_2(
                flat_residual
            )
            expert = self.experts(
                expert_input, indices, weights
            ).reshape_as(residual)
            expert = self.post_feedforward_layernorm_2(expert)
            dense = dense + expert
        hidden_states = self.post_feedforward_layernorm(dense)
        return (residual + hidden_states) * self.layer_scalar


class Gemma4ReferenceTextModel(nn.Module):
    """Tiny-configurable full text stack used for seam validation."""

    def __init__(self, config: Gemma4Config, tp_size: int = 1):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )
        self.layers = nn.ModuleList(
            Gemma4ReferenceDecoderLayer(config, idx, tp_size)
            for idx in range(config.num_hidden_layers)
        )
        self.norm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cache_layers: list[Gemma4PagedKVCache | None] | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * torch.tensor(
            math.sqrt(self.config.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[-1], device=input_ids.device
            )
        if cache_layers is None:
            cache_layers = [None] * len(self.layers)
        if len(cache_layers) != len(self.layers):
            raise ValueError(
                "cache_layers must contain one cache per decoder layer"
            )
        for layer, cache in zip(self.layers, cache_layers):
            hidden_states = layer(
                hidden_states, position_ids, cache, slot_mapping
            )
        return self.norm(hidden_states)


class Gemma4ReferenceLMHead(nn.Module):
    """Tied LM head honoring vLLM's sampling-position contract."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        embedding: nn.Embedding | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if embedding is None:
            self.weight = nn.Parameter(
                torch.empty(vocab_size, hidden_size, dtype=dtype)
            )
            nn.init.normal_(self.weight, std=hidden_size**-0.5)
            self._embedding = None
        else:
            self.register_parameter("weight", None)
            self._embedding = embedding

    def forward(
        self,
        hidden_states: torch.Tensor,
        sampling_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if sampling_positions is not None:
            positions = sampling_positions.to(
                device=flat.device, dtype=torch.long
            ).flatten()
            if positions.numel() and (
                positions.min() < 0 or positions.max() >= flat.shape[0]
            ):
                raise IndexError(
                    "sampling_positions exceeds hidden-state sequence"
                )
            flat = flat.index_select(0, positions)
        weight = (
            self._embedding.weight
            if self._embedding is not None
            else self.weight
        )
        return torch.matmul(
            flat.float(), weight.float().transpose(0, 1)
        ).to(hidden_states.dtype)


class Gemma4ReferenceCausalLM(nn.Module):
    """End-to-end reference model with selected-token logits."""

    def __init__(self, config: Gemma4Config, tp_size: int = 1):
        super().__init__()
        self.config = config
        self.model = Gemma4ReferenceTextModel(config, tp_size)
        self.lm_head = Gemma4ReferenceLMHead(
            config.hidden_size,
            config.vocab_size,
            self.model.embed_tokens if config.tie_word_embeddings else None,
            config.torch_dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        sampling_positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache_layers: list[Gemma4PagedKVCache | None] | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, position_ids, cache_layers, slot_mapping
        )
        logits = self.lm_head(hidden_states, sampling_positions)
        if self.config.final_logit_softcapping is not None:
            cap = torch.full_like(
                logits, self.config.final_logit_softcapping
            )
            logits = torch.tanh(logits / cap) * cap
        return logits
