"""Native Gemma 4 model implementation scaffold.

This module is the path-2 porting seam. It must own vLLM's paged KV-cache
writes and sampling-position contract; it must not route through NxDI model
registries or architecture-rewrite shims.
"""

import math

import torch
import torch.nn as nn


class Gemma4RMSNorm(nn.Module):
    """Gemma RMSNorm with the checkpoint's +1 weight convention."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight.float())).to(input_dtype)


class Gemma4ValueNorm(nn.Module):
    """RMS normalization used on attention values in Gemma 4."""

    def __init__(self, head_dim: int, eps: float = 1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim, dtype=dtype))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        x = values.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(values.dtype)


class Gemma4RotaryEmbedding(nn.Module):
    """Generate local/global rotary factors without owning KV-cache state."""

    def __init__(self, head_dim: int, theta: float, rotary_dim: int | None = None):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim or head_dim
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype=torch.bfloat16):
        positions = position_ids.float().reshape(-1)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class Gemma4PagedKVCache:
    """Explicit paged KV storage shared by local/global attention layers.

    The cache is deliberately a small tensor contract: callers provide
    `slot_mapping` and the layer's native head width; no layer may reshape a
    global cache into the local layout. The Neuron implementation can replace
    the backing tensors with device handles without changing this interface.
    """

    def __init__(self, num_slots: int, num_kv_heads: int, head_dim: int, dtype=torch.bfloat16):
        self.key = torch.zeros(num_slots, num_kv_heads, head_dim, dtype=dtype)
        self.value = torch.zeros_like(self.key)

    @property
    def shape(self):
        return self.key.shape

    def write(self, slot_mapping: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        slots = slot_mapping.to(device="cpu", dtype=torch.long).flatten()
        if key.shape != value.shape:
            raise ValueError(f"key/value shape mismatch: {key.shape} vs {value.shape}")
        if key.ndim != 3 or key.shape[0] != slots.numel():
            raise ValueError(
                f"cache write expects [num_slots, heads, head_dim], got {key.shape} "
                f"for {slots.numel()} mapped slots"
            )
        if slots.numel() and (slots.min() < 0 or slots.max() >= self.key.shape[0]):
            raise IndexError(f"slot_mapping exceeds cache size {self.key.shape[0]}")
        self.key.index_copy_(0, slots, key.to(self.key.dtype))
        self.value.index_copy_(0, slots, value.to(self.value.dtype))

    def read(self, slot_mapping: torch.Tensor):
        slots = slot_mapping.to(device="cpu", dtype=torch.long).flatten()
        if slots.numel() and (slots.min() < 0 or slots.max() >= self.key.shape[0]):
            raise IndexError(f"slot_mapping exceeds cache size {self.key.shape[0]}")
        return self.key.index_select(0, slots), self.value.index_select(0, slots)


class Gemma4ReferenceAttention(nn.Module):
    """Small CPU oracle for validating native attention/cache seams.

    This is not the serving kernel. It intentionally uses ordinary PyTorch
    operations so discrepancies can be localized before replacing it with a
    Neuron paged-attention implementation.
    """

    def __init__(self, head_dim: int, num_query_heads: int, num_kv_heads: int):
        super().__init__()
        if num_query_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        self.head_dim = head_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.scale = head_dim ** -0.5

    def forward(self, query, key, value, cache=None, slot_mapping=None):
        # Inputs are [tokens, heads, head_dim]. Cache writes preserve the
        # native per-layer head width and are performed before reading history.
        if cache is not None:
            if slot_mapping is None:
                raise ValueError("slot_mapping is required when using a KV cache")
            cache.write(slot_mapping, key, value)
            key, value = cache.read(slot_mapping)
        repeat = self.num_query_heads // self.num_kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        scores = torch.einsum("thd,shd->hts", query, key) * self.scale
        causal = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        return torch.einsum("hts,shd->thd", probs, value)


class Gemma4ReferenceMoE(nn.Module):
    """CPU oracle for Gemma 4 top-k router dispatch and expert combine."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size, bias=False),
                nn.GELU(approximate="tanh"),
                nn.Linear(intermediate_size, hidden_size, bias=False),
            )
            for _ in range(num_experts)
        )

    def forward(self, hidden_states: torch.Tensor):
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.router(flat).float()
        weights, indices = torch.topk(router_logits, self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1).to(flat.dtype)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_rows, choices = torch.where(indices == expert_id)
            if token_rows.numel() == 0:
                continue
            expert_output = expert(flat.index_select(0, token_rows))
            output.index_add_(0, token_rows, expert_output * weights[token_rows, choices].unsqueeze(-1))
        return output.reshape(original_shape)


class Gemma4WeightMapper:
    """Normalize Hugging Face Gemma 4 names to native module names."""

    _DIRECT = {
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "o_proj": "o_proj",
        "input_layernorm": "input_layernorm",
        "post_attention_layernorm": "post_attention_layernorm",
        "post_feedforward_layernorm": "post_feedforward_layernorm",
    }

    @classmethod
    def map_name(cls, name: str) -> str:
        """Return a stable native name while preserving expert indices."""
        name = name.removeprefix("model.")
        name = name.replace("layers.", "layers.")
        for source, target in cls._DIRECT.items():
            name = name.replace(f".{source}.", f".{target}.")
        name = name.replace(".block_sparse_moe.router.", ".moe.router.")
        name = name.replace(".block_sparse_moe.experts.", ".moe.experts.")
        name = name.replace(".gate_proj.", ".gate_proj.")
        name = name.replace(".up_proj.", ".up_proj.")
        name = name.replace(".down_proj.", ".down_proj.")
        return name

    @classmethod
    def is_expert_weight(cls, name: str) -> bool:
        return ".moe.experts." in cls.map_name(name)

    @classmethod
    def loader_kind(cls, name: str) -> str:
        """Classify a parameter for the native TP/EP loader policy."""
        mapped = cls.map_name(name)
        if cls.is_expert_weight(mapped):
            return "expert-local"
        if any(token in mapped for token in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight")):
            return "column"
        if any(token in mapped for token in ("o_proj.weight", "down_proj.weight")):
            return "row"
        return "replicated"

    @classmethod
    def make_loader(cls, name: str, shard_size: int, tp_size: int):
        """Build the vLLM-Neuron safetensors loader for a parameter role."""
        from vllm_neuron.utils.weight_loader import (
            SafetensorsWeightLoader,
            sharding_weight_loader,
        )

        role = cls.loader_kind(name)
        if role == "column":
            return sharding_weight_loader(0, shard_size, tp_size)
        if role == "row":
            return sharding_weight_loader(1, shard_size, tp_size)
        # Expert-local weights are selected by the EP layer; the loader must
        # not apply TP slicing a second time. Replicated parameters are copied
        # unchanged to every rank.
        return SafetensorsWeightLoader()


class Gemma4Linear(nn.Module):
    """Linear layer carrying its native vLLM-Neuron weight-loader policy."""

    def __init__(self, input_size: int, output_size: int, name: str, tp_size: int = 1):
        super().__init__()
        role = Gemma4WeightMapper.loader_kind(name)
        if role == "column":
            local_output = (output_size + tp_size - 1) // tp_size
            shape = (local_output, input_size)
        elif role == "row":
            local_input = (input_size + tp_size - 1) // tp_size
            shape = (output_size, local_input)
        else:
            shape = (output_size, input_size)
        self.weight = nn.Parameter(torch.empty(*shape))
        set_loader = Gemma4WeightMapper.make_loader(
            name,
            shard_size=shape[0] if role == "column" else shape[1] if role == "row" else 0,
            tp_size=tp_size,
        )
        setattr(self.weight, "weight_loader", set_loader)
        nn.init.normal_(self.weight, std=input_size**-0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden_states, self.weight)


class Gemma4ReferenceAttentionBlock(nn.Module):
    """Composable attention block used as the native numerical bridge."""

    def __init__(self, hidden_size: int, num_query_heads: int, num_kv_heads: int,
                 head_dim: int, layer_name: str = "layers.0.self_attn"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = Gemma4Linear(hidden_size, num_query_heads * head_dim, f"{layer_name}.q_proj.weight")
        self.k_proj = Gemma4Linear(hidden_size, num_kv_heads * head_dim, f"{layer_name}.k_proj.weight")
        self.v_proj = Gemma4Linear(hidden_size, num_kv_heads * head_dim, f"{layer_name}.v_proj.weight")
        self.o_proj = Gemma4Linear(num_query_heads * head_dim, hidden_size, f"{layer_name}.o_proj.weight")
        self.attention = Gemma4ReferenceAttention(head_dim, num_query_heads, num_kv_heads)

    def forward(self, hidden_states: torch.Tensor, cache=None, slot_mapping=None):
        tokens = hidden_states.reshape(-1, self.hidden_size)
        query = self.q_proj(tokens).reshape(-1, self.num_query_heads, self.head_dim)
        key = self.k_proj(tokens).reshape(-1, self.num_kv_heads, self.head_dim)
        value = self.v_proj(tokens).reshape(-1, self.num_kv_heads, self.head_dim)
        attended = self.attention(query, key, value, cache, slot_mapping)
        return self.o_proj(attended.reshape(-1, self.num_query_heads * self.head_dim)).reshape_as(hidden_states)


class Gemma4ReferenceDecoderLayer(nn.Module):
    """Reference decoder composition used to validate native layer seams."""

    def __init__(self, config: Gemma4Config, layer_idx: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        head_dim, num_kv_heads = config.attention_shape(layer_idx)
        self.attention = Gemma4ReferenceAttentionBlock(
            config.hidden_size,
            config.num_attention_heads,
            num_kv_heads,
            head_dim,
            f"layers.{layer_idx}.self_attn",
        )
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe = Gemma4ReferenceMoE(
            config.hidden_size, config.intermediate_size, num_experts, top_k
        )

    def forward(self, hidden_states, cache=None, slot_mapping=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.attention(hidden_states, cache, slot_mapping)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.moe(hidden_states)


class Gemma4MoeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        raise NotImplementedError(
            "Gemma4 native vLLM-Neuron layers are not implemented yet; "
            "use the committed serving baseline while this port is developed."
        )
