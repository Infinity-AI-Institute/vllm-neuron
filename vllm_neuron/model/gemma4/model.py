"""Native BF16 Gemma 4 text model for vLLM-Neuron.

This implementation deliberately starts with the smallest correctness surface
that can execute the released ``google/gemma-4-26B-A4B`` checkpoint:

* full-token tensor parallelism (no sequence parallelism);
* TP-sharded Q/K/V, dense MLP, experts, embedding, and LM head;
* ordinary PyTorch attention for Gemma's 256/512-wide attention heads;
* NKI dense/MoE operators where their public contracts support the shape;
* model-owned paged KV-cache updates.

The CPU numerical oracle remains isolated in :mod:`reference` and is selected
only by the explicit ``VLLM_NEURON_GEMMA4_REFERENCE=1`` test switch.
"""

from __future__ import annotations

import math
import os

import nki.language as nl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import Gemma4Config
from .weights import Gemma4WeightMapper


def _rms_norm(
    hidden_states: torch.Tensor,
    eps: float,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gemma 4 RMSNorm, including its intentional ``pow(-0.5)``."""
    input_dtype = hidden_states.dtype
    normalized = hidden_states.float()
    normalized = normalized * torch.pow(
        normalized.pow(2).mean(dim=-1, keepdim=True) + eps, -0.5
    )
    if weight is not None:
        normalized = normalized * weight.float()
    return normalized.to(input_dtype)


class Gemma4RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _rms_norm(hidden_states, self.eps, self.weight)


class Gemma4RotaryEmbedding(nn.Module):
    """Per-layer RoPE because Gemma uses different local/global head shapes."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.head_dim, _ = config.attention_shape(layer_idx)
        global_layer = config.layer_is_global(layer_idx)
        rope = config.rope_parameters.get(
            "full_attention" if global_layer else "sliding_attention", {}
        )
        self.theta = float(
            rope.get("rope_theta", 1_000_000.0 if global_layer else 10_000.0)
        )
        self.rotary_dim = int(
            self.head_dim * float(rope.get("partial_rotary_factor", 1.0))
        )

    def forward(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Transformers' Gemma 4 implementation divides by the full head
        # dimension even when only a prefix participates in RoPE.
        active = 1.0 / (
            self.theta
            ** (
                torch.arange(
                    0,
                    self.rotary_dim,
                    2,
                    dtype=torch.float32,
                    device=positions.device,
                )
                / self.head_dim
            )
        )
        inactive_pairs = (self.head_dim - self.rotary_dim) // 2
        if inactive_pairs:
            active = torch.cat(
                (
                    active,
                    torch.zeros(
                        inactive_pairs,
                        dtype=active.dtype,
                        device=active.device,
                    ),
                )
            )
        freqs = torch.outer(positions.float().reshape(-1), active)
        embedding = torch.cat((freqs, freqs), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return (
        hidden_states * cos[:, None, :] + _rotate_half(hidden_states) * sin[:, None, :]
    )


class Gemma4Attention(nn.Module):
    """TP-sharded hybrid attention with explicit paged-cache ownership."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_query_heads = config.num_attention_heads
        self.head_dim, self.num_kv_heads = config.attention_shape(layer_idx)
        self.sliding_window = (
            None if config.layer_is_global(layer_idx) else config.sliding_window
        )
        self.eps = config.rms_norm_eps

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        if self.num_query_heads % self.world_size:
            raise ValueError(
                "Gemma 4 query heads must be divisible by tensor parallel size"
            )
        self.num_query_heads_per_rank = self.num_query_heads // self.world_size
        if self.world_size >= self.num_kv_heads:
            if self.world_size % self.num_kv_heads:
                raise ValueError(
                    "tensor parallel size must be divisible by Gemma 4 KV heads"
                )
            self.num_kv_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_kv_heads
        else:
            if self.num_kv_heads % self.world_size:
                raise ValueError(
                    "Gemma 4 KV heads must be divisible by tensor parallel size"
                )
            self.num_kv_heads_per_rank = self.num_kv_heads // self.world_size
            self.num_kv_replicas = 1
        if self.num_query_heads_per_rank % self.num_kv_heads_per_rank:
            raise ValueError("local query heads must be divisible by local KV heads")

        self.q_size = self.num_query_heads_per_rank * self.head_dim
        self.kv_size = self.num_kv_heads_per_rank * self.head_dim
        self.qkv_split_indices = (self.q_size, self.q_size + self.kv_size)
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(
                self.hidden_size,
                self.q_size + 2 * self.kv_size,
                dtype=self.dtype,
            )
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.q_size, self.hidden_size, dtype=self.dtype)
        )
        # Q/K norm weights are replicated. V has scale-free RMSNorm.
        self.q_norm = Gemma4RMSNorm(self.head_dim, self.eps, self.dtype)
        self.k_norm = Gemma4RMSNorm(self.head_dim, self.eps, self.dtype)
        self.rotary = Gemma4RotaryEmbedding(config, layer_idx)
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.q_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    @property
    def layer_name(self) -> str:
        return f"layers.{self.layer_idx}.self_attn"

    def _project(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = hidden_states.to(self.dtype) @ self.qkv_proj_weight
        query, key, value = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        query = query.view(-1, self.num_query_heads_per_rank, self.head_dim)
        key = key.view(-1, self.num_kv_heads_per_rank, self.head_dim)
        value = value.view(-1, self.num_kv_heads_per_rank, self.head_dim)

        cos, sin = self.rotary(positions, query.dtype)
        query = _apply_rotary(self.q_norm(query), cos, sin)
        key = _apply_rotary(self.k_norm(key), cos, sin)
        value = _rms_norm(value, self.eps)
        return query, key, value

    def _write_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Write all real tokens while making padded ``-1`` entries harmless.

        ``index_put_`` requires a fixed-size index set during tracing. Padding
        entries therefore target slot zero, but every such write carries the
        desired final value for slot zero (new value when slot zero is present,
        otherwise its old value). The result is independent of duplicate-write
        ordering and padded entries cannot corrupt a real cache line.
        """
        if self.k_cache is None or self.v_cache is None:
            raise RuntimeError(f"KV cache is not bound for {self.layer_name}")
        num_blocks = self.k_cache.shape[0]
        max_slot = num_blocks * block_size
        slots = slot_mapping.reshape(-1).to(torch.long)
        valid = (slots >= 0) & (slots < max_slot)
        safe_slots = torch.where(valid, slots, torch.zeros_like(slots))

        old_key_zero = self.k_cache[0, :, 0, :]
        old_value_zero = self.v_cache[0, :, 0, :]
        writes_slot_zero = (valid & (safe_slots == 0)).to(key.dtype)
        has_slot_zero = writes_slot_zero.sum().clamp(max=1)
        new_key_zero = (key * writes_slot_zero[:, None, None]).sum(
            dim=0
        ) + old_key_zero.to(key.dtype) * (1 - has_slot_zero)
        new_value_zero = (value * writes_slot_zero[:, None, None]).sum(
            dim=0
        ) + old_value_zero.to(value.dtype) * (1 - has_slot_zero)
        safe_key = torch.where(valid[:, None, None], key, new_key_zero[None])
        safe_value = torch.where(valid[:, None, None], value, new_value_zero[None])

        block_indices = safe_slots // block_size
        block_offsets = safe_slots % block_size
        heads = torch.arange(
            self.num_kv_heads_per_rank,
            dtype=torch.long,
            device=slots.device,
        )
        block_indices = block_indices[:, None].expand(-1, heads.numel()).reshape(-1)
        block_offsets = block_offsets[:, None].expand(-1, heads.numel()).reshape(-1)
        head_indices = heads[None, :].expand(slots.numel(), -1).reshape(-1)
        self.k_cache.index_put_(
            (block_indices, head_indices, block_offsets),
            safe_key.reshape(-1, self.head_dim).to(self.k_cache.dtype),
        )
        self.v_cache.index_put_(
            (block_indices, head_indices, block_offsets),
            safe_value.reshape(-1, self.head_dim).to(self.v_cache.dtype),
        )
        return valid

    def _repeat_kv(self, states: torch.Tensor) -> torch.Tensor:
        repeat = self.num_query_heads_per_rank // self.num_kv_heads_per_rank
        return states.repeat_interleave(repeat, dim=2)

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        # [T, Hq, D] and [T, Hkv, D] -> [Hq, T, T].
        key = self._repeat_kv(key.unsqueeze(0)).squeeze(0)
        value = self._repeat_kv(value.unsqueeze(0)).squeeze(0)
        scores = torch.einsum("thd,shd->hts", query.float(), key.float())
        query_positions = positions.reshape(-1, 1)
        key_positions = positions.reshape(1, -1)
        allowed = (key_positions <= query_positions) & valid.reshape(1, -1)
        if self.sliding_window is not None:
            allowed &= key_positions > query_positions - self.sliding_window
        # Padded query rows are not consumed, but giving them one legal key
        # prevents an all-masked softmax from producing NaNs.
        safe_for_padding = (~valid).reshape(-1, 1) & (
            torch.arange(key.shape[0], device=key.device) == 0
        ).reshape(1, -1)
        allowed |= safe_for_padding
        scores = scores.masked_fill(
            ~allowed.unsqueeze(0), torch.finfo(scores.dtype).min
        )
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        output = torch.einsum("hts,shd->thd", probabilities, value.float()).to(
            query.dtype
        )
        return output * valid[:, None, None].to(output.dtype)

    def _decode_attention(
        self,
        query: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        position_offset: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.k_cache is None or self.v_cache is None:
            raise RuntimeError(f"KV cache is not bound for {self.layer_name}")
        batch, num_blocks = block_table.shape
        if query.shape[0] != batch:
            raise ValueError(
                "Gemma 4 correctness path requires one decode token per sequence"
            )
        block_valid = block_table >= 0
        safe_blocks = torch.where(
            block_valid, block_table, torch.zeros_like(block_table)
        ).to(torch.long)
        gathered_key = self.k_cache.index_select(0, safe_blocks.reshape(-1)).view(
            batch,
            num_blocks,
            self.num_kv_heads_per_rank,
            block_size,
            self.head_dim,
        )
        gathered_value = self.v_cache.index_select(0, safe_blocks.reshape(-1)).view_as(
            gathered_key
        )
        gathered_key = (
            gathered_key.permute(0, 1, 3, 2, 4)
            .reshape(
                batch,
                num_blocks * block_size,
                self.num_kv_heads_per_rank,
                self.head_dim,
            )
            .to(query.dtype)
        )
        gathered_value = (
            gathered_value.permute(0, 1, 3, 2, 4)
            .reshape_as(gathered_key)
            .to(query.dtype)
        )
        gathered_key = self._repeat_kv(gathered_key)
        gathered_value = self._repeat_kv(gathered_value)

        context_positions = torch.arange(
            num_blocks * block_size,
            device=query.device,
            dtype=positions.dtype,
        )[None, :].expand(batch, -1)
        if position_offset is not None:
            context_positions = context_positions + position_offset.reshape(-1, 1)
        key_valid = (
            block_valid[:, :, None].expand(-1, -1, block_size).reshape(batch, -1)
        )
        allowed = key_valid & (context_positions <= positions.reshape(-1, 1))
        if self.sliding_window is not None:
            allowed &= (
                context_positions > positions.reshape(-1, 1) - self.sliding_window
            )
        safe_for_padding = (~valid).reshape(-1, 1) & (
            torch.arange(num_blocks * block_size, device=query.device) == 0
        ).reshape(1, -1)
        allowed |= safe_for_padding

        scores = torch.einsum("bhd,bshd->bhs", query.float(), gathered_key.float())
        scores = scores.masked_fill(~allowed[:, None, :], torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        output = torch.einsum(
            "bhs,bshd->bhd", probabilities, gathered_value.float()
        ).to(query.dtype)
        return output * valid[:, None, None].to(output.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
    ) -> torch.Tensor:
        metadata = attn_metadata[self.layer_name]
        query, key, value = self._project(hidden_states, positions)
        is_decode = metadata["max_query_len"] <= metadata["decode_token_threshold"]
        valid = self._write_cache(
            key,
            value,
            metadata["slot_mapping"],
            metadata["block_size"],
        )
        if is_decode:
            attended = self._decode_attention(
                query,
                positions,
                valid,
                metadata["block_table_tensor"],
                metadata["block_size"],
                metadata.get("swa_kv_pos_offset"),
            )
        else:
            attended = self._prefill_attention(query, key, value, positions, valid)
        local_output = attended.reshape(-1, self.q_size) @ self.o_proj_weight
        if self.world_size > 1:
            local_output = self.tp_group.all_reduce(local_output)
        return local_output


def _padded_shard_size(size: int, tp_size: int, alignment: int = 128) -> int:
    return math.ceil(math.ceil(size / tp_size) / alignment) * alignment


def _padded_kernel_size(size: int, alignment: int = 512) -> int:
    """Pad a contracting dimension to the public Neuron MoE tile size."""
    return math.ceil(size / alignment) * alignment


class Gemma4DenseMLP(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.dtype = config.torch_dtype
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        actual_size = config.layer_intermediate_size(layer_idx)
        self.intermediate_size_per_rank = _padded_shard_size(
            actual_size, self.world_size
        )
        self.gate_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=self.dtype,
            )
        )
        self.up_proj_weight = nn.Parameter(torch.empty_like(self.gate_proj_weight))
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank,
                config.hidden_size,
                dtype=self.dtype,
            )
        )
        for parameter in (self.gate_proj_weight, self.up_proj_weight):
            set_weight_loader(
                parameter,
                sharding_weight_loader(
                    shard_dim=1,
                    shard_size=self.intermediate_size_per_rank,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                    pad_shard=True,
                ),
            )
        set_weight_loader(
            self.down_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=True,
                pad_shard=True,
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = NF.mlp(
            hidden=hidden_states.to(self.dtype),
            gate_w=self.gate_proj_weight,
            up_w=self.up_proj_weight,
            down_w=self.down_proj_weight,
            act_fn=ActFnType.GELU_Tanh_Approx,
        )
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output


def _expert_gate_up_loader(
    intermediate_size: int,
    shard_size: int,
    tp_size: int,
    padded_hidden_size: int,
) -> SafetensorsWeightLoader:
    def transform(slices, rank):
        source = slices[0]
        start = (rank % tp_size) * shard_size
        end = min(start + shard_size, intermediate_size)
        experts, _, hidden = source.get_shape()
        valid_size = max(0, end - start)
        if valid_size:
            gate = source[:, start:end, :]
            up = source[:, intermediate_size + start : intermediate_size + end, :]
        else:
            gate = torch.empty(experts, 0, hidden, dtype=source[:, :1, :].dtype)
            up = torch.empty_like(gate)
        if valid_size < shard_size:
            gate = F.pad(gate, (0, 0, 0, shard_size - valid_size))
            up = F.pad(up, (0, 0, 0, shard_size - valid_size))
        hidden_padding = padded_hidden_size - hidden
        if hidden_padding:
            gate = F.pad(gate, (0, hidden_padding))
            up = F.pad(up, (0, hidden_padding))
        return torch.stack((gate.transpose(1, 2), up.transpose(1, 2)), dim=2)

    return SafetensorsWeightLoader(transform=transform)


def _expert_down_loader(
    intermediate_size: int,
    shard_size: int,
    tp_size: int,
    padded_hidden_size: int,
) -> SafetensorsWeightLoader:
    def transform(slices, rank):
        source = slices[0]
        per_expert_scale = slices[1][:]
        start = (rank % tp_size) * shard_size
        end = min(start + shard_size, intermediate_size)
        experts, hidden, _ = source.get_shape()
        valid_size = max(0, end - start)
        if valid_size:
            tensor = source[:, :, start:end]
        else:
            tensor = torch.empty(experts, hidden, 0, dtype=source[:, :, :1].dtype)
        if valid_size < shard_size:
            tensor = F.pad(tensor, (0, shard_size - valid_size))
        hidden_padding = padded_hidden_size - hidden
        if hidden_padding:
            tensor = F.pad(tensor, (0, 0, 0, hidden_padding))
        # Gemma applies this learned scalar after routing, independently for
        # every selected expert. Folding it into the down projection is
        # algebraically identical and lets the fused decode block retain the
        # checkpoint's exact semantics without another affinity-kernel input.
        return tensor.transpose(1, 2) * per_expert_scale[:, None, None]

    return SafetensorsWeightLoader(transform=transform)


class Gemma4Experts(nn.Module):
    """Replicated experts with TP-sharded expert intermediate dimensions."""

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        # The public BF16 MoE kernels tile H in groups of 512. Gemma's 2816
        # hidden width therefore needs a zero-padded 3072-wide expert-only
        # view. Router and residual math remain at the checkpoint-native width.
        self.kernel_hidden_size = _padded_kernel_size(config.hidden_size)
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.eps = config.rms_norm_eps
        self.root_hidden_size = config.hidden_size**-0.5
        self.prefill_tkg_chunk_size = int(
            os.environ.get("NEURON_GEMMA4_PREFILL_TKG_CHUNK_SIZE", "128")
        )
        if not 1 <= self.prefill_tkg_chunk_size <= 128:
            raise ValueError("NEURON_GEMMA4_PREFILL_TKG_CHUNK_SIZE must be in [1, 128]")
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.intermediate_size_per_rank = _padded_shard_size(
            config.moe_intermediate_size, self.world_size
        )
        # Router selection is deliberately FP32 in the checkpoint reference.
        # The fused block requires its router input and weights to have the
        # same dtype, so retain these comparatively small matrices in FP32.
        self.router_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                config.num_experts,
                dtype=torch.float32,
            )
        )
        self.router_scale = nn.Parameter(
            torch.ones(config.hidden_size, dtype=self.dtype)
        )
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(
                config.num_experts,
                self.kernel_hidden_size,
                2,
                self.intermediate_size_per_rank,
                dtype=self.dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                config.num_experts,
                self.intermediate_size_per_rank,
                self.kernel_hidden_size,
                dtype=self.dtype,
            )
        )
        set_weight_loader(
            self.router_weight,
            SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:].T),
        )
        set_weight_loader(
            self.gate_up_proj_weight,
            _expert_gate_up_loader(
                config.moe_intermediate_size,
                self.intermediate_size_per_rank,
                self.world_size,
                self.kernel_hidden_size,
            ),
        )
        set_weight_loader(
            self.down_proj_weight,
            _expert_down_loader(
                config.moe_intermediate_size,
                self.intermediate_size_per_rank,
                self.world_size,
                self.kernel_hidden_size,
            ),
        )

    def _run_tkg(self, expert_hidden_states: torch.Tensor) -> torch.Tensor:
        """Execute Gemma's expert block for at most 128 tokens on Neuron.

        Expert MLPs are token-independent, so the same fused kernel used for
        decode is an exact prefill implementation when a static token bucket
        is partitioned into supported chunks.
        """
        if expert_hidden_states.shape[0] > 128:
            raise ValueError("Gemma 4 TKG expert chunks must contain <= 128 tokens")
        expert_rank = torch.zeros(
            (1, 1), dtype=torch.int32, device=expert_hidden_states.device
        )
        router_gamma = self.router_scale * self.root_hidden_size
        router_weight = self.router_weight
        hidden_padding = self.kernel_hidden_size - self.hidden_size
        if hidden_padding:
            router_gamma = F.pad(router_gamma, (0, hidden_padding))
            router_weight = F.pad(router_weight, (0, 0, 0, hidden_padding))
        return NF.moe_block_tkg(
            inp=expert_hidden_states.unsqueeze(0),
            gamma=router_gamma.unsqueeze(0).to(torch.float32),
            router_weights=router_weight,
            expert_gate_up_weights=self.gate_up_proj_weight,
            expert_down_weights=self.down_proj_weight,
            top_k=self.top_k,
            eps=self.eps,
            router_act_fn=RouterActFnType.SOFTMAX,
            router_pre_norm=True,
            norm_topk_prob=True,
            router_mm_dtype=nl.float32,
            is_all_expert=True,
            rank_id=expert_rank,
            expert_affinities_scaling_mode=(ExpertAffinityScaleMode.POST_SCALE),
            hidden_act_fn=ActFnType.GELU_Tanh_Approx,
            hidden_actual=self.hidden_size,
            skip_router_logits=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_decode: bool,
    ) -> torch.Tensor:
        expert_hidden_states = hidden_states
        hidden_padding = self.kernel_hidden_size - self.hidden_size
        if hidden_padding:
            expert_hidden_states = F.pad(hidden_states, (0, hidden_padding))
        if is_decode:
            # Use the stack's complete fused BF16 decode block. Calling the
            # lower-level moe_tkg seam directly produced a device-side
            # scatter/gather OOB for Gemma's E=128/K=8 shape in both selective
            # and all-expert modes. The block owns RMSNorm, routing, and the
            # expert-affinity layout consumed by its inner kernel.
            output = self._run_tkg(expert_hidden_states)
        else:
            output = torch.cat(
                [
                    self._run_tkg(chunk)
                    for chunk in torch.split(
                        expert_hidden_states,
                        self.prefill_tkg_chunk_size,
                        dim=0,
                    )
                ],
                dim=0,
            )
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output[..., : self.hidden_size]


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4DenseMLP(config, layer_idx)
        self.experts = Gemma4Experts(config) if config.enable_moe_block else None

        def norm() -> Gemma4RMSNorm:
            return Gemma4RMSNorm(
                config.hidden_size, config.rms_norm_eps, config.torch_dtype
            )

        self.input_layernorm = norm()
        self.post_attention_layernorm = norm()
        self.pre_feedforward_layernorm = norm()
        self.post_feedforward_layernorm = norm()
        if self.experts is not None:
            self.post_feedforward_layernorm_1 = norm()
            self.post_feedforward_layernorm_2 = norm()
            self.pre_feedforward_layernorm_2 = norm()
        self.layer_scalar = nn.Parameter(
            torch.ones(1, dtype=config.torch_dtype), requires_grad=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
    ) -> torch.Tensor:
        metadata = attn_metadata[self.self_attn.layer_name]
        is_decode = metadata["max_query_len"] <= metadata["decode_token_threshold"]
        residual = hidden_states
        attended = self.self_attn(
            self.input_layernorm(hidden_states), positions, attn_metadata
        )
        hidden_states = residual + self.post_attention_layernorm(attended)

        residual = hidden_states
        dense = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        if self.experts is not None:
            dense = self.post_feedforward_layernorm_1(dense)
            expert_input = self.pre_feedforward_layernorm_2(residual)
            expert = self.experts(expert_input, is_decode)
            expert = self.post_feedforward_layernorm_2(expert)
            dense = dense + expert
        hidden_states = self.post_feedforward_layernorm(dense)
        return (residual + hidden_states) * self.layer_scalar


class Gemma4TextModel(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.embed_tokens = VocabDimShardedEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList(
            Gemma4DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.embedding_scale = math.sqrt(config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        rank: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        is_token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(
            input_ids.reshape(-1), scatter_tokens=False, rank=rank
        )
        hidden_states = hidden_states * self.embedding_scale
        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, attn_metadata)
        return self.norm(hidden_states)


class Gemma4ForCausalLM(nn.Module):
    """Runner-facing native Gemma 4 causal language model."""

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self._reference_mode = os.environ.get("VLLM_NEURON_GEMMA4_REFERENCE") == "1"
        self._bound_kv_caches: dict[str, list[torch.Tensor]] = {}
        if self._reference_mode:
            from .reference import Gemma4ReferenceCausalLM

            self._reference = Gemma4ReferenceCausalLM(config)
            return

        neuron_config = config.neuron_config
        if neuron_config is None:
            raise ValueError("native Gemma 4 requires a NeuronConfig")
        unsupported = {
            "ep_degree": neuron_config.ep_degree,
            "attention_dp_size": neuron_config.attention_dp_size,
            "embedding_dp_size": neuron_config.embedding_dp_size,
            "lm_head_dp_size": neuron_config.lm_head_dp_size,
            "mlp_dp_size": neuron_config.mlp_dp_size,
        }
        invalid = {name: value for name, value in unsupported.items() if value != 1}
        if invalid:
            raise NotImplementedError(
                f"Gemma 4 correctness path currently supports TP only: {invalid}"
            )
        if neuron_config.on_device_sampling_config is not None:
            raise NotImplementedError(
                "Gemma 4 correctness path requires on-device sampling disabled"
            )
        self.model = Gemma4TextModel(config)
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=True,
            tp_group=self.tp_group.device_group,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del sampling_params, spec_decode_metadata, logit_mask, kwargs
        if self._reference_mode:
            return self._reference(
                input_ids,
                sampling_positions=sampling_positions,
                position_ids=positions,
            )
        if attn_metadata is None:
            raise ValueError("native Gemma 4 requires attention metadata")
        positions = positions.reshape(-1).to(torch.long)
        hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata,
            rank,
            inputs_embeds,
            is_token_ids,
        )
        if sampling_positions is None:
            raise ValueError("native Gemma 4 requires sampling_positions")
        hidden_states = torch.index_select(
            hidden_states,
            dim=0,
            index=sampling_positions.reshape(-1).to(torch.long),
        )
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits

    def get_kv_spec(self) -> KVSpec:
        layers = []
        for layer_idx in range(self.config.num_hidden_layers):
            head_dim, num_kv_heads = self.config.attention_shape(layer_idx)
            if not self._reference_mode:
                attention = self.model.layers[layer_idx].self_attn
                num_kv_heads = attention.num_kv_heads_per_rank
            layers.append(
                LayerSpec(
                    name=f"layers.{layer_idx}.self_attn",
                    num_kv_heads=num_kv_heads,
                    head_size=head_dim,
                    dtype=self.config.torch_dtype,
                    sliding_window_size=(
                        None
                        if self.config.layer_is_global(layer_idx)
                        else self.config.sliding_window
                    ),
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(
        self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]
    ) -> None:
        expected = {layer.name for layer in self.get_kv_spec().layers}
        missing = sorted(expected.difference(kv_caches))
        if missing:
            raise ValueError(f"KV cache missing layers: {missing}")
        self._bound_kv_caches = {name: kv_caches[name] for name in sorted(expected)}
        if not self._reference_mode:
            for layer_idx, layer in enumerate(self.model.layers):
                cache = kv_caches[f"layers.{layer_idx}.self_attn"]
                layer.self_attn.k_cache = cache[0]
                layer.self_attn.v_cache = cache[1]

    def _checkpoint_mappings(self) -> dict[str, str | list[str]]:
        prefix = Gemma4WeightMapper.CHECKPOINT_TEXT_PREFIX
        mappings: dict[str, str | list[str]] = {
            "model.embed_tokens.weight": f"{prefix}embed_tokens.weight",
            "model.norm.weight": f"{prefix}norm.weight",
            "lm_head.weight": f"{prefix}embed_tokens.weight",
        }
        for layer_idx, layer in enumerate(self.model.layers):
            native = f"model.layers.{layer_idx}"
            source = f"{prefix}layers.{layer_idx}"
            qkv_sources = [
                f"{source}.self_attn.q_proj.weight",
                f"{source}.self_attn.k_proj.weight",
                (
                    f"{source}.self_attn.k_proj.weight"
                    if self.config.layer_is_global(layer_idx)
                    and self.config.attention_k_eq_v
                    else f"{source}.self_attn.v_proj.weight"
                ),
            ]
            mappings[f"{native}.self_attn.qkv_proj_weight"] = qkv_sources
            mappings[f"{native}.self_attn.o_proj_weight"] = (
                f"{source}.self_attn.o_proj.weight"
            )
            mappings[f"{native}.self_attn.q_norm.weight"] = (
                f"{source}.self_attn.q_norm.weight"
            )
            mappings[f"{native}.self_attn.k_norm.weight"] = (
                f"{source}.self_attn.k_norm.weight"
            )
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "post_feedforward_layernorm_1",
                "post_feedforward_layernorm_2",
                "pre_feedforward_layernorm_2",
            ):
                parameter_name = f"{native}.{name}.weight"
                if parameter_name in dict(self.named_parameters()):
                    mappings[parameter_name] = f"{source}.{name}.weight"
            mappings[f"{native}.layer_scalar"] = f"{source}.layer_scalar"
            mappings[f"{native}.mlp.gate_proj_weight"] = (
                f"{source}.mlp.gate_proj.weight"
            )
            mappings[f"{native}.mlp.up_proj_weight"] = f"{source}.mlp.up_proj.weight"
            mappings[f"{native}.mlp.down_proj_weight"] = (
                f"{source}.mlp.down_proj.weight"
            )
            if layer.experts is not None:
                mappings[f"{native}.experts.router_weight"] = (
                    f"{source}.router.proj.weight"
                )
                mappings[f"{native}.experts.router_scale"] = f"{source}.router.scale"
                mappings[f"{native}.experts.gate_up_proj_weight"] = (
                    f"{source}.experts.gate_up_proj"
                )
                mappings[f"{native}.experts.down_proj_weight"] = [
                    f"{source}.experts.down_proj",
                    f"{source}.router.per_expert_scale",
                ]
        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        if self._reference_mode:
            mappings = Gemma4WeightMapper.build_mappings(
                (name for name, _ in self.named_parameters()),
                tied_lm_head=self.config.tie_word_embeddings,
            )
            checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
            result = checkpoint.load_sharded(
                rank=0,
                world_size=1,
                model=self,
                mappings=mappings,
                device=device,
                strict=True,
            )
            self.load_state_dict(result.state_dict, strict=False, assign=True)
            self._last_checkpoint_load_result = result
            return

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        result = checkpoint.load_sharded_pipelined(
            self.rank,
            self.world_size,
            self,
            self._checkpoint_mappings(),
            device,
            strict=True,
        )
        self.load_state_dict(result.state_dict, strict=False, assign=True)
        self._last_checkpoint_load_result = result

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        # BF16 Gemma has no checkpoint-derived cache scales or other constants.
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        checkpoint._ensure_indexed()
