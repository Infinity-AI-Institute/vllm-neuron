"""Native BF16 Inkling-Small text inference for vLLM-Neuron.

This is a correctness-first implementation of the released text backbone.  It
uses tensor parallel attention/dense projections, expert parallel routed MoE
weights, the public BF16 NKI expert kernel, and model-owned paged caches for
both attention K/V and Inkling's four short-convolution streams.

Vision, audio, MTP drafting, sequence parallelism, and quantized weights are
intentionally outside the initial text-serving contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import InklingConfig
from .routing import inkling_route


def _rms_norm(
    hidden_states: torch.Tensor,
    eps: float,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    states = hidden_states.float()
    states = states * torch.rsqrt(states.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        states = states * weight.float()
    return states.to(input_dtype)


def _last_selected_row_or(
    rows: torch.Tensor,
    selected: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Select the last marked row without a data-dependent tensor index."""

    row_numbers = torch.arange(rows.shape[0], device=rows.device, dtype=torch.long) + 1
    last_number = torch.where(
        selected, row_numbers, torch.zeros_like(row_numbers)
    ).amax()
    last_mask = selected & (row_numbers == last_number)
    broadcast_shape = (rows.shape[0],) + (1,) * (rows.ndim - 1)
    selected_row = (rows * last_mask.reshape(broadcast_shape).to(rows.dtype)).sum(dim=0)
    return torch.where(last_number > 0, selected_row, fallback)


class InklingRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _rms_norm(hidden_states, self.eps, self.weight)


def _head_column_loader(
    *,
    local_heads: int,
    head_dim: int,
    replicas: int = 1,
) -> SafetensorsWeightLoader:
    """Shard an HF ``[heads*D, H]`` projection into native ``[H, local]``."""

    def transform(slices, rank):
        source = slices[0]
        source_rank = rank // replicas
        start = source_rank * local_heads * head_dim
        width = local_heads * head_dim
        return source[start : start + width, :].T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _head_conv_loader(
    *,
    local_heads: int,
    head_dim: int,
    replicas: int,
) -> SafetensorsWeightLoader:
    """Shard an HF depthwise convolution ``[heads*D, 1, W]``."""

    def transform(slices, rank):
        source = slices[0]
        start = (rank // replicas) * local_heads * head_dim
        width = local_heads * head_dim
        return source[start : start + width, :, :].float().contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _dense_gate_up_loader(
    intermediate_size: int,
    shard_size: int,
    tp_size: int,
) -> SafetensorsWeightLoader:
    """Load contiguous interleaved gate/up pairs into ``[H, I, 2]``."""

    def transform(slices, rank):
        source = slices[0]
        start = (rank % tp_size) * 2 * shard_size
        rows = source[start : start + 2 * shard_size, :]
        return rows.reshape(shard_size, 2, rows.shape[-1]).permute(2, 0, 1)

    return SafetensorsWeightLoader(transform=transform)


def _dense_down_loader(
    shard_size: int,
    tp_size: int,
) -> SafetensorsWeightLoader:
    def transform(slices, rank):
        source = slices[0]
        start = (rank % tp_size) * shard_size
        return source[:, start : start + shard_size].T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _local_expert_gate_up_loader(
    *,
    num_local_experts: int,
    intermediate_size: int,
) -> SafetensorsWeightLoader:
    def transform(slices, rank):
        source = slices[0]
        start = rank * num_local_experts
        local = source[start : start + num_local_experts]
        return (
            local.reshape(
                num_local_experts,
                intermediate_size,
                2,
                local.shape[-1],
            )
            .permute(0, 3, 2, 1)
            .contiguous()
        )

    return SafetensorsWeightLoader(transform=transform)


def _local_expert_down_loader(*, num_local_experts: int) -> SafetensorsWeightLoader:
    def transform(slices, rank):
        source = slices[0]
        start = rank * num_local_experts
        return source[start : start + num_local_experts].transpose(1, 2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _shared_gate_up_loader(
    *,
    intermediate_size: int,
    shard_size: int,
    tp_size: int,
) -> SafetensorsWeightLoader:
    def transform(slices, rank):
        source = slices[0]
        start = (rank % tp_size) * 2 * shard_size
        rows = source[:, start : start + 2 * shard_size, :]
        return (
            rows.reshape(
                rows.shape[0],
                shard_size,
                2,
                rows.shape[-1],
            )
            .permute(0, 3, 2, 1)
            .contiguous()
        )

    return SafetensorsWeightLoader(transform=transform)


def _shared_down_loader(*, shard_size: int, tp_size: int) -> SafetensorsWeightLoader:
    def transform(slices, rank):
        source = slices[0]
        start = (rank % tp_size) * shard_size
        return source[:, :, start : start + shard_size].transpose(1, 2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


class InklingPagedConv(nn.Module):
    """Four model streams packed into one vLLM-managed paged cache.

    The updated cache is an explicit input/output of every stream operation.
    This is required for Neuron graph correctness: four independent in-place
    writes to the same module attribute can be traced as dead sibling writes,
    so later streams and decode invocations may observe stale state.
    """

    K = 0
    V = 1
    ATTN = 2
    MLP = 3

    def __init__(self, config: InklingConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.kernel_size = config.sconv_kernel_size
        self.hidden_size = config.hidden_size
        self.widths = (
            config.head_dim,
            config.head_dim,
            config.hidden_size,
            config.hidden_size,
        )
        offsets = [0]
        for width in self.widths[:-1]:
            offsets.append(offsets[-1] + width)
        self.offsets = tuple(offsets)
        self.total_width = sum(self.widths)
        self.cache: torch.Tensor | None = None
        # Full page-major allocation shape, including vLLM's otherwise-unused
        # second K/V stream. It is populated by bind_kv_cache_roots().
        self.cache_shape: tuple[int, ...] | None = None

    @property
    def layer_name(self) -> str:
        return f"layers.{self.layer_idx}.conv_state"

    def _write(
        self,
        cache: torch.Tensor,
        states: torch.Tensor,
        stream: int,
        metadata: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = int(metadata["block_size"])
        slots = metadata["slot_mapping"].reshape(-1).to(torch.long)
        max_slot = cache.shape[0] * block_size
        valid = (slots >= 0) & (slots < max_slot)
        safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
        block_indices = torch.div(safe_slots, block_size, rounding_mode="floor")
        block_offsets = safe_slots.remainder(block_size)

        start_head = self.offsets[stream] // self.head_dim
        num_heads = self.widths[stream] // self.head_dim
        head_indices = (
            torch.arange(num_heads, device=states.device, dtype=torch.long) + start_head
        )
        old_zero = cache[
            0,
            head_indices,
            torch.zeros_like(head_indices),
        ]
        has_real_zero = valid & (safe_slots == 0)
        state_rows = states.reshape(-1, num_heads, self.head_dim)
        zero_value = _last_selected_row_or(
            state_rows,
            has_real_zero,
            old_zero,
        )
        values = torch.where(
            valid[:, None, None],
            state_rows,
            zero_value[None, :, :],
        )
        updated_cache = cache.index_put_(
            (
                block_indices[:, None].expand(-1, num_heads).reshape(-1),
                head_indices[None, :].expand(states.shape[0], -1).reshape(-1),
                block_offsets[:, None].expand(-1, num_heads).reshape(-1),
            ),
            values.reshape(-1, self.head_dim).to(cache.dtype),
        )
        return valid, updated_cache

    def _read_positions(
        self,
        cache: torch.Tensor,
        absolute_positions: torch.Tensor,
        stream: int,
        metadata: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_table = metadata["block_table_tensor"].to(torch.long)
        batch = block_table.shape[0]
        tokens = absolute_positions.shape[0]
        if batch == 1:
            sequence_ids = torch.zeros(
                tokens, device=absolute_positions.device, dtype=torch.long
            )
        elif tokens == batch:
            sequence_ids = torch.arange(
                batch, device=absolute_positions.device, dtype=torch.long
            )
        else:
            raise ValueError(
                "Inkling initial serving path supports one prefill request or "
                "one decode token per request"
            )
        block_size = int(metadata["block_size"])
        position_offset = metadata.get("swa_kv_pos_offset")
        if position_offset is None:
            position_offset = torch.zeros(
                batch, device=absolute_positions.device, dtype=torch.long
            )
        else:
            position_offset = position_offset.reshape(-1).to(torch.long)

        relative_positions = absolute_positions.to(
            torch.long
        ) - position_offset.index_select(0, sequence_ids)
        table_columns = torch.div(relative_positions, block_size, rounding_mode="floor")
        valid = (
            (absolute_positions >= 0)
            & (table_columns >= 0)
            & (table_columns < block_table.shape[1])
        )
        safe_columns = table_columns.clamp(0, block_table.shape[1] - 1)
        block_ids = block_table[sequence_ids, safe_columns]
        valid &= block_ids >= 0
        safe_blocks = block_ids.clamp(0, cache.shape[0] - 1)
        offsets = relative_positions.remainder(block_size)

        start_head = self.offsets[stream] // self.head_dim
        num_heads = self.widths[stream] // self.head_dim
        selected = cache.index_select(0, safe_blocks)[
            :, start_head : start_head + num_heads
        ]
        gather_index = offsets[:, None, None, None].expand(
            -1, num_heads, 1, self.head_dim
        )
        values = torch.gather(selected, 2, gather_index).squeeze(2)
        values = values.reshape(tokens, self.widths[stream]).float()
        return values, valid

    def forward(
        self,
        states: torch.Tensor,
        positions: torch.Tensor,
        weight: torch.Tensor,
        stream: int,
        metadata: dict,
        cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_dtype = states.dtype
        states = states.reshape(-1, self.widths[stream])
        current_valid, cache = self._write(cache, states, stream, metadata)
        output = states.float()
        for tap in range(self.kernel_size):
            tap_positions = positions.reshape(-1) - (self.kernel_size - 1 - tap)
            cached, tap_valid = self._read_positions(
                cache, tap_positions, stream, metadata
            )
            output = output + (
                cached
                * weight.reshape(self.widths[stream], self.kernel_size)[:, tap]
                * tap_valid[:, None]
            )
        output = (output * current_valid[:, None].to(output.dtype)).to(input_dtype)
        return output, cache


class InklingAttention(nn.Module):
    """TP-sharded GQA with relative logits and paged K/V ownership."""

    def __init__(
        self,
        config: InklingConfig,
        layer_idx: int,
        conv: InklingPagedConv,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.head_dim = (
            config.swa_head_dim if config.layer_is_local(layer_idx) else config.head_dim
        )
        self.num_query_heads = (
            config.swa_num_attention_heads
            if config.layer_is_local(layer_idx)
            else config.num_attention_heads
        )
        self.num_kv_heads = (
            config.swa_num_key_value_heads
            if config.layer_is_local(layer_idx)
            else config.num_key_value_heads
        )
        self.sliding_window = (
            config.sliding_window_size if config.layer_is_local(layer_idx) else None
        )
        self.relative_extent = (
            config.sliding_window_size
            if config.layer_is_local(layer_idx)
            else config.rel_extent
        )
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        if self.num_query_heads % self.world_size:
            raise ValueError("Inkling query heads must divide tensor parallel size")
        self.num_query_heads_per_rank = self.num_query_heads // self.world_size
        if self.world_size >= self.num_kv_heads:
            if self.world_size % self.num_kv_heads:
                raise ValueError("Inkling tensor parallel size must divide KV replicas")
            self.num_kv_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_kv_heads
        else:
            if self.num_kv_heads % self.world_size:
                raise ValueError("Inkling KV heads must divide tensor parallel size")
            self.num_kv_heads_per_rank = self.num_kv_heads // self.world_size
            self.num_kv_replicas = 1
        if self.num_query_heads_per_rank != 1 or self.num_kv_heads_per_rank != 1:
            raise NotImplementedError(
                "Inkling-Small initial path is compiled for TP32 (one Q/KV head)"
            )
        self.q_size = self.num_query_heads_per_rank * self.head_dim
        self.kv_size = self.num_kv_heads_per_rank * self.head_dim
        self.r_size = self.num_query_heads_per_rank * config.d_rel
        self.q_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.q_size, dtype=self.dtype)
        )
        self.k_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.kv_size, dtype=self.dtype)
        )
        self.v_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.kv_size, dtype=self.dtype)
        )
        self.r_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.r_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.q_size, config.hidden_size, dtype=self.dtype)
        )
        self.q_norm = InklingRMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = InklingRMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.relative_projection = nn.Parameter(
            torch.empty(config.d_rel, self.relative_extent, dtype=self.dtype)
        )
        self.k_sconv_weight = nn.Parameter(
            torch.empty(self.kv_size, 1, config.sconv_kernel_size, dtype=torch.float32)
        )
        self.v_sconv_weight = nn.Parameter(torch.empty_like(self.k_sconv_weight))
        self.conv = conv
        # K and V remain one page-major graph input/output alias with adjacent
        # head ranges. The runner binds vLLM's authoritative allocation
        # through ``bind_kv_cache_roots``.
        self.kv_cache: torch.Tensor | None = None
        self.kv_cache_shape: tuple[int, ...] | None = None

        set_weight_loader(
            self.q_proj_weight,
            _head_column_loader(
                local_heads=self.num_query_heads_per_rank,
                head_dim=self.head_dim,
            ),
        )
        kv_loader = _head_column_loader(
            local_heads=self.num_kv_heads_per_rank,
            head_dim=self.head_dim,
            replicas=self.num_kv_replicas,
        )
        set_weight_loader(self.k_proj_weight, kv_loader)
        set_weight_loader(self.v_proj_weight, kv_loader)
        set_weight_loader(
            self.r_proj_weight,
            _head_column_loader(
                local_heads=self.num_query_heads_per_rank,
                head_dim=config.d_rel,
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
        conv_loader = _head_conv_loader(
            local_heads=self.num_kv_heads_per_rank,
            head_dim=self.head_dim,
            replicas=self.num_kv_replicas,
        )
        set_weight_loader(self.k_sconv_weight, conv_loader)
        set_weight_loader(self.v_sconv_weight, conv_loader)

    @property
    def layer_name(self) -> str:
        return f"layers.{self.layer_idx}.self_attn"

    def _write_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: dict,
        kv_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_cache is None:
            kv_cache = self.kv_cache
        if kv_cache is None:
            raise RuntimeError(f"KV cache is not bound for {self.layer_name}")
        block_size = int(metadata["block_size"])
        slots = metadata["slot_mapping"].reshape(-1).to(torch.long)
        max_slot = kv_cache.shape[0] * block_size
        valid = (slots >= 0) & (slots < max_slot)
        safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
        blocks = torch.div(safe_slots, block_size, rounding_mode="floor")
        offsets = safe_slots.remainder(block_size)
        packed_heads = torch.arange(
            2 * self.num_kv_heads_per_rank,
            device=key.device,
            dtype=torch.long,
        )
        has_real_zero = valid & (safe_slots == 0)
        # K and V share one cache tensor as adjacent head ranges:
        # [blocks, 2 * kv_heads, block_size, head_dim].  A single scatter over
        # that head axis is deliberate.  Neuron can drop the first half of a
        # scatter whose leading index selects a synthetic K/V stream axis,
        # even though the FX graph and input/output alias both look correct.
        # Packing K/V as heads makes every page update one ordinary cache-row
        # scatter, which is the same persistent-state pattern used by the
        # convolution cache.
        states = torch.cat((key, value), dim=1)
        zero_rows = _last_selected_row_or(
            states,
            has_real_zero,
            kv_cache[
                0,
                packed_heads,
                torch.zeros_like(packed_heads),
            ],
        )
        state_values = torch.where(
            valid[:, None, None],
            states,
            zero_rows[None, :, :],
        )
        cache_indices = (
            blocks[:, None].expand(-1, packed_heads.numel()).reshape(-1),
            packed_heads[None, :].expand(key.shape[0], -1).reshape(-1),
            offsets[:, None].expand(-1, packed_heads.numel()).reshape(-1),
        )
        updated_kv_cache = kv_cache.index_put_(
            cache_indices,
            state_values.reshape(-1, self.head_dim).to(kv_cache.dtype),
        )
        return valid, updated_kv_cache

    def _read_written_rows(
        self,
        cache: torch.Tensor,
        metadata: dict,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Read this step's rows from the post-write cache tensor.

        Besides making the state transition directly testable, this establishes
        a true dataflow edge from each cache update into prefill attention. The
        Neuron compiler must therefore preserve the same scatter value both for
        model math and for the input/output state alias.
        """
        block_size = int(metadata["block_size"])
        slots = metadata["slot_mapping"].reshape(-1).to(torch.long)
        max_slot = cache.shape[0] * block_size
        safe_slots = slots.clamp(0, max_slot - 1)
        blocks = torch.div(safe_slots, block_size, rounding_mode="floor")
        offsets = safe_slots.remainder(block_size)
        heads = torch.arange(
            self.num_kv_heads_per_rank,
            device=cache.device,
            dtype=torch.long,
        )
        rows = cache[
            blocks[:, None],
            heads[None, :],
            offsets[:, None],
        ]
        return rows * valid[:, None, None]

    def _relative_bias(
        self,
        relative_states: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        # profile: [Tq, H, extent].
        profile = torch.einsum(
            "thd,de->the",
            relative_states.float(),
            self.relative_projection.float(),
        )
        if key_positions.ndim == 1:
            distance = query_positions.reshape(-1, 1) - key_positions.reshape(1, -1)
            valid = (distance >= 0) & (distance < self.relative_extent)
            index = distance.clamp(0, self.relative_extent - 1)
            bias = torch.gather(
                profile.permute(1, 0, 2),
                -1,
                index[None, :, :].expand(profile.shape[1], -1, -1),
            )
            return bias * valid[None, :, :]
        distance = query_positions.reshape(-1, 1) - key_positions
        valid = (distance >= 0) & (distance < self.relative_extent)
        index = distance.clamp(0, self.relative_extent - 1)
        bias = torch.gather(
            profile,
            -1,
            index[:, None, :].expand(-1, profile.shape[1], -1),
        )
        return bias * valid[:, None, :]

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        relative: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        scores = (
            torch.einsum("thd,shd->hts", query.float(), key.float()) / self.head_dim
        )
        bias = self._relative_bias(relative, positions, positions)
        if self.sliding_window is None and self.config.log_scaling_n_floor:
            tau = 1.0 + self.config.log_scaling_alpha * torch.log(
                ((positions.float() + 1) / self.config.log_scaling_n_floor).clamp(
                    min=1.0
                )
            )
            scores = scores * tau[None, :, None]
            bias = bias * tau[None, :, None]
        scores = scores + bias
        qpos = positions.reshape(-1, 1)
        kpos = positions.reshape(1, -1)
        allowed = (kpos <= qpos) & valid.reshape(1, -1)
        if self.sliding_window is not None:
            allowed &= kpos > qpos - self.sliding_window
        safe_padding = (~valid).reshape(-1, 1) & (
            torch.arange(key.shape[0], device=key.device) == 0
        ).reshape(1, -1)
        allowed |= safe_padding
        scores = scores.masked_fill(~allowed[None, :, :], torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        output = torch.einsum("hts,shd->thd", probabilities, value.float()).to(
            query.dtype
        )
        return output * valid[:, None, None]

    def _decode_attention(
        self,
        query: torch.Tensor,
        relative: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
        metadata: dict,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        key_cache = kv_cache[:, : self.num_kv_heads_per_rank]
        value_cache = kv_cache[:, self.num_kv_heads_per_rank :]
        block_table = metadata["block_table_tensor"].to(torch.long)
        batch, num_blocks = block_table.shape
        if query.shape[0] != batch:
            raise ValueError("Inkling decode requires one token per request")
        block_valid = block_table >= 0
        safe_blocks = block_table.clamp(0, key_cache.shape[0] - 1)
        key = key_cache.index_select(0, safe_blocks.reshape(-1)).view(
            batch,
            num_blocks,
            self.num_kv_heads_per_rank,
            int(metadata["block_size"]),
            self.head_dim,
        )
        value = value_cache.index_select(0, safe_blocks.reshape(-1)).view_as(key)
        key = key.permute(0, 1, 3, 2, 4).reshape(
            batch, -1, self.num_kv_heads_per_rank, self.head_dim
        )
        value = value.permute(0, 1, 3, 2, 4).reshape_as(key)
        context_positions = torch.arange(
            key.shape[1], device=query.device, dtype=positions.dtype
        )[None, :].expand(batch, -1)
        offset = metadata.get("swa_kv_pos_offset")
        if offset is not None:
            context_positions = context_positions + offset.reshape(-1, 1)
        key_valid = (
            block_valid[:, :, None]
            .expand(-1, -1, int(metadata["block_size"]))
            .reshape(batch, -1)
        )
        allowed = key_valid & (context_positions <= positions.reshape(-1, 1))
        if self.sliding_window is not None:
            allowed &= (
                context_positions > positions.reshape(-1, 1) - self.sliding_window
            )
        safe_padding = (~valid).reshape(-1, 1) & (
            torch.arange(key.shape[1], device=query.device) == 0
        ).reshape(1, -1)
        allowed |= safe_padding
        scores = (
            torch.einsum("bhd,bshd->bhs", query.float(), key.float()) / self.head_dim
        )
        bias = self._relative_bias(relative, positions, context_positions)
        if self.sliding_window is None and self.config.log_scaling_n_floor:
            tau = 1.0 + self.config.log_scaling_alpha * torch.log(
                ((positions.float() + 1) / self.config.log_scaling_n_floor).clamp(
                    min=1.0
                )
            )
            scores = scores * tau[:, None, None]
            bias = bias * tau[:, None, None]
        scores = scores + bias
        scores = scores.masked_fill(~allowed[:, None, :], torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        output = torch.einsum("bhs,bshd->bhd", probabilities, value.float()).to(
            query.dtype
        )
        return output * valid[:, None, None]

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        cache_root: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = attn_metadata[self.layer_name]
        conv_metadata = attn_metadata[self.conv.layer_name]
        if self.conv.cache_shape is None or self.kv_cache_shape is None:
            raise RuntimeError(
                f"cache allocation shapes are not bound for {self.layer_name}"
            )
        # Attention and short-convolution layers can share one vLLM HMA
        # allocation. Carry the full allocation through both reinterpretations
        # so all writes form one explicit dataflow chain.
        conv_cache = cache_root.view(self.conv.cache_shape)
        hidden_states = hidden_states.to(self.dtype)
        query = (hidden_states @ self.q_proj_weight).view(
            -1, self.num_query_heads_per_rank, self.head_dim
        )
        key = hidden_states @ self.k_proj_weight
        value = hidden_states @ self.v_proj_weight
        relative = (hidden_states @ self.r_proj_weight).view(
            -1, self.num_query_heads_per_rank, self.config.d_rel
        )
        key, conv_cache = self.conv(
            key,
            positions,
            self.k_sconv_weight,
            InklingPagedConv.K,
            conv_metadata,
            conv_cache,
        )
        key = key.view(-1, self.num_kv_heads_per_rank, self.head_dim)
        value, conv_cache = self.conv(
            value,
            positions,
            self.v_sconv_weight,
            InklingPagedConv.V,
            conv_metadata,
            conv_cache,
        )
        value = value.view(-1, self.num_kv_heads_per_rank, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        kv_cache = conv_cache.reshape(self.kv_cache_shape)
        valid, kv_cache = self._write_cache(key, value, metadata, kv_cache)
        is_decode = metadata["max_query_len"] <= metadata["decode_token_threshold"]
        if is_decode:
            attended = self._decode_attention(
                query,
                relative,
                positions,
                valid,
                metadata,
                kv_cache,
            )
        else:
            key = self._read_written_rows(
                kv_cache[:, : self.num_kv_heads_per_rank],
                metadata,
                valid,
            )
            value = self._read_written_rows(
                kv_cache[:, self.num_kv_heads_per_rank :],
                metadata,
                valid,
            )
            attended = self._prefill_attention(
                query, key, value, relative, positions, valid
            )
        output = attended.reshape(-1, self.q_size) @ self.o_proj_weight
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output, kv_cache.reshape(-1)


class InklingDenseMLP(nn.Module):
    def __init__(self, config: InklingConfig):
        super().__init__()
        self.dtype = config.torch_dtype
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        if config.dense_intermediate_size % self.world_size:
            raise ValueError("Inkling dense width must divide TP")
        self.intermediate_per_rank = config.dense_intermediate_size // self.world_size
        self.gate_up_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_per_rank,
                2,
                dtype=self.dtype,
            )
        )
        self.down_weight = nn.Parameter(
            torch.empty(
                self.intermediate_per_rank,
                config.hidden_size,
                dtype=self.dtype,
            )
        )
        self.global_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        set_weight_loader(
            self.gate_up_weight,
            _dense_gate_up_loader(
                config.dense_intermediate_size,
                self.intermediate_per_rank,
                self.world_size,
            ),
        )
        set_weight_loader(
            self.down_weight,
            _dense_down_loader(self.intermediate_per_rank, self.world_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = torch.einsum(
            "th,hip->tip",
            hidden_states.to(self.dtype),
            self.gate_up_weight,
        )
        local = (F.silu(gate_up[..., 0]) * gate_up[..., 1]) @ self.down_weight
        local = local * self.global_scale
        if self.world_size > 1:
            local = self.tp_group.all_reduce(local)
        return local


class InklingMoE(nn.Module):
    """Exact router plus EP-sharded routed and TP-sharded shared experts."""

    def __init__(self, config: InklingConfig):
        super().__init__()
        self.config = config
        self.dtype = config.torch_dtype
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        if not get_current_vllm_config().parallel_config.enable_expert_parallel:
            raise ValueError("Inkling-Small requires --enable-expert-parallel")
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_ep_degree,
            get_neuron_ep_rank,
        )

        self.ep_degree = get_neuron_ep_degree()
        self.ep_rank = get_neuron_ep_rank()
        if self.ep_degree != self.world_size:
            raise NotImplementedError(
                "Inkling-Small initial path requires pure EP across TP32"
            )
        if config.n_routed_experts % self.ep_degree:
            raise ValueError("Inkling experts must divide expert parallel degree")
        self.num_local_experts = config.n_routed_experts // self.ep_degree
        self.router_weight = nn.Parameter(
            torch.empty(
                config.n_routed_experts + config.n_shared_experts,
                config.hidden_size,
                dtype=self.dtype,
            )
        )
        self.correction_bias = nn.Parameter(
            torch.zeros(config.n_routed_experts, dtype=torch.float32)
        )
        self.global_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.expert_gate_up_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                config.hidden_size,
                2,
                config.intermediate_size,
                dtype=self.dtype,
            )
        )
        self.expert_down_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                config.intermediate_size,
                config.hidden_size,
                dtype=self.dtype,
            )
        )
        if config.intermediate_size % self.world_size:
            raise ValueError("Inkling shared expert width must divide TP")
        self.shared_intermediate_per_rank = config.intermediate_size // self.world_size
        self.shared_gate_up_weight = nn.Parameter(
            torch.empty(
                config.n_shared_experts,
                config.hidden_size,
                2,
                self.shared_intermediate_per_rank,
                dtype=self.dtype,
            )
        )
        self.shared_down_weight = nn.Parameter(
            torch.empty(
                config.n_shared_experts,
                self.shared_intermediate_per_rank,
                config.hidden_size,
                dtype=self.dtype,
            )
        )
        set_weight_loader(
            self.expert_gate_up_weight,
            _local_expert_gate_up_loader(
                num_local_experts=self.num_local_experts,
                intermediate_size=config.intermediate_size,
            ),
        )
        set_weight_loader(
            self.expert_down_weight,
            _local_expert_down_loader(num_local_experts=self.num_local_experts),
        )
        set_weight_loader(
            self.shared_gate_up_weight,
            _shared_gate_up_loader(
                intermediate_size=config.intermediate_size,
                shard_size=self.shared_intermediate_per_rank,
                tp_size=self.world_size,
            ),
        )
        set_weight_loader(
            self.shared_down_weight,
            _shared_down_loader(
                shard_size=self.shared_intermediate_per_rank,
                tp_size=self.world_size,
            ),
        )
        self.prefill_chunk_size = 128

    def _run_chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, affinities, indices, shared_gammas = inkling_route(
            hidden_states,
            self.router_weight,
            self.correction_bias,
            self.global_scale,
            num_routed_experts=self.config.n_routed_experts,
            num_shared_experts=self.config.n_shared_experts,
            top_k=self.config.num_experts_per_tok,
            route_scale=self.config.route_scale,
        )
        rank_id = torch.tensor(
            [[self.ep_rank]], dtype=torch.int32, device=hidden_states.device
        )
        routed = NF.moe_tkg(
            hidden_input=hidden_states.to(self.dtype),
            expert_gate_up_weights=self.expert_gate_up_weight,
            expert_down_weights=self.expert_down_weight,
            expert_affinities=affinities.to(self.dtype),
            expert_index=indices,
            is_all_expert=True,
            rank_id=rank_id,
            mask_unselected_experts=True,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
            output_dtype=self.dtype,
        )
        shared_gate_up = torch.einsum(
            "th,ehpi->tepi",
            hidden_states.to(self.dtype),
            self.shared_gate_up_weight,
        )
        shared_hidden = F.silu(shared_gate_up[..., 0, :]) * shared_gate_up[..., 1, :]
        shared = torch.einsum("tei,eih->teh", shared_hidden, self.shared_down_weight)
        shared = (shared.float() * shared_gammas[..., None]).sum(dim=1).to(self.dtype)
        output = routed + shared
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] <= self.prefill_chunk_size:
            return self._run_chunk(hidden_states)
        return torch.cat(
            [
                self._run_chunk(chunk)
                for chunk in torch.split(hidden_states, self.prefill_chunk_size, dim=0)
            ],
            dim=0,
        )


class InklingDecoderLayer(nn.Module):
    def __init__(self, config: InklingConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.conv = InklingPagedConv(config, layer_idx)
        self.attention = InklingAttention(config, layer_idx, self.conv)
        self.attention_norm = InklingRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.mlp_norm = InklingRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.mlp: nn.Module = (
            InklingDenseMLP(config)
            if config.layer_is_dense(layer_idx)
            else InklingMoE(config)
        )
        self.attention_sconv_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                1,
                config.sconv_kernel_size,
                dtype=torch.float32,
            )
        )
        self.mlp_sconv_weight = nn.Parameter(
            torch.empty_like(self.attention_sconv_weight)
        )
        self.cache_root: torch.Tensor | None = None
        self.cache_allocation_index: int | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        cache_root: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv_metadata = attn_metadata[self.conv.layer_name]
        residual = hidden_states
        attention_output, cache_root = self.attention(
            self.attention_norm(hidden_states),
            positions,
            attn_metadata,
            cache_root,
        )
        if self.conv.cache_shape is None:
            raise RuntimeError(
                f"cache allocation shape is not bound for {self.conv.layer_name}"
            )
        conv_cache = cache_root.view(self.conv.cache_shape)
        attention_output, conv_cache = self.conv(
            attention_output,
            positions,
            self.attention_sconv_weight,
            InklingPagedConv.ATTN,
            conv_metadata,
            conv_cache,
        )
        hidden_states = residual + attention_output
        residual = hidden_states
        mlp_output = self.mlp(self.mlp_norm(hidden_states))
        mlp_output, conv_cache = self.conv(
            mlp_output,
            positions,
            self.mlp_sconv_weight,
            InklingPagedConv.MLP,
            conv_metadata,
            conv_cache,
        )
        return residual + mlp_output, conv_cache.reshape(-1)


class InklingTextModel(nn.Module):
    def __init__(self, config: InklingConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabDimShardedEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )
        self.embed_norm = InklingRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.layers = nn.ModuleList(
            [
                InklingDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = InklingRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        rank: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(
            input_ids.reshape(-1), scatter_tokens=False, rank=rank
        )
        hidden_states = self.embed_norm(hidden_states)
        cache_states: dict[int, torch.Tensor] = {}
        for layer in self.layers:
            allocation_index = layer.cache_allocation_index
            if allocation_index is None or layer.cache_root is None:
                raise RuntimeError(
                    f"cache allocation is not bound for layer {layer.layer_idx}"
                )
            cache_root = cache_states.get(allocation_index, layer.cache_root)
            hidden_states, cache_root = layer(
                hidden_states,
                positions,
                attn_metadata,
                cache_root,
            )
            cache_states[allocation_index] = cache_root
        return self.norm(hidden_states)


class InklingForConditionalGeneration(nn.Module):
    """Runner-facing text-only view of the multimodal checkpoint."""

    def __init__(self, config: InklingConfig):
        super().__init__()
        self.config = config
        neuron_config = config.neuron_config
        if neuron_config is None:
            raise ValueError("native Inkling requires a NeuronConfig")
        unsupported = {
            "attention_dp_size": neuron_config.attention_dp_size,
            "embedding_dp_size": neuron_config.embedding_dp_size,
            "lm_head_dp_size": neuron_config.lm_head_dp_size,
            "mlp_dp_size": neuron_config.mlp_dp_size,
        }
        invalid = {name: value for name, value in unsupported.items() if value != 1}
        if invalid:
            raise NotImplementedError(
                f"Inkling initial path supports TP32/EP32 only: {invalid}"
            )
        if neuron_config.on_device_sampling_config is not None:
            raise NotImplementedError(
                "Inkling correctness path requires on-device sampling disabled"
            )
        if neuron_config.quantization not in (None, "bf16"):
            raise NotImplementedError(
                "Trainium2 Inkling-Small supports the BF16 checkpoint only"
            )
        self.model = InklingTextModel(config)
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        if self.world_size != 32:
            raise NotImplementedError("Inkling-Small initial path requires TP32")
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.padded_vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=True,
            tp_group=self.tp_group.device_group,
        )
        self._bound_kv_caches: dict[str, list[torch.Tensor]] = {}
        self._bound_kv_cache_roots: dict[str, torch.Tensor] = {}

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
        del (
            inputs_embeds,
            is_token_ids,
            sampling_params,
            spec_decode_metadata,
            logit_mask,
            kwargs,
        )
        if attn_metadata is None:
            raise ValueError("native Inkling requires attention metadata")
        if sampling_positions is None:
            raise ValueError("native Inkling requires sampling_positions")
        hidden_states = self.model(
            input_ids,
            positions.reshape(-1).to(torch.long),
            attn_metadata,
            rank,
        )
        hidden_states = torch.index_select(
            hidden_states,
            0,
            sampling_positions.reshape(-1).to(torch.long),
        )
        if self.config.logits_mup_width_multiplier:
            hidden_states = hidden_states / self.config.logits_mup_width_multiplier
        logits = self.lm_head(hidden_states)
        return logits[..., : self.config.unpadded_vocab_size]

    def get_kv_spec(self) -> KVSpec:
        layers: list[LayerSpec] = []
        for layer_idx in range(self.config.num_hidden_layers):
            attention = self.model.layers[layer_idx].attention
            layers.append(
                LayerSpec(
                    name=attention.layer_name,
                    num_kv_heads=attention.num_kv_heads_per_rank,
                    head_size=attention.head_dim,
                    dtype=self.config.torch_dtype,
                    sliding_window_size=attention.sliding_window,
                    chunk_size=None,
                )
            )
            layers.append(
                LayerSpec(
                    name=self.model.layers[layer_idx].conv.layer_name,
                    num_kv_heads=self.config.conv_cache_heads,
                    head_size=self.config.head_dim,
                    dtype=self.config.torch_dtype,
                    sliding_window_size=self.config.sconv_kernel_size,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        expected = {layer.name for layer in self.get_kv_spec().layers}
        missing = sorted(expected.difference(kv_caches))
        if missing:
            raise ValueError(f"Inkling KV/conv cache missing layers: {missing}")
        self._bound_kv_caches = {name: kv_caches[name] for name in sorted(expected)}
        for layer in self.model.layers:
            layer.conv.cache = kv_caches[layer.conv.layer_name][0]

    def bind_kv_cache_roots(self, cache_roots: dict[str, torch.Tensor]) -> None:
        """Bind each K/V allocation as one page-major persistent alias.

        vLLM owns a raw allocation with enough bytes for K and V and exposes
        it conventionally as ``[2, blocks, heads, block, dim]``.  Inkling
        intentionally reinterprets the same bytes as
        ``[blocks, 2 * heads, block, dim]``.  That page-major view permits a
        single head-axis scatter for K and V, avoiding a Neuron lowering defect
        where only the final stream of a leading-axis scatter persists.
        """

        expected = {
            name
            for layer in self.model.layers
            for name in (
                layer.attention.layer_name,
                layer.conv.layer_name,
            )
        }
        missing = sorted(expected.difference(cache_roots))
        if missing:
            raise ValueError(
                f"Inkling cache allocation views missing layers: {missing}"
            )
        self._bound_kv_cache_roots = {}
        for layer in self.model.layers:
            root = cache_roots[layer.attention.layer_name]
            if root.ndim != 5 or root.shape[0] != 2:
                raise ValueError(
                    f"{layer.attention.layer_name} cache root must have shape "
                    f"[2, blocks, heads, block, dim], got {tuple(root.shape)}"
                )
            packed_root = root.view(
                root.shape[1],
                root.shape[0] * root.shape[2],
                root.shape[3],
                root.shape[4],
            )
            self._bound_kv_cache_roots[layer.attention.layer_name] = packed_root
            layer.attention.kv_cache = packed_root
            layer.attention.kv_cache_shape = tuple(packed_root.shape)

            conv_root = cache_roots[layer.conv.layer_name]
            if conv_root.ndim != 5 or conv_root.shape[0] != 2:
                raise ValueError(
                    f"{layer.conv.layer_name} cache root must have shape "
                    f"[2, blocks, heads, block, dim], got "
                    f"{tuple(conv_root.shape)}"
                )
            packed_conv_root = conv_root.view(
                conv_root.shape[1],
                conv_root.shape[0] * conv_root.shape[2],
                conv_root.shape[3],
                conv_root.shape[4],
            )
            layer.conv.cache_shape = tuple(packed_conv_root.shape)

    def bind_kv_cache_allocation_roots(
        self, allocation_roots: dict[str, torch.Tensor]
    ) -> None:
        """Bind and identify vLLM HMA's shared physical allocations.

        A hybrid cache tensor is shared by one layer from every cache group.
        The runner maps every such layer name to the exact same typed 1-D
        tensor object. The model threads each allocation through all layers
        that touch it, preventing independent output aliases from racing or
        clobbering disjoint cache-block updates.
        """

        expected = {
            name
            for layer in self.model.layers
            for name in (
                layer.attention.layer_name,
                layer.conv.layer_name,
            )
        }
        missing = sorted(expected.difference(allocation_roots))
        if missing:
            raise ValueError(
                f"Inkling cache allocation roots missing layers: {missing}"
            )

        allocation_indices: dict[int, int] = {}
        for layer in self.model.layers:
            attention_root = allocation_roots[layer.attention.layer_name]
            conv_root = allocation_roots[layer.conv.layer_name]
            if attention_root is not conv_root:
                raise ValueError(
                    f"{layer.attention.layer_name} and "
                    f"{layer.conv.layer_name} must share one HMA allocation"
                )
            root_id = id(attention_root)
            allocation_index = allocation_indices.setdefault(
                root_id, len(allocation_indices)
            )
            layer.cache_root = attention_root
            layer.cache_allocation_index = allocation_index

    def _checkpoint_mappings(self) -> dict[str, str]:
        mappings: dict[str, str] = {
            "model.embed_tokens.weight": "model.llm.embed.weight",
            "model.embed_norm.weight": "model.llm.embed_norm.weight",
            "model.norm.weight": "model.llm.norm.weight",
            "lm_head.weight": "model.llm.unembed.weight",
        }
        for layer_idx, layer in enumerate(self.model.layers):
            native = f"model.layers.{layer_idx}"
            source = f"model.llm.layers.{layer_idx}"
            mappings.update(
                {
                    f"{native}.attention_norm.weight": f"{source}.attn_norm.weight",
                    f"{native}.mlp_norm.weight": f"{source}.mlp_norm.weight",
                    f"{native}.attention.q_proj_weight": f"{source}.attn.wq_du.weight",
                    f"{native}.attention.k_proj_weight": f"{source}.attn.wk_dv.weight",
                    f"{native}.attention.v_proj_weight": f"{source}.attn.wv_dv.weight",
                    f"{native}.attention.r_proj_weight": f"{source}.attn.wr_du.weight",
                    f"{native}.attention.o_proj_weight": f"{source}.attn.wo_ud.weight",
                    f"{native}.attention.q_norm.weight": f"{source}.attn.q_norm.weight",
                    f"{native}.attention.k_norm.weight": f"{source}.attn.k_norm.weight",
                    f"{native}.attention.relative_projection": (
                        f"{source}.attn.rel_logits_proj.proj"
                    ),
                    f"{native}.attention.k_sconv_weight": (
                        f"{source}.attn.k_sconv.weight"
                    ),
                    f"{native}.attention.v_sconv_weight": (
                        f"{source}.attn.v_sconv.weight"
                    ),
                    f"{native}.attention_sconv_weight": f"{source}.attn_sconv.weight",
                    f"{native}.mlp_sconv_weight": f"{source}.mlp_sconv.weight",
                }
            )
            if isinstance(layer.mlp, InklingDenseMLP):
                mappings.update(
                    {
                        f"{native}.mlp.gate_up_weight": (f"{source}.mlp.w13_dn.weight"),
                        f"{native}.mlp.down_weight": f"{source}.mlp.w2_md.weight",
                        f"{native}.mlp.global_scale": f"{source}.mlp.global_scale",
                    }
                )
            else:
                mappings.update(
                    {
                        f"{native}.mlp.router_weight": f"{source}.mlp.gate.weight",
                        f"{native}.mlp.correction_bias": f"{source}.mlp.gate.bias",
                        f"{native}.mlp.global_scale": (
                            f"{source}.mlp.gate.global_scale"
                        ),
                        f"{native}.mlp.expert_gate_up_weight": (
                            f"{source}.mlp.experts.w13_weight"
                        ),
                        f"{native}.mlp.expert_down_weight": (
                            f"{source}.mlp.experts.w2_weight"
                        ),
                        f"{native}.mlp.shared_gate_up_weight": (
                            f"{source}.mlp.shared_experts.shared_w13_weight"
                        ),
                        f"{native}.mlp.shared_down_weight": (
                            f"{source}.mlp.shared_experts.shared_w2_weight"
                        ),
                    }
                )
        return mappings

    def load_weights(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None,
    ) -> None:
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
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None,
    ) -> None:
        # BF16 Inkling has no checkpoint-derived scales.  Keep this method so
        # CPU graph extraction does not page 532 GB of tensors into memory.
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        checkpoint._ensure_indexed()
