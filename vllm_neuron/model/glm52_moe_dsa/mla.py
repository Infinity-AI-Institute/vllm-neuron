# SPDX-License-Identifier: Apache-2.0
"""Expanded MLA attention for the GLM-5.2 TP64 correctness path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .attention import (
    apply_glm52_interleaved_rope,
    glm52_paged_sparse_attention,
)
from .cache_layout import Glm52CacheLayout
from .cache_ops import write_paged_cache
from .checkpoint_mapping import (
    _read_scalar,
    _to_neuron_legacy_fp8,
)
from .config import Glm52MoeDsaConfig
from .indexer import (
    Glm52FullIndexer,
    Glm52IndexShareState,
    advance_index_share_state,
)
from .sparse_mlp import glm52_rms_norm
from .static_fp8 import (
    OCP_E4M3FN_QMAX448,
    static_fp8_scale_multiplier,
)

if TYPE_CHECKING:
    from safetensors import PySafeSlice

_SCALE_ROWS = 128


def _projection_weight_loader(
    *,
    shard_dim: int,
    shard_size: int,
    num_shards: int,
    static_fp8: bool,
    weight_format: str = OCP_E4M3FN_QMAX448,
) -> SafetensorsWeightLoader:
    """Load checkpoint ``[out,in]`` storage into kernel ``[in,out]``."""
    base = sharding_weight_loader(
        shard_dim=shard_dim,
        shard_size=shard_size,
        num_shards=num_shards,
        is_storage_transposed=True,
    )
    if not static_fp8:
        return base

    def transform(
        slices: list["PySafeSlice"],
        rank: int,
    ) -> torch.Tensor:
        return _to_neuron_legacy_fp8(
            base.load(slices, rank),
            weight_format,
        )

    return SafetensorsWeightLoader(transform=transform)


def _scalar_scale_loader(
    *,
    compensate_weight_range: bool,
    columns: int = 1,
    weight_format: str = OCP_E4M3FN_QMAX448,
) -> SafetensorsWeightLoader:
    if columns <= 0:
        raise ValueError("scale columns must be positive")

    def transform(
        slices: list["PySafeSlice"],
        rank: int,
    ) -> torch.Tensor:
        del rank
        if len(slices) != 1:
            raise ValueError(f"expected one scalar scale, got {len(slices)}")
        scalar = _read_scalar(slices[0])
        if compensate_weight_range:
            scalar = scalar * static_fp8_scale_multiplier(weight_format)
        return scalar.reshape(1, 1).expand(_SCALE_ROWS, columns).contiguous()

    return SafetensorsWeightLoader(transform=transform)


class _ColumnProjection(nn.Module):
    """Column-style projection backed by the static-FP8 QKV kernel."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        shard_output: bool,
        world_size: int,
        static_fp8: bool,
        weight_format: str = OCP_E4M3FN_QMAX448,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.static_fp8 = static_fp8
        self.synthetic_d_head = None
        self.synthetic_q_heads = None
        self.synthetic_kv_heads = None
        if static_fp8:
            for candidate in (128, 64, 32, 16):
                if output_size % candidate:
                    continue
                total_heads = output_size // candidate
                if total_heads >= 3:
                    self.synthetic_d_head = candidate
                    self.synthetic_q_heads = total_heads - 2
                    self.synthetic_kv_heads = 1
                    break
            if self.synthetic_d_head is None:
                raise ValueError(
                    "static-FP8 projection output cannot be represented by "
                    "the QKV kernel's synthetic head contract"
                )
        weight_dtype = torch.float8_e4m3fn if static_fp8 else dtype
        self.weight = nn.Parameter(
            torch.empty(
                input_size,
                output_size,
                dtype=weight_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        set_weight_loader(
            self.weight,
            _projection_weight_loader(
                shard_dim=1,
                shard_size=output_size,
                num_shards=world_size if shard_output else 1,
                static_fp8=static_fp8,
                weight_format=weight_format,
            ),
        )
        if static_fp8:
            self.weight_scale = nn.Parameter(
                torch.empty(
                    _SCALE_ROWS,
                    3,
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )
            self.input_scale = nn.Parameter(
                torch.empty(
                    _SCALE_ROWS,
                    1,
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )
            set_weight_loader(
                self.weight_scale,
                _scalar_scale_loader(
                    compensate_weight_range=True,
                    columns=3,
                    weight_format=weight_format,
                ),
            )
            set_weight_loader(
                self.input_scale,
                _scalar_scale_loader(compensate_weight_range=False),
            )
        else:
            self.register_parameter("weight_scale", None)
            self.register_parameter("input_scale", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.static_fp8:
            return hidden_states @ self.weight

        from nkilib.core.utils.common_types import QuantizationType

        from vllm_neuron.functional.attention.qkv import qkv_proj

        leading_shape = hidden_states.shape[:-1]
        projected = qkv_proj(
            hidden=hidden_states.reshape(1, -1, hidden_states.shape[-1]),
            qkv_weights=self.weight,
            d_head=self.synthetic_d_head,
            num_q_heads=self.synthetic_q_heads,
            num_kv_heads=self.synthetic_kv_heads,
            quantization_type=QuantizationType.STATIC,
            qkv_w_scale=self.weight_scale,
            qkv_in_scale=self.input_scale,
        )
        return projected.reshape(*leading_shape, self.weight.shape[1])


class _OutputProjection(nn.Module):
    """Row-style output shard followed by the caller's TP all-reduce."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        world_size: int,
        static_fp8: bool,
        weight_format: str = OCP_E4M3FN_QMAX448,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.static_fp8 = static_fp8
        weight_dtype = torch.float8_e4m3fn if static_fp8 else dtype
        self.weight = nn.Parameter(
            torch.empty(
                input_size,
                output_size,
                dtype=weight_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        set_weight_loader(
            self.weight,
            _projection_weight_loader(
                shard_dim=0,
                shard_size=input_size,
                num_shards=world_size,
                static_fp8=static_fp8,
                weight_format=weight_format,
            ),
        )
        if static_fp8:
            self.weight_scale = nn.Parameter(
                torch.empty(
                    _SCALE_ROWS,
                    1,
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )
            self.input_scale = nn.Parameter(
                torch.empty(
                    _SCALE_ROWS,
                    1,
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )
            set_weight_loader(
                self.weight_scale,
                _scalar_scale_loader(
                    compensate_weight_range=True,
                    weight_format=weight_format,
                ),
            )
            set_weight_loader(
                self.input_scale,
                _scalar_scale_loader(compensate_weight_range=False),
            )
        else:
            self.register_parameter("weight_scale", None)
            self.register_parameter("input_scale", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.static_fp8:
            return hidden_states @ self.weight

        from nkilib.core.utils.common_types import QuantizationType

        from vllm_neuron.functional.attention.o_proj import o_proj

        leading_shape = hidden_states.shape[:-1]
        projected = o_proj(
            hidden_states.reshape(1, -1, hidden_states.shape[-1]),
            self.weight,
            quantization_type=QuantizationType.STATIC,
            weight_scales=self.weight_scale,
            input_scales=self.input_scale,
        )
        return projected.reshape(*leading_shape, self.weight.shape[1])


@dataclass(frozen=True)
class Glm52MlaProjection:
    q_resid: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


class Glm52MlaAttention(nn.Module):
    """One-head-per-rank expanded MLA/DSA attention for TP64."""

    def __init__(
        self,
        config: Glm52MoeDsaConfig,
        *,
        layer_idx: int,
        cache_layout: Glm52CacheLayout,
        world_size: int = 64,
        tp_group=None,
        static_fp8: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= layer_idx < config.num_hidden_layers:
            raise IndexError("attention layer index is outside the backbone")
        if config.num_attention_heads % world_size:
            raise ValueError("attention heads must divide over TP ranks")
        local_heads = config.num_attention_heads // world_size
        if local_heads != 1:
            raise ValueError(
                "the expanded GLM attention baseline requires exactly one "
                "attention head per rank"
            )
        if tp_group is not None and tp_group.world_size != world_size:
            raise ValueError("TP process group size must match world_size")

        self.config = config
        self.layer_idx = layer_idx
        self.cache_name = f"layers.{layer_idx}.self_attn"
        self.world_size = world_size
        self.local_heads = local_heads
        self.tp_group = tp_group
        self.dtype = dtype or config.torch_dtype
        self.scaling = config.qk_head_dim**-0.5

        self.q_a_proj = _ColumnProjection(
            config.hidden_size,
            config.q_lora_rank,
            shard_output=False,
            world_size=world_size,
            static_fp8=static_fp8,
            weight_format=config.static_fp8_weight_format,
            dtype=self.dtype,
            device=device,
        )
        self.q_a_layernorm = nn.Parameter(
            torch.empty(config.q_lora_rank, dtype=self.dtype, device=device),
            requires_grad=False,
        )
        self.q_b_proj = _ColumnProjection(
            config.q_lora_rank,
            local_heads * config.qk_head_dim,
            shard_output=True,
            world_size=world_size,
            static_fp8=static_fp8,
            weight_format=config.static_fp8_weight_format,
            dtype=self.dtype,
            device=device,
        )
        self.kv_a_proj_with_mqa = _ColumnProjection(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            shard_output=False,
            world_size=world_size,
            static_fp8=static_fp8,
            weight_format=config.static_fp8_weight_format,
            dtype=self.dtype,
            device=device,
        )
        self.kv_a_layernorm = nn.Parameter(
            torch.empty(config.kv_lora_rank, dtype=self.dtype, device=device),
            requires_grad=False,
        )
        self.kv_b_proj = _ColumnProjection(
            config.kv_lora_rank,
            local_heads * (config.qk_nope_head_dim + config.v_head_dim),
            shard_output=True,
            world_size=world_size,
            static_fp8=static_fp8,
            weight_format=config.static_fp8_weight_format,
            dtype=self.dtype,
            device=device,
        )
        self.o_proj = _OutputProjection(
            local_heads * config.v_head_dim,
            config.hidden_size,
            world_size=world_size,
            static_fp8=static_fp8,
            weight_format=config.static_fp8_weight_format,
            dtype=self.dtype,
            device=device,
        )

        self.indexer = (
            Glm52FullIndexer(
                config,
                layer_idx=layer_idx,
                cache_binding=cache_layout.indexer_binding(layer_idx),
                dtype=self.dtype,
                topk_backend="neuron" if static_fp8 else "torch",
                device=device,
            )
            if config.indexer_types[layer_idx] == "full"
            else None
        )
        self.register_buffer(
            "key_cache_quant_multiplier",
            torch.ones(1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "value_cache_quant_multiplier",
            torch.ones(1, dtype=torch.float32),
            persistent=False,
        )

    def set_cache_quant_multipliers(
        self,
        *,
        key: float,
        value: float,
    ) -> None:
        if key <= 0 or value <= 0:
            raise ValueError("cache quantization multipliers must be positive")
        self.key_cache_quant_multiplier.fill_(key)
        self.value_cache_quant_multiplier.fill_(value)

    def project(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Glm52MlaProjection:
        if hidden_states.ndim != 2:
            raise ValueError("MLA projection expects flattened token inputs")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden_states has an incorrect hidden dimension")

        q_resid = glm52_rms_norm(
            self.q_a_proj(hidden_states.to(self.dtype)),
            self.q_a_layernorm,
            eps=self.config.rms_norm_eps,
        )
        # Keep NKI projection outputs flat while selecting logical fields.
        # ``torch.split`` views taken after reshaping these custom-kernel
        # outputs preserve the logical shape but can bind the wrong physical
        # columns in a combined Neuron graph.
        query_flat = self.q_b_proj(q_resid)
        q_pass = query_flat[:, : self.config.qk_nope_head_dim].reshape(
            hidden_states.shape[0],
            self.local_heads,
            self.config.qk_nope_head_dim,
        )
        q_rot = query_flat[:, self.config.qk_nope_head_dim :].reshape(
            hidden_states.shape[0],
            self.local_heads,
            self.config.qk_rope_head_dim,
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states.to(self.dtype))
        kv_pass = compressed_kv[:, : self.config.kv_lora_rank]
        k_rot = compressed_kv[:, self.config.kv_lora_rank :]
        k_pass = glm52_rms_norm(
            kv_pass,
            self.kv_a_layernorm,
            eps=self.config.rms_norm_eps,
        )
        expanded_flat = self.kv_b_proj(k_pass)
        k_nope = expanded_flat[:, : self.config.qk_nope_head_dim].reshape(
            hidden_states.shape[0],
            self.local_heads,
            self.config.qk_nope_head_dim,
        )
        value = expanded_flat[:, self.config.qk_nope_head_dim :].reshape(
            hidden_states.shape[0],
            self.local_heads,
            self.config.v_head_dim,
        )

        q_rot, k_rot = apply_glm52_interleaved_rope(
            q_rot,
            k_rot.unsqueeze(-2),
            cos,
            sin,
            unsqueeze_dim=-2,
        )
        key = torch.cat(
            (k_nope, k_rot.expand(-1, self.local_heads, -1)),
            dim=-1,
        )
        query = torch.cat((q_pass, q_rot), dim=-1)
        return Glm52MlaProjection(
            q_resid=q_resid,
            query=query,
            key=key,
            value=value,
        )

    def forward_paged_decode(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attn_metadata: dict[str, dict[str, torch.Tensor | int]],
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        previous_index_state: Glm52IndexShareState | None,
        indexer_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Glm52IndexShareState]:
        """Project, update caches, run DSA, and propagate IndexShare state."""
        cos, sin = position_embeddings
        projection = self.project(hidden_states, cos, sin)

        if self.indexer is not None:
            if indexer_cache is None:
                raise RuntimeError("full indexer layer requires its paged key cache")
            computed_topk = self.indexer.forward_paged(
                hidden_states,
                projection.q_resid,
                cos,
                sin,
                position_ids=position_ids.reshape(-1),
                attn_metadata=attn_metadata,
                key_cache=indexer_cache,
            )
        else:
            computed_topk = None
        index_state = advance_index_share_state(
            self.config,
            layer_idx=self.layer_idx,
            previous=previous_index_state,
            computed_topk=computed_topk,
        )

        metadata = attn_metadata[self.cache_name]
        slot_mapping = metadata["slot_mapping"]
        block_size = metadata["block_size"]
        block_table = metadata["block_table_tensor"]
        if not isinstance(slot_mapping, torch.Tensor):
            raise TypeError("slot_mapping metadata must be a tensor")
        if not isinstance(block_table, torch.Tensor):
            raise TypeError("block_table_tensor metadata must be a tensor")
        if not isinstance(block_size, int):
            raise TypeError("block_size metadata must be an integer")

        key_multiplier = (
            self.key_cache_quant_multiplier
            if key_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            else None
        )
        value_multiplier = (
            self.value_cache_quant_multiplier
            if value_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            else None
        )
        write_paged_cache(
            key_cache,
            projection.key,
            slot_mapping,
            block_size=block_size,
            quant_multiplier=key_multiplier,
        )
        write_paged_cache(
            value_cache,
            projection.value,
            slot_mapping,
            block_size=block_size,
            quant_multiplier=value_multiplier,
        )

        batch_size = block_table.shape[0]
        tokens = hidden_states.shape[0]
        if tokens % batch_size:
            raise ValueError("flattened token count must divide over requests")
        query_tokens = tokens // batch_size
        query = projection.query.reshape(
            batch_size,
            query_tokens,
            self.local_heads,
            self.config.qk_head_dim,
        )
        topk_indices = index_state.topk_indices.reshape(
            batch_size,
            query_tokens,
            -1,
        )
        output = glm52_paged_sparse_attention(
            query,
            key_cache,
            value_cache,
            topk_indices,
            block_table,
            block_size=block_size,
            position_ids=position_ids.reshape(batch_size, query_tokens),
            scaling=self.scaling,
            key_quant_multiplier=key_multiplier,
            value_quant_multiplier=value_multiplier,
        )
        output = self.o_proj(
            output.reshape(tokens, self.local_heads * self.config.v_head_dim)
        )
        if self.tp_group is not None and self.tp_group.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output, index_state

    def forward_paged_prefill(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attn_metadata: dict[str, dict[str, torch.Tensor | int]],
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        previous_index_state: Glm52IndexShareState | None,
        indexer_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Glm52IndexShareState]:
        """Exact bounded prefill over paged DSA caches.

        The initial serving gate accepts only contexts no longer than
        ``index_topk``. In that envelope the selected set contains every
        causally visible token, so sparse attention is mathematically equal
        to dense causal attention without materializing dense K/V history.
        """
        metadata = attn_metadata[self.cache_name]
        max_query_len = metadata.get("max_query_len")
        if not isinstance(max_query_len, int):
            raise TypeError("prefill max_query_len must be an integer")
        if max_query_len > self.config.index_topk:
            raise NotImplementedError(
                "GLM-5.2 exact prefill is limited to index_topk tokens"
            )
        if metadata.get("kv_segment_size"):
            raise NotImplementedError(
                "GLM-5.2 segmented/chunked prefill is not integrated"
            )

        full_hidden = hidden_states
        if self.tp_group is not None and self.tp_group.world_size > 1:
            full_hidden = self.tp_group.all_gather(hidden_states, dim=0)
        if full_hidden.shape[0] != position_ids.numel():
            raise ValueError(
                "sequence-parallel attention gather must reconstruct all "
                "prefill positions"
            )

        cos, sin = position_embeddings
        projection = self.project(full_hidden, cos, sin)
        if self.indexer is not None:
            if indexer_cache is None:
                raise RuntimeError("full indexer layer requires its paged key cache")
            computed_topk = self.indexer.forward_paged(
                full_hidden,
                projection.q_resid,
                cos,
                sin,
                position_ids=position_ids.reshape(-1),
                attn_metadata=attn_metadata,
                key_cache=indexer_cache,
            )
        else:
            computed_topk = None
        index_state = advance_index_share_state(
            self.config,
            layer_idx=self.layer_idx,
            previous=previous_index_state,
            computed_topk=computed_topk,
        )

        slot_mapping = metadata["slot_mapping"]
        block_size = metadata["block_size"]
        block_table = metadata["block_table_tensor"]
        if not isinstance(slot_mapping, torch.Tensor):
            raise TypeError("slot_mapping metadata must be a tensor")
        if not isinstance(block_table, torch.Tensor):
            raise TypeError("block_table_tensor metadata must be a tensor")
        if not isinstance(block_size, int):
            raise TypeError("block_size metadata must be an integer")

        key_multiplier = (
            self.key_cache_quant_multiplier
            if key_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            else None
        )
        value_multiplier = (
            self.value_cache_quant_multiplier
            if value_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            else None
        )
        write_paged_cache(
            key_cache,
            projection.key,
            slot_mapping,
            block_size=block_size,
            quant_multiplier=key_multiplier,
        )
        write_paged_cache(
            value_cache,
            projection.value,
            slot_mapping,
            block_size=block_size,
            quant_multiplier=value_multiplier,
        )

        batch_size = block_table.shape[0]
        tokens = full_hidden.shape[0]
        if tokens % batch_size:
            raise ValueError("flattened token count must divide over requests")
        query_tokens = tokens // batch_size
        positions = position_ids.reshape(batch_size, query_tokens)
        query = projection.query.reshape(
            batch_size,
            query_tokens,
            self.local_heads,
            self.config.qk_head_dim,
        )
        topk_indices = index_state.topk_indices.reshape(
            batch_size,
            query_tokens,
            -1,
        )
        key_lengths = positions.amax(dim=1).to(torch.int64) + 1
        output = glm52_paged_sparse_attention(
            query,
            key_cache,
            value_cache,
            topk_indices,
            block_table,
            block_size=block_size,
            position_ids=positions,
            scaling=self.scaling,
            key_lengths=key_lengths,
            key_quant_multiplier=key_multiplier,
            value_quant_multiplier=value_multiplier,
        )
        output = self.o_proj(
            output.reshape(tokens, self.local_heads * self.config.v_head_dim)
        )
        if self.tp_group is not None and self.tp_group.world_size > 1:
            output = self.tp_group.reduce_scatter(output, dim=0)
        return output, index_state
