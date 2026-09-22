# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 (text) for vllm-neuron
==============================

<-- MODEL-SPECIFIC: a hybrid of 24 Gated DeltaNet layers and 8 gated,
partial-rotary full-attention layers (every 4th). See ``config.py``.

What is verified where
----------------------
All the arithmetic this file performs is delegated to CPU-tested modules:

- ``gated_deltanet.py`` / ``layer_step.py``: the Gated DeltaNet mixer and its
  flat-token/state plumbing, tested against an independent float64 reference
  and transformers' own layer.
- ``attention_math.py``: zero-centred norm, partial rotary, gated output,
  tested against transformers' ``Qwen3_5Attention``.
- ``runner_state.py``: the runner contract, tested on vllm-neuron's exact
  padded layout.

What is NOT verified until it runs on a Trainium2: everything here that
touches vllm-neuron infrastructure (TP collectives, sequence-parallel
all-gather/reduce-scatter, VocabDimShardedEmbedding, ColumnParallelLinear,
the NF attention kernels, the paged KV layout) and ``torch.compile`` through
the ``vllm_neuron`` backend. Those seams mirror ``qwen3/model.py`` exactly
wherever the semantics are the same.

Two deliberate differences from Qwen3
-------------------------------------
- The prefill/decode switch reads the first *full-attention* layer's
  metadata. Qwen3 reads ``layers.0``, but here layer 0 is Gated DeltaNet and
  has no KV spec, so no metadata entry.
- Decode attention is unfused torch over the paged cache. Qwen3's fused
  ``NF.attention_decode`` folds ``o_proj`` into the kernel, and Qwen3.5 must
  multiply by ``sigmoid(gate)`` *before* ``o_proj``. First performance target.
"""

import logging
import os

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
    with_rank_override,
)

from ..qwen3.model import Qwen3MLP
from . import attention_math as am
from .config import Qwen3_5Config
from .layer_step import gdn_layer_step
from .runner_state import GdnStepInputs
from .state import RecurrentStateBuffers

logger = logging.getLogger(__name__)

# Recurrent state slots, one per concurrent request. ~50 MiB per slot for the
# 9B at TP=1 (~12.5 MiB per rank at TP=4). Defaults to max_num_seqs.
STATE_SLOTS_ENV = "VLLM_NEURON_QWEN35_STATE_SLOTS"

# Checkpoint tensors kept in their stored precision. A_log is published as
# F32; casting it to BF16 perturbs every decay rate exp(A_log).
KEEP_STORED_DTYPE = ("A_log",)


# =============================================================================
# Section 1: Norms
# <-- MODEL-SPECIFIC: zero-centred, x * (1 + w). Not Qwen3's x * w.
# =============================================================================


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return am.zero_centred_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


# =============================================================================
# Section 2: Gated, partial-rotary full attention
# Mixed: PARALLELISM (TP head sharding, SP) + MODEL-SPECIFIC (gate, partial RoPE)
# =============================================================================


def _decode_meta(attn_metadata, meta_layer: str) -> bool:
    meta = attn_metadata[meta_layer]
    return meta["max_query_len"] <= meta["decode_token_threshold"]


class Qwen3_5Attention(nn.Module):
    """>>> PARALLELISM: TP <<<  Q heads and their gates sharded together;
    KV heads sharded, or replicated when TP exceeds the KV head count.

    <-- MODEL-SPECIFIC: q_proj emits [query | gate] per head; zero-centred
    per-head q/k norm before RoPE; RoPE on the first 64 of 256 channels;
    output * sigmoid(gate) before o_proj.
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.self_attn"
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.scaling = config.head_dim**-0.5
        self.rotary_dim = config.rotary_dim

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.num_heads_per_rank = config.num_attention_heads // self.world_size
        if self.world_size >= config.num_key_value_heads:
            self.num_kv_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // config.num_key_value_heads
        else:
            self.num_kv_heads_per_rank = config.num_key_value_heads // self.world_size
            self.num_kv_replicas = 1
        self.num_kv_groups = self.num_heads_per_rank // self.num_kv_heads_per_rank

        q_out = self.num_heads_per_rank * 2 * self.head_dim  # [query | gate] per head
        kv_out = self.num_kv_heads_per_rank * self.head_dim
        # Stored [in, out] like qwen3, so projections are `hidden @ weight`.
        self.q_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_out, dtype=self.dtype)
        )
        self.k_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, kv_out, dtype=self.dtype)
        )
        self.v_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, kv_out, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(
                self.num_heads_per_rank * self.head_dim,
                self.hidden_size,
                dtype=self.dtype,
            )
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

        self.k_cache = None
        self.v_cache = None
        self._setup_weight_loaders(config, q_out, kv_out)

    def _setup_weight_loaders(self, config, q_out, kv_out):
        # Each head's rows are [query | gate] (2 * head_dim, contiguous), so a
        # contiguous shard of whole heads keeps every query with its gate.
        set_weight_loader(
            self.q_proj_weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=q_out,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        kv_shards = config.num_key_value_heads // self.num_kv_heads_per_rank
        for weight in (self.k_proj_weight, self.v_proj_weight):
            set_weight_loader(
                weight,
                with_rank_override(
                    sharding_weight_loader(
                        shard_dim=1,
                        shard_size=kv_out,
                        num_shards=kv_shards,
                        is_storage_transposed=True,
                    ),
                    self.rank // self.num_kv_replicas,
                ),
            )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.num_heads_per_rank * self.head_dim,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def _project(self, hidden_states, cos, sin):
        tokens = hidden_states.shape[0]
        query, gate = am.split_query_gate(
            hidden_states @ self.q_proj_weight, self.num_heads_per_rank, self.head_dim
        )
        key = (hidden_states @ self.k_proj_weight).view(
            tokens, self.num_kv_heads_per_rank, self.head_dim
        )
        value = (hidden_states @ self.v_proj_weight).view(
            tokens, self.num_kv_heads_per_rank, self.head_dim
        )
        query = self.q_norm(query).transpose(0, 1)  # [Hq, T, Dh]
        key = self.k_norm(key).transpose(0, 1)  # [Hkv, T, Dh]
        value = value.transpose(0, 1)  # [Hkv, T, Dh]
        query = am.apply_partial_rotary(query, cos, sin)
        key = am.apply_partial_rotary(key, cos, sin)
        return query, key, value, gate

    def _write_cache(self, key, value, attn_metadata):
        if self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise NotImplementedError("FP8 KV cache is not wired for Qwen3.5 yet")
        meta = attn_metadata[self.layer_name]
        slot_mapping, block_size = meta["slot_mapping"], meta["block_size"]
        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        heads = self.num_kv_heads_per_rank
        head_indices = torch.arange(
            heads, dtype=torch.long, device=key.device
        ).repeat_interleave(slot_mapping.shape[0])
        index = (
            block_indices.repeat(heads),
            head_indices,
            position_indices.repeat(heads),
        )
        self.k_cache.index_put_(
            index, key.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        )
        self.v_cache.index_put_(
            index, value.reshape(-1, self.head_dim).to(self.v_cache.dtype)
        )

    def forward(self, hidden_states, positions, cos, sin, attn_metadata, is_decode):
        if not is_decode and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        hidden_states = hidden_states.to(self.dtype)
        query, key, value, gate = self._project(hidden_states, cos, sin)
        self._write_cache(key, value, attn_metadata)
        if is_decode:
            attn = self._decode_attention(query, positions, attn_metadata)
        else:
            attn = self._prefill_attention(query, key, value, attn_metadata)
        attn = am.gate_output(attn, gate)  # before o_proj
        out = NF.o_proj(attn.unsqueeze(0), self.o_proj_weight, None).squeeze(0)
        if self.world_size > 1:
            if is_decode:
                self.tp_group.all_reduce(out)
            else:
                out = self.tp_group.reduce_scatter(out, dim=0)
        return out.contiguous()

    def _prefill_attention(self, query, key, value, attn_metadata):
        """Mirrors qwen3's prefill kernels. Returns ``[T, Hq * Dh]``."""
        meta = attn_metadata[self.layer_name]
        if meta.get("kv_segment_size"):
            attn = NF.segmented_attention(
                query,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=meta["block_table_tensor"],
                prior_tokens=meta.get("cached_seq_len"),
                block_size=meta["block_size"],
                kv_segment_size=meta["kv_segment_size"],
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
            )
        else:
            key = key.repeat_interleave(self.num_kv_groups, dim=0)
            value = value.repeat_interleave(self.num_kv_groups, dim=0)
            attn = NF.flash_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value,
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )
        # TODO(trn2): confirm tp_out=True yields [T, Hq * Dh] as qwen3's
        # o_proj call implies; the gate multiplies that layout.
        return attn

    def _decode_attention(self, query, positions, attn_metadata):
        """Unfused paged attention; the new token's K/V is already written.

        ``query`` is ``[Hq, T, Dh]`` with ``T = B * S`` (S = 1 without
        speculation). Returns ``[T, Hq * Dh]``.
        """
        meta = attn_metadata[self.layer_name]
        block_table = meta["block_table_tensor"]  # [B, max_blocks]
        block_size = meta["block_size"]
        batch = block_table.shape[0]
        tokens = query.shape[1]
        steps = tokens // batch
        # Paged cache is [num_blocks, Hkv, block_size, Dh].
        keys = (
            self.k_cache[block_table].permute(0, 2, 1, 3, 4).flatten(2, 3)
        )  # [B, Hkv, S_ctx, Dh]
        values = self.v_cache[block_table].permute(0, 2, 1, 3, 4).flatten(2, 3)
        keys = keys.repeat_interleave(self.num_kv_groups, dim=1).to(query.dtype)
        values = values.repeat_interleave(self.num_kv_groups, dim=1).to(query.dtype)
        q = query.view(self.num_heads_per_rank, batch, steps, self.head_dim).permute(
            1, 0, 2, 3
        )
        scores = (
            q @ keys.transpose(-1, -2)
        ).float() * self.scaling  # [B, Hq, S, S_ctx]
        context = torch.arange(keys.shape[2], device=query.device)
        current = positions.view(batch, 1, steps, 1)
        scores = scores.masked_fill(context.view(1, 1, 1, -1) > current, float("-inf"))
        probs = scores.softmax(-1).to(values.dtype)
        out = (probs @ values).permute(0, 2, 1, 3)  # [B, S, Hq, Dh]
        return out.reshape(tokens, self.num_heads_per_rank * self.head_dim)


# =============================================================================
# Section 3: Gated DeltaNet
# Mixed: PARALLELISM (head sharding, SP) + MODEL-SPECIFIC (delta rule, conv)
# =============================================================================


def _segment_loader(segments, rank, squeeze_dim=None):
    """Take this rank's rows from each ``(start, per_rank)`` segment of one tensor.

    The fused ``in_proj_qkv`` and ``conv1d`` weights stack [q | k | v] along
    dim 0, so a contiguous shard would split q/k/v unevenly. Rank ``r`` takes
    rows ``start + r*per_rank .. start + (r+1)*per_rank`` from every segment.
    """

    def transform(slices, _rank):
        source = slices[0]
        parts = [
            source[start + rank * size : start + (rank + 1) * size]
            for start, size in segments
        ]
        tensor = torch.cat(parts, dim=0)
        return tensor.squeeze(squeeze_dim) if squeeze_dim is not None else tensor

    return SafetensorsWeightLoader(transform=transform)


class Qwen3_5GatedDeltaNet(nn.Module):
    """>>> PARALLELISM: TP by head <<<  Value head ``j`` reads key head
    ``j // (V/K)``, so giving rank ``r`` value heads ``[r*vr, (r+1)*vr)`` needs
    exactly key heads ``[r*kr, (r+1)*kr)``: self-contained per rank.

    Weights stay in transformers' ``[out, in]`` orientation because the tested
    mixer uses ``F.linear``.
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        k_heads, v_heads = config.linear_num_key_heads, config.linear_num_value_heads
        if k_heads % self.world_size or v_heads % self.world_size:
            raise ValueError(
                f"TP={self.world_size} must divide {k_heads} key and {v_heads} value heads"
            )
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.num_k_heads = k_heads // self.world_size
        self.num_v_heads = v_heads // self.world_size
        self.eps = config.rms_norm_eps

        key_r = self.num_k_heads * self.head_k_dim
        value_r = self.num_v_heads * self.head_v_dim
        conv_r = 2 * key_r + value_r
        dtype, hidden = config.torch_dtype, config.hidden_size
        self.in_proj_qkv = nn.Parameter(torch.empty(conv_r, hidden, dtype=dtype))
        self.in_proj_z = nn.Parameter(torch.empty(value_r, hidden, dtype=dtype))
        self.in_proj_b = nn.Parameter(
            torch.empty(self.num_v_heads, hidden, dtype=dtype)
        )
        self.in_proj_a = nn.Parameter(
            torch.empty(self.num_v_heads, hidden, dtype=dtype)
        )
        self.conv1d = nn.Parameter(
            torch.empty(conv_r, config.linear_conv_kernel_dim, dtype=dtype)
        )
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads, dtype=dtype))
        self.norm = nn.Parameter(torch.ones(self.head_v_dim, dtype=dtype))
        self.out_proj = nn.Parameter(torch.empty(hidden, value_r, dtype=dtype))

        full_key = k_heads * self.head_k_dim
        segments = [(0, key_r), (full_key, key_r), (2 * full_key, value_r)]
        set_weight_loader(self.in_proj_qkv, _segment_loader(segments, self.rank))
        set_weight_loader(
            self.conv1d, _segment_loader(segments, self.rank, squeeze_dim=1)
        )
        for weight, size in (
            (self.in_proj_z, value_r),
            (self.in_proj_b, self.num_v_heads),
            (self.in_proj_a, self.num_v_heads),
            (self.A_log, self.num_v_heads),
            (self.dt_bias, self.num_v_heads),
        ):
            set_weight_loader(
                weight,
                sharding_weight_loader(
                    shard_dim=0, shard_size=size, num_shards=self.world_size
                ),
            )
        set_weight_loader(
            self.out_proj,
            sharding_weight_loader(
                shard_dim=1, shard_size=value_r, num_shards=self.world_size
            ),
        )

    def weights(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name)
            for name in (
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "conv1d",
                "A_log",
                "dt_bias",
                "norm",
                "out_proj",
            )
        }

    def forward(
        self,
        hidden_states,
        step: GdnStepInputs,
        buffers: RecurrentStateBuffers,
        is_decode,
    ):
        if not is_decode and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        out = gdn_layer_step(
            hidden_states,
            self.weights(),
            buffers,
            self.layer_idx,
            step,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            eps=self.eps,
            decode=is_decode,
        )
        if self.world_size > 1:
            if is_decode:
                self.tp_group.all_reduce(out)
            else:
                out = self.tp_group.reduce_scatter(out, dim=0)
        return out


# =============================================================================
# Section 4: Decoder layer
# =============================================================================


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)

    def forward(
        self,
        hidden_states,
        positions,
        cos,
        sin,
        attn_metadata,
        is_decode,
        step,
        buffers,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, step, buffers, is_decode)
        else:
            hidden_states = self.self_attn(
                hidden_states, positions, cos, sin, attn_metadata, is_decode
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states, is_prefill=not is_decode)


# =============================================================================
# Section 5: Backbone
# =============================================================================


class Qwen3_5Model(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        # Metadata exists only for full-attention layers; layer 0 is Gated DeltaNet.
        self.meta_layer = f"layers.{config.full_attention_layers[0]}.self_attn"
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )

    def forward(self, input_ids, positions, attn_metadata, step, buffers, rank=None):
        is_decode = _decode_meta(attn_metadata, self.meta_layer)
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=not is_decode, rank=rank
        )
        cos, sin = am.rotary_cos_sin(
            positions,
            self.config.rotary_dim,
            self.config.rope_theta,
            hidden_states.dtype,
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions,
                cos,
                sin,
                attn_metadata,
                is_decode,
                step,
                buffers,
            )
        hidden_states = self.norm(hidden_states)
        if not is_decode and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states


# =============================================================================
# Section 6: Causal LM, KV spec, recurrent state, weight loading
# =============================================================================


class Qwen3_5ForCausalLM(nn.Module):
    """Text-only Qwen3.5. Serve with ``--no-enable-prefix-caching`` and with
    image/video disabled (``--limit-mm-per-prompt '{"image":0,"video":0}'``)."""

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        if config.tie_word_embeddings:
            raise NotImplementedError(
                "Qwen3.5-9B ships an untied lm_head; tied is not wired"
            )

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config, process_group=self.tp_group.device_group
            )
        self.gdn_state: RecurrentStateBuffers | None = None

    @classmethod
    def from_configs(cls, hf_config, neuron_config):
        return cls(Qwen3_5Config.from_configs(hf_config, neuron_config))

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        inputs_embeds=None,
        is_token_ids=None,
        attn_metadata=None,
        sampling_positions=None,
        sampling_params=None,
        spec_decode_metadata=None,
        logit_mask=None,
        rank=None,
        gdn_request_of_token=None,
        gdn_lengths=None,
        gdn_state_slots=None,
        gdn_state_reset=None,
        gdn_max_len: int = 1,
        **kwargs,
    ):
        if inputs_embeds is not None and is_token_ids is not None:
            raise NotImplementedError(
                "multimodal embeddings are not served by this text model"
            )
        if self.gdn_state is None:
            raise RuntimeError(
                "bind_kv_cache() must run before forward: it allocates GDN state"
            )
        step = GdnStepInputs(
            request_of_token=gdn_request_of_token,
            lengths=gdn_lengths,
            state_slots=gdn_state_slots,
            state_reset=gdn_state_reset,
            max_len=gdn_max_len,
        )
        hidden_states = self.model(
            input_ids,
            positions.to(torch.int32),
            attn_metadata,
            step,
            self.gdn_state,
            rank=rank,
        )
        logits = self.lm_head(torch.index_select(hidden_states, 0, sampling_positions))
        if self.on_device_sampling_config is None:
            return logits
        sampled = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled)
        return sampled, None

    # ── KV cache: full-attention layers only ─────────────────────────────

    def get_kv_spec(self):
        layers = []
        for i in self.config.full_attention_layers:
            attn = self.model.layers[i].self_attn
            layers.append(
                LayerSpec(
                    name=attn.layer_name,
                    num_kv_heads=attn.num_kv_heads_per_rank,
                    head_size=attn.head_dim,
                    dtype=attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches):
        for i in self.config.full_attention_layers:
            attn = self.model.layers[i].self_attn
            if attn.layer_name not in kv_caches:
                raise KeyError(f"KV cache for layer {attn.layer_name} not initialized")
            attn.k_cache, attn.v_cache = kv_caches[attn.layer_name]
        self._allocate_gdn_state(device=attn.k_cache.device)

    def _allocate_gdn_state(self, device):
        from vllm.config import get_current_vllm_config

        slots = int(
            os.environ.get(
                STATE_SLOTS_ENV, get_current_vllm_config().scheduler_config.max_num_seqs
            )
        )
        gdn = self.model.layers[self.config.linear_attention_layers[0]].linear_attn
        per_slot = RecurrentStateBuffers.bytes_per_slot(
            num_layers=len(self.config.linear_attention_layers),
            conv_dim=gdn.in_proj_qkv.shape[0],
            conv_width=self.config.linear_conv_kernel_dim - 1,
            num_v_heads=gdn.num_v_heads,
            head_k_dim=gdn.head_k_dim,
            head_v_dim=gdn.head_v_dim,
            conv_itemsize=torch.tensor(
                [], dtype=self.config.torch_dtype
            ).element_size(),
        )
        logger.info(
            "Qwen3.5 Gated DeltaNet state: %d slots x %.1f MiB = %.2f GiB per rank",
            slots,
            per_slot / 2**20,
            slots * per_slot / 2**30,
        )
        self.gdn_state = RecurrentStateBuffers(
            self.config.linear_attention_layers,
            num_slots=slots,
            conv_dim=gdn.in_proj_qkv.shape[0],
            conv_width=self.config.linear_conv_kernel_dim - 1,
            num_v_heads=gdn.num_v_heads,
            head_k_dim=gdn.head_k_dim,
            head_v_dim=gdn.head_v_dim,
            conv_dtype=self.config.torch_dtype,
            device=device,
        )

    # ── Weights ──────────────────────────────────────────────────────────

    def weight_mappings(self) -> dict[str, str]:
        """Model parameter -> checkpoint key. Skips model.visual.* and mtp.*."""
        src = "model.language_model"
        m = {
            "model.embed_tokens.weight": f"{src}.embed_tokens.weight",
            "model.norm.weight": f"{src}.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        for i, kind in enumerate(self.config.layer_types):
            ours, theirs = f"model.layers.{i}", f"{src}.layers.{i}"
            for norm in ("input_layernorm", "post_attention_layernorm"):
                m[f"{ours}.{norm}.weight"] = f"{theirs}.{norm}.weight"
            for proj in ("gate", "up", "down"):
                m[f"{ours}.mlp.{proj}_proj_weight"] = f"{theirs}.mlp.{proj}_proj.weight"
            if kind == "linear_attention":
                for name in (
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                    "conv1d",
                    "norm",
                    "out_proj",
                ):
                    m[f"{ours}.linear_attn.{name}"] = (
                        f"{theirs}.linear_attn.{name}.weight"
                    )
                for name in ("A_log", "dt_bias"):
                    m[f"{ours}.linear_attn.{name}"] = f"{theirs}.linear_attn.{name}"
            else:
                for proj in ("q", "k", "v", "o"):
                    m[f"{ours}.self_attn.{proj}_proj_weight"] = (
                        f"{theirs}.self_attn.{proj}_proj.weight"
                    )
                for norm in ("q_norm", "k_norm"):
                    m[f"{ours}.self_attn.{norm}.weight"] = (
                        f"{theirs}.self_attn.{norm}.weight"
                    )
        return m

    def load_weights(self, checkpoint_path, device, cache_dir):
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        state = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, self.weight_mappings(), device
        ).state_dict
        target = self.config.torch_dtype
        for name, tensor in state.items():
            if name.endswith(KEEP_STORED_DTYPE):
                continue  # A_log stays F32
            if tensor.dtype != target:
                state[name] = tensor.to(target)
        missing, _ = self.load_state_dict(state, strict=False, assign=True)
        missing = [k for k in missing if not k.startswith("gdn_state")]
        if missing:
            raise RuntimeError(
                f"Qwen3.5 checkpoint left parameters unloaded: {missing[:8]}"
            )

    def load_weights_lite(self, checkpoint_path, device, cache_dir):
        """CPU-compile path: no KV-cache scales to bake (FP8 KV is not wired)."""
        return
