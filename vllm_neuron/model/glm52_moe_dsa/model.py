# SPDX-License-Identifier: Apache-2.0
"""Decode-first GLM-5.2 model integration for the Trn2 TP64 baseline."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
    sharding_weight_loader,
)

from .artifact_preflight import preflight_checkpoint_artifact
from .cache_layout import Glm52CacheLayout
from .checkpoint_mapping import build_checkpoint_contract
from .config import Glm52MoeDsaConfig
from .dense_mlp import Glm52DenseMlp
from .indexer import Glm52IndexShareState
from .mla import Glm52MlaAttention
from .parallelism import RoutedExpertPlan
from .sparse_mlp import Glm52SparseMlp, glm52_rms_norm


def _is_compile_stub_directory(checkpoint_path: str) -> bool:
    config_path = Path(checkpoint_path) / "config.json"
    if not config_path.is_file():
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = config.get("glm52_artifact")
    return isinstance(artifact, dict) and artifact.get("compile_stub") is True


class _Glm52CompileStubCheckpoint:
    """Index-backed constants reader used only for CPU graph compilation."""

    def __init__(self, checkpoint_path: str) -> None:
        self.root = Path(checkpoint_path)
        config = json.loads(
            (self.root / "config.json").read_text(encoding="utf-8")
        )
        artifact = config.get("glm52_artifact")
        if (
            not isinstance(artifact, dict)
            or artifact.get("compile_stub") is not True
            or artifact.get("loader_ready") is not False
        ):
            raise ValueError("invalid non-serving GLM-5.2 compile stub marker")
        index = json.loads(
            (self.root / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("GLM-5.2 compile stub has no indexed tensor names")
        self.weight_map = {
            str(key): str(file_name) for key, file_name in weight_map.items()
        }
        self._open_files = {}

    def get_tensor_names(self) -> set[str]:
        return set(self.weight_map)

    def _get_slice(self, key: str):
        file_name = self.weight_map[key]
        path = self.root / file_name
        if not path.is_file():
            raise FileNotFoundError(
                f"compile-time constant {key!r} is not materialized in {path}"
            )
        if path not in self._open_files:
            self._open_files[path] = safe_open(
                path,
                framework="pt",
                device="cpu",
            )
        return self._open_files[path].get_slice(key)


class Glm52RotaryEmbedding(nn.Module):
    """Frozen default RoPE used by both MLA and the DSA indexer."""

    def __init__(self, config: Glm52MoeDsaConfig) -> None:
        super().__init__()
        rope_type = config.rope_parameters.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(f"unsupported GLM-5.2 rope_type: {rope_type!r}")
        self.rotary_dim = config.qk_rope_head_dim
        theta = float(config.rope_parameters["rope_theta"])
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(
                    0,
                    self.rotary_dim,
                    2,
                    dtype=torch.float32,
                    device="cpu",
                )
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = positions.reshape(-1, 1).to(
            torch.float32
        ) * self.inv_freq.reshape(1, -1)
        # apply_glm52_interleaved_rope consumes the first rotary_dim / 2
        # entries and expects the conventional duplicated [T, rotary_dim] ABI.
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


class _Glm52VocabParallelEmbedding(nn.Module):
    """Explicit-rank vocabulary sharding usable in runtime and meta tests."""

    def __init__(
        self,
        config: Glm52MoeDsaConfig,
        *,
        world_size: int,
        global_rank: int,
        tp_group,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.world_size = world_size
        self.global_rank = global_rank
        self.tp_group = tp_group
        self.vocab_size_per_rank = math.ceil(config.vocab_size / world_size)
        self.weight = nn.Parameter(
            torch.empty(
                self.vocab_size_per_rank,
                config.hidden_size,
                dtype=config.torch_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        set_weight_loader(
            self.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.vocab_size_per_rank,
                num_shards=world_size,
                pad_shard=True,
            ),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        rank: torch.Tensor | None,
        scatter_tokens: bool,
    ) -> torch.Tensor:
        active_rank = rank if rank is not None else self.global_rank
        start = active_rank * self.vocab_size_per_rank
        end = torch.as_tensor(
            start + self.vocab_size_per_rank,
            device=input_ids.device,
        ).clamp(max=self.vocab_size)
        mask = (input_ids >= start) & (input_ids < end)
        local_ids = torch.where(mask, input_ids - start, 0)
        output = F.embedding(local_ids, self.weight)
        output = torch.where(mask.unsqueeze(-1), output, torch.zeros_like(output))
        if self.tp_group is not None and self.world_size > 1:
            if scatter_tokens:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                output = self.tp_group.all_reduce(output)
        return output


class _Glm52VocabParallelHead(nn.Module):
    def __init__(
        self,
        config: Glm52MoeDsaConfig,
        *,
        world_size: int,
        tp_group,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.world_size = world_size
        self.tp_group = tp_group
        self.vocab_size_per_rank = math.ceil(config.vocab_size / world_size)
        self.weight = nn.Parameter(
            torch.empty(
                self.vocab_size_per_rank,
                config.hidden_size,
                dtype=config.torch_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        set_weight_loader(
            self.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.vocab_size_per_rank,
                num_shards=world_size,
                pad_shard=True,
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_states, self.weight)
        if self.tp_group is not None and self.world_size > 1:
            logits = self.tp_group.all_gather(logits, dim=-1)
        return logits[..., : self.vocab_size]


class Glm52DecoderLayer(nn.Module):
    """One pre-norm GLM block with exact attention/MLP residual ordering."""

    def __init__(
        self,
        config: Glm52MoeDsaConfig,
        *,
        layer_idx: int,
        cache_layout: Glm52CacheLayout,
        plan: RoutedExpertPlan,
        world_size: int,
        global_rank: int,
        tp_group,
        expert_tp_group,
        static_fp8: bool = True,
        device: torch.device | str | None = None,
        attention_module: nn.Module | None = None,
        mlp_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm = nn.Module()
        self.input_layernorm.weight = nn.Parameter(
            torch.empty(config.hidden_size, dtype=config.torch_dtype, device=device),
            requires_grad=False,
        )
        self.post_attention_layernorm = nn.Module()
        self.post_attention_layernorm.weight = nn.Parameter(
            torch.empty(config.hidden_size, dtype=config.torch_dtype, device=device),
            requires_grad=False,
        )
        self.self_attn = attention_module or Glm52MlaAttention(
            config,
            layer_idx=layer_idx,
            cache_layout=cache_layout,
            world_size=world_size,
            tp_group=tp_group,
            static_fp8=static_fp8,
            device=device,
        )
        if mlp_module is not None:
            self.mlp = mlp_module
        elif config.mlp_layer_types[layer_idx] == "dense":
            self.mlp = Glm52DenseMlp(
                config,
                world_size=world_size,
                global_rank=global_rank,
                tp_group=tp_group,
                static_fp8=static_fp8,
                device=device,
            )
        else:
            self.mlp = Glm52SparseMlp(
                config,
                plan,
                global_rank=global_rank,
                tp_group=tp_group,
                expert_tp_group=expert_tp_group,
                static_fp8=static_fp8,
                device=device,
            )
        self.key_cache: torch.Tensor | None = None
        self.value_cache: torch.Tensor | None = None
        self.indexer_cache: torch.Tensor | None = None

    def _is_decode(
        self,
        attn_metadata: dict[str, dict[str, torch.Tensor | int]],
    ) -> bool:
        metadata = attn_metadata[self.self_attn.cache_name]
        max_query_len = metadata.get("max_query_len")
        decode_threshold = metadata.get("decode_token_threshold")
        if not isinstance(max_query_len, int) or not isinstance(decode_threshold, int):
            raise TypeError(
                "GLM attention metadata requires integer max_query_len and "
                "decode_token_threshold"
            )
        return max_query_len <= decode_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict[str, dict[str, torch.Tensor | int]],
        previous_index_state: Glm52IndexShareState | None,
    ) -> tuple[torch.Tensor, Glm52IndexShareState]:
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError(f"KV cache for layer {self.layer_idx} is not bound")

        residual = hidden_states
        normalized = glm52_rms_norm(
            hidden_states,
            self.input_layernorm.weight,
            eps=self.config.rms_norm_eps,
        )
        is_decode = self._is_decode(attn_metadata)
        attention_forward = (
            self.self_attn.forward_paged_decode
            if is_decode
            else self.self_attn.forward_paged_prefill
        )
        attention_output, index_state = attention_forward(
            normalized,
            position_embeddings,
            positions,
            attn_metadata,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            previous_index_state=previous_index_state,
            indexer_cache=self.indexer_cache,
        )
        hidden_states = residual + attention_output

        residual = hidden_states
        if is_decode:
            mlp_output = self.mlp.forward_decode(
                hidden_states,
                norm_weight=self.post_attention_layernorm.weight,
            )
        elif isinstance(self.mlp, Glm52SparseMlp):
            metadata = attn_metadata[self.self_attn.cache_name]
            slot_mapping = metadata.get("slot_mapping")
            if not isinstance(slot_mapping, torch.Tensor):
                raise TypeError("slot_mapping metadata must be a tensor")
            local_slot_mapping = slot_mapping
            if self.self_attn.world_size > 1:
                # GLM prefill DCP is rejected by the factory. Embedding and
                # attention both sequence-parallelize with reduce_scatter on
                # dim 0, so each TP rank owns one contiguous token chunk.
                local_tokens = hidden_states.shape[0]
                start = (self.self_attn.tp_group.rank_in_group) * local_tokens
                local_slot_mapping = slot_mapping[start : start + local_tokens]
            if local_slot_mapping.numel() != hidden_states.shape[0]:
                raise ValueError(
                    "sequence-parallel slot_mapping slice must match the "
                    "rank-local hidden-state token count"
                )
            mlp_output = self.mlp.forward_prefill(
                hidden_states,
                norm_weight=self.post_attention_layernorm.weight,
                slot_mapping=local_slot_mapping,
            )
        else:
            mlp_output = self.mlp.forward_prefill(
                hidden_states,
                norm_weight=self.post_attention_layernorm.weight,
            )
        hidden_states = residual + mlp_output
        return hidden_states, index_state


class Glm52Model(nn.Module):
    """MTP-off 78-layer GLM-5.2 backbone."""

    def __init__(
        self,
        config: Glm52MoeDsaConfig,
        *,
        world_size: int,
        global_rank: int,
        tp_group,
        expert_tp_group,
        cache_dtype: torch.dtype = torch.bfloat16,
        static_fp8: bool = True,
        device: torch.device | str | None = None,
        layer_factory: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.world_size = world_size
        self.global_rank = global_rank
        self.tp_group = tp_group
        self.cache_dtype = cache_dtype
        self.cache_layout = Glm52CacheLayout.build(
            config,
            world_size=world_size,
            cache_dtype=cache_dtype,
        )
        self.plan = RoutedExpertPlan(
            world_size=world_size,
            ep_degree=config.neuron_config.ep_degree if config.neuron_config else 1,
            num_experts=config.n_routed_experts,
            expert_intermediate_size=config.moe_intermediate_size,
        )
        self.embed_tokens = _Glm52VocabParallelEmbedding(
            config,
            world_size=world_size,
            global_rank=global_rank,
            tp_group=tp_group,
            device=device,
        )
        make_layer = layer_factory or Glm52DecoderLayer
        self.layers = nn.ModuleList(
            [
                make_layer(
                    config,
                    layer_idx=layer_idx,
                    cache_layout=self.cache_layout,
                    plan=self.plan,
                    world_size=world_size,
                    global_rank=global_rank,
                    tp_group=tp_group,
                    expert_tp_group=expert_tp_group,
                    static_fp8=static_fp8,
                    device=device,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(
            torch.empty(config.hidden_size, dtype=config.torch_dtype, device=device),
            requires_grad=False,
        )
        self.rotary_emb = Glm52RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        *,
        attn_metadata: dict[str, dict[str, torch.Tensor | int]],
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first = attn_metadata["layers.0.self_attn"]
        is_prefill = first["max_query_len"] > first["decode_token_threshold"]
        if is_prefill and first["max_query_len"] > self.config.index_topk:
            raise NotImplementedError(
                "GLM-5.2's initial exact prefill path is bounded to "
                f"{self.config.index_topk} tokens; longer/segmented prefill "
                "requires the sparse streaming kernel"
            )
        hidden_states = self.embed_tokens(
            input_ids,
            rank=rank,
            scatter_tokens=is_prefill,
        )
        if inputs_embeds is not None:
            if is_token_ids is None:
                raise ValueError("is_token_ids is required with inputs_embeds")
            if is_prefill and self.world_size > 1:
                local_tokens = hidden_states.shape[0]
                start = self.global_rank * local_tokens
                inputs_embeds = inputs_embeds[start : start + local_tokens]
                is_token_ids = is_token_ids[start : start + local_tokens]
            hidden_states = torch.where(
                is_token_ids.reshape(-1, 1).to(torch.bool),
                hidden_states,
                inputs_embeds.to(hidden_states.dtype),
            )
        position_embeddings = self.rotary_emb(
            positions,
            dtype=hidden_states.dtype,
        )
        index_state: Glm52IndexShareState | None = None
        for layer in self.layers:
            hidden_states, index_state = layer(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
                index_state,
            )
        hidden_states = glm52_rms_norm(
            hidden_states,
            self.norm.weight,
            eps=self.config.rms_norm_eps,
        )
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states


class Glm52MoeDsaForCausalLM(nn.Module):
    """Concrete Trn2 GLM-5.2 implementation selected by the public factory."""

    def __init__(
        self,
        config: Glm52MoeDsaConfig,
        *,
        world_size: int,
        global_rank: int,
        tp_group,
        expert_tp_group,
        cache_dtype: torch.dtype = torch.bfloat16,
        static_fp8: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if world_size != config.num_attention_heads:
            raise ValueError(
                "GLM-5.2 expanded MLA requires TP64/one attention head per rank"
            )
        if config.tie_word_embeddings:
            raise ValueError("GLM-5.2 requires an untied language-model head")
        self.config = config
        self.world_size = world_size
        self.global_rank = global_rank
        self.tp_group = tp_group
        self.cache_dtype = cache_dtype
        self.model = Glm52Model(
            config,
            world_size=world_size,
            global_rank=global_rank,
            tp_group=tp_group,
            expert_tp_group=expert_tp_group,
            cache_dtype=cache_dtype,
            static_fp8=static_fp8,
            device=device,
        )
        self.lm_head = _Glm52VocabParallelHead(
            config,
            world_size=world_size,
            tp_group=tp_group,
            device=device,
        )

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig,
    ) -> "Glm52MoeDsaForCausalLM":
        from vllm.distributed.parallel_state import get_tp_group
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_ep_tp_group,
        )

        tp_group = get_tp_group()
        config = Glm52MoeDsaConfig.from_configs(hf_config, neuron_config)
        from vllm.config import get_current_vllm_config
        from vllm_neuron.utils.dtype_utils import kv_cache_dtype_str_to_dtype

        vllm_config = get_current_vllm_config()
        cache_dtype = kv_cache_dtype_str_to_dtype(
            vllm_config.cache_config.cache_dtype,
            vllm_config.model_config,
        )
        return cls(
            config,
            world_size=tp_group.world_size,
            global_rank=tp_group.rank_in_group,
            tp_group=tp_group,
            expert_tp_group=get_neuron_ep_tp_group(),
            cache_dtype=cache_dtype,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: dict[str, dict[str, torch.Tensor | int]] | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del sampling_params, kwargs
        if attn_metadata is None:
            raise ValueError("attn_metadata is required")
        if spec_decode_metadata is not None:
            raise NotImplementedError("GLM-5.2 MTP/speculative decode is not supported")
        if logit_mask is not None:
            raise NotImplementedError(
                "GLM-5.2 structured logit masks are not supported"
            )
        hidden_states = self.model(
            input_ids,
            positions.to(torch.int32),
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )
        if sampling_positions is not None:
            # Neuron's vector DGE path reinterprets a one-element int64 index
            # as two int32 lanes.  The synthetic upper lane then trips the
            # gather OOB check even when the requested token is valid.  Keep
            # the runtime index (short prefill requests are right-padded, so a
            # static last-row slice is incorrect) but lower it as one int32
            # lane, which is sufficient for every bounded token bucket.
            sampling_positions = sampling_positions.to(dtype=torch.int32)
            hidden_states = torch.index_select(
                hidden_states,
                0,
                sampling_positions,
            )
        return self.lm_head(hidden_states)

    def get_kv_spec(self):
        return self.model.cache_layout.kv_spec

    def bind_kv_cache(
        self,
        kv_caches: dict[str, Sequence[torch.Tensor]],
    ) -> None:
        for layer in self.model.layers:
            cache_name = layer.self_attn.cache_name
            if cache_name not in kv_caches:
                raise KeyError(f"KV cache for layer {cache_name!r} is not initialized")
            cache_pair = kv_caches[cache_name]
            if len(cache_pair) != 2:
                raise ValueError(f"KV cache {cache_name!r} must contain K and V")
            layer.key_cache, layer.value_cache = cache_pair
            if layer.self_attn.indexer is not None:
                layer.indexer_cache = self.model.cache_layout.indexer_cache_tensor(
                    kv_caches,
                    layer.layer_idx,
                )
                layer.self_attn.indexer.bind_key_cache(layer.indexer_cache)

    @staticmethod
    def _loader_mappings(contract) -> dict[str, str | list[str]]:
        mappings = dict(contract.mappings)
        for suffix in ("q_a_layernorm", "kv_a_layernorm"):
            for destination in tuple(mappings):
                marker = f".self_attn.{suffix}.weight"
                if destination.endswith(marker):
                    mappings[destination.removesuffix(".weight")] = mappings.pop(
                        destination
                    )
        return mappings

    def load_weights(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None,
    ) -> None:
        if _is_compile_stub_directory(checkpoint_path):
            raise RuntimeError(
                "a GLM-5.2 compile stub contains no serving weights and "
                "cannot be loaded for execution"
            )
        preflight_checkpoint_artifact(
            checkpoint_path,
            expected_weight_format=self.config.static_fp8_weight_format,
            expected_shared_expert_dtype=self.config.shared_expert_dtype,
        )
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        contract = build_checkpoint_contract(
            self.config,
            self.model.plan,
            global_rank=self.global_rank,
        )
        contract.validate_source_keys(checkpoint.get_tensor_names())
        self._load_cache_quant_multipliers(checkpoint, device)
        loaded = checkpoint.load_sharded_pipelined(
            self.global_rank,
            self.world_size,
            self,
            self._loader_mappings(contract),
            device,
        )
        self.load_state_dict(loaded.state_dict, strict=True, assign=True)

    def load_weights_lite(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None,
    ) -> None:
        del device
        if _is_compile_stub_directory(checkpoint_path):
            checkpoint = _Glm52CompileStubCheckpoint(checkpoint_path)
        else:
            preflight_checkpoint_artifact(
                checkpoint_path,
                expected_weight_format=self.config.static_fp8_weight_format,
                expected_shared_expert_dtype=self.config.shared_expert_dtype,
            )
            checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        contract = build_checkpoint_contract(
            self.config,
            self.model.plan,
            global_rank=self.global_rank,
        )
        contract.validate_source_keys(checkpoint.get_tensor_names())
        self._load_cache_quant_multipliers(checkpoint, torch.device("cpu"))

    @staticmethod
    def _read_positive_scalar(
        checkpoint: SafetensorsCheckpoint,
        key: str,
    ) -> float:
        slice_obj = checkpoint._get_slice(key)
        tensor = (
            slice_obj[()] if not tuple(slice_obj.get_shape()) else slice_obj[:]
        ).to(dtype=torch.float32)
        if tensor.numel() != 1:
            raise ValueError(f"cache multiplier {key!r} must be scalar")
        value = float(tensor.reshape(-1)[0])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"cache multiplier {key!r} must be finite and positive")
        return value

    def _load_cache_quant_multipliers(
        self,
        checkpoint: SafetensorsCheckpoint,
        device: torch.device,
    ) -> None:
        if self.cache_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            for layer in self.model.layers:
                layer.self_attn.key_cache_quant_multiplier = torch.ones(
                    1, dtype=torch.float32, device=device
                )
                layer.self_attn.value_cache_quant_multiplier = torch.ones(
                    1, dtype=torch.float32, device=device
                )
                if layer.self_attn.indexer is not None:
                    layer.self_attn.indexer.cache_quant_multiplier = torch.ones(
                        1, dtype=torch.float32, device=device
                    )
            return
        names = checkpoint.get_tensor_names()
        required: list[str] = []
        for layer in self.model.layers:
            prefix = f"model.layers.{layer.layer_idx}.self_attn"
            required.extend(
                (
                    f"{prefix}.k_cache_quant_multiplier",
                    f"{prefix}.v_cache_quant_multiplier",
                )
            )
            if layer.self_attn.indexer is not None:
                required.append(f"{prefix}.indexer.cache_quant_multiplier")
        missing = sorted(set(required).difference(names))
        if missing:
            preview = ", ".join(repr(key) for key in missing[:8])
            suffix = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
            raise KeyError(
                "FP8 GLM caches require calibrated quantization multipliers: "
                f"{preview}{suffix}"
            )
        for layer in self.model.layers:
            prefix = f"model.layers.{layer.layer_idx}.self_attn"
            key = self._read_positive_scalar(
                checkpoint,
                f"{prefix}.k_cache_quant_multiplier",
            )
            value = self._read_positive_scalar(
                checkpoint,
                f"{prefix}.v_cache_quant_multiplier",
            )
            layer.self_attn.key_cache_quant_multiplier = torch.ones(
                1, dtype=torch.float32, device=device
            )
            layer.self_attn.value_cache_quant_multiplier = torch.ones(
                1, dtype=torch.float32, device=device
            )
            layer.self_attn.set_cache_quant_multipliers(key=key, value=value)
            if layer.self_attn.indexer is not None:
                indexer = self._read_positive_scalar(
                    checkpoint,
                    f"{prefix}.indexer.cache_quant_multiplier",
                )
                layer.self_attn.indexer.cache_quant_multiplier = torch.ones(
                    1, dtype=torch.float32, device=device
                )
                layer.self_attn.indexer.set_cache_quant_multiplier(indexer)
