# SPDX-License-Identifier: Apache-2.0
"""Exact Glm5Next TP-rank inventory and bounded lazy conversion plan.

The plan is generated from the pinned production config and index.  It maps
every non-vision, non-MTP text weight exactly once into the module names and
per-rank shapes declared by the Glm5Next NxDI graph.  Execution yields one
bounded ``TensorChunk`` at a time and never constructs a full model or rank
dictionary.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from .checkpoint_converter import (
    GLM53_CHECKPOINT_REVISION,
    GLM53_CONFIG_SHA256,
    GLM53_INDEX_SHA256,
    Glm53ArchitectureMismatch,
    classify_tensor,
    kda_conv1d_per_head_layout,
    preflight_checkpoint_dir,
)
from .streaming_rank_writer import (
    Glm53StreamingError,
    IndexedTensorReader,
    RankInventory,
    StreamingRankWriter,
    TensorChunk,
    TensorSpec,
)

PlanKind = Literal["copy", "shard0", "shard1", "kda_conv", "moe_gate_up", "moe_down"]


@dataclass(frozen=True)
class TargetTensorPlan:
    target: TensorSpec
    kind: PlanKind
    source_keys: tuple[str, ...]
    source_shapes: tuple[tuple[int, ...], ...]

    def canonical(self) -> dict[str, Any]:
        return {
            "target": {
                "name": self.target.name,
                "dtype": str(self.target.dtype),
                "shape": list(self.target.shape),
            },
            "kind": self.kind,
            "source_keys": list(self.source_keys),
            "source_shapes": [list(shape) for shape in self.source_shapes],
        }


@dataclass(frozen=True)
class PlannedSourceSpec:
    key: str
    shape: tuple[int, ...]
    role: Literal["weight", "reciprocal_scale"]

    def canonical(self) -> dict[str, Any]:
        return {"key": self.key, "shape": list(self.shape), "role": self.role}


@dataclass(frozen=True)
class Glm53RankPlan:
    inventory: RankInventory
    operations: tuple[TargetTensorPlan, ...]
    max_chunk_bytes: int
    source_specs: tuple[PlannedSourceSpec, ...] = ()

    @property
    def contract_sha256(self) -> str:
        payload = {
            "schema": "glm53-target-rank-plan-v1",
            "source": {
                "revision": GLM53_CHECKPOINT_REVISION,
                "config_sha256": GLM53_CONFIG_SHA256,
                "index_sha256": GLM53_INDEX_SHA256,
            },
            "rank_inventory_sha256": self.inventory.contract_sha256,
            "max_chunk_bytes": self.max_chunk_bytes,
            "source_shape_contract": [spec.canonical() for spec in self.source_specs],
            "operations": [operation.canonical() for operation in self.operations],
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _validate_source(
        self, reader: IndexedTensorReader, operation: TargetTensorPlan
    ) -> None:
        for key, expected_shape in zip(
            operation.source_keys, operation.source_shapes, strict=True
        ):
            spec = reader.source_specs.get(key)
            if spec is None:
                raise Glm53StreamingError(
                    f"planned source is absent after audit: {key}"
                )
            if spec.shape != expected_shape:
                raise Glm53StreamingError(
                    f"source shape drift for {key}: expected {expected_shape}, got {spec.shape}"
                )

    def _validate_source_contract(self, reader: IndexedTensorReader) -> None:
        for planned in self.source_specs:
            actual = reader.source_specs.get(planned.key)
            if actual is None:
                raise Glm53StreamingError(
                    f"planned {planned.role} is absent after audit: {planned.key}"
                )
            if actual.shape != planned.shape:
                raise Glm53StreamingError(
                    f"{planned.role} shape drift for {planned.key}: "
                    f"expected {planned.shape}, got {actual.shape}"
                )

    def _split_target(self, name: str, start: int, tensor: torch.Tensor):
        flat = tensor.contiguous().view(-1)
        max_elements = self.max_chunk_bytes // flat.element_size()
        if max_elements <= 0:
            raise Glm53StreamingError(
                "max_chunk_bytes is smaller than one target element"
            )
        for offset in range(0, flat.numel(), max_elements):
            yield TensorChunk(
                name, start + offset, flat[offset : offset + max_elements]
            )

    def _simple_chunks(self, reader: IndexedTensorReader, operation: TargetTensorPlan):
        source_shape = operation.source_shapes[0]
        key = operation.source_keys[0]
        dtype = operation.target.dtype
        rank = self.inventory.rank
        tp = self.inventory.tp_degree
        if len(source_shape) == 1:
            source_start, source_stop = 0, source_shape[0]
            if operation.kind == "shard0":
                width = source_shape[0] // tp
                source_start, source_stop = rank * width, (rank + 1) * width
            element_size = torch.empty((), dtype=dtype).element_size()
            width = max(1, self.max_chunk_bytes // element_size)
            target_start = 0
            for start in range(source_start, source_stop, width):
                stop = min(source_stop, start + width)
                value = reader.read_converted_slice(
                    key, (slice(start, stop),), out_dtype=dtype
                ).contiguous()
                yield TensorChunk(operation.target.name, target_start, value)
                target_start += value.numel()
            return
        if len(source_shape) != 2:
            raise Glm53StreamingError(
                f"simple plan only supports rank-1/2 sources: {key}={source_shape}"
            )
        rows, cols = source_shape
        row_start, row_stop = 0, rows
        col_start, col_stop = 0, cols
        if operation.kind == "shard0":
            local = rows // tp
            row_start, row_stop = rank * local, (rank + 1) * local
        elif operation.kind == "shard1":
            local = cols // tp
            col_start, col_stop = rank * local, (rank + 1) * local
        target_cols = col_stop - col_start
        row_bytes = target_cols * torch.empty((), dtype=dtype).element_size()
        rows_per_chunk = max(1, self.max_chunk_bytes // row_bytes)
        target_start = 0
        for start in range(row_start, row_stop, rows_per_chunk):
            stop = min(row_stop, start + rows_per_chunk)
            value = reader.read_converted_slice(
                key,
                (slice(start, stop), slice(col_start, col_stop)),
                out_dtype=dtype,
            ).contiguous()
            yield TensorChunk(operation.target.name, target_start, value)
            target_start += value.numel()

    def iter_chunks(self, reader: IndexedTensorReader):
        if self.max_chunk_bytes <= 0:
            raise Glm53StreamingError("max_chunk_bytes must be positive")
        self._validate_source_contract(reader)
        for operation in self.operations:
            self._validate_source(reader, operation)
            if operation.kind in ("copy", "shard0", "shard1"):
                yield from self._simple_chunks(reader, operation)
                continue
            if operation.kind == "kda_conv":
                values = [
                    reader.read_converted_slice(
                        key,
                        tuple(slice(0, size) for size in shape),
                        out_dtype=operation.target.dtype,
                    )
                    for key, shape in zip(
                        operation.source_keys, operation.source_shapes, strict=True
                    )
                ]
                num_heads = operation.source_shapes[0][0] // 128
                full = kda_conv1d_per_head_layout(
                    values[0], values[1], values[2], num_heads=num_heads, head_dim=128
                ).view(num_heads, 3 * 128, 1, operation.source_shapes[0][-1])
                local_heads = num_heads // self.inventory.tp_degree
                rank = self.inventory.rank
                value = full[rank * local_heads : (rank + 1) * local_heads].reshape(
                    operation.target.shape
                )
                yield from self._split_target(operation.target.name, 0, value)
                continue
            experts = len(operation.source_keys) // (
                2 if operation.kind == "moe_gate_up" else 1
            )
            per_expert = operation.target.numel // experts
            if operation.kind == "moe_gate_up":
                for expert in range(experts):
                    gate_key = operation.source_keys[2 * expert]
                    up_key = operation.source_keys[2 * expert + 1]
                    gate_shape = operation.source_shapes[2 * expert]
                    up_shape = operation.source_shapes[2 * expert + 1]
                    gate = reader.read_converted_slice(
                        gate_key,
                        tuple(slice(0, n) for n in gate_shape),
                        out_dtype=operation.target.dtype,
                    )
                    up = reader.read_converted_slice(
                        up_key,
                        tuple(slice(0, n) for n in up_shape),
                        out_dtype=operation.target.dtype,
                    )
                    intermediate = gate_shape[0]
                    local = intermediate // self.inventory.tp_degree
                    start = self.inventory.rank * local
                    value = torch.cat(
                        (
                            gate.t()[:, start : start + local],
                            up.t()[:, start : start + local],
                        ),
                        dim=1,
                    ).contiguous()
                    yield from self._split_target(
                        operation.target.name, expert * per_expert, value
                    )
            else:
                for expert, (key, shape) in enumerate(
                    zip(operation.source_keys, operation.source_shapes, strict=True)
                ):
                    down = reader.read_converted_slice(
                        key,
                        tuple(slice(0, n) for n in shape),
                        out_dtype=operation.target.dtype,
                    )
                    intermediate = shape[1]
                    local = intermediate // self.inventory.tp_degree
                    start = self.inventory.rank * local
                    value = down.t()[start : start + local].contiguous()
                    yield from self._split_target(
                        operation.target.name, expert * per_expert, value
                    )


def _require_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        raise Glm53ArchitectureMismatch(f"text_config.{key} must be a positive integer")
    return value


def build_glm53_rank_plan(
    checkpoint_dir: str | Path,
    *,
    rank: int,
    tp_degree: int = 32,
    max_chunk_bytes: int = 64 * 1024 * 1024,
) -> Glm53RankPlan:
    """Derive the complete production rank plan from pinned metadata only."""
    preflight_checkpoint_dir(checkpoint_dir)
    root = Path(checkpoint_dir).resolve(strict=True)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    index = json.loads(
        (root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    text = config["text_config"]
    weight_map = index["weight_map"]
    if not isinstance(text, dict) or not isinstance(weight_map, dict):
        raise Glm53ArchitectureMismatch("pinned config/index schema changed")
    if tp_degree not in (32, 64):
        raise Glm53ArchitectureMismatch(
            "the qualified GLM-5.3 target contract requires TP=32 or TP=64"
        )
    if rank < 0 or rank >= tp_degree or max_chunk_bytes <= 0:
        raise ValueError("invalid rank, TP degree, or max_chunk_bytes")

    hidden = _require_int(text, "hidden_size")
    vocab = _require_int(text, "vocab_size")
    layers = _require_int(text, "num_hidden_layers")
    dense_i = _require_int(text, "intermediate_size")
    moe_i = _require_int(text, "moe_intermediate_size")
    experts = _require_int(text, "n_routed_experts")
    heads = _require_int(text, "num_attention_heads")
    q_lora = _require_int(text, "q_lora_rank")
    kv_lora = _require_int(text, "kv_lora_rank")
    qk = _require_int(text, "qk_head_dim")
    qk_nope = _require_int(text, "qk_nope_head_dim")
    v_head = _require_int(text, "v_head_dim")
    index_heads = _require_int(text, "index_n_heads")
    index_dim = _require_int(text, "index_head_dim")
    index_kpool = _require_int(text, "index_kpool")
    hc_mult = _require_int(text, "hc_mult")
    linear = text["linear_attn_config"]
    kda_heads = _require_int(linear, "num_heads")
    kda_dim = _require_int(linear, "head_dim")
    kernel = _require_int(linear, "short_conv_kernel_size")
    divisibles = (
        vocab,
        dense_i,
        moe_i,
        heads,
        kda_heads,
        q_lora,
        kv_lora,
        index_dim,
        index_heads * index_dim,
    )
    if any(value % tp_degree for value in divisibles):
        raise Glm53ArchitectureMismatch(
            f"production dimensions are not TP={tp_degree} divisible: {divisibles}"
        )

    operations: list[TargetTensorPlan] = []
    consumed: set[str] = set()

    def add(
        target: str,
        dtype: torch.dtype,
        target_shape: tuple[int, ...],
        kind: PlanKind,
        source_keys: tuple[str, ...],
        source_shapes: tuple[tuple[int, ...], ...],
    ) -> None:
        for key in source_keys:
            if key not in weight_map:
                raise Glm53ArchitectureMismatch(
                    f"required production source is absent: {key}"
                )
            if key in consumed:
                raise Glm53ArchitectureMismatch(
                    f"production source is mapped twice: {key}"
                )
            consumed.add(key)
        operations.append(
            TargetTensorPlan(
                target=TensorSpec(target, dtype, target_shape),
                kind=kind,
                source_keys=source_keys,
                source_shapes=source_shapes,
            )
        )

    add(
        "embed_tokens.weight",
        torch.bfloat16,
        (vocab // tp_degree, hidden),
        "shard0",
        ("model.language_model.embed_tokens.weight",),
        ((vocab, hidden),),
    )
    add(
        "final_norm_weight",
        torch.bfloat16,
        (hidden,),
        "copy",
        ("model.language_model.norm.weight",),
        ((hidden,),),
    )
    add(
        "lm_head.weight",
        torch.bfloat16,
        (vocab // tp_degree, hidden),
        "shard0",
        ("lm_head.weight",),
        ((vocab, hidden),),
    )

    mix_rows = (2 + hc_mult) * hc_mult
    for layer in range(layers):
        base = f"model.language_model.layers.{layer}."
        out = f"layers.{layer}."
        add(
            f"{out}input_norm_weight",
            torch.bfloat16,
            (hidden,),
            "copy",
            (f"{base}input_layernorm.weight",),
            ((hidden,),),
        )
        add(
            f"{out}post_attention_norm_weight",
            torch.bfloat16,
            (hidden,),
            "copy",
            (f"{base}post_attention_layernorm.weight",),
            ((hidden,),),
        )
        for hf_stem, target_stem in (("hc_attn", "hc_attn"), ("hc_ffn", "hc_mlp")):
            for leaf, dtype, shape in (
                ("base", torch.float32, (mix_rows,)),
                ("fn", torch.bfloat16, (mix_rows, hc_mult * hidden)),
                ("scale", torch.float32, (3,)),
            ):
                add(
                    f"{out}{target_stem}.{leaf}",
                    dtype,
                    shape,
                    "copy",
                    (f"{base}{hf_stem}_{leaf}",),
                    (shape,),
                )

        attn = f"{base}self_attn."
        target = f"{out}self_attn."
        if text["layer_types"][layer] == "linear_attention":
            qkv = kda_heads * kda_dim
            for suffix, shape, kind in (
                ("q_proj.weight", (qkv, hidden), "shard0"),
                ("k_proj.weight", (qkv, hidden), "shard0"),
                ("v_proj.weight", (qkv, hidden), "shard0"),
                ("b_proj.weight", (kda_heads, hidden), "shard0"),
                ("f_a_proj.weight", (kda_dim, hidden), "shard0"),
                ("f_b_proj.weight", (qkv, kda_dim), "shard0"),
                ("g_a_proj.weight", (kda_dim, hidden), "shard0"),
                ("g_b_proj.weight", (qkv, kda_dim), "shard0"),
                ("o_proj.weight", (hidden, qkv), "shard1"),
            ):
                target_shape = list(shape)
                target_shape[0 if kind == "shard0" else 1] //= tp_degree
                add(
                    f"{target}{suffix}",
                    torch.bfloat16,
                    tuple(target_shape),
                    kind,
                    (f"{attn}{suffix}",),
                    (shape,),
                )
            add(
                f"{target}A_log",
                torch.float32,
                (kda_heads,),
                "copy",
                (f"{attn}A_log",),
                ((kda_heads,),),
            )
            add(
                f"{target}dt_bias",
                torch.float32,
                (qkv,),
                "copy",
                (f"{attn}dt_bias",),
                ((qkv,),),
            )
            add(
                f"{target}o_norm_weight",
                torch.bfloat16,
                (kda_dim,),
                "copy",
                (f"{attn}o_norm.weight",),
                ((kda_dim,),),
            )
            conv_keys = tuple(
                f"{attn}{stream}_conv1d.weight" for stream in ("q", "k", "v")
            )
            conv_shape = (qkv, 1, kernel)
            add(
                f"{target}conv1d.weight",
                torch.bfloat16,
                (3 * qkv // tp_degree, 1, kernel),
                "kda_conv",
                conv_keys,
                (conv_shape,) * 3,
            )
        else:
            dsa_specs = (
                ("q_a_proj.weight", "mla.q_a_proj.weight", (q_lora, hidden), 0),
                ("q_b_proj.weight", "mla.q_b_proj.weight", (heads * qk, q_lora), 0),
                (
                    "kv_a_proj_with_mqa.weight",
                    "mla.kv_a_proj.weight",
                    (kv_lora, hidden),
                    0,
                ),
                (
                    "kv_b_proj.weight",
                    "mla.kv_b_proj.weight",
                    (heads * (qk_nope + v_head), kv_lora),
                    0,
                ),
                ("o_proj.weight", "mla.o_proj.weight", (hidden, heads * v_head), 1),
            )
            for source_suffix, target_suffix, shape, dim in dsa_specs:
                target_shape = list(shape)
                target_shape[dim] //= tp_degree
                add(
                    f"{target}{target_suffix}",
                    torch.bfloat16,
                    tuple(target_shape),
                    f"shard{dim}",
                    (f"{attn}{source_suffix}",),
                    (shape,),
                )
            add(
                f"{target}mla.q_a_norm",
                torch.bfloat16,
                (q_lora,),
                "copy",
                (f"{attn}q_a_layernorm.weight",),
                ((q_lora,),),
            )
            add(
                f"{target}mla.kv_a_norm",
                torch.bfloat16,
                (kv_lora,),
                "copy",
                (f"{attn}kv_a_layernorm.weight",),
                ((kv_lora,),),
            )
            index_specs = (
                ("wq_b.weight", (index_heads * index_dim, q_lora), "shard0"),
                ("wk.weight", (index_dim, hidden), "shard0"),
                (
                    "weights_proj.weight",
                    (index_heads, hidden),
                    "shard0" if tp_degree == 32 else "copy",
                ),
                ("k_norm.weight", (index_dim,), "copy"),
                ("k_norm.bias", (index_dim,), "copy"),
                ("index_kpool_compress_ape", (index_kpool, index_dim), "copy"),
                ("index_kpool_compress_gate", (index_dim, hidden), "copy"),
            )
            for suffix, shape, kind in index_specs:
                target_shape = list(shape)
                if kind == "shard0":
                    target_shape[0] //= tp_degree
                add(
                    f"{target}indexer.{suffix}",
                    torch.bfloat16,
                    tuple(target_shape),
                    kind,
                    (f"{attn}indexer.{suffix}",),
                    (shape,),
                )

        mlp = f"{base}mlp."
        target_mlp = f"{out}mlp."
        if text["mlp_layer_types"][layer] == "dense":
            for suffix, shape, kind in (
                ("gate_proj.weight", (dense_i, hidden), "shard0"),
                ("up_proj.weight", (dense_i, hidden), "shard0"),
                ("down_proj.weight", (hidden, dense_i), "shard1"),
            ):
                target_shape = list(shape)
                target_shape[0 if kind == "shard0" else 1] //= tp_degree
                add(
                    f"{target_mlp}{suffix}",
                    torch.bfloat16,
                    tuple(target_shape),
                    kind,
                    (f"{mlp}{suffix}",),
                    (shape,),
                )
        else:
            add(
                f"{target_mlp}router.weight",
                torch.float32,
                (experts, hidden),
                "copy",
                (f"{mlp}gate.weight",),
                ((experts, hidden),),
            )
            add(
                f"{target_mlp}e_score_correction_bias",
                torch.float32,
                (experts,),
                "copy",
                (f"{mlp}gate.e_score_correction_bias",),
                ((experts,),),
            )
            for source_suffix, target_suffix, shape, kind in (
                (
                    "shared_experts.gate_proj.weight",
                    "shared_expert.gate_proj.weight",
                    (moe_i, hidden),
                    "shard0",
                ),
                (
                    "shared_experts.up_proj.weight",
                    "shared_expert.up_proj.weight",
                    (moe_i, hidden),
                    "shard0",
                ),
                (
                    "shared_experts.down_proj.weight",
                    "shared_expert.down_proj.weight",
                    (hidden, moe_i),
                    "shard1",
                ),
            ):
                target_shape = list(shape)
                target_shape[0 if kind == "shard0" else 1] //= tp_degree
                add(
                    f"{target_mlp}{target_suffix}",
                    torch.bfloat16,
                    tuple(target_shape),
                    kind,
                    (f"{mlp}{source_suffix}",),
                    (shape,),
                )
            gate_up_keys: list[str] = []
            gate_up_shapes: list[tuple[int, ...]] = []
            down_keys: list[str] = []
            down_shapes: list[tuple[int, ...]] = []
            for expert in range(experts):
                for projection in ("gate_proj", "up_proj"):
                    gate_up_keys.append(f"{mlp}experts.{expert}.{projection}.weight")
                    gate_up_shapes.append((moe_i, hidden))
                down_keys.append(f"{mlp}experts.{expert}.down_proj.weight")
                down_shapes.append((hidden, moe_i))
            add(
                f"{target_mlp}expert_mlps.mlp_op.gate_up_proj.weight",
                torch.bfloat16,
                (experts, hidden, 2 * moe_i // tp_degree),
                "moe_gate_up",
                tuple(gate_up_keys),
                tuple(gate_up_shapes),
            )
            add(
                f"{target_mlp}expert_mlps.mlp_op.down_proj.weight",
                torch.bfloat16,
                (experts, moe_i // tp_degree, hidden),
                "moe_down",
                tuple(down_keys),
                tuple(down_shapes),
            )

    emittable = {
        key
        for key in weight_map
        if classify_tensor(key, weight_map)
        not in ("drop_mtp", "drop_vision", "block_fp8_scale")
    }
    missing = sorted(emittable - consumed)
    extra = sorted(consumed - emittable)
    if missing or extra:
        raise Glm53ArchitectureMismatch(
            f"target plan does not bijectively cover production text weights: "
            f"unmapped={missing[:8]} invalid={extra[:8]}"
        )
    expected_weight_shapes = {
        key: shape
        for operation in operations
        for key, shape in zip(
            operation.source_keys, operation.source_shapes, strict=True
        )
    }
    planned_sources: list[PlannedSourceSpec] = []
    for key in sorted(expected_weight_shapes):
        shape = expected_weight_shapes[key]
        planned_sources.append(PlannedSourceSpec(key, shape, "weight"))
        scale_key = f"{key}_scale_inv"
        if scale_key in weight_map:
            scale_shape = shape[:-2] + (
                math.ceil(shape[-2] / 128),
                math.ceil(shape[-1] / 128),
            )
            planned_sources.append(
                PlannedSourceSpec(scale_key, scale_shape, "reciprocal_scale")
            )
    inventory = RankInventory(
        rank=rank, tp_degree=tp_degree, tensors=tuple(op.target for op in operations)
    )
    return Glm53RankPlan(
        inventory=inventory,
        operations=tuple(operations),
        max_chunk_bytes=max_chunk_bytes,
        source_specs=tuple(planned_sources),
    )


def stream_glm53_rank_checkpoint(
    checkpoint_dir: str | Path,
    output_path: str | Path,
    *,
    rank: int,
    tp_degree: int = 32,
    max_chunk_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Audit, plan, and transactionally emit one complete production rank."""
    plan = build_glm53_rank_plan(
        checkpoint_dir,
        rank=rank,
        tp_degree=tp_degree,
        max_chunk_bytes=max_chunk_bytes,
    )
    reader = IndexedTensorReader(checkpoint_dir)
    with StreamingRankWriter(
        output_path,
        plan.inventory,
        source_report=reader.preflight_report,
        max_chunk_bytes=max_chunk_bytes,
        plan_contract_sha256=plan.contract_sha256,
    ) as writer:
        for chunk in plan.iter_chunks(reader):
            writer.write_chunk(chunk)
        return writer.finalize(source_reader=reader)


__all__ = [
    "Glm53RankPlan",
    "PlannedSourceSpec",
    "TargetTensorPlan",
    "build_glm53_rank_plan",
    "stream_glm53_rank_checkpoint",
]
