# SPDX-License-Identifier: Apache-2.0
"""Stream a BF16 GLM-5.2 checkpoint into the Trn2 static-FP8 contract.

The converted weights use OCP E4M3FN values with one scalar dequantization
scale per projection.  They intentionally retain the OCP ``[-448, 448]``
range on disk.  :mod:`checkpoint_mapping` applies the paired ``240/448``
weight downscale and ``448/240`` scale compensation when loading on Trn2.

Activation scales are calibration data, not weight statistics.  The
converter therefore never invents them.  An optional calibration manifest
can supply the required scalar ``*.input_scale`` tensors; otherwise the
artifact provenance explicitly records that it is not loader-ready.

Conversion is deterministic and bounded by one source tensor, one output
tensor, one FP32 quantization chunk, and the configured output-shard buffer.
The unavoidable exception is a single tensor larger than the shard target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

import safetensors
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .checkpoint_mapping import (
    MTP_IGNORED_PREFIX,
    STATIC_WEIGHT_SCALE_SUFFIX,
    _NEURON_LEGACY_E4M3_MAX,
    _OCP_E4M3_MAX,
    _SCALE_COMPENSATION,
    _WEIGHT_DOWNSCALE,
    build_checkpoint_contract,
)
from .config import Glm52MoeDsaConfig
from .parallelism import RoutedExpertPlan

ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_VERSION = "glm52-trn2-static-fp8-v1"
CONVERTER_VERSION = "1.0.0"
MANIFEST_FILENAME = "glm52-static-fp8-manifest.json"
INDEX_FILENAME = "model.safetensors.index.json"
DEFAULT_MAX_SHARD_BYTES = 2 * 1024**3
DEFAULT_QUANTIZATION_CHUNK_ELEMENTS = 4 * 1024**2
ZERO_TENSOR_SCALE = 1.0

_STATIC_WEIGHT_PATTERNS = (
    re.compile(
        r"^model\.layers\.\d+\.self_attn\."
        r"(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)\.weight$"
    ),
    re.compile(
        r"^model\.layers\.\d+\.mlp\."
        r"(?:gate_proj|up_proj|down_proj)\.weight$"
    ),
    re.compile(
        r"^model\.layers\.\d+\.mlp\.shared_experts\."
        r"(?:gate_proj|up_proj|down_proj)\.weight$"
    ),
    re.compile(
        r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
        r"(?:gate_proj|up_proj|down_proj)\.weight$"
    ),
)
_ROUTED_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
)
_GATE_OR_UP_PATTERN = re.compile(
    r"^(model\.layers\.\d+\.mlp(?:\.shared_experts)?)\."
    r"(gate_proj|up_proj)\.input_scale$"
)

_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

_AUXILIARY_FILES = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)

_BF16_EXCLUDE_MODULES = (
    "model.embed_tokens",
    "lm_head",
    "model.norm",
    "model.layers.*.input_layernorm",
    "model.layers.*.post_attention_layernorm",
    "model.layers.*.self_attn.q_a_layernorm",
    "model.layers.*.self_attn.kv_a_layernorm",
    "model.layers.*.self_attn.indexer.*",
    "model.layers.*.mlp.gate",
)


@dataclass(frozen=True)
class SourceCheckpoint:
    weight_map: dict[str, str]
    shard_names: tuple[str, ...]
    index_sha256: str | None
    declared_total_size: int | None


@dataclass
class ConversionStats:
    source_tensor_count: int = 0
    source_total_tensor_bytes: int = 0
    output_tensor_count: int = 0
    output_total_tensor_bytes: int = 0
    quantized_weight_count: int = 0
    preserved_tensor_count: int = 0
    generated_weight_scale_count: int = 0
    generated_input_scale_count: int = 0
    generated_cache_scale_count: int = 0
    excluded_tensor_count: int = 0
    largest_source_tensor_bytes: int = 0
    largest_output_group_bytes: int = 0


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _materialize_source_shard(
    source_dir: Path,
    shard_name: str,
    *,
    hf_streaming: bool,
    source_model_id: str,
    source_revision: str,
    download_staging_dir: Path,
):
    """Yield one shard and remove any transient HF cache before returning."""

    local_path = source_dir / shard_name
    if local_path.is_file():
        yield local_path, False
        return
    if not hf_streaming:
        raise FileNotFoundError(
            f"missing source shard {local_path}; use hf_streaming for a "
            "metadata-only source directory"
        )

    from huggingface_hub import hf_hub_download

    download_staging_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{shard_name}.",
        dir=download_staging_dir,
    ) as ephemeral_cache:
        downloaded = Path(
            hf_hub_download(
                repo_id=source_model_id,
                filename=shard_name,
                revision=source_revision,
                cache_dir=ephemeral_cache,
            )
        )
        if not downloaded.is_file():
            raise FileNotFoundError(
                f"Hugging Face download did not materialize {shard_name}"
            )
        yield downloaded, True
    # TemporaryDirectory removes the snapshot link, blob, lock, and metadata
    # before the caller is allowed to request the next source shard.


def _key_digest(keys: Sequence[str]) -> str:
    return _sha256_bytes("".join(f"{key}\n" for key in sorted(keys)).encode())


def _validate_relative_shard_name(name: str) -> str:
    path = Path(name)
    if (
        not name
        or path.is_absolute()
        or len(path.parts) != 1
        or path.suffix != ".safetensors"
    ):
        raise ValueError(f"unsafe safetensors shard name: {name!r}")
    return name


def _discover_source_checkpoint(source_dir: Path) -> SourceCheckpoint:
    index_path = source_dir / INDEX_FILENAME
    if index_path.is_file():
        index_bytes = index_path.read_bytes()
        raw_index = json.loads(index_bytes)
        raw_weight_map = raw_index.get("weight_map")
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise ValueError(f"{index_path} has no non-empty weight_map")
        weight_map: dict[str, str] = {}
        for key, shard_name in raw_weight_map.items():
            if not isinstance(key, str) or not isinstance(shard_name, str):
                raise ValueError("checkpoint weight_map must map strings to strings")
            weight_map[key] = _validate_relative_shard_name(shard_name)
        shard_names = tuple(sorted(set(weight_map.values())))
        index_sha256 = _sha256_bytes(index_bytes)
        metadata = raw_index.get("metadata")
        declared_total_size = (
            metadata.get("total_size") if isinstance(metadata, dict) else None
        )
        if not isinstance(declared_total_size, int) or declared_total_size <= 0:
            raise ValueError(
                f"{index_path} must declare a positive metadata.total_size"
            )
    else:
        shard_names = tuple(
            sorted(path.name for path in source_dir.glob("*.safetensors"))
        )
        if not shard_names:
            raise FileNotFoundError(
                f"no {INDEX_FILENAME} or safetensors shards under {source_dir}"
            )
        weight_map = {}
        index_sha256 = None
        declared_total_size = None
        for shard_name in shard_names:
            with safe_open(
                source_dir / shard_name,
                framework="pt",
                device="cpu",
            ) as shard:
                for key in shard.keys():
                    if key in weight_map:
                        raise ValueError(
                            f"tensor {key!r} occurs in multiple source shards"
                        )
                    weight_map[key] = shard_name

    return SourceCheckpoint(
        weight_map=weight_map,
        shard_names=shard_names,
        index_sha256=index_sha256,
        declared_total_size=declared_total_size,
    )


def _validate_source_config(source_dir: Path) -> tuple[dict[str, Any], str]:
    config_path = source_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing GLM-5.2 config: {config_path}")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    architectures = config.get("architectures", ())
    if "GlmMoeDsaForCausalLM" not in architectures:
        raise ValueError("source config must declare architecture GlmMoeDsaForCausalLM")
    dtype = config.get("dtype", config.get("torch_dtype"))
    if dtype not in ("bfloat16", "bf16"):
        raise ValueError(f"source config must declare BF16, got {dtype!r}")
    if config.get("num_hidden_layers") != 78:
        raise ValueError("GLM-5.2 conversion requires 78 backbone layers")
    return config, _sha256_bytes(config_bytes)


def _is_generated_scale_key(key: str) -> bool:
    return key.endswith((".weight_scale", ".input_scale"))


def validate_bf16_index_closure(
    config_dict: Mapping[str, Any],
    source_keys: Sequence[str],
    *,
    world_size: int = 64,
    ep_degree: int = 16,
) -> dict[str, Any]:
    """Prove the BF16 index is exactly the MTP-off loader source contract."""

    config = Glm52MoeDsaConfig.from_configs(dict(config_dict), None)
    plan = RoutedExpertPlan(
        world_size=world_size,
        ep_degree=ep_degree,
        num_experts=config.n_routed_experts,
        expert_intermediate_size=config.moe_intermediate_size,
    )
    expected: set[str] = set()
    for rank in range(world_size):
        contract = build_checkpoint_contract(config, plan, global_rank=rank)
        expected.update(
            key
            for key in contract.required_source_keys
            if not _is_generated_scale_key(key)
        )

    actual = {key for key in source_keys if not key.startswith(MTP_IGNORED_PREFIX)}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "BF16 checkpoint index does not close over the GLM loader "
            f"contract: missing={missing[:8]!r}, extra={extra[:8]!r}"
        )
    return {
        "status": "passed",
        "world_size": world_size,
        "ep_degree": ep_degree,
        "expected_non_mtp_key_count": len(expected),
        "actual_non_mtp_key_count": len(actual),
        "expected_keys_sha256": _key_digest(tuple(expected)),
        "actual_keys_sha256": _key_digest(tuple(actual)),
        "missing_key_count": 0,
        "extra_key_count": 0,
    }


def is_static_fp8_weight(key: str) -> bool:
    """Return whether ``key`` is consumed by a static-FP8 GLM projection."""

    return any(pattern.fullmatch(key) for pattern in _STATIC_WEIGHT_PATTERNS)


def _weight_scale_key(weight_key: str) -> str:
    return weight_key.removesuffix(".weight") + STATIC_WEIGHT_SCALE_SUFFIX


def _input_scale_key(weight_key: str) -> str | None:
    if _ROUTED_EXPERT_PATTERN.fullmatch(weight_key):
        return None
    return weight_key.removesuffix(".weight") + ".input_scale"


def required_activation_scale_keys(source_keys: Sequence[str]) -> tuple[str, ...]:
    """Return deterministic scalar activation-scale names for this artifact."""

    return tuple(
        sorted(
            input_scale
            for key in source_keys
            if not key.startswith(MTP_IGNORED_PREFIX)
            and is_static_fp8_weight(key)
            and (input_scale := _input_scale_key(key)) is not None
        )
    )


def required_cache_quant_multiplier_keys(
    config_dict: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return every scalar multiplier needed by the production FP8 caches."""

    config = Glm52MoeDsaConfig.from_configs(dict(config_dict), None)
    keys: list[str] = []
    for layer_idx, indexer_type in enumerate(config.indexer_types):
        attention = f"model.layers.{layer_idx}.self_attn"
        keys.extend(
            (
                f"{attention}.k_cache_quant_multiplier",
                f"{attention}.v_cache_quant_multiplier",
            )
        )
        if indexer_type == "full":
            keys.append(f"{attention}.indexer.cache_quant_multiplier")
    return tuple(sorted(keys))


def _float32_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite scalar")
    try:
        scalar = torch.tensor(value, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be a positive finite scalar") from error
    if scalar.numel() != 1:
        raise ValueError(f"{name} must be a scalar")
    result = float(scalar)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def _load_calibration(
    path: Path | None,
    required_projection_keys: Sequence[str],
    required_cache_keys: Sequence[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    if path is None:
        return (
            {},
            {},
            {
                "status": "missing",
                "loader_ready": False,
                "projection_input_scales": {
                    "status": "missing",
                    "required_scale_count": len(required_projection_keys),
                    "required_keys_sha256": _key_digest(required_projection_keys),
                    "range_max": _NEURON_LEGACY_E4M3_MAX,
                },
                "cache_quant_multipliers": {
                    "status": "missing",
                    "required_scale_count": len(required_cache_keys),
                    "required_keys_sha256": _key_digest(required_cache_keys),
                    "range_max": _NEURON_LEGACY_E4M3_MAX,
                    "semantics": "quantized_cache = value * multiplier",
                },
                "note": (
                    "Calibrate projection inputs and main/indexer cache values on "
                    "representative prompts; weight maxima determine neither."
                ),
            },
        )

    calibration_bytes = path.read_bytes()
    payload = json.loads(calibration_bytes)
    if not isinstance(payload, dict):
        raise ValueError("calibration manifest must be a JSON object")
    raw_projection = payload.get("projection_input_scales")
    raw_cache = payload.get("cache_quant_multipliers")
    if not isinstance(raw_projection, dict) or not isinstance(raw_cache, dict):
        raise ValueError(
            "calibration must contain 'projection_input_scales' and "
            "'cache_quant_multipliers' objects"
        )

    required_projection = set(required_projection_keys)
    supplied_projection = set(raw_projection)
    missing_projection = sorted(required_projection - supplied_projection)
    extra_projection = sorted(supplied_projection - required_projection)
    required_cache = set(required_cache_keys)
    supplied_cache = set(raw_cache)
    missing_cache = sorted(required_cache - supplied_cache)
    extra_cache = sorted(supplied_cache - required_cache)
    if missing_projection or extra_projection or missing_cache or extra_cache:
        raise ValueError(
            "activation calibration key mismatch: "
            f"projection_missing={missing_projection[:8]!r}, "
            f"projection_extra={extra_projection[:8]!r}, "
            f"cache_missing={missing_cache[:8]!r}, "
            f"cache_extra={extra_cache[:8]!r}"
        )
    projection_scales = {
        key: _float32_scalar(raw_projection[key], name=key)
        for key in required_projection_keys
    }
    cache_multipliers = {
        key: _float32_scalar(raw_cache[key], name=key) for key in required_cache_keys
    }

    gate_up: dict[str, dict[str, float]] = defaultdict(dict)
    for key, value in projection_scales.items():
        match = _GATE_OR_UP_PATTERN.fullmatch(key)
        if match:
            gate_up[match.group(1)][match.group(2)] = value
    for prefix, pair in gate_up.items():
        if set(pair) != {"gate_proj", "up_proj"}:
            raise ValueError(f"incomplete gate/up calibration pair for {prefix}")
        if pair["gate_proj"] != pair["up_proj"]:
            raise ValueError(
                f"gate/up input scales must match for {prefix}: "
                f"{pair['gate_proj']} != {pair['up_proj']}"
            )

    return (
        projection_scales,
        cache_multipliers,
        {
            "status": "complete",
            "loader_ready": True,
            "source_file": path.name,
            "source_sha256": _sha256_bytes(calibration_bytes),
            "projection_input_scales": {
                "status": "complete",
                "required_scale_count": len(required_projection_keys),
                "required_keys_sha256": _key_digest(required_projection_keys),
                "range_max": _NEURON_LEGACY_E4M3_MAX,
            },
            "cache_quant_multipliers": {
                "status": "complete",
                "required_scale_count": len(required_cache_keys),
                "required_keys_sha256": _key_digest(required_cache_keys),
                "range_max": _NEURON_LEGACY_E4M3_MAX,
                "semantics": "quantized_cache = value * multiplier",
            },
        },
    )


def quantize_bf16_per_tensor_ocp(
    weight: torch.Tensor,
    *,
    chunk_elements: int = DEFAULT_QUANTIZATION_CHUNK_ELEMENTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one contiguous BF16 tensor to deterministic OCP E4M3FN.

    The scale is ``max(abs(weight)) / 448``.  An all-zero tensor uses a
    canonical scale of ``1.0`` so no divide-by-zero or non-finite metadata is
    emitted.
    """

    if weight.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 source weight, got {weight.dtype}")
    if weight.ndim != 2:
        raise ValueError(f"static-FP8 projection must be rank-2, got {weight.shape}")
    if not weight.is_contiguous():
        raise ValueError("source safetensors weight must be contiguous")
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")

    flat = weight.view(-1)
    maximum = 0.0
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start : start + chunk_elements]
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("source BF16 tensor contains NaN or Inf")
        if chunk.numel():
            chunk_maximum = float(chunk.abs().max())
            maximum = max(maximum, chunk_maximum)

    scale_value = maximum / _OCP_E4M3_MAX if maximum else ZERO_TENSOR_SCALE
    scale = torch.tensor(scale_value, dtype=torch.float32)
    output = torch.empty(weight.shape, dtype=torch.float8_e4m3fn)
    flat_output = output.view(-1)
    for start in range(0, flat.numel(), chunk_elements):
        source_chunk = flat[start : start + chunk_elements]
        quantized = (
            (source_chunk.to(torch.float32) / scale)
            .clamp(-_OCP_E4M3_MAX, _OCP_E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
        flat_output[start : start + source_chunk.numel()].copy_(quantized)
    return output, scale


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _slice_bytes(slice_obj: Any) -> int:
    dtype = slice_obj.get_dtype()
    try:
        element_bytes = _SAFETENSORS_DTYPE_BYTES[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}") from error
    return prod(slice_obj.get_shape()) * element_bytes


class _StreamingShardWriter:
    def __init__(self, staging_dir: Path, max_shard_bytes: int) -> None:
        if max_shard_bytes <= 0:
            raise ValueError("max_shard_bytes must be positive")
        self.staging_dir = staging_dir
        self.max_shard_bytes = max_shard_bytes
        self.buffer: dict[str, torch.Tensor] = {}
        self.buffered_bytes = 0
        self.maximum_buffered_bytes = 0
        self.parts: list[tuple[Path, tuple[str, ...], int]] = []

    def prepare(self, output_group_bytes: int) -> None:
        if self.buffer and self.buffered_bytes + output_group_bytes > (
            self.max_shard_bytes
        ):
            self.flush()

    def add(self, tensors: Mapping[str, torch.Tensor]) -> None:
        duplicate = set(tensors).intersection(self.buffer)
        if duplicate:
            raise ValueError(f"duplicate output tensors: {sorted(duplicate)!r}")
        for key, tensor in tensors.items():
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            self.buffer[key] = tensor
            self.buffered_bytes += _tensor_bytes(tensor)
        self.maximum_buffered_bytes = max(
            self.maximum_buffered_bytes,
            self.buffered_bytes,
        )
        if self.buffered_bytes >= self.max_shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        part_number = len(self.parts) + 1
        part_path = self.staging_dir / f".part-{part_number:05d}.safetensors"
        ordered = {key: self.buffer[key] for key in sorted(self.buffer)}
        save_file(
            ordered,
            part_path,
            # Multiple metadata entries are serialized through a Rust
            # HashMap and can change byte order between identical runs.
            # Keep the shard metadata single-key; artifact versioning lives
            # in the deterministic index/config/provenance manifests.
            metadata={"format": "pt"},
        )
        keys = tuple(ordered)
        self.parts.append((part_path, keys, self.buffered_bytes))
        self.buffer.clear()
        self.buffered_bytes = 0

    def finalize(self) -> tuple[dict[str, str], list[dict[str, Any]]]:
        self.flush()
        if not self.parts:
            raise ValueError("conversion produced no output tensors")
        count = len(self.parts)
        weight_map: dict[str, str] = {}
        shards: list[dict[str, Any]] = []
        for number, (part_path, keys, tensor_bytes) in enumerate(
            self.parts,
            start=1,
        ):
            final_name = f"model-{number:05d}-of-{count:05d}.safetensors"
            final_path = self.staging_dir / final_name
            part_path.replace(final_path)
            for key in keys:
                weight_map[key] = final_name
            shards.append(
                {
                    "file": final_name,
                    "file_bytes": final_path.stat().st_size,
                    "tensor_bytes": tensor_bytes,
                    "sha256": _sha256_file(final_path),
                }
            )
        return weight_map, shards


def _copy_auxiliary_files(
    source_dir: Path,
    staging_dir: Path,
) -> list[dict[str, Any]]:
    copied = []
    for name in _AUXILIARY_FILES:
        source = source_dir / name
        if not source.is_file():
            continue
        destination = staging_dir / name
        shutil.copyfile(source, destination)
        copied.append(
            {
                "file": name,
                "bytes": destination.stat().st_size,
                "sha256": _sha256_file(destination),
            }
        )
    return copied


def _write_output_config(
    source_config: Mapping[str, Any],
    staging_dir: Path,
    *,
    calibration_manifest: Mapping[str, Any],
    cache_quant_multipliers: Mapping[str, float],
    index_closure: Mapping[str, Any],
) -> dict[str, Any]:
    output_config = dict(source_config)
    output_config["dtype"] = "bfloat16"
    output_config["torch_dtype"] = "bfloat16"
    output_config["quantization_config"] = {
        "quant_method": "modelopt",
        "artifact_version": ARTIFACT_VERSION,
        "quantization": {
            "quant_algo": "FP8",
            "kv_cache_quant_algo": "FP8",
            "weight_scale_granularity": "per_tensor",
            "activation_scheme": "static",
            "exclude_modules": list(_BF16_EXCLUDE_MODULES),
        },
    }
    output_config["glm52_artifact"] = {
        "artifact_version": ARTIFACT_VERSION,
        "manifest_file": MANIFEST_FILENAME,
        "mtp_enabled": False,
        "loader_ready": bool(calibration_manifest["loader_ready"])
        and index_closure["status"] == "passed",
        "index_closure_status": index_closure["status"],
        "calibration_status": calibration_manifest["status"],
        "projection_input_scales": dict(
            calibration_manifest["projection_input_scales"]
        ),
        "cache_quant_multipliers": {
            "contract": dict(calibration_manifest["cache_quant_multipliers"]),
            "values": dict(sorted(cache_quant_multipliers.items())),
        },
    }
    config_bytes = _json_bytes(output_config)
    path = staging_dir / "config.json"
    path.write_bytes(config_bytes)
    return {
        "file": path.name,
        "bytes": len(config_bytes),
        "sha256": _sha256_bytes(config_bytes),
    }


def convert_checkpoint(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    source_revision: str,
    source_model_id: str = "zai-org/GLM-5.2",
    activation_scales_path: str | Path | None = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    quantization_chunk_elements: int = DEFAULT_QUANTIZATION_CHUNK_ELEMENTS,
    hf_streaming: bool = False,
    download_staging_dir: str | Path | None = None,
    strict_index_closure: bool = True,
    world_size: int = 64,
    ep_degree: int = 16,
) -> dict[str, Any]:
    """Convert a local BF16 checkpoint without loading more than one tensor."""

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not source_revision.strip():
        raise ValueError("source_revision is required for provenance")
    if not source_model_id.strip():
        raise ValueError("source_model_id is required for provenance")
    if source_dir == output_dir:
        raise ValueError("source and output directories must differ")
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    config, config_sha256 = _validate_source_config(source_dir)
    source = _discover_source_checkpoint(source_dir)
    source_keys = tuple(sorted(source.weight_map))
    if strict_index_closure:
        index_closure = validate_bf16_index_closure(
            config,
            source_keys,
            world_size=world_size,
            ep_degree=ep_degree,
        )
    else:
        index_closure = {
            "status": "not_checked",
            "world_size": world_size,
            "ep_degree": ep_degree,
            "actual_non_mtp_key_count": sum(
                not key.startswith(MTP_IGNORED_PREFIX) for key in source_keys
            ),
            "actual_keys_sha256": _key_digest(
                tuple(
                    key for key in source_keys if not key.startswith(MTP_IGNORED_PREFIX)
                )
            ),
        }
    target_keys = tuple(
        key
        for key in source_keys
        if not key.startswith(MTP_IGNORED_PREFIX) and is_static_fp8_weight(key)
    )
    if not target_keys:
        raise ValueError("source checkpoint contains no GLM static-FP8 weights")

    generated_weight_scales = {_weight_scale_key(key) for key in target_keys}
    required_input_scales = required_activation_scale_keys(source_keys)
    required_cache_scales = required_cache_quant_multiplier_keys(config)
    generated_input_scales = set(required_input_scales)
    generated_cache_scales = set(required_cache_scales)
    generated_keys = (
        generated_weight_scales | generated_input_scales | generated_cache_scales
    )
    collisions = generated_keys.intersection(source_keys)
    if collisions:
        raise ValueError(
            "source checkpoint already contains generated scale keys: "
            f"{sorted(collisions)[:8]!r}"
        )

    calibration_path = (
        Path(activation_scales_path).resolve()
        if activation_scales_path is not None
        else None
    )
    (
        activation_scales,
        cache_quant_multipliers,
        calibration_manifest,
    ) = _load_calibration(
        calibration_path,
        required_input_scales,
        required_cache_scales,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    stats = ConversionStats(source_tensor_count=len(source_keys))
    excluded_keys: list[str] = []
    writer = _StreamingShardWriter(staging_dir, max_shard_bytes)
    source_shards = []
    download_root = (
        Path(download_staging_dir).resolve()
        if download_staging_dir is not None
        else staging_dir / ".source-downloads"
    )
    try:
        keys_by_shard: dict[str, list[str]] = defaultdict(list)
        for key, shard_name in source.weight_map.items():
            keys_by_shard[shard_name].append(key)

        for shard_name in source.shard_names:
            with _materialize_source_shard(
                source_dir,
                shard_name,
                hf_streaming=hf_streaming,
                source_model_id=source_model_id,
                source_revision=source_revision,
                download_staging_dir=download_root,
            ) as (shard_path, transient):
                source_shards.append(
                    {
                        "file": shard_name,
                        "file_bytes": shard_path.stat().st_size,
                        "sha256": _sha256_file(shard_path),
                        "transient_download": transient,
                    }
                )
                with safe_open(
                    shard_path,
                    framework="pt",
                    device="cpu",
                ) as shard:
                    expected_shard_keys = set(keys_by_shard[shard_name])
                    actual_shard_keys = set(shard.keys())
                    if actual_shard_keys != expected_shard_keys:
                        raise ValueError(
                            f"source index disagrees with {shard_name}: "
                            f"missing={sorted(expected_shard_keys - actual_shard_keys)[:4]!r}, "
                            f"unindexed={sorted(actual_shard_keys - expected_shard_keys)[:4]!r}"
                        )
                    for key in sorted(expected_shard_keys):
                        slice_obj = shard.get_slice(key)
                        source_bytes = _slice_bytes(slice_obj)
                        stats.source_total_tensor_bytes += source_bytes
                        stats.largest_source_tensor_bytes = max(
                            stats.largest_source_tensor_bytes,
                            source_bytes,
                        )
                        if key.startswith(MTP_IGNORED_PREFIX):
                            excluded_keys.append(key)
                            stats.excluded_tensor_count += 1
                            del slice_obj
                            continue

                        if is_static_fp8_weight(key):
                            output_group_bytes = prod(slice_obj.get_shape()) + 4
                            input_scale_key = _input_scale_key(key)
                            if input_scale_key in activation_scales:
                                output_group_bytes += 4
                        else:
                            output_group_bytes = source_bytes
                        writer.prepare(output_group_bytes)

                        tensor = shard.get_tensor(key)
                        if is_static_fp8_weight(key):
                            quantized, weight_scale = quantize_bf16_per_tensor_ocp(
                                tensor,
                                chunk_elements=(quantization_chunk_elements),
                            )
                            output_group = {
                                key: quantized,
                                _weight_scale_key(key): weight_scale,
                            }
                            input_scale_key = _input_scale_key(key)
                            if input_scale_key in activation_scales:
                                output_group[input_scale_key] = torch.tensor(
                                    activation_scales[input_scale_key],
                                    dtype=torch.float32,
                                )
                                stats.generated_input_scale_count += 1
                            stats.quantized_weight_count += 1
                            stats.generated_weight_scale_count += 1
                        else:
                            output_group = {key: tensor}
                            stats.preserved_tensor_count += 1

                        actual_group_bytes = sum(
                            _tensor_bytes(value) for value in output_group.values()
                        )
                        if actual_group_bytes != output_group_bytes:
                            raise AssertionError(
                                f"output byte estimate drift for {key}: "
                                f"{output_group_bytes} != {actual_group_bytes}"
                            )
                        stats.largest_output_group_bytes = max(
                            stats.largest_output_group_bytes,
                            actual_group_bytes,
                        )
                        stats.output_tensor_count += len(output_group)
                        stats.output_total_tensor_bytes += actual_group_bytes
                        writer.add(output_group)
                        del output_group, tensor, slice_obj
                # Deterministically end every output shard group at a source
                # shard boundary. This also persists all mmap-backed
                # preserved tensors before an ephemeral HF snapshot/blob is
                # removed and makes local and HF-streaming output identical.
                writer.flush()

        if (
            source.declared_total_size is not None
            and stats.source_total_tensor_bytes != source.declared_total_size
        ):
            raise ValueError(
                "source index metadata.total_size does not match tensor "
                f"headers: {source.declared_total_size} != "
                f"{stats.source_total_tensor_bytes}"
            )

        if cache_quant_multipliers:
            cache_scale_tensors = {
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in sorted(cache_quant_multipliers.items())
            }
            cache_scale_bytes = 4 * len(cache_scale_tensors)
            writer.prepare(cache_scale_bytes)
            writer.add(cache_scale_tensors)
            stats.generated_cache_scale_count = len(cache_scale_tensors)
            stats.output_tensor_count += len(cache_scale_tensors)
            stats.output_total_tensor_bytes += cache_scale_bytes
            stats.largest_output_group_bytes = max(
                stats.largest_output_group_bytes,
                cache_scale_bytes,
            )
        if download_staging_dir is None and download_root.exists():
            download_root.rmdir()

        weight_map, output_shards = writer.finalize()
        index = {
            "metadata": {
                "artifact_version": ARTIFACT_VERSION,
                "total_size": stats.output_total_tensor_bytes,
                "weight_scale_shape": "scalar",
            },
            "weight_map": dict(sorted(weight_map.items())),
        }
        index_bytes = _json_bytes(index)
        (staging_dir / INDEX_FILENAME).write_bytes(index_bytes)
        output_config_file = _write_output_config(
            config,
            staging_dir,
            calibration_manifest=calibration_manifest,
            cache_quant_multipliers=cache_quant_multipliers,
            index_closure=index_closure,
        )
        auxiliary_files = [
            output_config_file,
            *_copy_auxiliary_files(source_dir, staging_dir),
        ]
        loader_ready = (
            calibration_manifest["loader_ready"] and index_closure["status"] == "passed"
        )

        identity = {
            "artifact_version": ARTIFACT_VERSION,
            "source_model_id": source_model_id,
            "source_revision": source_revision,
            "source_index_sha256": source.index_sha256,
            "source_shards": source_shards,
            "output_index_sha256": _sha256_bytes(index_bytes),
            "output_shards": output_shards,
            "output_config_sha256": output_config_file["sha256"],
            "calibration_source_sha256": calibration_manifest.get("source_sha256"),
        }
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "artifact_id": _sha256_bytes(json.dumps(identity, sort_keys=True).encode()),
            "converter": {
                "name": ("vllm_neuron.model.glm52_moe_dsa.checkpoint_converter"),
                "version": CONVERTER_VERSION,
                "torch_version": torch.__version__,
                "safetensors_version": safetensors.__version__,
            },
            "model": {
                "architecture": "GlmMoeDsaForCausalLM",
                "backbone_layers": config["num_hidden_layers"],
                "mtp_enabled": False,
            },
            "source": {
                "model_id": source_model_id,
                "revision": source_revision,
                "config_sha256": config_sha256,
                "index_file": INDEX_FILENAME
                if source.index_sha256 is not None
                else None,
                "index_sha256": source.index_sha256,
                "shards": source_shards,
                "tensor_count": stats.source_tensor_count,
                "total_tensor_bytes": stats.source_total_tensor_bytes,
                "declared_total_tensor_bytes": source.declared_total_size,
                "index_closure": index_closure,
            },
            "quantization": {
                "algorithm": "symmetric_per_tensor_absmax",
                "source_dtype": "bfloat16",
                "weight_dtype": "float8_e4m3fn",
                "format": "OCP E4M3FN",
                "qmax": _OCP_E4M3_MAX,
                "scale_dtype": "float32",
                "scale_shape": [],
                "zero_tensor_scale": ZERO_TENSOR_SCALE,
                "loader_compensation": {
                    "weight_multiplier": _WEIGHT_DOWNSCALE,
                    "scale_multiplier": _SCALE_COMPENSATION,
                    "neuron_kernel_qmax": _NEURON_LEGACY_E4M3_MAX,
                },
            },
            "calibration": calibration_manifest,
            "exclusions": {
                "prefixes": [MTP_IGNORED_PREFIX],
                "tensor_count": len(excluded_keys),
                "keys_sha256": _key_digest(excluded_keys),
            },
            "output": {
                "index_file": INDEX_FILENAME,
                "index_sha256": _sha256_bytes(index_bytes),
                "shards": output_shards,
                "auxiliary_files": auxiliary_files,
                "tensor_count": stats.output_tensor_count,
                "total_tensor_bytes": stats.output_total_tensor_bytes,
                "quantized_weight_count": stats.quantized_weight_count,
                "preserved_tensor_count": stats.preserved_tensor_count,
                "generated_weight_scale_count": (stats.generated_weight_scale_count),
                "generated_input_scale_count": (stats.generated_input_scale_count),
                "generated_cache_scale_count": (stats.generated_cache_scale_count),
            },
            "streaming_bounds": {
                "hf_streaming": hf_streaming,
                "transient_hf_cache_per_shard": hf_streaming,
                "source_tensors_loaded_at_once": 1,
                "source_shards_materialized_at_once": 1,
                "configured_max_shard_bytes": max_shard_bytes,
                "maximum_buffered_output_bytes": (writer.maximum_buffered_bytes),
                "largest_source_tensor_bytes": (stats.largest_source_tensor_bytes),
                "largest_output_group_bytes": (stats.largest_output_group_bytes),
                "quantization_chunk_elements": (quantization_chunk_elements),
                "quantization_fp32_chunk_bytes": (quantization_chunk_elements * 4),
            },
            "loader_validation": {
                "loader_ready": loader_ready,
                "required_artifact_version": ARTIFACT_VERSION,
                "required_weight_scale_suffix": (STATIC_WEIGHT_SCALE_SUFFIX),
                "required_weight_scale_shape": [],
                "required_projection_input_scale_keys_sha256": (
                    _key_digest(required_input_scales)
                ),
                "required_cache_quant_multiplier_keys_sha256": (
                    _key_digest(required_cache_scales)
                ),
                "required_config_marker": "glm52_artifact.loader_ready=true",
                "ignored_prefixes": [MTP_IGNORED_PREFIX],
            },
        }
        (staging_dir / MANIFEST_FILENAME).write_bytes(_json_bytes(manifest))
        staging_dir.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-model-id", default="zai-org/GLM-5.2")
    parser.add_argument(
        "--calibration",
        "--activation-scales",
        dest="activation_scales",
        type=Path,
    )
    parser.add_argument(
        "--hf-streaming",
        action="store_true",
        help=(
            "download each missing indexed shard into an ephemeral HF cache "
            "and remove the snapshot/blob before fetching the next"
        ),
    )
    parser.add_argument("--download-staging-dir", type=Path)
    parser.add_argument("--world-size", type=_positive_int, default=64)
    parser.add_argument("--ep-degree", type=_positive_int, default=16)
    parser.add_argument(
        "--skip-index-closure",
        action="store_true",
        help="development-only: do not prove exact loader/index key closure",
    )
    parser.add_argument(
        "--max-shard-bytes",
        type=_positive_int,
        default=DEFAULT_MAX_SHARD_BYTES,
    )
    parser.add_argument(
        "--quantization-chunk-elements",
        type=_positive_int,
        default=DEFAULT_QUANTIZATION_CHUNK_ELEMENTS,
    )
    args = parser.parse_args(argv)
    manifest = convert_checkpoint(
        args.source_dir,
        args.output_dir,
        source_revision=args.source_revision,
        source_model_id=args.source_model_id,
        activation_scales_path=args.activation_scales,
        max_shard_bytes=args.max_shard_bytes,
        quantization_chunk_elements=args.quantization_chunk_elements,
        hf_streaming=args.hf_streaming,
        download_staging_dir=args.download_staging_dir,
        strict_index_closure=not args.skip_index_closure,
        world_size=args.world_size,
        ep_degree=args.ep_degree,
    )
    print(
        json.dumps(
            {
                "artifact_id": manifest["artifact_id"],
                "loader_ready": manifest["loader_validation"]["loader_ready"],
                "manifest": str(args.output_dir / MANIFEST_FILENAME),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
