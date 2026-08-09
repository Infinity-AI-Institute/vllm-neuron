# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

from vllm_neuron import envs
from vllm_neuron.compile.backend import (
    _apply_platform_compiler_args,
    preprocess_and_validate_inputs,
)
from vllm_neuron.fx_passes import get_default_pass_manager
from vllm_neuron.fx_passes.pass_manager import (
    _format_replica_groups_header,
)
from vllm_neuron.utils.timer import timer

logger = logging.getLogger(__name__)

_PREFLIGHT_HLO_RECEIPT_SCHEMA = 1


class CaptureComplete(Exception):
    """
    Exception to bail out after successful HLO capture.
    """

    pass


def setup_workdir_common(
    gm: torch.fx.GraphModule, example_inputs, options: dict, per_rank: bool = False
) -> tuple[str, str]:
    """Setup working directory and compute cache hash.

    Common logic shared by both the compile and capture backends.

    Args:
        gm: The PyTorch FX GraphModule
        example_inputs: Example inputs
        options: Compilation options
        per_rank: If True, append a rank-specific subdirectory (used by capture).

    Returns:
        tuple: (workdir path, cache hash key, base compile workdir)
    """
    from vllm_neuron.compile import cache

    base_workdir = options.get("compiler_workdir", envs.get_neuron_compile_cache_dir())
    base_workdir = base_workdir.rstrip("/")
    compilation_hash = cache.create_cache_hash(gm, example_inputs, options)
    workdir = f"{base_workdir}/{compilation_hash}"
    if per_rank and dist.is_initialized():
        from vllm.distributed.parallel_state import get_tp_group

        from vllm_neuron.compile.platform import get_server_prefix

        tp_group = get_tp_group()
        tp_local_rank = tp_group.rank_in_group
        prefix = get_server_prefix()
        if prefix:
            workdir = f"{workdir}/{prefix}.rank{tp_local_rank}"
        else:
            workdir = f"{workdir}/rank{tp_local_rank}"
    return workdir, compilation_hash, base_workdir


def run_fx_to_hlo_pipeline(
    gm: torch.fx.GraphModule, example_inputs, options: dict, workdir: str
):
    """Run the full FX→HLO conversion pipeline.

    Shared sequence used by both compile and capture backends:
    setup logging → run FX passes → convert to HLO.

    Args:
        gm: The PyTorch FX GraphModule
        example_inputs: Example inputs
        options: Compilation options
        workdir: Working directory for artifacts

    Returns:
        tuple: (hlo_module, unused_input_indices, has_rng_seed_parameter,
                io_map, output_count, fx_to_hlo_time)
    """
    from vllm_neuron.compile.hlo import convert_fx_to_hlo

    _setup_compilation_logging(gm, example_inputs, workdir)
    processed_gm, io_map, output_count = _run_fx_passes(gm, options, workdir)

    with timer() as fx_timer:
        hlo_module, unused_input_indices, has_rng_seed_parameter = convert_fx_to_hlo(
            processed_gm,
            example_inputs,
            log_path=f"{workdir}/hlo_passes/",
            aliasing_map=io_map,
        )
    fx_to_hlo_time = fx_timer()

    return (
        hlo_module,
        unused_input_indices,
        has_rng_seed_parameter,
        io_map,
        output_count,
        fx_to_hlo_time,
    )


def capture(gm, example_inputs, options={}):
    """Capture backend: trace FX→HLO, store HLO to cache dir, return no-op."""
    if envs.VLLM_NEURON_CPU_MODE:
        raise RuntimeError(
            "Capture backend is not compatible with VLLM_NEURON_CPU_MODE."
        )
    gm, example_inputs = preprocess_and_validate_inputs(gm, example_inputs)

    # Return no-op
    def bail(*args, **kwargs):
        raise CaptureComplete()

    # Representative-rank staging normally validates only the expensive
    # Python/FakeTensor trace. A separate default-off experiment may continue
    # the same in-process GraphModule through FX-to-HLO. It still cannot
    # publish cache metadata, compile a NEFF, or bypass the normal all-rank
    # extraction that follows the representative child.
    if envs.VLLM_NEURON_TRACE_PREFLIGHT_ONLY:
        from vllm_neuron.compile.trace_milestones import emit_trace_milestone

        rank = dist.get_rank() if dist.is_initialized() else 0
        emit_trace_milestone(
            "capture_backend_reached",
            parent_rank=rank,
            stage="preflight",
            fx_nodes=sum(1 for _ in gm.graph.nodes),
        )
        if getattr(envs, "VLLM_NEURON_TRACE_PREFLIGHT_LOWER_HLO", False):
            options = _apply_platform_compiler_args(options)
            receipt = _lower_preflight_hlo_receipt(
                gm,
                example_inputs,
                options,
                parent_rank=rank,
            )
            emit_trace_milestone(
                "preflight_hlo_persisted",
                parent_rank=rank,
                stage="preflight",
                receipt_path=receipt["receipt_path"],
                hlo_sha256=receipt["hlo"]["sha256"],
                fx_to_hlo_seconds=receipt["lowering"]["fx_to_hlo_seconds"],
            )
        return bail

    from vllm_neuron.compile import cache

    # Mirror the injection the main compile() path does before hashing, so
    # capture and forward agree on the cache key.
    options = _apply_platform_compiler_args(options)

    # FX→HLO (reuse existing pipeline)
    workdir, hash_key, base_workdir = setup_workdir_common(
        gm, example_inputs, options, per_rank=True
    )
    artifacts = cache.get_local(hash_key, base_workdir)
    if artifacts:
        logger.info(f"Local cache hit for key: {hash_key}. Skipping graph capture.")
        return bail
    os.makedirs(workdir, exist_ok=True)
    (
        hlo_module,
        unused_input_indices,
        has_rng_seed_parameter,
        io_map,
        output_count,
        fx_to_hlo_time,
    ) = run_fx_to_hlo_pipeline(gm, example_inputs, options, workdir)

    # Write graph.hlo to cache dir
    from vllm_neuron.compile.schema import create_metadata

    hlo_path = os.path.join(workdir, "graph.hlo")
    with open(hlo_path, "wb") as f:
        f.write(hlo_module.SerializeToString())
        f.flush()

    metadata = create_metadata(
        cache_key=hash_key,
        output_count=output_count,
        unused_input_indices=unused_input_indices,
        has_rng_seed_parameter=has_rng_seed_parameter,
        io_map=io_map,
    )
    cache.save_artifact_metadata(workdir, metadata)

    return bail


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _input_abi(example_inputs) -> dict[str, Any]:
    aliases: dict[int, int] = {}
    entries = []
    for index, value in enumerate(example_inputs):
        if not isinstance(value, torch.Tensor):
            entries.append(
                {
                    "index": index,
                    "kind": "python",
                    "type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "value": repr(value),
                }
            )
            continue
        try:
            storage_id = int(value.untyped_storage()._cdata)
            storage_offset = int(value.storage_offset())
        except (AttributeError, RuntimeError, TypeError) as error:
            raise RuntimeError(
                f"cannot establish preflight input alias ABI for input {index}"
            ) from error
        entries.append(
            {
                "index": index,
                "kind": "tensor",
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(value.dtype),
                "stride": [int(dim) for dim in value.stride()],
                "layout": str(value.layout),
                "device_type": value.device.type,
                "requires_grad": bool(value.requires_grad),
                "alias_group": aliases.setdefault(storage_id, len(aliases)),
                "storage_offset": storage_offset,
            }
        )
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "entries": entries,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "tensor_payload_values_included": False,
    }


def _write_bytes_atomic(path: str, payload: bytes) -> None:
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: str, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    _write_bytes_atomic(path, payload)


def _lower_preflight_hlo_receipt(
    gm: torch.fx.GraphModule,
    example_inputs,
    options: dict,
    *,
    parent_rank: int,
) -> dict[str, Any]:
    """Lower the live preflight FX graph without publishing runtime artifacts."""

    receipt_root = getattr(
        envs,
        "VLLM_NEURON_TRACE_PREFLIGHT_HLO_RECEIPT_DIR",
        None,
    )
    if not receipt_root:
        raise RuntimeError(
            "VLLM_NEURON_TRACE_PREFLIGHT_HLO_RECEIPT_DIR is required when "
            "VLLM_NEURON_TRACE_PREFLIGHT_LOWER_HLO=1"
        )
    receipt_root = os.path.abspath(os.path.expanduser(receipt_root))
    os.makedirs(receipt_root, mode=0o700, exist_ok=True)
    if os.path.islink(receipt_root) or not os.path.isdir(receipt_root):
        raise RuntimeError(f"unsafe preflight HLO receipt directory: {receipt_root}")

    workdir = tempfile.mkdtemp(
        prefix=f"rank-{parent_rank}-pid-{os.getpid()}-",
        dir=receipt_root,
    )
    fx_text = str(gm.graph)
    fx_payload = fx_text.encode()
    header = _format_replica_groups_header(gm) or ""
    input_abi = _input_abi(example_inputs)
    started_ns = time.time_ns()
    (
        hlo_module,
        unused_input_indices,
        has_rng_seed_parameter,
        io_map,
        output_count,
        fx_to_hlo_time,
    ) = run_fx_to_hlo_pipeline(gm, example_inputs, options, workdir)
    hlo_payload = hlo_module.SerializeToString()
    hlo_path = os.path.join(workdir, "graph.hlo")
    _write_bytes_atomic(hlo_path, hlo_payload)

    receipt_path = os.path.join(workdir, "receipt.json")
    receipt = {
        "schema_version": _PREFLIGHT_HLO_RECEIPT_SCHEMA,
        "status": "complete",
        "artifact_scope": "diagnostic_preflight_hlo",
        "receipt_path": receipt_path,
        "workdir": workdir,
        "parent_rank": parent_rank,
        "pid": os.getpid(),
        "started_wall_time_ns": started_ns,
        "completed_wall_time_ns": time.time_ns(),
        "same_process_fx_to_hlo": True,
        "cross_process_fx_handoff": False,
        "normal_all_rank_extraction_still_required": True,
        "cache_key": None,
        "cache_lookup_performed": False,
        "cache_metadata_written": False,
        "cache_published": False,
        "neff_written": False,
        "runtime_bypass_enabled": False,
        "fx": {
            "node_count": sum(1 for _ in gm.graph.nodes),
            "sha256": hashlib.sha256(fx_payload).hexdigest(),
            "replica_groups_header": header,
            "replica_groups_header_sha256": hashlib.sha256(
                header.encode()
            ).hexdigest(),
        },
        "inputs": input_abi,
        "hlo": {
            "path": hlo_path,
            "bytes": len(hlo_payload),
            "sha256": hashlib.sha256(hlo_payload).hexdigest(),
        },
        "lowering": {
            "fx_to_hlo_seconds": fx_to_hlo_time,
            "unused_input_indices": _json_safe(unused_input_indices),
            "has_rng_seed_parameter": bool(has_rng_seed_parameter),
            "io_map": _json_safe(io_map),
            "output_count": output_count,
        },
    }
    _write_json_atomic(receipt_path, receipt)
    return receipt


def _run_fx_passes(
    gm: torch.fx.GraphModule, options: dict, workdir: str
) -> Tuple[torch.fx.GraphModule, Optional[Dict[str, Any]]]:
    """Run FX passes and return results.

    Args:
        gm: The PyTorch FX GraphModule
        options: Compilation options
        workdir: Working directory

    Returns:
        tuple: (processed_gm, io_map) tuple containing processed GraphModule and I/O mapping
    """
    pass_manager = get_default_pass_manager()
    gm, pass_metadata = pass_manager.run_passes(
        gm,
        target_device=options.get("target_device", "xla"),
        compiler_workdir=workdir,
    )
    gm.recompile()

    io_map = None
    if "aliasing_output_rewrite" in pass_metadata:
        io_map = pass_metadata["aliasing_output_rewrite"]["io_map"]
        output_count = pass_metadata["aliasing_output_rewrite"]["original_output_count"]

    # Log pass metadata for debugging
    logger.info(f"FX Pass metadata: {pass_metadata}")

    return gm, io_map, output_count


def _setup_compilation_logging(gm: torch.fx.GraphModule, example_inputs, workdir: str):
    """Setup logging and metadata files for compilation.

    Args:
        gm: The PyTorch FX GraphModule
        example_inputs: Example inputs
        workdir: Working directory for logs
    """
    os.makedirs(os.path.join(workdir, "hlo_passes"), exist_ok=True)

    # Log FX graph
    fx_filename = os.path.join(workdir, "fxgraph.txt")
    with open(fx_filename, "w") as f:
        header = _format_replica_groups_header(gm)
        if header:
            f.write(header + "\n")
        print(gm.graph, file=f)
    logger.info(f"FX graph - {fx_filename}")

    # Log example inputs metadata
    input_metadata = []
    for i, inp in enumerate(example_inputs):
        if hasattr(inp, "shape") and hasattr(inp, "dtype"):
            input_metadata.append(
                {
                    "index": i,
                    "shape": tuple(inp.shape),
                    "dtype": str(inp.dtype).split(".")[-1],
                    "device": str(inp.device),
                }
            )

    inputs_filename = os.path.join(workdir, "example_inputs.txt")
    with open(inputs_filename, "w") as f:
        f.write("Example Inputs Metadata\n")
        f.write("=" * 50 + "\n\n")
        for meta in input_metadata:
            f.write(f"Input {meta['index']}:\n")
            f.write(f"  Shape: {meta['shape']}\n")
            f.write(f"  Dtype: {meta['dtype']}\n")
            f.write(f"  Device: {meta['device']}\n\n")
    logger.info(f"Example inputs - {inputs_filename}")
