#!/usr/bin/env python3
"""Emit canonical GLM-5.3-Flash TP32 rank checkpoints on the host.

This runner only drives the existing transactional rank writer.  It never
compiles, opens Neuron devices, or authorizes a card/runtime launch.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

GLM53_TP = 32
GLM53_MAX_CHUNK_BYTES = 64 * 1024 * 1024


def _install_source_namespaces(source_root: Path) -> None:
    """Import source modules without executing the vllm_neuron facade.

    The pinned NxDI image contains safetensors and torch but not the base
    ``vllm`` package.  The rank-plan modules only need their own package
    namespace, so avoid a misleading import failure at the CLI boundary.
    """

    package_root = source_root / "vllm_neuron"
    names = (
        ("vllm_neuron", package_root),
        ("vllm_neuron.model", package_root / "model"),
        ("vllm_neuron.model.glm53_flash", package_root / "model" / "glm53_flash"),
    )
    for name, path in names:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _load_streamer() -> Callable[..., dict[str, Any]]:
    source_root = Path(__file__).resolve().parents[1]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    _install_source_namespaces(source_root)
    from vllm_neuron.model.glm53_flash.rank_plan import (
        stream_glm53_rank_checkpoint,
    )

    return stream_glm53_rank_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="stream one or more canonical GLM-5.3 TP32 rank checkpoints"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rank",
        type=int,
        action="append",
        required=True,
        help="TP rank to emit; repeat for multiple ranks (0..31)",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    streamer: Callable[..., dict[str, Any]] | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    ranks = args.rank
    if len(set(ranks)) != len(ranks):
        parser.error("duplicate --rank values are refused before any write")
    if any(rank < 0 or rank >= GLM53_TP for rank in ranks):
        parser.error("--rank must be in the closed interval 0..31")

    checkpoint_dir = args.checkpoint_dir.resolve(strict=True)
    if not checkpoint_dir.is_dir():
        parser.error("--checkpoint-dir must be a directory")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        rank: output_dir / f"tp{rank}_sharded_checkpoint.safetensors" for rank in ranks
    }
    conflicts: list[str] = []
    for rank, output in outputs.items():
        manifest = output.with_suffix(output.suffix + ".manifest.json")
        partials = list(output.parent.glob(f".{output.name}.partial-*"))
        if output.exists():
            conflicts.append(f"rank {rank}: output exists ({output.name})")
        if manifest.exists():
            conflicts.append(f"rank {rank}: manifest exists ({manifest.name})")
        if partials:
            conflicts.append(f"rank {rank}: transactional partial exists")
    if conflicts:
        parser.error("; ".join(conflicts))

    emit = streamer or _load_streamer()
    rows: list[dict[str, Any]] = []
    for rank in ranks:
        output = outputs[rank]
        result = emit(
            checkpoint_dir,
            output,
            rank=rank,
            tp_degree=GLM53_TP,
            max_chunk_bytes=GLM53_MAX_CHUNK_BYTES,
        )
        rows.append(
            {
                "rank": rank,
                "checkpoint": output.name,
                "manifest": output.with_suffix(output.suffix + ".manifest.json").name,
                "writer": result,
            }
        )

    payload = {
        "schema": "glm53-canonical-rank-runner-v1",
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "tp_degree": GLM53_TP,
        "max_chunk_bytes": GLM53_MAX_CHUNK_BYTES,
        "ranks": rows,
        "claims": {
            "canonical_rank_files_emitted": True,
            "card_launch_authorized": False,
            "runtime_permitted": False,
            "correctness_40_of_40": False,
            "performance": False,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
