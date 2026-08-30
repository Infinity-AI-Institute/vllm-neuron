#!/usr/bin/env python3
"""Fail-fast host-only GLM-5.3 original-target reference producer.

The loader and prompt runner are deliberately injected: this command does not
invent a CPU implementation for the 600+ GiB released checkpoint. It first
validates only the pinned checkpoint metadata, then asks the supplied provider
to load the exact snapshot and emit four prompts x ten positions of full
154880-wide rows. No Neuron device, compile, rank bundle, or card is used.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

GLM53_VOCAB_SIZE = 154_880
GLM53_PROMPTS = ("feedback-0", "feedback-1", "feedback-2", "feedback-3")
GLM53_POSITIONS = tuple(range(10))


def _install_source_namespaces(source_root: Path) -> None:
    package_root = source_root / "vllm_neuron"
    for name, path in (
        ("vllm_neuron", package_root),
        ("vllm_neuron.model", package_root / "model"),
        ("vllm_neuron.model.glm53_flash", package_root / "model" / "glm53_flash"),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module


def _load_contract():
    source_root = Path(__file__).resolve().parents[1]
    _install_source_namespaces(source_root)
    from vllm_neuron.model.glm53_flash.checkpoint_converter import (
        preflight_checkpoint_dir,
    )
    from vllm_neuron.model.glm53_flash.reference_producer import (
        Glm53OriginalTargetProducer,
        Glm53OriginalTargetProducerSpec,
    )

    return (
        preflight_checkpoint_dir,
        Glm53OriginalTargetProducer,
        Glm53OriginalTargetProducerSpec,
    )


def _callable(value: str, name: str) -> Callable[..., Any]:
    module_name, separator, attribute = value.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"{name} must be MODULE:CALLABLE")
    module = importlib.import_module(module_name)
    result = getattr(module, attribute, None)
    if not callable(result):
        raise TypeError(f"{name} is not callable: {value}")
    return result


def _loader_versions(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, version = value.partition("=")
        if not separator or not key or not version or key in result:
            raise ValueError("--loader-version must be unique KEY=VERSION entries")
        result[key] = version
    if not result:
        raise ValueError("at least one explicit --loader-version is required")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--semantics", required=True)
    parser.add_argument("--loader", required=True, help="MODULE:CALLABLE")
    parser.add_argument("--runner", required=True, help="MODULE:CALLABLE")
    parser.add_argument("--loader-version", action="append", default=[])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate pinned metadata and the 4x10 contract without loading weights",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    preflight: Callable[[str | Path], Any] | None = None,
    producer_cls: Any | None = None,
    spec_cls: Any | None = None,
) -> int:
    args = _parser().parse_args(argv)
    preflight_fn, producer_type, spec_type = _load_contract()
    preflight_fn = preflight or preflight_fn
    producer_type = producer_cls or producer_type
    spec_type = spec_cls or spec_type
    checkpoint_dir = args.checkpoint_dir.resolve(strict=True)
    if not checkpoint_dir.is_dir():
        raise ValueError("--checkpoint-dir must be a directory")
    preflight_fn(checkpoint_dir)
    versions = _loader_versions(args.loader_version)
    spec = spec_type(
        reference_id=args.reference_id,
        checkpoint_dir=checkpoint_dir,
        loader_versions=versions,
        semantics=args.semantics,
        prompt_ids=GLM53_PROMPTS,
        positions=GLM53_POSITIONS,
        vocab_size=GLM53_VOCAB_SIZE,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": "glm53-reference-producer-preflight-v1",
                    "checkpoint_dir": str(checkpoint_dir),
                    "reference_id": spec.reference_id,
                    "semantics": spec.semantics,
                    "loader_versions": dict(spec.loader_versions),
                    "vocab_size": spec.vocab_size,
                    "prompt_ids": list(spec.prompt_ids),
                    "positions": list(spec.positions),
                    "expected_rows": len(spec.expected_rows),
                    "weights_loaded": False,
                    "device_used": False,
                    "claims": {"canonical": False, "correctness_40_of_40": False},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    loader = _callable(args.loader, "--loader")
    runner = _callable(args.runner, "--runner")
    manifest = producer_type(spec).produce(
        loader=loader,
        run_prompt=runner,
        output_dir=args.output_dir,
    )
    print(json.dumps({"manifest": str(manifest), "rows": 40}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
