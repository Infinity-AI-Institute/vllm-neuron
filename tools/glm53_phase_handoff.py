#!/usr/bin/env python3
"""Verify the serialized GLM TKG/CTE weight and hybrid-state handoff."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path


def _load_verifier():
    root = Path(__file__).resolve().parents[1]
    package_root = root / "vllm_neuron"
    for name, path in (
        ("vllm_neuron", package_root),
        ("vllm_neuron.model", package_root / "model"),
        ("vllm_neuron.model.glm53_flash", package_root / "model" / "glm53_flash"),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    from vllm_neuron.model.glm53_flash.phase_handoff import inspect_phase_handoff

    return inspect_phase_handoff


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tkg-artifact-root", type=Path, required=True)
    parser.add_argument("--cte-artifact-root", type=Path, required=True)
    parser.add_argument("--compose-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    inspect = _load_verifier()
    receipt = inspect(
        args.tkg_artifact_root,
        args.cte_artifact_root,
        compose_receipt_path=args.compose_receipt,
    )
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded)
    print(encoded.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
