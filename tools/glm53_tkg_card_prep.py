#!/usr/bin/env python3
"""Emit a host-only GLM-5.3 TKG card-preparation receipt.

This command never compiles, opens Neuron devices, writes checkpoint shards, or
changes the supplied artifact.  It is intended to be run inside the pinned
NxDI/vLLM-Neuron image with the checkpoint mounted read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_neuron.model.glm53_flash.card_prep import inspect_tkg_artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--rank-dir", type=Path)
    parser.add_argument("--cte-artifact-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = inspect_tkg_artifact(
        args.artifact_root,
        checkpoint_dir=args.checkpoint_dir,
        rank_dir=args.rank_dir,
        cte_artifact_root=args.cte_artifact_root,
    )
    payload = receipt.to_mapping()
    payload["receipt_sha256"] = receipt.sha256()
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded)
    print(encoded.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
