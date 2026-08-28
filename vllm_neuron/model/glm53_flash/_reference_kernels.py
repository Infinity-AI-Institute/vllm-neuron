# SPDX-License-Identifier: Apache-2.0
"""Lazy imports for the session-qualified GLM-5.3 CPU golden kernels.

The goldens deliberately remain owned by the handoff bundle.  Keeping this
loader tiny avoids creating a second, drifting implementation in this tree.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

_KERNEL_FILES = {
    "dsa": "dsa_lightning_indexer.py",
    "kda": "kda_state_v2.py",
    "moe": "moe_dispatch.py",
    "fp8": "glm52_indexer_fp8_scale_fix.py",
}


def _kernel_dir() -> Path:
    override = os.environ.get("GLM53_REFERENCE_KERNEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    repo = Path(__file__).resolve().parents[3]
    return (
        repo.parent
        / "gemma4-trn2-handoff"
        / "harness-v2"
        / "staging"
        / "reference-sweep-20260826T2150Z"
        / "kernels"
    )


@lru_cache(maxsize=len(_KERNEL_FILES))
def load_reference_kernel(name: str) -> ModuleType:
    """Import one canonical golden by path, once, with an actionable error."""
    if name not in _KERNEL_FILES:
        raise KeyError(f"unknown GLM-5.3 reference kernel: {name}")
    path = _kernel_dir() / _KERNEL_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(
            f"missing GLM-5.3 {name} golden at {path}; set "
            "GLM53_REFERENCE_KERNEL_DIR to the canonical kernels directory"
        )
    return _load_module(path, f"_glm53_reference_{name}")


@lru_cache(maxsize=1)
def load_glm52_moe() -> ModuleType:
    """Load the reusable 5.2 router without importing its package facade."""
    repo = Path(__file__).resolve().parents[3]
    path = repo / "vllm_neuron" / "model" / "glm52_moe_dsa" / "moe.py"
    return _load_module(path, "_glm53_reused_glm52_moe")


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["load_glm52_moe", "load_reference_kernel"]
