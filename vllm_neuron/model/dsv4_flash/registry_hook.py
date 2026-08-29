# SPDX-License-Identifier: Apache-2.0
"""External-facing registration hook for DeepSeek-V4-Flash.

Mirror of ``glm53_flash/registry_hook.py`` — same three integration
surfaces (dict, ``.register()`` method, ``.add()`` method); same
``install_dsv4_flash_sys_path`` fall-back for the immutable-container
case.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from .neuron_wrapper import NeuronDeepseekV4FlashForCausalLM

DSV4_FLASH_ARCHITECTURE = "DeepseekV4ForCausalLM"


def register_dsv4_flash(registry: Any) -> None:
    """Register the DeepSeek-V4-Flash wrapper into ``registry``.

    Idempotent: registering the same class twice is a no-op; registering
    a different class under the same key raises ``RuntimeError``.
    """
    architecture = DSV4_FLASH_ARCHITECTURE
    cls = NeuronDeepseekV4FlashForCausalLM

    if isinstance(registry, dict):
        existing = registry.get(architecture)
        if existing is cls:
            return
        if existing is not None:
            raise RuntimeError(
                f"registry[{architecture!r}] is already bound to {existing!r}; "
                f"refusing to overwrite with {cls!r}."
            )
        registry[architecture] = cls
        return

    for method_name in ("register", "add"):
        method = getattr(registry, method_name, None)
        if callable(method):
            method(architecture, cls)
            return

    raise TypeError(
        f"registry {registry!r} is neither a dict nor an object exposing "
        "`register(architecture, cls)` / `add(architecture, cls)`"
    )


def install_dsv4_flash_sys_path(worktree_root: str | os.PathLike[str]) -> None:
    """Prepend ``worktree_root`` to ``sys.path`` so imports pick the alpha tree."""
    root = os.fspath(worktree_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"worktree_root {root!r} is not a directory")
    package_dir = os.path.join(root, "vllm_neuron")
    if not os.path.isdir(package_dir):
        raise FileNotFoundError(
            f"{root!r} does not contain a `vllm_neuron` package directory"
        )
    if root not in sys.path:
        sys.path.insert(0, root)


__all__ = [
    "DSV4_FLASH_ARCHITECTURE",
    "install_dsv4_flash_sys_path",
    "register_dsv4_flash",
]
