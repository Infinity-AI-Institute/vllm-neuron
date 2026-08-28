# SPDX-License-Identifier: Apache-2.0
"""External-facing registration hook for GLM-5.3-Flash.

NxDI itself has no `MODEL_REGISTRY` — a caller picks a `Neuron{X}ForCausalLM`
class directly (`NeuronQwen3ForCausalLM(model_path, config).compile(...)`).
The vllm-neuron side wires the HF architecture id (`Glm5NextForConditionalGeneration`)
to a specific class via `vllm_neuron.model.registry.get_models()`; this
package's `.registry.get_models()` returns the wrapper for that dispatch, and
the parent module already collects our entry via
`from .glm53_flash import NeuronGlm53FlashForCausalLM`.

This module exposes two additional integration hooks so out-of-tree callers
(vLLM launcher shims, alternative registries, integration-test harnesses)
can pin the wrapper without importing the whole package:

- `register_glm53_flash(registry)` — merge the GLM-5.3-Flash entry into an
  external mapping (e.g. a dict, or any object with a
  `register(architecture, cls)` method).  Idempotent.
- `install_glm53_flash_sys_path(worktree_root)` — the fallback shim
  documented in the wrapper prompt: if a container-level registration hook
  cannot be modified inside the immutable image, insert the alpha worktree
  at `sys.path[0]` so the `vllm_neuron.model.glm53_flash` package resolves
  from the mounted source tree instead of from any pre-installed copy.
  Safe to call multiple times.

The registration surface is deliberately minimal — the wrapper class is the
single source of truth for compile behaviour; this module only helps the
launcher find it.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from .neuron_wrapper import NeuronGlm53FlashForCausalLM

GLM53_FLASH_ARCHITECTURE = "Glm5NextForConditionalGeneration"


def register_glm53_flash(registry: Any) -> None:
    """Register the GLM-5.3-Flash wrapper into ``registry``.

    Supported shapes:
      * A plain dict: ``registry[GLM53_FLASH_ARCHITECTURE] = cls``.
      * An object exposing ``register(architecture, cls)``.
      * An object exposing ``add(architecture, cls)``.

    Idempotent: registering the same class twice is a no-op; registering a
    different class under the same key raises ``RuntimeError`` so accidental
    downgrade from wrapper to reference impl is loud.
    """
    architecture = GLM53_FLASH_ARCHITECTURE
    cls = NeuronGlm53FlashForCausalLM

    if isinstance(registry, dict):
        existing = registry.get(architecture)
        if existing is cls:
            return
        if existing is not None:
            raise RuntimeError(
                f"registry[{architecture!r}] is already bound to {existing!r}; "
                f"refusing to overwrite with {cls!r}. Remove the prior entry "
                "explicitly if the swap is intentional."
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


def install_glm53_flash_sys_path(worktree_root: str | os.PathLike[str]) -> None:
    """Prepend ``worktree_root`` to ``sys.path`` so imports pick alpha's tree.

    Documented fall-back for the container-level registration case: when the
    NxDI container is immutable and no registration hook can be pushed
    inside, the launcher can mount the alpha worktree read-only into the
    container and call this function from a launcher shim (or a python
    inline `-c 'import ...; install_glm53_flash_sys_path(...)'` command)
    before instantiating any GLM-5.3-Flash class.
    """
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
    "GLM53_FLASH_ARCHITECTURE",
    "install_glm53_flash_sys_path",
    "register_glm53_flash",
]
