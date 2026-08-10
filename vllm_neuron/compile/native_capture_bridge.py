# SPDX-License-Identifier: Apache-2.0
"""Bind native libtorch capture to source-overlay FX passes."""

from collections.abc import Callable
from types import ModuleType
from typing import Any


def bind_source_pass_manager_to_native_capture(
    native_capture_backend: ModuleType | Any | None = None,
    pass_manager_factory: Callable[[], Any] | None = None,
) -> None:
    """Make the registered native capture backend use source-overlay passes.

    Native mode intentionally keeps the image-bundled libtorch compile and
    capture backends.  The latter imports ``get_default_pass_manager`` into a
    module global, however, so it would otherwise bypass graph-affecting fixes
    delivered by the vLLM-Neuron source overlay.

    Optional arguments keep the import-order mutation independently testable.
    """
    if native_capture_backend is None:
        from libtorch_neuronx_lite.compile import (
            capture_backend as native_capture_backend,
        )

    if pass_manager_factory is None:
        from vllm_neuron.fx_passes import get_default_pass_manager

        pass_manager_factory = get_default_pass_manager

    if not hasattr(native_capture_backend, "get_default_pass_manager"):
        raise RuntimeError(
            "native capture backend does not expose get_default_pass_manager"
        )

    native_capture_backend.get_default_pass_manager = pass_manager_factory

    if native_capture_backend.get_default_pass_manager is not pass_manager_factory:
        raise RuntimeError("failed to bind source FX pass manager to native capture")
