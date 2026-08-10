# SPDX-License-Identifier: Apache-2.0
"""Bind native libtorch capture to source-overlay FX passes."""

from collections.abc import Callable
from types import ModuleType
from typing import Any


def bind_source_pass_manager_to_native_capture(
    native_capture_backend: ModuleType | Any | None = None,
    pass_manager_factory: Callable[[], Any] | None = None,
    lookup_backend: Callable[[str], Callable[..., Any]] | None = None,
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

    if lookup_backend is None:
        import torch._dynamo.backends.registry as registry

        lookup_backend = registry.lookup_backend

    expected_capture = getattr(native_capture_backend, "capture", None)
    if not callable(expected_capture):
        raise RuntimeError("native capture backend does not expose capture")

    compatibility_capture = lookup_backend("vllm_neuron_graph_capture")
    native_capture = lookup_backend("neuron_libtorch_graph_capture")
    if compatibility_capture is not expected_capture:
        raise RuntimeError(
            "vllm_neuron_graph_capture is not bound to the native capture backend"
        )
    if native_capture is not expected_capture:
        raise RuntimeError(
            "neuron_libtorch_graph_capture is not bound to the native capture backend"
        )

    capture_globals = getattr(expected_capture, "__globals__", None)
    if capture_globals is not native_capture_backend.__dict__:
        raise RuntimeError("native capture callable has unexpected module globals")
    if "get_default_pass_manager" not in capture_globals:
        raise RuntimeError(
            "native capture backend does not expose get_default_pass_manager"
        )

    capture_globals["get_default_pass_manager"] = pass_manager_factory

    if (
        native_capture_backend.get_default_pass_manager is not pass_manager_factory
        or expected_capture.__globals__["get_default_pass_manager"]
        is not pass_manager_factory
    ):
        raise RuntimeError("failed to bind source FX pass manager to native capture")
