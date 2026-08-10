# SPDX-License-Identifier: Apache-2.0
"""Bind native libtorch capture to source-overlay graph transformations."""

import logging
from collections.abc import Callable
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)


def bind_source_pass_manager_to_native_capture(
    native_capture_backend: ModuleType | Any | None = None,
    native_hlo_module: ModuleType | Any | None = None,
    pass_manager_factory: Callable[[], Any] | None = None,
    source_hlo_converter: Callable[..., Any] | None = None,
    lookup_backend: Callable[[str], Callable[..., Any]] | None = None,
) -> None:
    """Make native capture use source-overlay FX passes and FX-to-HLO.

    Native mode intentionally keeps the image-bundled libtorch compile and
    capture backends.  The latter imports ``get_default_pass_manager`` into a
    module global, however, so it would otherwise bypass graph-affecting fixes
    delivered by the vLLM-Neuron source overlay.  Its FX-to-HLO pipeline also
    imports the bundled converter dynamically; rebinding only the pass manager
    leaves the bundled sync-before-clear PJRT reset active.

    Optional arguments keep the import-order mutation independently testable.
    """
    if native_capture_backend is None:
        from libtorch_neuronx_lite.compile import (
            capture_backend as native_capture_backend,
        )

    if native_hlo_module is None:
        from libtorch_neuronx_lite.compile import hlo as native_hlo_module

    if pass_manager_factory is None:
        from vllm_neuron.fx_passes import get_default_pass_manager

        pass_manager_factory = get_default_pass_manager

    if source_hlo_converter is None:
        from vllm_neuron.compile.hlo import convert_fx_to_hlo

        source_hlo_converter = convert_fx_to_hlo

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

    bundled_hlo_converter = getattr(native_hlo_module, "convert_fx_to_hlo", None)
    if not callable(bundled_hlo_converter):
        raise RuntimeError("native HLO module does not expose convert_fx_to_hlo")
    if not callable(source_hlo_converter):
        raise RuntimeError("source HLO converter is not callable")

    capture_globals["get_default_pass_manager"] = pass_manager_factory
    native_hlo_module.convert_fx_to_hlo = source_hlo_converter

    if (
        native_capture_backend.get_default_pass_manager is not pass_manager_factory
        or expected_capture.__globals__["get_default_pass_manager"]
        is not pass_manager_factory
    ):
        raise RuntimeError("failed to bind source FX pass manager to native capture")
    if native_hlo_module.convert_fx_to_hlo is not source_hlo_converter:
        raise RuntimeError(
            "failed to bind source FX-to-HLO converter to native capture"
        )

    capture_file = getattr(
        getattr(expected_capture, "__code__", None), "co_filename", None
    )
    hlo_file = getattr(
        getattr(source_hlo_converter, "__code__", None), "co_filename", None
    )
    logger.info(
        "Native capture bridge active: capture=%s fx_to_hlo=%s",
        capture_file or "<native>",
        hlo_file or "<native>",
    )
