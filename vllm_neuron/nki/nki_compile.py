# SPDX-License-Identifier: Apache-2.0
"""NKI kernel compilation bridge.

Provides compile_nki() which compiles NKI kernels via CompileKernel
and returns the fields the HOP infrastructure needs.
"""

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch
from nki.framework.compiled import CompileKernel
from nki.language.buffers import shared_hbm
from nki.language.tensor import NkiTensor

from vllm_neuron import envs

from ..compile.platform import get_platform_target
from .nki_dtype import nki_dtype_to_torch, torch_to_nki_dtype

logger = logging.getLogger(__name__)

_NKI_EVENT_PREFIX = "NKI_COMPILE_EVENT "


def _log_nki_event(
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit one machine-readable NKI compile/cache event.

    Parallel trace children already prefix log records with their actual trace
    rank and lane. Keeping that existing mechanism avoids introducing a second
    source of process context while the PID and cache key make standalone
    records attributable outside the trace pool.
    """
    if not logger.isEnabledFor(level):
        return
    payload = {"event": event, "pid": os.getpid(), **fields}
    logger.log(
        level,
        "%s%s",
        _NKI_EVENT_PREFIX,
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )


@dataclass
class NKICompileResult:
    """Compilation output consumed by the HOP dispatch implementations."""

    dumped_config: str
    return_types: tuple[tuple[torch.dtype, tuple[int, ...]], ...]
    operand_output_aliases: dict[int, int]


def compile_nki(
    func: Callable,
    args: dict[str, Any],
    grid: tuple[int, ...],
) -> NKICompileResult:
    """Compile an NKI kernel via CompileKernel.

    Results are persisted to the vLLM Neuron compile cache so that
    warm starts skip the expensive BIR compilation step.

    Args:
        func: Raw NKI kernel function.
        args: Parameter name → value (torch.Tensor or scalar).
        grid: LNC grid, e.g. (2,).

    Returns:
        NKICompileResult with everything the HOP needs.
    """
    from .nki_cache import compile_with_cache, create_nki_cache_key

    cache_key = create_nki_cache_key(func, args, grid)

    def _do_compile() -> NKICompileResult:
        lnc = grid[0] if grid else 1
        kernel = CompileKernel(func=func, lnc=lnc, target=get_platform_target())
        inputs = {name: _convert_input(v, name) for name, v in args.items()}

        def compile_to_nir(active_kernel: CompileKernel):
            compile_opts = active_kernel._compile_opts()
            if envs.VLLM_NEURON_KERNEL_DEVICE_DUMP:
                # Runtime env vars (NEURON_RT_DEBUG_OUTPUT_DIR,
                # NEURON_RT_DEBUG_SAVE_BINARY) are set in executor.py.
                from nki.compiler.frontend import TracerFrontend

                compile_opts = replace(compile_opts, enable_device_dump=True)
                frontend = TracerFrontend(
                    enable_backend_opt=active_kernel._enable_backend_opt
                )
            else:
                frontend = active_kernel._frontend_cls(
                    enable_backend_opt=active_kernel._enable_backend_opt,
                )
            frontend_name = type(frontend).__name__
            started = time.perf_counter()
            try:
                nir = active_kernel._cached_compile_to_bir(
                    frontend=frontend,
                    inputs=inputs,
                    compile_opts=compile_opts,
                )
            except Exception as error:
                _log_nki_event(
                    "frontend_compile",
                    level=logging.WARNING,
                    cache_key=cache_key,
                    frontend=frontend_name,
                    outcome="error",
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                    error_type=type(error).__name__,
                )
                raise
            _log_nki_event(
                "frontend_compile",
                cache_key=cache_key,
                frontend=frontend_name,
                outcome="success",
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            return nir

        try:
            nir = compile_to_nir(kernel)
        except RuntimeError as error:
            # ParserFrontend serializes called Python helpers as nested defs.
            # TracerFrontend executes those helpers while constructing NKI IR
            # and is the supported route for such kernels. Keep this fallback
            # diagnostic-specific so unrelated parser failures remain fatal.
            if "inner function definitions" not in str(error):
                raise
            from nki.compiler.frontend import TracerFrontend

            fallback_reason = "parser_inner_function_definitions"
            logger.warning(
                "NKI parser rejected helper functions; retrying with tracer frontend"
            )
            kernel = replace(kernel, _frontend_cls=TracerFrontend)
            fallback_started = time.perf_counter()
            try:
                nir = compile_to_nir(kernel)
            except Exception as fallback_error:
                _log_nki_event(
                    "tracer_fallback",
                    level=logging.WARNING,
                    cache_key=cache_key,
                    reason=fallback_reason,
                    outcome="error",
                    duration_ms=round(
                        (time.perf_counter() - fallback_started) * 1000, 3
                    ),
                    error_type=type(fallback_error).__name__,
                )
                raise
            _log_nki_event(
                "tracer_fallback",
                cache_key=cache_key,
                reason=fallback_reason,
                outcome="success",
                duration_ms=round((time.perf_counter() - fallback_started) * 1000, 3),
            )

        config = nir.build_config()

        return NKICompileResult(
            dumped_config=config.backend_config_b64.decode("ascii"),
            return_types=tuple(
                (nki_dtype_to_torch(s.dtype), tuple(s.shape))
                for s in config.output_specs
            ),
            operand_output_aliases=config.operand_output_aliases,
        )

    return compile_with_cache(cache_key, _do_compile)


def _convert_input(x: Any, name: str) -> Any:
    if isinstance(x, torch.Tensor):
        # convert to NkiTensor
        return NkiTensor(
            name=name,
            shape=x.shape,
            dtype=torch_to_nki_dtype(x.dtype),
            storage=None,
            buffer=shared_hbm,
        )

    return x
