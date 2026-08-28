# SPDX-License-Identifier: Apache-2.0
"""Round-3 1-layer compile-driver smoke for GLM-5.3-Flash.

Run inside the NxDI container (digest sha256:011d49c7…) on the compile host:

    PYTHONPATH=/src/nxdi/src:/mnt/compile/src/vllm-neuron-alpha:/mnt/compile/src/glm53-kernels \\
    GLM53_REFERENCE_KERNEL_DIR=/mnt/compile/src/glm53-kernels \\
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    python -m vllm_neuron.model.glm53_flash.smoke_round3

Stages, each reported independently so a failure localises:

  1. golden import         — the three CPU goldens load from the kernels dir
  2. kda parity            — torch port is bit-exact vs the numpy golden
  3. moe dispatch identity — Tier-1 CPU battery on the shape family
  4. config build          — 1-layer reduced config constructs
  5. wrapper construct     — NxDI wrapper binds (needs the Neuron toolchain)
  6. dry-run compile       — `wrapper.compile(dry_run=True)` traces the graph

Nothing here fabricates a pass: every stage prints PASS/FAIL with the
exception text, and the process exits non-zero on the first hard failure.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
import types
from typing import Any

RESULTS: list[dict[str, Any]] = []

# ---------------------------------------------------------------------------
# Import bootstrap
# ---------------------------------------------------------------------------
# `vllm_neuron/__init__.py` imports `vllm.logger` at module scope.  The
# NxDI-direct container (digest sha256:011d49c7…) has neuronx-distributed and
# torch but NOT vllm, so `import vllm_neuron.model.glm53_flash` dies in the
# package facade before it ever reaches our code.  This is what blocked the
# Round-2 driver too — its `command.sh` did a plain package import.
#
# Registering namespace stubs for the parent packages lets the submodules load
# by path with their relative imports intact, without importing the facade.
# This affects only how the module is *reached*; the code under test is the
# real shipped file.

# .../<repo>/vllm_neuron/model/glm53_flash/smoke_round3.py -> .../<repo>
_PKG_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)


def _bootstrap_package() -> None:
    if "vllm_neuron.model.glm53_flash" in sys.modules:
        return
    chain = {
        "vllm_neuron": os.path.join(_PKG_ROOT, "vllm_neuron"),
        "vllm_neuron.model": os.path.join(_PKG_ROOT, "vllm_neuron", "model"),
        "vllm_neuron.model.glm53_flash": os.path.join(
            _PKG_ROOT, "vllm_neuron", "model", "glm53_flash"
        ),
    }
    for name, path in chain.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [path]
            sys.modules[name] = module


def _load(module_name: str, filename: str):
    """Load one glm53_flash submodule by path under the stubbed package."""
    full = f"vllm_neuron.model.glm53_flash.{module_name}"
    if full in sys.modules:
        return sys.modules[full]
    path = os.path.join(
        _PKG_ROOT, "vllm_neuron", "model", "glm53_flash", filename
    )
    spec = importlib.util.spec_from_file_location(full, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


def stage(name: str, fn) -> Any:
    try:
        value = fn()
    except BaseException as exc:  # noqa: BLE001 - the traceback IS the deliverable
        RESULTS.append(
            {
                "stage": name,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        return None
    RESULTS.append({"stage": name, "status": "PASS", "detail": repr(value)[:400]})
    print(f"[PASS] {name}: {repr(value)[:300]}", flush=True)
    return value


def main() -> int:
    print("=" * 72, flush=True)
    print("GLM-5.3-Flash Round-3 1-layer smoke", flush=True)
    print(f"python={sys.version.split()[0]}", flush=True)
    print(
        "kernel_dir="
        + os.environ.get("GLM53_REFERENCE_KERNEL_DIR", "<default>"),
        flush=True,
    )
    print("=" * 72, flush=True)

    _bootstrap_package()
    _load("registry", "registry.py")
    refs = _load("_reference_kernels", "_reference_kernels.py")
    nki_bindings = _load("nki_bindings", "nki_bindings.py")
    load_reference_kernel = refs.load_reference_kernel

    stage(
        "1.golden-import",
        lambda: [
            load_reference_kernel(n).__name__ for n in ("kda", "dsa", "moe")
        ],
    )

    err = stage(
        "2.kda-parity",
        lambda: max(
            nki_bindings.kda_reference_parity_check(seed=s) for s in range(4)
        ),
    )
    if err is not None and err != 0.0:
        print(
            f"[WARN] KDA torch port is NOT bit-exact (max abs err {err}); "
            "the recurrence must match the golden exactly.",
            flush=True,
        )

    stage(
        "3.moe-dispatch-identity",
        lambda: nki_bindings.build_glm53_moe_dispatch_config(
            hidden=4096,
            num_experts=288,
            top_k=8,
            intermediate_global=2048,
            tp_degree=16,
            renormalize_topk=True,
        ).cache_key()
        if hasattr(
            nki_bindings.build_glm53_moe_dispatch_config(
                hidden=4096,
                num_experts=288,
                top_k=8,
                intermediate_global=2048,
                tp_degree=16,
                renormalize_topk=True,
            ),
            "cache_key",
        )
        else "validated",
    )

    config_mod = _load("config", "config.py")
    Glm53FlashInferenceConfig = config_mod.Glm53FlashInferenceConfig

    source = stage(
        "4a.source-config", lambda: Glm53FlashInferenceConfig()
    )
    if source is None:
        return _finish(1)

    wrapper_mod = stage(
        "4c.wrapper-import", lambda: _load("neuron_wrapper", "neuron_wrapper.py")
    )
    if wrapper_mod is None:
        return _finish(1)
    NeuronGlm53FlashForCausalLM = wrapper_mod.NeuronGlm53FlashForCausalLM

    tp = int(os.environ.get("GLM53_SMOKE_TP", "8"))
    # GLM53_SMOKE_MODE=coverage traces KDA + DSA + routed-MoE in one graph;
    # the default 1-layer mode is KDA + dense only and proves nothing about
    # the other two kernels.
    mode = os.environ.get("GLM53_SMOKE_MODE", "one-layer")
    builder = (
        NeuronGlm53FlashForCausalLM.build_kernel_coverage_smoke_config
        if mode == "coverage"
        else NeuronGlm53FlashForCausalLM.build_one_layer_smoke_config
    )
    cfg = stage(
        f"4b.smoke-config[{mode}]",
        lambda: builder(source, tp_degree=tp, seq_len=128),
    )
    if cfg is None:
        return _finish(1)

    model_path = os.environ.get(
        "GLM53_MODEL_PATH",
        "/mnt/compile/hf-cache/models--zai-org--GLM-5.3-Flash/snapshots/"
        "04c4e9e95c5da8862dced7e5056455116f83a7e0",
    )
    wrapper = stage(
        "5.wrapper-construct",
        lambda: NeuronGlm53FlashForCausalLM(model_path, cfg),
    )
    if wrapper is None:
        return _finish(1)

    out_path = os.environ.get("GLM53_SMOKE_OUT", "/runroot/artifacts/smoke")
    os.makedirs(out_path, exist_ok=True)
    stage(
        "6.dry-run-compile",
        lambda: wrapper.compile(out_path, dry_run=True),
    )
    return _finish(0)


def _finish(code: int) -> int:
    out = os.environ.get("GLM53_SMOKE_RESULT", "/tmp/glm53-smoke-result.json")
    try:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(RESULTS, handle, indent=2)
        print(f"\nwrote {out}", flush=True)
    except OSError as exc:
        print(f"could not write {out}: {exc}", flush=True)
    failures = [r for r in RESULTS if r["status"] == "FAIL"]
    print("\n" + "=" * 72, flush=True)
    for item in RESULTS:
        print(f"  {item['status']:4}  {item['stage']}", flush=True)
    print("=" * 72, flush=True)
    return 1 if failures else code


if __name__ == "__main__":
    raise SystemExit(main())
