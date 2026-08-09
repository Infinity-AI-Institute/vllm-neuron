"""Load NKI cache modules on development hosts without Neuron/vLLM wheels."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str, path: Path | None = None) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = [] if path is None else [str(path)]
    sys.modules[name] = module
    return module


def _install_local_stubs() -> None:
    vllm_neuron = _package("vllm_neuron", ROOT / "vllm_neuron")
    envs = _load("vllm_neuron.envs", ROOT / "vllm_neuron" / "envs.py")
    vllm_neuron.envs = envs

    _package("vllm_neuron.compile", ROOT / "vllm_neuron" / "compile")
    _load(
        "vllm_neuron.compile.platform",
        ROOT / "vllm_neuron" / "compile" / "platform.py",
    )
    _package("vllm_neuron.nki", ROOT / "vllm_neuron" / "nki")

    nki = _package("nki")
    nki._version = type("Version", (), {"__version__": "test"})()
    nki_dtype = ModuleType("nki.dtype")
    nki_dtype.float8_e4m3 = "float8_e4m3fn"
    sys.modules["nki.dtype"] = nki_dtype
    nki.dtype = nki_dtype

    language = _package("nki.language")
    nki.language = language
    buffers = ModuleType("nki.language.buffers")
    buffers.shared_hbm = object()
    sys.modules["nki.language.buffers"] = buffers
    tensor = ModuleType("nki.language.tensor")

    class NkiTensor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    tensor.NkiTensor = NkiTensor
    sys.modules["nki.language.tensor"] = tensor

    framework = _package("nki.framework")
    nki.framework = framework
    compiled = ModuleType("nki.framework.compiled")

    class CompileKernel:
        pass

    compiled.CompileKernel = CompileKernel
    sys.modules["nki.framework.compiled"] = compiled

    compiler = _package("nki.compiler")
    nki.compiler = compiler
    frontend = ModuleType("nki.compiler.frontend")

    class TracerFrontend:
        def __init__(self, enable_backend_opt=False):
            self.enable_backend_opt = enable_backend_opt

    frontend.TracerFrontend = TracerFrontend
    sys.modules["nki.compiler.frontend"] = frontend
    compiler.frontend = frontend


def load_nki_modules():
    loaded = tuple(
        sys.modules.get(name)
        for name in (
            "vllm_neuron.nki.nki_cache",
            "vllm_neuron.nki.nki_compile",
            "vllm_neuron.nki.nki_dtype",
        )
    )
    if all(loaded):
        return loaded

    try:
        importlib.import_module("vllm")
        importlib.import_module("nki")
    except ModuleNotFoundError:
        _install_local_stubs()

    nki_dtype = _load(
        "vllm_neuron.nki.nki_dtype",
        ROOT / "vllm_neuron" / "nki" / "nki_dtype.py",
    )
    nki_compile = _load(
        "vllm_neuron.nki.nki_compile",
        ROOT / "vllm_neuron" / "nki" / "nki_compile.py",
    )
    nki_cache = _load(
        "vllm_neuron.nki.nki_cache",
        ROOT / "vllm_neuron" / "nki" / "nki_cache.py",
    )
    return nki_cache, nki_compile, nki_dtype
