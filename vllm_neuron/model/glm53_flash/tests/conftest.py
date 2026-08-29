"""Let the CPU-only tests run when the optional vLLM package is absent."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

if importlib.util.find_spec("vllm") is None:
    root = Path(__file__).resolve().parents[4]
    package = ModuleType("vllm_neuron")
    package.__path__ = [str(root / "vllm_neuron")]
    model_package = ModuleType("vllm_neuron.model")
    model_package.__path__ = [str(root / "vllm_neuron" / "model")]
    sys.modules["vllm_neuron"] = package
    sys.modules["vllm_neuron.model"] = model_package
