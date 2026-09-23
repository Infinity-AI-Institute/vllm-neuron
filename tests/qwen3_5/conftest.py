# SPDX-License-Identifier: Apache-2.0
"""Let the pure-torch Qwen3.5 modules import on a machine without vLLM.

``vllm_neuron/__init__.py`` imports ``vllm`` at package import time. The Gated
DeltaNet math (``gated_deltanet.py``) and state (``state.py``) import only
torch, so on a CPU development machine without vLLM this registers bare
package shells pointing at the real directories. On the Neuron host, where
vLLM is installed, nothing is stubbed and the real package loads.
"""

import importlib.util
import sys
import types
from pathlib import Path

if importlib.util.find_spec("vllm") is None:
    _root = Path(__file__).resolve().parents[2] / "vllm_neuron"
    for _name, _path in (
        ("vllm_neuron", _root),
        ("vllm_neuron.model", _root / "model"),
        ("vllm_neuron.model.qwen3_5", _root / "model" / "qwen3_5"),
    ):
        if _name not in sys.modules:
            _shell = types.ModuleType(_name)
            _shell.__path__ = [str(_path)]
            sys.modules[_name] = _shell
