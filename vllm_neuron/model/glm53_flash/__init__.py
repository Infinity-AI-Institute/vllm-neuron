# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash source-qualified model package.

The package exposes two class names shaped like `NeuronGlm53Flash…`:

- `NeuronGlm53FlashForCausalLM` (from `.neuron_wrapper`) — the NxDI
  compile-integration wrapper subclassing `NeuronBaseForCausalLM`.  This is
  what the top-level `vllm_neuron.model.registry.get_models()` dispatcher
  hands to NxDI for the `Glm5NextForConditionalGeneration` architecture id.
- `NeuronGlm53FlashForCausalLMImpl` (from `.model`) — the CPU-only source
  reference (pure Python autoregressive with reference kernels).  The unit
  tests in `tests/` import this directly.

The old `NeuronGlm53FlashForCausalLM` alias in `.model` continues to point at
the Impl so no test needs to change.  New code should prefer importing from
the package namespace directly.
"""

from .config import Glm53FlashInferenceConfig
from .model import NeuronGlm53FlashForCausalLMImpl
from .neuron_wrapper import (
    GLM53_BLOCKWISE_MATMUL_WORKAROUND,
    Glm53FlashNeuronInferenceConfig,
    NeuronGlm53FlashForCausalLM,
    build_neuron_config,
)
from .registry import _GLM53_GRAPH_ID, GLM53_SOURCE_CACHE_ABI

__all__ = [
    "GLM53_BLOCKWISE_MATMUL_WORKAROUND",
    "GLM53_SOURCE_CACHE_ABI",
    "_GLM53_GRAPH_ID",
    "Glm53FlashInferenceConfig",
    "Glm53FlashNeuronInferenceConfig",
    "NeuronGlm53FlashForCausalLM",
    "NeuronGlm53FlashForCausalLMImpl",
    "build_neuron_config",
]
