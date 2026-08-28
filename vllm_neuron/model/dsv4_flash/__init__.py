# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash source-qualified model package (Round 1 skeleton).

The package exposes:

- ``NeuronDeepseekV4FlashForCausalLM`` (from ``.neuron_wrapper``) — the NxDI
  compile-integration wrapper subclassing ``NeuronBaseForCausalLM``.  Round 1:
  init raises ``NotImplementedError``.  Round 2 wires the block scaffold.
- ``DeepseekV4FlashInferenceConfig`` (from ``.config``) — the frozen source
  config with fail-loud architecture validation.
- ``DeepseekV4FlashForCausalLM`` (from ``.factory``) — the registry-facing
  factory that validates HF/neuron config pairs before instantiation.

See ``README.md`` for the current implementation-status matrix, and the
enablement draft at
``harness-v2/staging/reference-sweep-20260826T2150Z/lanes/deepseek-v4-flash/
ENABLEMENT-DRAFT-2026-08-28.md`` for the block-by-block delta vs
GLM-5.3-Flash and the first-fire blockers.
"""

from .config import (
    HF_REPO_ID,
    HF_SNAPSHOT_SHA,
    DeepseekV4FlashInferenceConfig,
    DeepseekV4QuantizationConfig,
    DeepseekV4RopeScalingConfig,
    validate_ue8m0_scale,
)
from .factory import DeepseekV4FlashForCausalLM
from .neuron_wrapper import (
    DSV4_BLOCKWISE_MATMUL_WORKAROUND,
    DeepseekV4FlashNeuronInferenceConfig,
    NeuronDeepseekV4FlashForCausalLM,
    build_neuron_config,
)
from .registry import DSV4_SOURCE_CACHE_ABI, _DSV4_GRAPH_ID

__all__ = [
    "DSV4_BLOCKWISE_MATMUL_WORKAROUND",
    "DSV4_SOURCE_CACHE_ABI",
    "DeepseekV4FlashForCausalLM",
    "DeepseekV4FlashInferenceConfig",
    "DeepseekV4FlashNeuronInferenceConfig",
    "DeepseekV4QuantizationConfig",
    "DeepseekV4RopeScalingConfig",
    "HF_REPO_ID",
    "HF_SNAPSHOT_SHA",
    "NeuronDeepseekV4FlashForCausalLM",
    "_DSV4_GRAPH_ID",
    "build_neuron_config",
    "validate_ue8m0_scale",
]
