"""GLM-5.3-Flash conversion, artifact, and compile adapter surfaces."""

from .checkpoint_converter import (
    GLM53_CHECKPOINT_REVISION,
    Glm53CheckpointReport,
    classify_tensor,
    dequantize_block_fp8,
    kda_conv1d_per_head_layout,
    preflight_checkpoint_metadata,
)
from .compile_adapter import (
    GLM53_COMPILE_ADAPTER_SCHEMA,
    Glm53CompileAdapterError,
    assert_emitted_neuron_config,
    compile_kwargs,
)
from .config import Glm53FlashInferenceConfig
from .model import NeuronGlm53FlashForCausalLMImpl
from .neuron_wrapper import (
    GLM53_BLOCKWISE_MATMUL_WORKAROUND,
    Glm53FlashNeuronInferenceConfig,
    NeuronGlm53FlashForCausalLM,
    build_neuron_config,
)
from .phase_runtime import (
    PHASE_RUNTIME_SCHEMA,
    Glm53PairedPhaseRuntime,
    Glm53PhaseRuntimeError,
    Glm53PhaseState,
    initialize_phase,
)
from .rank_plan import (
    Glm53RankPlan,
    PlannedSourceSpec,
    TargetTensorPlan,
    build_glm53_rank_plan,
    stream_glm53_rank_checkpoint,
)
from .reference_target import (
    Glm53ReferenceTarget,
    Glm53ReferenceTargetError,
)
from .registry import _GLM53_GRAPH_ID, GLM53_SOURCE_CACHE_ABI
from .runtime_config import (
    GLM53_ARCHITECTURE,
    GLM53_RUNTIME_CONFIG_SCHEMA,
    Glm53RuntimeConfig,
    Glm53RuntimeConfigError,
)
from .runtime_factory import (
    GLM53_RUNTIME_ADAPTER,
    GLM53_RUNTIME_BUNDLE_SCHEMA,
    GLM53_RUNTIME_FACTORY_ABI,
    Glm53CompileLaunchPolicy,
    Glm53RuntimeArtifactBundle,
    Glm53RuntimeFactory,
    Glm53RuntimeFactoryError,
    Glm53RuntimeRank,
    get_runtime_factories,
)
from .streaming_rank_writer import (
    Glm53StreamingError,
    IndexedTensorReader,
    RankInventory,
    StreamingRankWriter,
    TensorChunk,
    TensorSpec,
    stream_rank_checkpoint,
)

__all__ = [
    "GLM53_ARCHITECTURE",
    "GLM53_BLOCKWISE_MATMUL_WORKAROUND",
    "GLM53_CHECKPOINT_REVISION",
    "GLM53_COMPILE_ADAPTER_SCHEMA",
    "GLM53_RUNTIME_ADAPTER",
    "GLM53_RUNTIME_BUNDLE_SCHEMA",
    "GLM53_RUNTIME_CONFIG_SCHEMA",
    "GLM53_RUNTIME_FACTORY_ABI",
    "GLM53_SOURCE_CACHE_ABI",
    "PHASE_RUNTIME_SCHEMA",
    "_GLM53_GRAPH_ID",
    "Glm53CheckpointReport",
    "Glm53CompileAdapterError",
    "Glm53CompileLaunchPolicy",
    "Glm53FlashInferenceConfig",
    "Glm53FlashNeuronInferenceConfig",
    "Glm53PairedPhaseRuntime",
    "Glm53PhaseRuntimeError",
    "Glm53PhaseState",
    "Glm53RankPlan",
    "Glm53ReferenceTarget",
    "Glm53ReferenceTargetError",
    "Glm53RuntimeArtifactBundle",
    "Glm53RuntimeConfig",
    "Glm53RuntimeConfigError",
    "Glm53RuntimeFactory",
    "Glm53RuntimeFactoryError",
    "Glm53RuntimeRank",
    "Glm53StreamingError",
    "IndexedTensorReader",
    "NeuronGlm53FlashForCausalLM",
    "NeuronGlm53FlashForCausalLMImpl",
    "PlannedSourceSpec",
    "RankInventory",
    "StreamingRankWriter",
    "TargetTensorPlan",
    "TensorChunk",
    "TensorSpec",
    "assert_emitted_neuron_config",
    "build_glm53_rank_plan",
    "build_neuron_config",
    "classify_tensor",
    "compile_kwargs",
    "dequantize_block_fp8",
    "get_runtime_factories",
    "initialize_phase",
    "kda_conv1d_per_head_layout",
    "preflight_checkpoint_metadata",
    "stream_glm53_rank_checkpoint",
    "stream_rank_checkpoint",
]
