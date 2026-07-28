# SPDX-License-Identifier: Apache-2.0
"""Validated public factory for the frozen GLM-5.2 Trn2 model."""

from __future__ import annotations

import os
from fnmatch import fnmatch

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.compile.platform import get_platform_target
from vllm_neuron.model.neuron_config import NeuronConfig

from .config import Glm52MoeDsaConfig

GLM52_ARTIFACT_VERSION = "glm52-trn2-static-fp8-v1"


def _get_tp_world_size() -> int:
    from vllm.distributed.parallel_state import get_tp_group

    return get_tp_group().world_size


class GlmMoeDsaForCausalLM(nn.Module):
    """Registry-facing factory matching the checkpoint architecture string."""

    def __init__(
        self,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)
        from .model import Glm52MoeDsaForCausalLM

        return Glm52MoeDsaForCausalLM.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> None:
        if neuron_config is None:
            raise ValueError("GLM-5.2 requires an explicit neuron_config")
        if get_platform_target() != "trn2":
            raise ValueError("GLM-5.2 is currently qualified only for Trn2")

        config = Glm52MoeDsaConfig.from_configs(hf_config, neuron_config)
        expected = Glm52MoeDsaConfig()
        frozen_fields = (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "rms_norm_eps",
            "num_attention_heads",
            "num_key_value_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_head_dim",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "attention_bias",
            "attention_dropout",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "index_skip_topk_offset",
            "index_topk_freq",
            "indexer_rope_interleave",
            "indexer_types",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "first_k_dense_replace",
            "moe_layer_freq",
            "n_group",
            "topk_group",
            "scoring_func",
            "topk_method",
            "norm_topk_prob",
            "routed_scaling_factor",
            "moe_router_dtype",
            "mlp_layer_types",
            "max_position_embeddings",
            "rope_interleave",
            "rope_parameters",
            "hidden_act",
            "tie_word_embeddings",
            "num_nextn_predict_layers",
            "torch_dtype",
        )
        mismatches = [
            f"{field}={getattr(config, field)!r}"
            for field in frozen_fields
            if getattr(config, field) != getattr(expected, field)
        ]
        if mismatches:
            raise ValueError(
                "GLM-5.2 integration targets only the frozen architecture: "
                + ", ".join(mismatches)
            )

        config_dict = hf_config.to_dict()
        artifact = config_dict.get("glm52_artifact")
        if not isinstance(artifact, dict):
            raise ValueError("GLM-5.2 requires a converted glm52_artifact marker")
        if artifact.get("artifact_version") != GLM52_ARTIFACT_VERSION:
            raise ValueError(
                f"GLM-5.2 artifact_version must be {GLM52_ARTIFACT_VERSION!r}"
            )
        compile_stub = artifact.get("compile_stub") is True
        cpu_compile = os.environ.get("VLLM_NEURON_CPU_COMPILE", "").lower() in (
            "1",
            "true",
        )
        if compile_stub:
            if not cpu_compile:
                raise ValueError(
                    "GLM-5.2 compile stubs are accepted only with "
                    "VLLM_NEURON_CPU_COMPILE=1"
                )
            if artifact.get("loader_ready") is not False:
                raise ValueError("a GLM-5.2 compile stub must not be loader_ready")
        elif artifact.get("loader_ready") is not True:
            raise ValueError("GLM-5.2 artifact is not loader_ready")
        if artifact.get("mtp_enabled") is not False:
            raise ValueError("GLM-5.2 serving requires MTP disabled")
        if artifact.get("index_closure_status") != "passed":
            raise ValueError("GLM-5.2 artifact index closure has not passed")

        quantization = getattr(hf_config, "quantization_config", None)
        if quantization is None:
            quantization = config_dict.get("quantization_config")
        cls._validate_static_fp8_artifact(quantization)

        if neuron_config.quantization is not None:
            raise ValueError(
                "static FP8 is checkpoint-driven; neuron_config.quantization "
                "must be unset"
            )
        if neuron_config.ep_degree not in (8, 16, 32, 64):
            raise ValueError("GLM-5.2 requires ep_degree in {8, 16, 32, 64}")
        for field in (
            "attention_dp_size",
            "embedding_dp_size",
            "lm_head_dp_size",
            "mlp_dp_size",
        ):
            if getattr(neuron_config, field) != 1:
                raise ValueError(f"GLM-5.2 does not yet support {field} != 1")
        if neuron_config.apply_prefill_dcp:
            raise ValueError("GLM-5.2 prefill DCP is not integrated")
        if neuron_config.fp8_packed_kv:
            raise ValueError("GLM-5.2 requires the standard four-dimensional KV ABI")
        if neuron_config.on_device_sampling_config is not None:
            raise ValueError(
                "GLM-5.2 on-device sampling is not integrated; set "
                "on_device_sampling_config to null"
            )

        if _get_tp_world_size() != 64:
            raise ValueError("GLM-5.2 expanded MLA requires tensor parallelism 64")

    @staticmethod
    def _validate_static_fp8_artifact(quantization: object) -> None:
        if not isinstance(quantization, dict):
            raise ValueError(
                "GLM-5.2 requires the converted ModelOpt static-FP8 artifact"
            )
        if str(quantization.get("quant_method", "")).lower() != "modelopt":
            raise ValueError(
                "native block-FP8/BF16 checkpoints are unsupported; use the "
                "converted ModelOpt static-FP8 artifact"
            )
        inner = quantization.get("quantization")
        details = inner if isinstance(inner, dict) else quantization
        if str(details.get("quant_algo", "")).upper() != "FP8":
            raise ValueError("converted GLM-5.2 artifact must declare quant_algo='FP8'")
        if details.get("weight_block_size") is not None:
            raise ValueError(
                "native block-FP8 scales are unsupported; per-projection scalar "
                "static-FP8 scales are required"
            )
        excluded = details.get("exclude_modules") or []
        if not isinstance(excluded, (list, tuple)) or not any(
            pattern == "lm_head" or fnmatch("lm_head", str(pattern))
            for pattern in excluded
        ):
            raise ValueError(
                "converted GLM-5.2 artifact must exclude the BF16 lm_head from FP8"
            )
