# SPDX-License-Identifier: Apache-2.0
"""Validated public factory for DeepSeek-V4-Flash on Trn2.

Mirrors the GLM-5.2 factory template at ``vllm_neuron/model/glm52_moe_dsa/
factory.py`` for the same reasons: the checkpoint pins the architecture,
the neuron_config pins the runtime, and any deviation must fail loudly
rather than let the compile flow silently drop a field.

The FP8-KV guard below (``FORBIDDEN_FP8_KV_KEYS``) is required by the
same silent-drop failure class that motivated its inclusion in GLM-5.2
and GLM-5.3-Flash: this wrapper replaces NxDI's ``KVCacheManager`` with
a per-attention-type state cache (sliding-window KV + compressor pool +
optional indexer pool), whose aliased tensors are declared bf16
explicitly, and any FP8-KV request would emit a mismatched
``neuron_config.json`` — the exact "requested vs emitted" split that
this port refuses to inherit.  See user memory
``nxdi-fp8-kv-wireup-requirements-20260828`` for the failure signature.
"""

from __future__ import annotations

from torch import nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .config import HF_SNAPSHOT_SHA, DeepseekV4FlashInferenceConfig

DSV4_ARTIFACT_VERSION = "dsv4-flash-trn2-v0-scaffold"

FORBIDDEN_FP8_KV_KEYS: tuple[str, ...] = (
    "fp8_packed_kv",
    "kv_cache_quant",
    "kv_quant_config",
)


def _get_tp_world_size() -> int:
    from vllm.distributed.parallel_state import get_tp_group

    return get_tp_group().world_size


class DeepseekV4FlashForCausalLM(nn.Module):
    """Registry-facing factory matching the HF architecture string.

    The actual NxDI wrapper class is
    ``vllm_neuron.model.dsv4_flash.neuron_wrapper.NeuronDeepseekV4FlashForCausalLM``
    (Round 2+).  This factory validates the config/neuron_config pair and
    then hands off — a single fail-loud gate keeps every downstream
    Round-N wrapper free of validation-drift.
    """

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
        from .neuron_wrapper import NeuronDeepseekV4FlashForCausalLM

        return NeuronDeepseekV4FlashForCausalLM.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> None:
        if neuron_config is None:
            raise ValueError("DeepSeek-V4-Flash requires an explicit neuron_config")
        try:
            from vllm_neuron.compile.platform import get_platform_target
        except ImportError:  # pragma: no cover - CPU-only guard
            get_platform_target = None
        if get_platform_target is not None and get_platform_target() != "trn2":
            raise ValueError("DeepSeek-V4-Flash is currently qualified only for Trn2")

        # Architecture identity — refuses out-of-band clones with different
        # constants.  Reads the same "frozen fields" the GLM-5.2 factory does.
        config = DeepseekV4FlashInferenceConfig.from_configs(hf_config, neuron_config)
        expected = DeepseekV4FlashInferenceConfig()
        frozen_fields = (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "q_lora_rank",
            "qk_rope_head_dim",
            "o_groups",
            "o_lora_rank",
            "moe_intermediate_size",
            "n_routed_experts",
            "num_experts_per_tok",
            "n_shared_experts",
            "scoring_func",
            "topk_method",
            "routed_scaling_factor",
            "norm_topk_prob",
            "num_hash_layers",
            "hc_mult",
            "hc_sinkhorn_iters",
            "swiglu_limit",
            "sliding_window",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "max_position_embeddings",
            "rope_theta",
            "compress_rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "expert_dtype",
            "torch_dtype",
            "layer_types",
            "mlp_layer_types",
        )
        mismatches = [
            f"{field}={getattr(config, field)!r}"
            for field in frozen_fields
            if getattr(config, field) != getattr(expected, field)
        ]
        if mismatches:
            raise ValueError(
                "DeepSeek-V4-Flash integration targets only the frozen architecture: "
                + ", ".join(mismatches)
            )

        # Snapshot pin — a mismatched SHA does not fail-hard here (the
        # HF metadata may be absent on non-hydrated flows) but leaves a
        # trace field the caller can inspect.
        config_dict = hf_config.to_dict()
        declared_sha = config_dict.get("hf_snapshot_sha") or config_dict.get(
            "_commit_hash"
        )
        if declared_sha is not None and declared_sha != HF_SNAPSHOT_SHA:
            raise ValueError(
                "DeepSeek-V4-Flash HF snapshot pin mismatch: expected "
                f"{HF_SNAPSHOT_SHA!r}, got {declared_sha!r}."
            )

        # FP8-KV guard: silent-drop-immune per user memory nxdi-fp8-kv-wireup-
        # requirements-20260828.  This wrapper does not use NxDI's
        # KVCacheManager (it owns per-attention-type state) so any FP8-KV
        # request would emit a mismatched neuron_config.json.
        offenders = sorted(
            key for key in FORBIDDEN_FP8_KV_KEYS if getattr(neuron_config, key, None)
        )
        if offenders:
            raise ValueError(
                "DeepSeek-V4-Flash refuses FP8-packed KV configuration: "
                f"{offenders!r}. This wrapper replaces NxDI's KVCacheManager "
                "with its own per-attention-type state cache (sliding-window "
                "KV + compressor pool + optional indexer pool); the aliased "
                "state tensors are declared bf16 explicitly and any FP8-KV "
                "request would silently mismatch the emitted neuron_config.json "
                "against the actual tensor dtypes."
            )
        # Also refuse a raw NxDI FP8-KV dict on the neuron_config extras.
        extras = getattr(neuron_config, "extra", None) or {}
        offenders_extra = sorted(k for k in FORBIDDEN_FP8_KV_KEYS if k in extras)
        if offenders_extra:
            raise ValueError(
                "DeepSeek-V4-Flash refuses FP8-packed KV in neuron_config.extra: "
                f"{offenders_extra!r}."
            )

        # Speculative-decode / MTP is out of scope campaign-wide.
        if getattr(neuron_config, "on_device_sampling_config", None) is not None:
            raise ValueError(
                "DeepSeek-V4-Flash on-device sampling is not integrated; set "
                "on_device_sampling_config to null"
            )
        if getattr(neuron_config, "apply_prefill_dcp", False):
            raise ValueError("DeepSeek-V4-Flash prefill DCP is not integrated")
        if getattr(neuron_config, "quantization", None) is not None:
            raise ValueError(
                "DeepSeek-V4-Flash static FP8/FP4 is checkpoint-driven; "
                "neuron_config.quantization must be unset"
            )

        # TP / EP degree.  See the enablement doc for divisibility math;
        # first-fire lands at tp_degree=32, LNC=2.
        if _get_tp_world_size() != 32:
            raise ValueError(
                "DeepSeek-V4-Flash first-fire contract is TP=32; other degrees "
                "must be enabled via a separate lane after the first NEFF "
                "correctness gate closes."
            )
        for field_ in (
            "attention_dp_size",
            "embedding_dp_size",
            "lm_head_dp_size",
            "mlp_dp_size",
        ):
            if getattr(neuron_config, field_, 1) != 1:
                raise ValueError(
                    f"DeepSeek-V4-Flash does not yet support {field_} != 1"
                )


__all__ = [
    "DSV4_ARTIFACT_VERSION",
    "FORBIDDEN_FP8_KV_KEYS",
    "DeepseekV4FlashForCausalLM",
]
