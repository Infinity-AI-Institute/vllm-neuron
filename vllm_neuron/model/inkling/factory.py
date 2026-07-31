"""ModelRegistry factory for native Inkling-Small vLLM-Neuron onboarding."""

from __future__ import annotations

from torch import nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .config import InklingConfig


class InklingForConditionalGeneration(nn.Module):
    """Validate the public checkpoint contract and build its text path."""

    def __init__(
        self,
        hf_config: PretrainedConfig | InklingConfig | None = None,
        neuron_config: NeuronConfig | None = None,
        *,
        vllm_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        del prefix
        if hf_config is None:
            if vllm_config is None:
                raise ValueError("Inkling requires hf_config or vllm_config")
            hf_config = vllm_config.model_config.hf_config
            raw_neuron = vllm_config.additional_config.get("neuron_config", {})
            neuron_config = NeuronConfig.from_dict(raw_neuron)
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, input_ids, positions, *args, **kwargs):
        return self._model(input_ids, positions, *args, **kwargs)

    def embed_input_ids(self, input_ids):
        return self._model.model.embed_tokens(
            input_ids.reshape(-1),
            scatter_tokens=False,
            rank=None,
        )

    def compute_logits(self, hidden_states):
        return self._model.lm_head(hidden_states)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | InklingConfig,
        neuron_config: NeuronConfig | None = None,
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig | InklingConfig,
        neuron_config: NeuronConfig | None,
    ) -> nn.Module:
        config = InklingConfig.from_configs(hf_config, neuron_config)
        cls._validate_config(config)
        from .model import InklingForConditionalGeneration as Model

        return Model(config)

    @staticmethod
    def _validate_config(config: InklingConfig) -> None:
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("Inkling query heads must be divisible by KV heads")
        if config.swa_num_attention_heads % config.swa_num_key_value_heads:
            raise ValueError("Inkling SWA query heads must be divisible by KV heads")
        if len(set(config.local_layer_ids)) != len(config.local_layer_ids):
            raise ValueError("Inkling local_layer_ids contains duplicates")
        if any(
            layer < 0 or layer >= config.num_hidden_layers
            for layer in config.local_layer_ids
        ):
            raise ValueError("Inkling local_layer_ids contains an invalid layer")
        if not 0 < config.num_experts_per_tok <= config.n_routed_experts:
            raise ValueError("Inkling routed top-k is invalid")
        if config.n_shared_experts != 2 or not config.shared_expert_sink:
            raise NotImplementedError(
                "native Inkling currently requires the two sink experts"
            )
        if (
            config.gate_activation != "sigmoid"
            or not config.norm_after_topk
            or not config.use_gate_bias
            or not config.use_global_scale
        ):
            raise NotImplementedError(
                "native Inkling requires its published sigmoid router contract"
            )
        if not config.use_sconv or config.sconv_kernel_size != 4:
            raise NotImplementedError(
                "native Inkling requires four-tap short convolutions"
            )
