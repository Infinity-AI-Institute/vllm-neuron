"""Native vLLM-Neuron Gemma 4 model boundary.

The public class in this module follows the runner-facing model contract used
by the other native vLLM-Neuron models.  CPU numerical oracles live in
``reference.py`` and are never selected implicitly for serving.
"""

import os

import torch
import torch.nn as nn

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

from .config import Gemma4Config
from .weights import Gemma4WeightMapper


class Gemma4ForCausalLM(nn.Module):
    """Native Gemma 4 causal-LM interface.

    The production Neuron layer implementation is still in development.  The
    explicit reference mode exists only so registry and runner contracts can be
    unit-tested without claiming device enablement.
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self._bound_kv_caches: dict[str, list[torch.Tensor]] = {}
        if os.environ.get("VLLM_NEURON_GEMMA4_REFERENCE") == "1":
            from .reference import Gemma4ReferenceCausalLM

            self._reference = Gemma4ReferenceCausalLM(config)
        else:
            raise NotImplementedError(
                "Gemma 4 native Neuron layers are not implemented yet. "
                "VLLM_NEURON_GEMMA4_REFERENCE=1 is only for CPU contract tests."
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del (
            inputs_embeds,
            is_token_ids,
            attn_metadata,
            sampling_params,
            spec_decode_metadata,
            logit_mask,
            rank,
            kwargs,
        )
        return self._reference(
            input_ids,
            sampling_positions=sampling_positions,
            position_ids=positions,
        )

    def get_kv_spec(self) -> KVSpec:
        layers = []
        for layer_idx in range(self.config.num_hidden_layers):
            head_dim, num_kv_heads = self.config.attention_shape(layer_idx)
            sliding_window = (
                None
                if self.config.layer_is_global(layer_idx)
                else self.config.sliding_window
            )
            layers.append(
                LayerSpec(
                    name=f"layers.{layer_idx}.self_attn",
                    num_kv_heads=num_kv_heads,
                    head_size=head_dim,
                    dtype=self.config.torch_dtype,
                    sliding_window_size=sliding_window,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(
        self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]
    ) -> None:
        expected = {layer.name for layer in self.get_kv_spec().layers}
        missing = sorted(expected.difference(kv_caches))
        if missing:
            raise ValueError(f"KV cache missing layers: {missing}")
        self._bound_kv_caches = {
            name: kv_caches[name] for name in sorted(expected)
        }

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        if not hasattr(self, "_reference"):
            raise NotImplementedError(
                "Gemma 4 production Neuron parameter storage is not "
                "implemented yet."
            )
        # The CPU oracle intentionally uses the simple loader: it is a TP1
        # correctness seam and does not require a distributed Store.  The
        # production model will use load_sharded_pipelined once its TP/EP
        # parameter storage is present.
        mappings = Gemma4WeightMapper.build_mappings(
            (name for name, _ in self.named_parameters()),
            tied_lm_head=self.config.tie_word_embeddings,
        )
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        result = checkpoint.load_sharded(
            rank=0,
            world_size=1,
            model=self,
            mappings=mappings,
            device=device,
            strict=True,
        )
        self.load_state_dict(result.state_dict, strict=False, assign=True)
        self._last_checkpoint_load_result = result
