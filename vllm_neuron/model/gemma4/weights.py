"""Gemma 4 checkpoint names and parameter loader policies.

Keep checkpoint policy outside the CPU oracle.  Both the reference model and
the production Neuron model use the same Hugging Face key contract, while the
production layers remain free to fuse or shard their storage differently.
"""

from __future__ import annotations

from collections.abc import Iterable

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    sharding_weight_loader,
)


class Gemma4WeightMapper:
    """Map native parameter names to the Gemma 4 text checkpoint."""

    CHECKPOINT_TEXT_PREFIX = "model.language_model."

    @classmethod
    def native_name(cls, name: str) -> str:
        """Remove wrapper prefixes without rewriting architectural names."""
        return name.removeprefix("_reference.")

    @classmethod
    def checkpoint_name(cls, name: str, *, tied_lm_head: bool = True) -> str:
        """Return the safetensors key for a native parameter name."""
        name = cls.native_name(name)
        if name == "lm_head.weight" and tied_lm_head:
            return f"{cls.CHECKPOINT_TEXT_PREFIX}embed_tokens.weight"
        if name.startswith("model."):
            return f"{cls.CHECKPOINT_TEXT_PREFIX}{name.removeprefix('model.')}"
        return name

    @classmethod
    def build_mappings(
        cls, parameter_names: Iterable[str], *, tied_lm_head: bool = True
    ) -> dict[str, str]:
        """Build the mapping consumed by :class:`SafetensorsCheckpoint`."""
        return {
            name: cls.checkpoint_name(name, tied_lm_head=tied_lm_head)
            for name in parameter_names
        }

    @classmethod
    def loader_kind(cls, name: str) -> str:
        """Classify a parameter for TP/EP storage policy."""
        name = cls.native_name(name)
        if ".experts." in name:
            return "expert-local"
        if name.endswith(
            (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "gate_proj.weight",
                "up_proj.weight",
                "lm_head.weight",
            )
        ):
            return "column"
        if name.endswith(("o_proj.weight", "down_proj.weight")):
            return "row"
        return "replicated"

    @classmethod
    def make_loader(
        cls, name: str, shard_size: int, tp_size: int
    ) -> SafetensorsWeightLoader:
        """Build the vLLM-Neuron safetensors loader for a parameter role."""
        role = cls.loader_kind(name)
        if role == "column":
            return sharding_weight_loader(0, shard_size, tp_size)
        if role == "row":
            return sharding_weight_loader(1, shard_size, tp_size)
        # Expert tensors require an EP-aware loader in the production model.
        # The CPU oracle is TP1, so identity is the only honest default here.
        return SafetensorsWeightLoader()
