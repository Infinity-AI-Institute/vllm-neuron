# SPDX-License-Identifier: Apache-2.0
"""Utility functions for Neuron."""

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_num_layers_from_hf_config(hf_config) -> int:
    """
    Extract the number of layers from a HuggingFace model config.

    First tries top level. If not found, sums layers from nested configs
    (text_config, vision_config, etc.) for multimodal models.

    Args:
        hf_config: HuggingFace model configuration object

    Returns:
        Total number of layers

    Raises:
        RuntimeError: If layer count cannot be determined
    """
    # Try top level first
    num_layers = getattr(hf_config, "num_hidden_layers", None) or getattr(
        hf_config, "num_layers", None
    )

    if num_layers is not None:
        return num_layers

    # Sum nested configs (multimodal models)
    total = 0
    for attr in ["text_config", "vision_config"]:
        config = getattr(hf_config, attr, None)
        if config is not None:
            layers = getattr(config, "num_hidden_layers", None) or getattr(
                config, "num_layers", None
            )
            if layers is not None:
                total += layers

    if total == 0:
        raise RuntimeError("Could not determine number of layers from model config.")

    return total
