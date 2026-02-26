# SPDX-License-Identifier: Apache-2.0
# test/unit/worker/test_utils.py
"""Unit tests for Neuron utility functions."""

import pytest
from unittest.mock import Mock

from vllm_neuron.worker.utils import get_num_layers_from_hf_config


class TestGetNumLayersFromHfConfig:
    """Tests for get_num_layers_from_hf_config function."""

    def test_num_hidden_layers_at_top_level(self):
        """Test extraction when num_hidden_layers is at top level."""
        hf_config = Mock()
        hf_config.num_hidden_layers = 32
        hf_config.num_layers = None

        result = get_num_layers_from_hf_config(hf_config)

        assert result == 32

    def test_num_layers_at_top_level(self):
        """Test extraction when num_layers is at top level (fallback)."""
        hf_config = Mock()
        hf_config.num_hidden_layers = None
        hf_config.num_layers = 24

        result = get_num_layers_from_hf_config(hf_config)

        assert result == 24

    def test_num_hidden_layers_takes_precedence(self):
        """Test that num_hidden_layers is preferred over num_layers."""
        hf_config = Mock()
        hf_config.num_hidden_layers = 32
        hf_config.num_layers = 24

        result = get_num_layers_from_hf_config(hf_config)

        assert result == 32

    def test_multimodal_text_config_only(self):
        """Test multimodal model with only text_config."""
        hf_config = Mock(spec=[])
        # No top-level attributes
        del hf_config.num_hidden_layers
        del hf_config.num_layers

        # Set up text_config
        text_config = Mock()
        text_config.num_hidden_layers = 24
        hf_config.text_config = text_config
        hf_config.vision_config = None

        result = get_num_layers_from_hf_config(hf_config)

        assert result == 24

    def test_multimodal_vision_config_only(self):
        """Test multimodal model with only vision_config."""
        hf_config = Mock(spec=[])
        # No top-level attributes
        del hf_config.num_hidden_layers
        del hf_config.num_layers

        # Set up vision_config
        vision_config = Mock()
        vision_config.num_hidden_layers = 12
        hf_config.text_config = None
        hf_config.vision_config = vision_config

        result = get_num_layers_from_hf_config(hf_config)

        assert result == 12

    def test_multimodal_both_configs(self):
        """Test multimodal model with both text_config and vision_config."""
        hf_config = Mock(spec=[])
        # No top-level attributes
        del hf_config.num_hidden_layers
        del hf_config.num_layers

        # Set up text_config
        text_config = Mock()
        text_config.num_hidden_layers = 24
        hf_config.text_config = text_config

        # Set up vision_config
        vision_config = Mock()
        vision_config.num_hidden_layers = 12
        hf_config.vision_config = vision_config

        result = get_num_layers_from_hf_config(hf_config)

        # Should sum both
        assert result == 36

    def test_multimodal_nested_num_layers_fallback(self):
        """Test multimodal model using num_layers as fallback in nested config."""
        hf_config = Mock(spec=[])
        # No top-level attributes
        del hf_config.num_hidden_layers
        del hf_config.num_layers

        # Set up text_config with num_layers (not num_hidden_layers)
        text_config = Mock()
        text_config.num_hidden_layers = None
        text_config.num_layers = 20
        hf_config.text_config = text_config
        hf_config.vision_config = None

        result = get_num_layers_from_hf_config(hf_config)

        assert result == 20

    def test_multimodal_mixed_attributes(self):
        """Test multimodal model with mixed attribute names in nested configs."""
        hf_config = Mock(spec=[])
        # No top-level attributes
        del hf_config.num_hidden_layers
        del hf_config.num_layers

        # text_config uses num_hidden_layers
        text_config = Mock()
        text_config.num_hidden_layers = 24
        text_config.num_layers = None
        hf_config.text_config = text_config

        # vision_config uses num_layers
        vision_config = Mock()
        vision_config.num_hidden_layers = None
        vision_config.num_layers = 8
        hf_config.vision_config = vision_config

        result = get_num_layers_from_hf_config(hf_config)

        assert result == 32

    def test_no_layers_found_raises_error(self):
        """Test that RuntimeError is raised when no layers can be determined."""
        hf_config = Mock(spec=[])
        # No top-level attributes
        del hf_config.num_hidden_layers
        del hf_config.num_layers

        # No nested configs
        hf_config.text_config = None
        hf_config.vision_config = None

        with pytest.raises(
            RuntimeError, match="Could not determine number of layers from model config"
        ):
            get_num_layers_from_hf_config(hf_config)

    def test_nested_configs_with_no_layer_attributes(self):
        """Test RuntimeError when nested configs exist but have no layer attributes."""
        hf_config = Mock(spec=[])
        # No top-level attributes
        del hf_config.num_hidden_layers
        del hf_config.num_layers

        # text_config exists but has no layer attributes
        text_config = Mock()
        text_config.num_hidden_layers = None
        text_config.num_layers = None
        hf_config.text_config = text_config
        hf_config.vision_config = None

        with pytest.raises(
            RuntimeError, match="Could not determine number of layers from model config"
        ):
            get_num_layers_from_hf_config(hf_config)

    def test_zero_layers_at_top_level(self):
        """Test behavior when num_hidden_layers is 0 (falsy but valid)."""
        # Note: 0 layers is unlikely in practice, but tests edge case behavior
        hf_config = Mock()
        hf_config.num_hidden_layers = 0
        hf_config.num_layers = None

        # 0 is falsy, so it falls through to nested configs
        hf_config.text_config = None
        hf_config.vision_config = None

        # Should raise error since 0 is treated as not found (falsy)
        with pytest.raises(RuntimeError):
            get_num_layers_from_hf_config(hf_config)

    def test_typical_llama_config(self):
        """Test with a typical LLaMA-style config."""
        hf_config = Mock()
        hf_config.num_hidden_layers = 32
        hf_config.hidden_size = 4096
        hf_config.num_attention_heads = 32
        hf_config.vocab_size = 32000

        result = get_num_layers_from_hf_config(hf_config)

        assert result == 32

    def test_typical_llava_config(self):
        """Test with a typical LLaVA multimodal config."""
        hf_config = Mock(spec=[])
        # LLaVA typically has top-level config but may also have nested
        del hf_config.num_hidden_layers
        del hf_config.num_layers

        # Text config (LLaMA backbone)
        text_config = Mock()
        text_config.num_hidden_layers = 32
        hf_config.text_config = text_config

        # Vision config (CLIP ViT)
        vision_config = Mock()
        vision_config.num_hidden_layers = 24
        hf_config.vision_config = vision_config

        result = get_num_layers_from_hf_config(hf_config)

        assert result == 56  # 32 + 24

    def test_getattr_fallback_behavior(self):
        """Test that getattr with default None works correctly."""

        # Create config without the attributes at all
        class MinimalConfig:
            pass

        hf_config = MinimalConfig()

        with pytest.raises(RuntimeError):
            get_num_layers_from_hf_config(hf_config)

    def test_partial_nested_config(self):
        """Test when only one nested config has layer info."""
        hf_config = Mock(spec=[])
        del hf_config.num_hidden_layers
        del hf_config.num_layers

        # text_config has layers
        text_config = Mock()
        text_config.num_hidden_layers = 32
        hf_config.text_config = text_config

        # vision_config exists but has no layer info
        vision_config = Mock()
        vision_config.num_hidden_layers = None
        vision_config.num_layers = None
        hf_config.vision_config = vision_config

        result = get_num_layers_from_hf_config(hf_config)

        # Should return just text_config layers
        assert result == 32


if __name__ == "__main__":
    pytest.main([__file__])
