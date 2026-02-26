# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for vllm_neuron.attention.neuron_attn module.

This test suite provides comprehensive coverage for the NeuronAttentionBackend
class, which serves as the Neuron-specific attention backend implementation
for vLLM integration with AWS Neuron devices.
"""

from unittest.mock import Mock, patch

import pytest


class TestNeuronAttentionBackend:
    """Test suite for NeuronAttentionBackend class."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks for vLLM modules that cause import issues."""
        with patch.dict(
            "sys.modules",
            {
                "vllm.model_executor.layers.quantization.utils.quant_utils": Mock(),
                "vllm.attention.layer": Mock(),
                "vllm.attention.selector": Mock(),
            },
        ):
            # Import the modules within the mocked context
            from vllm.attention.backends.abstract import AttentionBackend

            from vllm_neuron.attention.neuron_attn import NeuronAttentionBackend

            # Store references for use in tests
            self.NeuronAttentionBackend = NeuronAttentionBackend
            self.AttentionBackend = AttentionBackend
            yield

    def test_get_name(self):
        """Test get_name static method returns correct backend name."""
        name = self.NeuronAttentionBackend.get_name()

        assert isinstance(name, str)
        assert name == "NEURON_ATTN"

        # Test that it's consistent across multiple calls
        assert self.NeuronAttentionBackend.get_name() == name

    def test_get_impl_cls(self):
        """Test get_impl_cls static method returns the correct class."""
        impl_cls = self.NeuronAttentionBackend.get_impl_cls()

        assert impl_cls is self.NeuronAttentionBackend
        assert issubclass(impl_cls, self.AttentionBackend)

        # Test that it's consistent across multiple calls
        assert self.NeuronAttentionBackend.get_impl_cls() is impl_cls

    def test_get_kv_cache_shape_not_implemented(self):
        """Test that get_kv_cache_shape raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            self.NeuronAttentionBackend.get_kv_cache_shape(
                num_blocks=100, block_size=16, num_kv_heads=32, head_size=128
            )
