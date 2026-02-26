# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for CPU sampling functionality in NeuronxDistributedModelRunner.
"""

from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import torch

from vllm_neuron.worker.neuronx_distributed_model_runner import (
    ModelInputForNeuron,
    NeuronxDistributedModelRunner,
)


class TestCPUSampling:
    """Test CPU sampling functionality."""

    @pytest.fixture
    def mock_model_runner(self):
        """Create a mock model runner for testing."""
        with patch.object(
            NeuronxDistributedModelRunner,
            "__init__",
            lambda self, *args, **kwargs: None,
        ):
            runner = NeuronxDistributedModelRunner.__new__(
                NeuronxDistributedModelRunner
            )

            # Set up required attributes
            runner.model = Mock()
            runner.model.neuron_config = Mock()
            runner.model.neuron_config.vocab_size = 32000
            runner.model.neuron_config.on_device_sampling_config = None
            runner.model.neuron_config.is_block_kv_layout = False
            runner.model.neuron_config.is_prefix_caching = False
            runner.model.neuron_config.chunked_prefill_config = None
            runner.model.sample = Mock()

            runner.cpu_sampler = Mock()

            # Use MagicMock for input_batch to handle item assignment
            runner.input_batch = MagicMock()
            runner.input_batch.sampling_metadata = Mock()
            runner.input_batch.req_ids = ["req1", "req2"]
            runner.input_batch.req_id_to_index = {"req1": 0, "req2": 1}
            runner.input_batch.num_tokens_no_spec = [0, 0]
            runner.input_batch.num_tokens = [0, 0]
            # Use MagicMock for token_ids_cpu to allow item assignment
            runner.input_batch.token_ids_cpu = MagicMock()

            runner._cached_logits = None
            runner._cached_model_input = None
            runner.speculative_config = None
            runner.max_model_len = 1024
            runner.requests = {
                "req1": Mock(output_token_ids=[]),
                "req2": Mock(output_token_ids=[]),
            }

            return runner

    def test_sample_tokens_success(self, mock_model_runner):
        """Test successful CPU sampling via sample_tokens."""
        batch_size, vocab_size = 2, 32000
        logits = torch.randn(batch_size, vocab_size)

        model_input = ModelInputForNeuron(
            request_ids=["req1", "req2"],
            prefill_completion_state=None,
        )

        # Set up cached state (simulating execute_model was called)
        mock_model_runner._cached_logits = logits
        mock_model_runner._cached_model_input = model_input

        # Mock sampler output
        expected_output = Mock()
        expected_output.sampled_token_ids = torch.tensor([[1], [2]])
        expected_output.logprobs_tensors = None
        mock_model_runner.cpu_sampler.return_value = expected_output

        # Test sampling
        result = mock_model_runner.sample_tokens(grammar_output=None)

        assert result is not None
        mock_model_runner.cpu_sampler.assert_called_once()

    def test_sample_tokens_without_cached_logits_raises_error(self, mock_model_runner):
        """Test sample_tokens raises error when called without cached logits."""
        mock_model_runner._cached_logits = None

        with pytest.raises(
            RuntimeError, match="sample_tokens.*called without prior execute_model"
        ):
            mock_model_runner.sample_tokens(grammar_output=None)

    def test_sample_tokens_clears_cache_after_sampling(self, mock_model_runner):
        """Test that sample_tokens clears cached logits after sampling."""
        logits = torch.randn(2, 32000)
        model_input = ModelInputForNeuron(request_ids=["req1", "req2"])

        mock_model_runner._cached_logits = logits
        mock_model_runner._cached_model_input = model_input

        expected_output = Mock()
        expected_output.sampled_token_ids = torch.tensor([[1], [2]])
        expected_output.logprobs_tensors = None
        mock_model_runner.cpu_sampler.return_value = expected_output

        mock_model_runner.sample_tokens(grammar_output=None)

        # Cache should be cleared
        assert mock_model_runner._cached_logits is None
        assert mock_model_runner._cached_model_input is None

    def test_sample_tokens_with_on_device_sampling(self, mock_model_runner):
        """Test sample_tokens uses hardware sampling when on_device_sampling_config is set."""
        logits = torch.randn(2, 32000)
        model_input = ModelInputForNeuron(
            request_ids=["req1", "req2"],
            prefill_completion_state=None,
        )

        mock_model_runner._cached_logits = logits
        mock_model_runner._cached_model_input = model_input

        # Enable on-device sampling
        mock_model_runner.model.neuron_config.on_device_sampling_config = Mock()

        expected_output = Mock()
        expected_output.sampled_token_ids = torch.tensor([[1], [2]])
        expected_output.logprobs_tensors = None
        mock_model_runner.model.sample = Mock(return_value=expected_output)

        result = mock_model_runner.sample_tokens(grammar_output=None)

        assert result is not None
        mock_model_runner.model.sample.assert_called_once()
        # CPU sampler should NOT be called when using hardware sampling
        mock_model_runner.cpu_sampler.assert_not_called()

    def test_sample_tokens_cpu_sampling_when_no_on_device_config(
        self, mock_model_runner
    ):
        """Test sample_tokens uses CPU sampling when on_device_sampling_config is None."""
        logits = torch.randn(2, 32000)
        model_input = ModelInputForNeuron(
            request_ids=["req1", "req2"],
            prefill_completion_state=None,
        )

        mock_model_runner._cached_logits = logits
        mock_model_runner._cached_model_input = model_input
        mock_model_runner.model.neuron_config.on_device_sampling_config = None

        expected_output = Mock()
        expected_output.sampled_token_ids = torch.tensor([[1], [2]])
        expected_output.logprobs_tensors = None
        mock_model_runner.cpu_sampler.return_value = expected_output

        result = mock_model_runner.sample_tokens(grammar_output=None)

        assert result is not None
        mock_model_runner.cpu_sampler.assert_called_once()

    def test_sample_tokens_no_sampling_metadata_raises_error(self, mock_model_runner):
        """Test sample_tokens raises error when sampling metadata is None."""
        logits = torch.randn(2, 32000)
        model_input = ModelInputForNeuron(request_ids=["req1", "req2"])

        mock_model_runner._cached_logits = logits
        mock_model_runner._cached_model_input = model_input
        mock_model_runner.model.neuron_config.on_device_sampling_config = None
        mock_model_runner.input_batch.sampling_metadata = None

        with pytest.raises(RuntimeError, match="Sampling metadata not available"):
            mock_model_runner.sample_tokens(grammar_output=None)

    def test_sampling_configuration_validation_cpu_mode(self, mock_model_runner):
        """Test sampling configuration validation for CPU mode."""
        mock_model_runner.model.neuron_config.on_device_sampling_config = None
        mock_model_runner.model.sample = Mock()

        # Should not raise any exceptions
        mock_model_runner._validate_sampling_configuration()

    def test_sampling_configuration_validation_hardware_mode(self, mock_model_runner):
        """Test sampling configuration validation for hardware mode."""
        mock_config = Mock()
        mock_config.global_topk = 50
        mock_model_runner.model.neuron_config.on_device_sampling_config = mock_config
        mock_model_runner.model.sample = Mock()

        # Should not raise any exceptions
        mock_model_runner._validate_sampling_configuration()

    def test_sampling_configuration_validation_missing_cpu_sampler(
        self, mock_model_runner
    ):
        """Test validation fails when CPU sampler is missing."""
        mock_model_runner.model.neuron_config.on_device_sampling_config = None
        mock_model_runner.cpu_sampler = None
        mock_model_runner.model.sample = Mock()

        with pytest.raises(
            RuntimeError,
            match="CPU sampling is required but cpu_sampler is not initialized",
        ):
            mock_model_runner._validate_sampling_configuration()

    def test_sampling_configuration_validation_missing_sample_method(
        self, mock_model_runner
    ):
        """Test validation fails when model sample method is missing."""
        mock_model_runner.model.neuron_config.on_device_sampling_config = None
        if hasattr(mock_model_runner.model, "sample"):
            delattr(mock_model_runner.model, "sample")

        with pytest.raises(
            RuntimeError, match="Model does not have required 'sample' method"
        ):
            mock_model_runner._validate_sampling_configuration()

    def test_prepare_logits_for_sampling_reorders_correctly(self, mock_model_runner):
        """Test _prepare_logits_for_sampling reorders logits to match input batch order."""
        # Logits in order: req2, req1
        hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        model_input = ModelInputForNeuron(
            request_ids=["req2", "req1"],  # Different order than input_batch
            prefill_completion_state=None,
        )
        # Input batch order: req1, req2
        mock_model_runner.input_batch.req_ids = ["req1", "req2"]

        result = mock_model_runner._prepare_logits_for_sampling(
            hidden_states, model_input
        )

        # Should be reordered to match input_batch order (req1, req2)
        # req1 is at index 1 in model_input (value [3.0, 4.0]), should be first in result
        assert torch.allclose(result[0], torch.tensor([3.0, 4.0]))
        # req2 is at index 0 in model_input (value [1.0, 2.0]), should be second in result
        assert torch.allclose(result[1], torch.tensor([1.0, 2.0]))

    def test_prepare_logits_for_sampling_masks_incomplete_prefills(
        self, mock_model_runner
    ):
        """Test _prepare_logits_for_sampling masks incomplete prefill requests."""
        hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        model_input = ModelInputForNeuron(
            request_ids=["req1", "req2", "req3"],
            prefill_completion_state=torch.tensor([True, False, True]),
        )
        mock_model_runner.input_batch.req_ids = ["req1", "req2", "req3"]
        mock_model_runner.input_batch.req_id_to_index = {
            "req1": 0,
            "req2": 1,
            "req3": 2,
        }

        result = mock_model_runner._prepare_logits_for_sampling(
            hidden_states, model_input
        )

        # req2 (index 1) should have -inf logits because prefill is incomplete
        assert torch.all(result[1] == float("-inf"))
        # req1 and req3 should be unchanged
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))
        assert torch.allclose(result[2], torch.tensor([5.0, 6.0]))

    def test_sample_on_device_success(self, mock_model_runner):
        """Test _sample_on_device performs hardware sampling correctly."""
        hidden_states = torch.tensor([[1], [2]])
        model_input = ModelInputForNeuron(
            request_ids=["req1", "req2"],
            prefill_completion_state=None,
        )
        mock_model_runner.input_batch.req_ids = ["req1", "req2"]

        expected_output = Mock()
        expected_output.sampled_token_ids = torch.tensor([[10], [20]])
        mock_model_runner.model.sample = Mock(return_value=expected_output)

        result = mock_model_runner._sample_on_device(hidden_states, model_input)

        assert result == expected_output
        mock_model_runner.model.sample.assert_called_once()

    def test_sample_on_device_masks_incomplete_prefills(self, mock_model_runner):
        """Test _sample_on_device masks incomplete prefills with -1."""
        hidden_states = torch.tensor([[1], [2], [3]])
        model_input = ModelInputForNeuron(
            request_ids=["req1", "req2", "req3"],
            prefill_completion_state=torch.tensor([True, False, True]),
        )
        mock_model_runner.input_batch.req_ids = ["req1", "req2", "req3"]

        expected_output = Mock()
        mock_model_runner.model.sample = Mock(return_value=expected_output)

        mock_model_runner._sample_on_device(hidden_states, model_input)

        mock_model_runner.model.sample.assert_called_once()
        call_kwargs = mock_model_runner.model.sample.call_args.kwargs
        passed_logits = call_kwargs.get("logits")
        assert passed_logits[1].item() == -1

    def test_sample_on_device_error_handling(self, mock_model_runner):
        """Test _sample_on_device handles hardware sampling errors."""
        hidden_states = torch.tensor([[1], [2]])
        model_input = ModelInputForNeuron(
            request_ids=["req1", "req2"],
            prefill_completion_state=None,
        )
        mock_model_runner.input_batch.req_ids = ["req1", "req2"]
        mock_model_runner.model.sample = Mock(
            side_effect=RuntimeError("Hardware error")
        )

        with pytest.raises(RuntimeError, match="Hardware sampling operation failed"):
            mock_model_runner._sample_on_device(hidden_states, model_input)


class TestCPUSamplingIntegration:
    """Integration tests for CPU sampling with realistic scenarios."""

    @pytest.fixture
    def integration_model_runner(self):
        """Create a model runner for integration testing."""
        with patch.object(
            NeuronxDistributedModelRunner,
            "__init__",
            lambda self, *args, **kwargs: None,
        ):
            runner = NeuronxDistributedModelRunner.__new__(
                NeuronxDistributedModelRunner
            )

            runner.model = Mock()
            runner.model.neuron_config = Mock()
            runner.model.neuron_config.vocab_size = 32000
            runner.model.neuron_config.on_device_sampling_config = None
            runner.model.neuron_config.is_block_kv_layout = False
            runner.model.neuron_config.is_prefix_caching = False
            runner.model.neuron_config.chunked_prefill_config = None

            runner.cpu_sampler = Mock()

            # Use MagicMock for input_batch
            runner.input_batch = MagicMock()
            runner.input_batch.sampling_metadata = Mock()
            runner.input_batch.req_ids = ["req1", "req2"]
            runner.input_batch.req_id_to_index = {"req1": 0, "req2": 1}
            runner.input_batch.num_tokens_no_spec = [0, 0]
            runner.input_batch.num_tokens = [0, 0]
            runner.input_batch.token_ids_cpu = MagicMock()

            runner._cached_logits = None
            runner._cached_model_input = None
            runner.speculative_config = None
            runner.max_model_len = 1024
            runner.requests = {
                "req1": Mock(output_token_ids=[]),
                "req2": Mock(output_token_ids=[]),
            }

            return runner

    def test_end_to_end_cpu_sampling_flow(self, integration_model_runner):
        """Test the complete CPU sampling flow from cached logits to output."""
        batch_size, vocab_size = 2, 32000
        logits = torch.randn(batch_size, vocab_size)

        model_input = ModelInputForNeuron(
            request_ids=["req1", "req2"],
            prefill_completion_state=None,
        )

        # Simulate execute_model caching the logits
        integration_model_runner._cached_logits = logits
        integration_model_runner._cached_model_input = model_input

        # Mock sampler output
        sampler_output = Mock()
        sampler_output.sampled_token_ids = torch.tensor([[123], [456]])
        sampler_output.logprobs_tensors = None
        integration_model_runner.cpu_sampler.return_value = sampler_output

        # Call sample_tokens
        result = integration_model_runner.sample_tokens(grammar_output=None)

        assert result is not None
        integration_model_runner.cpu_sampler.assert_called_once()

    def test_cpu_sampling_with_different_batch_sizes(self, integration_model_runner):
        """Test CPU sampling with various batch sizes."""
        test_cases = [1, 2, 4, 8]

        for batch_size in test_cases:
            logits = torch.randn(batch_size, 32000)
            model_input = ModelInputForNeuron(
                request_ids=[f"req{i}" for i in range(batch_size)],
                prefill_completion_state=None,
            )

            # Update input_batch to match batch size
            integration_model_runner.input_batch.req_ids = [
                f"req{i}" for i in range(batch_size)
            ]
            integration_model_runner.input_batch.num_tokens_no_spec = [0] * batch_size
            integration_model_runner.input_batch.num_tokens = [0] * batch_size
            integration_model_runner.requests = {
                f"req{i}": Mock(output_token_ids=[]) for i in range(batch_size)
            }

            integration_model_runner._cached_logits = logits
            integration_model_runner._cached_model_input = model_input

            sampler_output = Mock()
            sampler_output.sampled_token_ids = torch.tensor(
                [[i] for i in range(batch_size)]
            )
            sampler_output.logprobs_tensors = None
            integration_model_runner.cpu_sampler.return_value = sampler_output

            result = integration_model_runner.sample_tokens(grammar_output=None)

            assert result is not None

            # Reset for next iteration
            integration_model_runner.cpu_sampler.reset_mock()


class TestGrammarBitmask:
    """Tests for grammar bitmask functionality."""

    @pytest.fixture
    def mock_model_runner(self):
        """Create a mock model runner for grammar bitmask testing."""
        with patch.object(
            NeuronxDistributedModelRunner,
            "__init__",
            lambda self, *args, **kwargs: None,
        ):
            runner = NeuronxDistributedModelRunner.__new__(
                NeuronxDistributedModelRunner
            )

            runner.model = Mock()
            runner.model.neuron_config = Mock()
            runner.model.neuron_config.on_device_sampling_config = None

            runner.input_batch = Mock()
            runner.input_batch.req_ids = ["req1", "req2"]

            return runner

    def test_apply_grammar_bitmask_no_bitmask(self, mock_model_runner):
        """Test that logits are unchanged when no bitmask is provided."""
        logits = torch.randn(2, 100)
        original_logits = logits.clone()

        grammar_output = Mock()
        grammar_output.grammar_bitmask = None
        grammar_output.structured_output_request_ids = []

        result = mock_model_runner._apply_grammar_bitmask_from_output(
            logits, grammar_output
        )

        assert torch.equal(result, original_logits)

    def test_apply_grammar_bitmask_empty_req_ids(self, mock_model_runner):
        """Test that logits are unchanged when structured_output_request_ids is empty."""
        logits = torch.randn(2, 100)
        original_logits = logits.clone()

        grammar_output = Mock()
        grammar_output.grammar_bitmask = np.ones((1, 4), dtype=np.int32)
        grammar_output.structured_output_request_ids = []

        result = mock_model_runner._apply_grammar_bitmask_from_output(
            logits, grammar_output
        )

        assert torch.equal(result, original_logits)

    def test_apply_packed_bitmask_to_row(self, mock_model_runner):
        """Test applying packed bitmask to a single row of logits."""
        logits_row = torch.ones(64)
        # Create bitmask that allows only first 32 tokens
        # Use -1 for all 1s in signed int32 representation
        packed_bitmask = torch.tensor([-1, 0], dtype=torch.int32)

        result = mock_model_runner._apply_packed_bitmask_to_row(
            logits_row, packed_bitmask
        )

        # First 32 tokens should be unchanged (1.0)
        assert torch.all(result[:32] == 1.0)
        # Last 32 tokens should be -inf
        assert torch.all(result[32:] == float("-inf"))

    def test_apply_grammar_bitmask_with_matching_request(self, mock_model_runner):
        """Test grammar bitmask is applied to matching request."""
        logits = torch.ones(2, 64)

        grammar_output = Mock()
        # Use -1 for all 1s in signed int32 (equivalent to 0xFFFFFFFF unsigned)
        grammar_output.grammar_bitmask = np.array([[-1, 0]], dtype=np.int32)
        grammar_output.structured_output_request_ids = ["req1"]

        result = mock_model_runner._apply_grammar_bitmask_from_output(
            logits, grammar_output
        )

        # req1 (index 0) should have bitmask applied
        assert torch.all(result[0, :32] == 1.0)
        assert torch.all(result[0, 32:] == float("-inf"))
        # req2 (index 1) should be unchanged
        assert torch.all(result[1] == 1.0)


if __name__ == "__main__":
    pytest.main([__file__])
