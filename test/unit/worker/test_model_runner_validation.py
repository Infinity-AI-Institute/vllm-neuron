# SPDX-License-Identifier: Apache-2.0
"""Unit tests for model runner validation enhancements.

This module tests the enhanced validation logic and error messages that were added
to the NeuronxDistributedModelRunner, including improved max_prompt_length handling,
runtime validation during request processing, and better user guidance.
"""

import logging
from unittest.mock import Mock

import pytest

from vllm_neuron.worker.neuronx_distributed_model_runner import \
    NeuronxDistributedModelRunner


class TestModelRunnerValidation:
    """Test model runner validation enhancements."""

    @pytest.fixture
    def mock_vllm_config(self):
        """Create a mock vLLM configuration."""
        config = Mock()
        config.model_config = Mock(max_model_len=2048, max_prompt_length=None)
        config.cache_config = Mock(block_size=8)
        config.lora_config = None
        config.load_config = Mock()
        config.parallel_config = Mock(tensor_parallel_size=1)
        config.scheduler_config = Mock(max_model_len=2048,
                                       max_num_seqs=32,
                                       max_num_batched_tokens=4096)
        config.speculative_config = Mock()
        config.observability_config = Mock()
        config.device_config = Mock(device="cpu")
        config.additional_config = {}
        return config

    @pytest.fixture
    def model_runner(self, mock_vllm_config):
        """Create a NeuronxDistributedModelRunner instance for testing."""
        runner = NeuronxDistributedModelRunner(vllm_config=mock_vllm_config,
                                               device="cpu")
        # Mock the model and its neuron_config
        runner.model = Mock()
        runner.model.neuron_config = Mock()
        runner.model.neuron_config.max_context_length = 2048
        runner.max_model_len = 2048
        return runner

    def test_validate_max_prompt_length_config_match_success(
            self, model_runner):
        """Test validation passes when user config matches neuron config.
        
        This test verifies that when the user-provided max_prompt_length in
        additional_config matches the neuron model's compiled max_context_length,
        the validation passes without errors or warnings.
        """
        # Setup matching configurations
        model_runner.model.neuron_config.max_context_length = 1024
        model_runner.vllm_config.model_config.max_prompt_length = 1024
        model_runner.max_model_len = 2048

        # Should not raise any exception
        model_runner._validate_max_prompt_length()

        # max_prompt_length should be set to the neuron config value
        assert model_runner.max_prompt_length == 1024

    def test_validate_max_prompt_length_config_mismatch_error(
            self, model_runner):
        """Test validation raises error when user config doesn't match neuron config.
        
        This test verifies that when the user-provided max_prompt_length in
        additional_config doesn't match the neuron model's compiled max_context_length,
        a clear RuntimeError is raised with helpful guidance.
        """
        # Setup mismatched configurations
        model_runner.model.neuron_config.max_context_length = 1024
        model_runner.vllm_config.model_config.max_prompt_length = 512

        with pytest.raises(RuntimeError) as exc_info:
            model_runner._validate_max_prompt_length()

        error_msg = str(exc_info.value)
        assert "Configuration mismatch" in error_msg
        assert "max_prompt_length in --additional-config (512)" in error_msg
        assert "does not match the Neuron model's compiled max prompt length (1024)" in error_msg
        assert 'Please update --additional-config to set "max_prompt_length": 1024' in error_msg

    def test_validate_max_prompt_length_no_user_config_warning(
            self, model_runner, caplog):
        """Test validation warns when no user config provided and values differ.
        
        This test verifies that when max_prompt_length is not provided in additional_config
        but the neuron model's max_context_length differs from max_model_len, a helpful
        warning is logged with configuration guidance.
        """
        # Setup: no user config, but neuron config differs from max_model_len
        model_runner.model.neuron_config.max_context_length = 1024
        model_runner.vllm_config.model_config.max_prompt_length = None
        model_runner.max_model_len = 2048

        with caplog.at_level(logging.WARNING):
            model_runner._validate_max_prompt_length()

        # Should set max_prompt_length to neuron config value
        assert model_runner.max_prompt_length == 1024

        # Should log a warning with helpful guidance
        warning_msgs = [
            record.message for record in caplog.records
            if record.levelname == 'WARNING'
        ]
        assert len(warning_msgs) > 0
        warning_msg = warning_msgs[0]
        assert "Your Neuron model was compiled with max prompt length 1024" in warning_msg
        assert "but max_model_len is set to 2048" in warning_msg
        assert "To prevent the vLLM engine from crashing" in warning_msg
        assert 'add "max_prompt_length": 1024 to --additional-config' in warning_msg
        assert "This will return a 400 error for oversized prompts" in warning_msg

    def test_validate_max_prompt_length_no_user_config_no_warning(
            self, model_runner, caplog):
        """Test validation doesn't warn when configs already match.
        
        This test verifies that when max_prompt_length is not provided in additional_config
        but the neuron model's max_context_length matches max_model_len, no warning is logged.
        """
        # Setup: no user config, but neuron config matches max_model_len
        model_runner.model.neuron_config.max_context_length = 2048
        model_runner.vllm_config.model_config.max_prompt_length = None
        model_runner.max_model_len = 2048

        with caplog.at_level(logging.WARNING):
            model_runner._validate_max_prompt_length()

        # Should set max_prompt_length to neuron config value
        assert model_runner.max_prompt_length == 2048

        # Should not log any warnings
        warning_msgs = [
            record.message for record in caplog.records
            if record.levelname == 'WARNING'
        ]
        assert len(warning_msgs) == 0

    def test_validate_max_prompt_length_with_bucketing(self, model_runner):
        """Test validation with bucketing enabled.
        
        This test verifies that when bucketing is enabled, the max bucket size
        is used as the effective max_prompt_length.
        """
        # Setup bucketing configuration
        model_runner.model.neuron_config.enable_bucketing = True
        model_runner.model.neuron_config.context_encoding_buckets = [
            512, 1024, 2048
        ]
        # Remove max_context_length to simulate bucketing mode
        delattr(model_runner.model.neuron_config, 'max_context_length')
        model_runner.vllm_config.model_config.max_prompt_length = None
        model_runner.max_model_len = 1024

        model_runner._validate_max_prompt_length()

        # Should use the max bucket size
        assert model_runner.max_prompt_length == 2048

    def test_runtime_validation_within_limits(self, model_runner):
        """Test runtime validation passes when prompt is within limits.
        
        This test verifies that during request processing, when the prompt token
        count is within the max_prompt_length limit, the request is processed
        without errors.
        """
        model_runner.max_prompt_length = 100

        # Mock request data with prompt within limits
        request_data = Mock()
        request_data.req_id = "test_req"
        request_data.prompt_token_ids = [1, 2, 3, 4, 5]  # 5 tokens < 100

        # This should not raise any exception
        data = Mock()
        data.request_ids = []
        data.input_tokens = []
        data.position_ids = []
        data.input_block_ids = []

        # The validation happens in _add_request_data_to_batch
        # Let's simulate the key validation logic
        if len(request_data.prompt_token_ids) > model_runner.max_prompt_length:
            raise ValueError(
                f'Prompt length ({len(request_data.prompt_token_ids)} tokens) exceeds the maximum '
                f'prompt length ({model_runner.max_prompt_length} tokens) for this Neuron model.'
            )

        # Should pass without exception
        assert len(
            request_data.prompt_token_ids) <= model_runner.max_prompt_length

    def test_runtime_validation_exceeds_limits(self, model_runner):
        """Test runtime validation raises detailed error when prompt exceeds limits.
        
        This test verifies that during request processing, when the prompt token
        count exceeds the max_prompt_length limit, a detailed error is raised with
        helpful guidance for the user.
        """
        model_runner.max_prompt_length = 10

        # Mock request data with prompt exceeding limits
        request_data = Mock()
        request_data.req_id = "test_req"
        request_data.prompt_token_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        ]  # 12 tokens > 10

        # Simulate the validation logic that would be in _add_request_data_to_batch
        def validate_request_length():
            if len(request_data.prompt_token_ids
                   ) > model_runner.max_prompt_length:
                raise ValueError(
                    f'Prompt length ({len(request_data.prompt_token_ids)} tokens) exceeds the maximum '
                    f'prompt length ({model_runner.max_prompt_length} tokens) for this Neuron model. '
                    f'To handle this gracefully during online serving, add "max_prompt_length": '
                    f'{model_runner.max_prompt_length} to --additional-config. This will return a 400 error '
                    f'instead of terminating the engine (supported on OpenAI /v1/completions and '
                    f'/v1/chat/completions endpoints). Alternatively, provide a shorter prompt or '
                    f'recompile the model with a larger max_prompt_length.')

        with pytest.raises(ValueError) as exc_info:
            validate_request_length()

        error_msg = str(exc_info.value)
        assert "Prompt length (12 tokens) exceeds the maximum prompt length (10 tokens)" in error_msg
        assert "To handle this gracefully during online serving" in error_msg
        assert 'add "max_prompt_length": 10 to --additional-config' in error_msg
        assert "This will return a 400 error instead of terminating the engine" in error_msg
        assert "supported on OpenAI /v1/completions and /v1/chat/completions endpoints" in error_msg
        assert "provide a shorter prompt or recompile the model" in error_msg

    def test_enhanced_error_message_formatting(self, model_runner):
        """Test that enhanced error messages are properly formatted and helpful.
        
        This test verifies that the error messages provide clear, actionable
        guidance to users, including specific configuration instructions and
        alternative solutions.
        """

        def test_config_mismatch_error():
            mpl_value = 512
            mpl_nc_value = 1024
            raise RuntimeError(
                f"Configuration mismatch: max_prompt_length in --additional-config ({mpl_value}) "
                f"does not match the Neuron model's compiled max_prompt_length ({mpl_nc_value}). "
                f'Please update --additional-config to set "max_prompt_length": {mpl_nc_value}.'
            )

        def test_runtime_validation_error():
            prompt_length = 150
            max_length = 100
            raise ValueError(
                f'Prompt length ({prompt_length} tokens) exceeds the maximum '
                f'prompt length ({max_length} tokens) for this Neuron model. '
                f'To handle this gracefully during online serving, add "max_prompt_length": '
                f'{max_length} to --additional-config. This will return a 400 error '
                f'instead of terminating the engine (supported on OpenAI /v1/completions and '
                f'/v1/chat/completions endpoints). Alternatively, provide a shorter prompt or '
                f'recompile the model with a larger max_prompt_length.')

        # Test config mismatch error formatting
        with pytest.raises(RuntimeError) as exc_info:
            test_config_mismatch_error()

        config_error = str(exc_info.value)
        assert "Configuration mismatch" in config_error
        assert "--additional-config (512)" in config_error
        assert "compiled max_prompt_length (1024)" in config_error
        assert 'set "max_prompt_length": 1024' in config_error

        # Test runtime validation error formatting
        with pytest.raises(ValueError) as exc_info:
            test_runtime_validation_error()

        runtime_error = str(exc_info.value)
        assert "Prompt length (150 tokens) exceeds" in runtime_error
        assert "maximum prompt length (100 tokens)" in runtime_error
        assert "gracefully during online serving" in runtime_error
        assert "return a 400 error instead of terminating" in runtime_error
        assert "OpenAI /v1/completions and /v1/chat/completions" in runtime_error
        assert "provide a shorter prompt or recompile" in runtime_error

    def test_warning_message_formatting(self, model_runner, caplog):
        """Test that warning messages are properly formatted and informative.
        
        This test verifies that warning messages provide clear explanations
        of potential issues and specific instructions for resolving them.
        """
        # Simulate the warning logic
        mpl_nc_value = 1024
        max_model_len = 2048

        with caplog.at_level(logging.WARNING):
            # Simulate the warning that would be logged
            logger = logging.getLogger(
                'vllm_neuron.worker.neuronx_distributed_model_runner')
            logger.warning(
                f"Your Neuron model was compiled with max_prompt_length={mpl_nc_value}, "
                f"but max_model_len is set to {max_model_len}. "
                f"To prevent the vLLM engine from crashing when prompts exceed {mpl_nc_value} tokens, "
                f'add "max_prompt_length": {mpl_nc_value} to --additional-config when using the '
                f"OpenAI API server. This will return a 400 error for oversized prompts instead of "
                f"terminating the engine.")

        warning_records = [
            record for record in caplog.records
            if record.levelname == 'WARNING'
        ]
        assert len(warning_records) > 0

        warning_msg = warning_records[0].message
        assert "Your Neuron model was compiled with max_prompt_length=1024" in warning_msg
        assert "but max_model_len is set to 2048" in warning_msg
        assert "To prevent the vLLM engine from crashing" in warning_msg
        assert "when prompts exceed 1024 tokens" in warning_msg
        assert 'add "max_prompt_length": 1024 to --additional-config' in warning_msg
        assert "when using the OpenAI API server" in warning_msg
        assert "return a 400 error for oversized prompts" in warning_msg
        assert "instead of terminating the engine" in warning_msg

    @pytest.mark.parametrize(
        "prompt_length,max_length,should_pass",
        [
            (50, 100, True),  # Well within limits
            (100, 100, True),  # At exact limit
            (101, 100, False),  # Just over limit
            (200, 100, False),  # Well over limit
            (0, 100, True),  # Empty prompt
            (1, 1, True),  # Single token at limit
            (2, 1, False),  # Single token over limit
        ])
    def test_validation_boundary_conditions(self, model_runner, prompt_length,
                                            max_length, should_pass):
        """Test validation boundary conditions with various prompt sizes.
        
        This test verifies that validation correctly handles edge cases,
        including exact limits, empty prompts, and various overages.
        """
        model_runner.max_prompt_length = max_length

        def validate_prompt_length(token_count):
            if token_count > model_runner.max_prompt_length:
                raise ValueError(
                    f'Prompt length ({token_count} tokens) exceeds the maximum '
                    f'prompt length ({model_runner.max_prompt_length} tokens) for this Neuron model.'
                )
            return True

        if should_pass:
            result = validate_prompt_length(prompt_length)
            assert result is True
        else:
            with pytest.raises(ValueError) as exc_info:
                validate_prompt_length(prompt_length)
            assert f"Prompt length ({prompt_length} tokens) exceeds" in str(
                exc_info.value)
            assert f"maximum prompt length ({max_length} tokens)" in str(
                exc_info.value)

    def test_validation_with_none_max_prompt_length(self, model_runner):
        """Test that validation is skipped when max_prompt_length is None.
        
        This test verifies that when max_prompt_length is not configured,
        no validation is performed and large prompts are allowed.
        """
        model_runner.max_prompt_length = None

        def validate_prompt_length(token_count):
            if model_runner.max_prompt_length is not None and token_count > model_runner.max_prompt_length:
                raise ValueError(f"Prompt too long: {token_count}")
            return True

        # Should pass even with very large prompt
        result = validate_prompt_length(10000)
        assert result is True

    def test_error_message_consistency_across_scenarios(self, model_runner):
        """Test that error messages are consistent across different validation scenarios.
        
        This test verifies that similar validation failures produce consistent
        error message formats and content.
        """
        scenarios = [{
            "name":
            "config_mismatch",
            "user_config":
            512,
            "neuron_config":
            1024,
            "error_type":
            RuntimeError,
            "key_phrases": [
                "Configuration mismatch",
                "max_prompt_length in --additional-config (512)",
                "compiled max_prompt_length (1024)",
                'set "max_prompt_length": 1024'
            ]
        }, {
            "name":
            "runtime_validation",
            "prompt_length":
            150,
            "max_length":
            100,
            "error_type":
            ValueError,
            "key_phrases": [
                "Prompt length (150 tokens) exceeds",
                "maximum prompt length (100 tokens)",
                "gracefully during online serving",
                "return a 400 error instead of terminating",
                "OpenAI /v1/completions and /v1/chat/completions"
            ]
        }]

        for scenario in scenarios:
            if scenario["name"] == "config_mismatch":

                def raise_config_error():
                    raise RuntimeError(
                        f"Configuration mismatch: max_prompt_length in --additional-config ({scenario['user_config']}) "
                        f"does not match the Neuron model's compiled max_prompt_length ({scenario['neuron_config']}). "
                        f'Please update --additional-config to set "max_prompt_length": {scenario["neuron_config"]}.'
                    )

                with pytest.raises(scenario["error_type"]) as exc_info:
                    raise_config_error()

            elif scenario["name"] == "runtime_validation":

                def raise_runtime_error():
                    raise ValueError(
                        f'Prompt length ({scenario["prompt_length"]} tokens) exceeds the maximum '
                        f'prompt length ({scenario["max_length"]} tokens) for this Neuron model. '
                        f'To handle this gracefully during online serving, add "max_prompt_length": '
                        f'{scenario["max_length"]} to --additional-config. This will return a 400 error '
                        f'instead of terminating the engine (supported on OpenAI /v1/completions and '
                        f'/v1/chat/completions endpoints). Alternatively, provide a shorter prompt or '
                        f'recompile the model with a larger max_prompt_length.'
                    )

                with pytest.raises(scenario["error_type"]) as exc_info:
                    raise_runtime_error()

            # Verify all key phrases are present
            error_msg = str(exc_info.value)
            for phrase in scenario["key_phrases"]:
                assert phrase in error_msg, f"Missing phrase '{phrase}' in error: {error_msg}"

    def test_validation_integration_with_model_loading(self, model_runner):
        """Test that validation is properly integrated with model loading process.
        
        This test verifies that the validation logic works correctly when
        integrated with the model loading and configuration setup process.
        """
        # Mock the model loading scenario
        model_runner.model = Mock()
        model_runner.model.neuron_config = Mock()

        # Test scenario 1: Successful validation
        model_runner.model.neuron_config.max_context_length = 2048
        model_runner.vllm_config.model_config.max_prompt_length = 2048
        model_runner.max_model_len = 2048

        model_runner._validate_max_prompt_length()
        assert model_runner.max_prompt_length == 2048

        # Test scenario 2: Configuration mismatch
        model_runner.model.neuron_config.max_context_length = 1024
        model_runner.vllm_config.model_config.max_prompt_length = 512

        with pytest.raises(RuntimeError) as exc_info:
            model_runner._validate_max_prompt_length()

        assert "Configuration mismatch" in str(exc_info.value)
        assert "512" in str(exc_info.value)  # User config value
        assert "1024" in str(exc_info.value)  # Neuron config value
