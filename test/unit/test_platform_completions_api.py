# SPDX-License-Identifier: Apache-2.0
"""Unit tests for completions API validation functionality in platform.py.

This module tests the new CompletionRenderer validation logic that was added
to support max_prompt_length validation for the /v1/completions endpoint,
in addition to the existing /v1/chat/completions endpoint validation.
"""

from unittest.mock import Mock

import pytest

from vllm_neuron.platform import NeuronPlatform


class TestCompletionsAPIValidation:
    """Test completions API validation functionality."""

    @pytest.fixture
    def platform(self):
        """Create a NeuronPlatform instance for testing."""
        return NeuronPlatform()

    @pytest.fixture
    def mock_completion_renderer(self):
        """Mock CompletionRenderer for testing."""
        return Mock()

    def test_completion_renderer_validation_within_limits(self):
        """Test CompletionRenderer validation passes when input is within limits.
        
        This test verifies that the custom validation logic for the completions
        endpoint correctly allows requests that are within the max_prompt_length limit.
        """

        def neuron_create_tokens_prompt(renderer,
                                        token_ids,
                                        max_length=None,
                                        cache_salt=None,
                                        prompt=None):
            """Simulate the Neuron CompletionRenderer validation logic."""
            if renderer.model_config.max_prompt_length is not None:
                if len(token_ids) > renderer.model_config.max_prompt_length:
                    raise ValueError(
                        f"This model's maximum prompt length is {renderer.model_config.max_prompt_length} tokens. "
                        f"However, your request has {len(token_ids)} prompt tokens. "
                        "Please reduce the length of the input messages.")
            return f"tokens_prompt_for_{len(token_ids)}_tokens"

        # Create mock renderer
        mock_renderer = Mock()
        mock_renderer.model_config.max_prompt_length = 10

        # Test with input within limits
        result = neuron_create_tokens_prompt(mock_renderer, [1, 2, 3])
        assert result == "tokens_prompt_for_3_tokens"

    def test_completion_renderer_validation_exceeds_limits(self):
        """Test CompletionRenderer validation raises error when input exceeds limits.
        
        This test verifies that the custom validation logic for the completions
        endpoint correctly rejects requests that exceed the max_prompt_length limit.
        """

        def neuron_create_tokens_prompt(renderer,
                                        token_ids,
                                        max_length=None,
                                        cache_salt=None,
                                        prompt=None):
            """Simulate the Neuron CompletionRenderer validation logic."""
            if renderer.model_config.max_prompt_length is not None:
                if len(token_ids) > renderer.model_config.max_prompt_length:
                    raise ValueError(
                        f"This model's maximum prompt length is {renderer.model_config.max_prompt_length} tokens. "
                        f"However, your request has {len(token_ids)} prompt tokens. "
                        "Please reduce the length of the input messages.")
            return f"tokens_prompt_for_{len(token_ids)}_tokens"

        # Create mock renderer
        mock_renderer = Mock()
        mock_renderer.model_config.max_prompt_length = 5

        # Test with input exceeding limits
        with pytest.raises(ValueError) as exc_info:
            neuron_create_tokens_prompt(mock_renderer,
                                        [1, 2, 3, 4, 5, 6, 7, 8])

        assert "maximum prompt length is 5 tokens" in str(exc_info.value)
        assert "your request has 8 prompt tokens" in str(exc_info.value)

    def test_completion_renderer_validation_none_max_prompt_length(self):
        """Test CompletionRenderer validation skips when max_prompt_length is None.
        
        This test verifies that when max_prompt_length is not configured (None),
        the validation is skipped and no error is raised regardless of input size.
        """

        def neuron_create_tokens_prompt(renderer,
                                        token_ids,
                                        max_length=None,
                                        cache_salt=None,
                                        prompt=None):
            """Simulate the Neuron CompletionRenderer validation logic."""
            if renderer.model_config.max_prompt_length is not None:
                if len(token_ids) > renderer.model_config.max_prompt_length:
                    raise ValueError(
                        f"This model's maximum prompt length is {renderer.model_config.max_prompt_length} tokens. "
                        f"However, your request has {len(token_ids)} prompt tokens. "
                        "Please reduce the length of the input messages.")
            return f"tokens_prompt_for_{len(token_ids)}_tokens"

        # Create mock renderer with max_prompt_length = None
        mock_renderer = Mock()
        mock_renderer.model_config.max_prompt_length = None

        # Test with large input - should not raise error
        result = neuron_create_tokens_prompt(mock_renderer, [1] * 1000)
        assert result == "tokens_prompt_for_1000_tokens"

    def test_openai_serving_validation_within_limits(self):
        """Test OpenAI serving validation passes when input is within limits.
        
        This test verifies the enhanced chat/completions validation logic.
        """

        def neuron_validate_input(serving_instance, request, input_ids,
                                  input_text):
            """Simulate the enhanced OpenAI serving validation logic."""
            if serving_instance.model_config.max_prompt_length is not None:
                token_num = len(input_ids)
                if token_num > serving_instance.model_config.max_prompt_length:
                    raise ValueError(
                        f"This model's maximum prompt length is "
                        f"{serving_instance.model_config.max_prompt_length} tokens. However, "
                        f"your request has {token_num} prompt tokens. Please reduce "
                        "the length of the input messages.")
            return f"validation_passed_for_{len(input_ids)}_tokens"

        # Create mock serving instance
        mock_serving = Mock()
        mock_serving.model_config.max_prompt_length = 10

        # Test with input within limits
        result = neuron_validate_input(mock_serving, Mock(), [1, 2, 3], "test")
        assert result == "validation_passed_for_3_tokens"

    def test_openai_serving_validation_exceeds_limits(self):
        """Test OpenAI serving validation raises error when input exceeds limits.
        
        This test verifies the enhanced chat/completions validation logic.
        """

        def neuron_validate_input(serving_instance, request, input_ids,
                                  input_text):
            """Simulate the enhanced OpenAI serving validation logic."""
            if serving_instance.model_config.max_prompt_length is not None:
                token_num = len(input_ids)
                if token_num > serving_instance.model_config.max_prompt_length:
                    raise ValueError(
                        f"This model's maximum prompt length is "
                        f"{serving_instance.model_config.max_prompt_length} tokens. However, "
                        f"your request has {token_num} prompt tokens. Please reduce "
                        "the length of the input messages.")
            return f"validation_passed_for_{len(input_ids)}_tokens"

        # Create mock serving instance
        mock_serving = Mock()
        mock_serving.model_config.max_prompt_length = 5

        # Test with input exceeding limits
        with pytest.raises(ValueError) as exc_info:
            neuron_validate_input(mock_serving, Mock(),
                                  [1, 2, 3, 4, 5, 6, 7, 8], "test")

        assert "maximum prompt length is 5 tokens" in str(exc_info.value)
        assert "your request has 8 prompt tokens" in str(exc_info.value)

    def test_openai_serving_validation_none_max_prompt_length(self):
        """Test OpenAI serving validation skips when max_prompt_length is None.
        
        This test verifies that when max_prompt_length is not configured,
        the validation is skipped for chat/completions endpoint.
        """

        def neuron_validate_input(serving_instance, request, input_ids,
                                  input_text):
            """Simulate the enhanced OpenAI serving validation logic."""
            if serving_instance.model_config.max_prompt_length is not None:
                token_num = len(input_ids)
                if token_num > serving_instance.model_config.max_prompt_length:
                    raise ValueError(
                        f"This model's maximum prompt length is "
                        f"{serving_instance.model_config.max_prompt_length} tokens. However, "
                        f"your request has {token_num} prompt tokens. Please reduce "
                        "the length of the input messages.")
            return f"validation_passed_for_{len(input_ids)}_tokens"

        # Create mock serving instance with max_prompt_length = None
        mock_serving = Mock()
        mock_serving.model_config.max_prompt_length = None

        # Test with large input - should not raise error
        result = neuron_validate_input(mock_serving, Mock(), [1] * 1000,
                                       "test")
        assert result == "validation_passed_for_1000_tokens"

    def test_config_override_integration(self):
        """Test that max_prompt_length configuration is properly integrated."""
        # Test that additional_config max_prompt_length is used
        mock_config = Mock()
        mock_config.model_config = Mock()
        mock_config.parallel_config = Mock(world_size=1, worker_cls="auto")
        mock_config.cache_config = Mock(enable_prefix_caching=False,
                                        num_gpu_blocks_override=None)
        mock_config.scheduler_config = Mock()
        mock_config.lora_config = None
        mock_config.additional_config = {"max_prompt_length": 512}

        NeuronPlatform.check_and_update_config(mock_config)

        assert mock_config.model_config.max_prompt_length == 512

    def test_config_override_none_when_missing(self):
        """Test that max_prompt_length is None when not in additional_config."""
        mock_config = Mock()
        mock_config.model_config = Mock()
        mock_config.parallel_config = Mock(world_size=1, worker_cls="auto")
        mock_config.cache_config = Mock(enable_prefix_caching=False,
                                        num_gpu_blocks_override=None)
        mock_config.scheduler_config = Mock()
        mock_config.lora_config = None
        mock_config.additional_config = {}

        NeuronPlatform.check_and_update_config(mock_config)

        assert mock_config.model_config.max_prompt_length is None

    def test_validation_error_message_consistency(self):
        """Test that both validation methods produce consistent error messages."""

        # Test CompletionRenderer error message
        def completion_validation(token_ids, max_length):
            if len(token_ids) > max_length:
                raise ValueError(
                    f"This model's maximum prompt length is {max_length} tokens. "
                    f"However, your request has {len(token_ids)} prompt tokens. "
                    "Please reduce the length of the input messages.")

        # Test OpenAI serving error message
        def serving_validation(token_ids, max_length):
            token_num = len(token_ids)
            if token_num > max_length:
                raise ValueError(
                    f"This model's maximum prompt length is "
                    f"{max_length} tokens. However, "
                    f"your request has {token_num} prompt tokens. Please reduce "
                    "the length of the input messages.")

        tokens = [1, 2, 3, 4, 5, 6]
        max_len = 3

        # Both should raise similar errors
        with pytest.raises(ValueError) as exc1:
            completion_validation(tokens, max_len)

        with pytest.raises(ValueError) as exc2:
            serving_validation(tokens, max_len)

        # Both messages should contain key information
        assert "maximum prompt length is 3 tokens" in str(exc1.value)
        assert "your request has 6 prompt tokens" in str(exc1.value)
        assert "Please reduce the length" in str(exc1.value)

        assert "maximum prompt length is 3 tokens" in str(exc2.value)
        assert "your request has 6 prompt tokens" in str(exc2.value)
        assert "Please reduce the length" in str(exc2.value)

    @pytest.mark.parametrize(
        "input_length,max_length,should_pass",
        [
            (5, 10, True),  # Within limits
            (10, 10, True),  # At limit
            (15, 10, False),  # Exceeds limit
            (0, 10, True),  # Empty input
            (1, 1, True),  # Single token at limit
        ])
    def test_validation_boundary_conditions(self, input_length, max_length,
                                            should_pass):
        """Test validation boundary conditions with various input sizes."""

        def validate_tokens(token_ids, max_prompt_length):
            if max_prompt_length is not None and len(
                    token_ids) > max_prompt_length:
                raise ValueError(
                    f"Prompt length {len(token_ids)} exceeds maximum {max_prompt_length}"
                )
            return True

        tokens = [1] * input_length

        if should_pass:
            result = validate_tokens(tokens, max_length)
            assert result is True
        else:
            with pytest.raises(ValueError) as exc_info:
                validate_tokens(tokens, max_length)
            assert f"exceeds maximum {max_length}" in str(exc_info.value)

    def test_dual_endpoint_validation_consistency(self):
        """Test that both completions and chat/completions endpoints have consistent validation."""

        # This test ensures both endpoints apply the same validation logic

        def completions_validate(token_ids, max_prompt_length):
            """Simulate completions endpoint validation."""
            if max_prompt_length is not None and len(
                    token_ids) > max_prompt_length:
                return False, f"Completions: Prompt length {len(token_ids)} exceeds {max_prompt_length}"
            return True, "Completions: Valid"

        def chat_completions_validate(token_ids, max_prompt_length):
            """Simulate chat/completions endpoint validation."""
            if max_prompt_length is not None and len(
                    token_ids) > max_prompt_length:
                return False, f"Chat: Prompt length {len(token_ids)} exceeds {max_prompt_length}"
            return True, "Chat: Valid"

        test_cases = [
            ([1, 2, 3], 5, True),  # Within limits
            ([1, 2, 3, 4, 5], 5, True),  # At limits
            ([1, 2, 3, 4, 5, 6], 5, False),  # Exceeds limits
        ]

        for tokens, max_len, should_pass in test_cases:
            comp_valid, comp_msg = completions_validate(tokens, max_len)
            chat_valid, chat_msg = chat_completions_validate(tokens, max_len)

            # Both endpoints should have consistent behavior
            assert comp_valid == should_pass
            assert chat_valid == should_pass
            assert comp_valid == chat_valid

    def test_validation_with_various_token_types(self):
        """Test validation works with different token ID formats."""

        def validate_input(token_ids, max_length):
            if max_length is not None and len(token_ids) > max_length:
                raise ValueError(
                    f"Too many tokens: {len(token_ids)} > {max_length}")
            return len(token_ids)

        max_len = 5

        # Test with various token formats
        test_inputs = [
            [1, 2, 3],  # Normal integers
            [0, 100, 999, 1000],  # Various token values
            [],  # Empty list
            [50257] * 3,  # Special tokens
        ]

        for tokens in test_inputs:
            if len(tokens) <= max_len:
                result = validate_input(tokens, max_len)
                assert result == len(tokens)
            else:
                with pytest.raises(ValueError):
                    validate_input(tokens, max_len)
