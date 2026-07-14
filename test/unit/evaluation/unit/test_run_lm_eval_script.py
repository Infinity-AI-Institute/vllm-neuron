# SPDX-License-Identifier: Apache-2.0
"""Unit tests for run_lm_eval.py script functions."""

import os
from unittest.mock import Mock, patch

import pytest

from test.evaluation.clients.lm_eval.scripts.run_lm_eval import (
    build_lm_eval_command,
    run_lm_eval,
    str_to_bool,
)


class TestStrToBool:
    """Test str_to_bool helper function."""

    def test_true_values(self):
        """Test values that should return True."""
        assert str_to_bool("true") is True
        assert str_to_bool("TRUE") is True
        assert str_to_bool("1") is True
        assert str_to_bool("yes") is True
        assert str_to_bool("on") is True

    def test_false_values(self):
        """Test values that should return False."""
        assert str_to_bool("false") is False
        assert str_to_bool("FALSE") is False
        assert str_to_bool("0") is False
        assert str_to_bool("no") is False
        assert str_to_bool("off") is False
        assert str_to_bool("random") is False


class TestBuildLmEvalCommand:
    """Test build_lm_eval_command function."""

    def test_all_parameters(self):
        """Test command building with all parameters."""
        cmd = build_lm_eval_command(
            model="test_model",
            model_path="/path/to/model",
            max_concurrent_requests=4,
            port=9000,
            task_name="custom_task",
            results_dir="custom_results",
            timeout=3600,
            limit=100,
            use_chat=True,
            gen_kwargs='{"max_tokens": 512}',
            max_length=4096,
            system_prompt=True,
            fewshot_as_multiturn=True,
        )

        # Verify all parameters are included
        cmd_str = str(cmd)
        assert "/path/to/model" in cmd_str
        assert "num_concurrent=4" in cmd_str
        assert "localhost:9000" in cmd_str
        assert "custom_task" in cmd
        assert "custom_results" in cmd
        assert "timeout=3600" in cmd_str
        assert "100" in cmd
        assert "local-chat-completions" in cmd
        assert "--apply_chat_template" in cmd
        assert "max_length=4096" in cmd_str
        assert "--system_instruction" in cmd
        assert "--fewshot_as_multiturn" in cmd
        assert "--gen_kwargs" in cmd
        assert '{"max_tokens": 512}' in cmd

    def test_basic_command_build(self):
        """Test basic command building."""
        cmd = build_lm_eval_command(
            model="test_model",
            model_path="/path/to/model",
            max_concurrent_requests=1,
            port=8000,
            task_name="gsm8k_cot",
            results_dir="results",
            timeout=7200,
            limit=200,
            use_chat=False,
            gen_kwargs="{}",
        )

        assert cmd[0] == "python"
        assert "--model" in cmd
        assert "local-completions" in cmd
        assert "--tasks" in cmd
        assert "gsm8k_cot" in cmd

    def test_chat_completions_mode(self):
        """Test command building with chat completions."""
        cmd = build_lm_eval_command(
            model="test_model",
            model_path="/path/to/model",
            max_concurrent_requests=1,
            port=8000,
            task_name="gsm8k_cot",
            results_dir="results",
            timeout=7200,
            limit=200,
            use_chat=True,
            gen_kwargs="{}",
        )

        assert "local-chat-completions" in cmd
        assert "--apply_chat_template" in cmd

    def test_mbpp_task_handling(self):
        """Test MBPP task specific handling."""
        cmd = build_lm_eval_command(
            model="test_model",
            model_path="/path/to/model",
            max_concurrent_requests=1,
            port=8000,
            task_name="mbpp",
            results_dir="results",
            timeout=7200,
            limit=200,
            use_chat=False,
            gen_kwargs="{}",
        )

        assert "--confirm_run_unsafe_code" in cmd

    def test_system_prompt_flag(self):
        """Test system prompt flag inclusion."""
        cmd = build_lm_eval_command(
            model="test_model",
            model_path="/path/to/model",
            max_concurrent_requests=1,
            port=8000,
            task_name="gsm8k_cot",
            results_dir="results",
            timeout=7200,
            limit=200,
            use_chat=False,
            gen_kwargs="{}",
            system_prompt=True,
        )

        assert "--system_instruction" in cmd
        assert "true" in cmd

    def test_system_prompt_disabled(self):
        """Test system prompt flag disabled."""
        cmd = build_lm_eval_command(
            model="test_model",
            model_path="/path/to/model",
            max_concurrent_requests=1,
            port=8000,
            task_name="gsm8k_cot",
            results_dir="results",
            timeout=7200,
            limit=200,
            use_chat=False,
            gen_kwargs="{}",
            system_prompt=False,
        )

        assert "--system_instruction" not in cmd

    def test_fewshot_as_multiturn_flag(self):
        """Test fewshot_as_multiturn flag inclusion."""
        cmd = build_lm_eval_command(
            model="test_model",
            model_path="/path/to/model",
            max_concurrent_requests=1,
            port=8000,
            task_name="gsm8k_cot",
            results_dir="results",
            timeout=7200,
            limit=200,
            use_chat=False,
            gen_kwargs="{}",
            fewshot_as_multiturn=True,
        )

        assert "--fewshot_as_multiturn" in cmd

    def test_fewshot_as_multiturn_disabled(self):
        """Test fewshot_as_multiturn flag disabled."""
        cmd = build_lm_eval_command(
            model="test_model",
            model_path="/path/to/model",
            max_concurrent_requests=1,
            port=8000,
            task_name="gsm8k_cot",
            results_dir="results",
            timeout=7200,
            limit=200,
            use_chat=False,
            gen_kwargs="{}",
            fewshot_as_multiturn=False,
        )

        assert "--fewshot_as_multiturn" not in cmd

    def test_max_length_inclusion(self):
        """Test max_length parameter inclusion."""
        cmd = build_lm_eval_command(
            model="test_model",
            model_path="/path/to/model",
            max_concurrent_requests=1,
            port=8000,
            task_name="gsm8k_cot",
            results_dir="results",
            timeout=7200,
            limit=200,
            use_chat=False,
            gen_kwargs="{}",
            max_length=2048,
        )

        model_args_idx = cmd.index("--model_args") + 1
        assert "max_length=2048" in cmd[model_args_idx]

    def test_max_length_none(self):
        """Test max_length parameter when None."""
        cmd = build_lm_eval_command(
            model="test_model",
            model_path="/path/to/model",
            max_concurrent_requests=1,
            port=8000,
            task_name="gsm8k_cot",
            results_dir="results",
            timeout=7200,
            limit=200,
            use_chat=False,
            gen_kwargs="{}",
            max_length=None,
        )

        model_args_idx = cmd.index("--model_args") + 1
        assert "max_length=" not in cmd[model_args_idx]

    def test_gen_kwargs_inclusion(self):
        """Test gen_kwargs parameter inclusion."""
        gen_kwargs = '{"max_tokens": 256}'
        cmd = build_lm_eval_command(
            model="test_model",
            model_path="/path/to/model",
            max_concurrent_requests=1,
            port=8000,
            task_name="gsm8k_cot",
            results_dir="results",
            timeout=7200,
            limit=200,
            use_chat=False,
            gen_kwargs=gen_kwargs,
        )

        assert "--gen_kwargs" in cmd
        assert gen_kwargs in cmd

    def test_gen_kwargs_empty(self):
        """Test gen_kwargs parameter when empty."""
        cmd = build_lm_eval_command(
            model="test_model",
            model_path="/path/to/model",
            max_concurrent_requests=1,
            port=8000,
            task_name="gsm8k_cot",
            results_dir="results",
            timeout=7200,
            limit=200,
            use_chat=False,
            gen_kwargs="{}",
        )

        assert "--gen_kwargs" not in cmd

    def test_different_ports(self):
        """Test different port values."""
        for port in [8000, 8080, 9000]:
            cmd = build_lm_eval_command(
                model="test_model",
                model_path="/path/to/model",
                max_concurrent_requests=1,
                port=port,
                task_name="gsm8k_cot",
                results_dir="results",
                timeout=7200,
                limit=200,
                use_chat=False,
                gen_kwargs="{}",
            )
            assert f"localhost:{port}" in str(cmd)

    def test_different_limits(self):
        """Test different limit values."""
        for limit in [50, 100, 500]:
            cmd = build_lm_eval_command(
                model="test_model",
                model_path="/path/to/model",
                max_concurrent_requests=1,
                port=8000,
                task_name="gsm8k_cot",
                results_dir="results",
                timeout=7200,
                limit=limit,
                use_chat=False,
                gen_kwargs="{}",
            )
            assert str(limit) in cmd


class TestRunLmEval:
    """Test run_lm_eval function."""

    @patch("test.evaluation.clients.lm_eval.scripts.run_lm_eval.subprocess.run")
    @patch("test.evaluation.clients.lm_eval.scripts.run_lm_eval.Path.exists")
    def test_run_lm_eval_basic(self, mock_exists, mock_subprocess):
        """Test basic run_lm_eval execution."""
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock()

        run_lm_eval(
            model="test_model",
            model_path="/path/to/model",
        )

        mock_subprocess.assert_called_once()
        assert os.environ.get("OPENAI_API_KEY") == "EMPTY"
        assert os.environ.get("HF_HUB_OFFLINE") is None

    @patch("test.evaluation.clients.lm_eval.scripts.run_lm_eval.subprocess.run")
    @patch("test.evaluation.clients.lm_eval.scripts.run_lm_eval.Path.exists")
    def test_run_lm_eval_all_parameters(self, mock_exists, mock_subprocess):
        """Test run_lm_eval with all parameters."""
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock()

        run_lm_eval(
            model="test_model",
            model_path="/path/to/model",
            max_concurrent_requests=4,
            port=9000,
            task_name="custom_task",
            results_dir="custom_results",
            timeout=3600,
            limit=100,
            use_chat=False,
            gen_kwargs='{"max_tokens": 512}',
            max_length=4096,
            system_prompt=False,
            fewshot_as_multiturn=True,
        )

        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]

        # Verify all parameters are passed through
        assert any("custom_task" in arg for arg in call_args)
        assert any("custom_results" in arg for arg in call_args)
        assert any("100" in arg for arg in call_args)
        assert any("max_length=4096" in arg for arg in call_args)

    @patch("test.evaluation.clients.lm_eval.scripts.run_lm_eval.subprocess.run")
    @patch("test.evaluation.clients.lm_eval.scripts.run_lm_eval.Path.exists")
    def test_run_lm_eval_with_custom_params(self, mock_exists, mock_subprocess):
        """Test run_lm_eval with custom parameters."""
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock()

        run_lm_eval(
            model="test_model",
            model_path="/path/to/model",
            system_prompt=False,
            fewshot_as_multiturn=True,
            max_length=1024,
        )

        mock_subprocess.assert_called_once()
        # Verify the command was built with custom parameters
        call_args = mock_subprocess.call_args[0][0]
        assert any("max_length=1024" in arg for arg in call_args)

    @patch("test.evaluation.clients.lm_eval.scripts.run_lm_eval.Path.exists")
    def test_run_lm_eval_venv_not_found(self, mock_exists):
        """Test run_lm_eval raises error when venv not found."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError):
            run_lm_eval(
                model="test_model",
                model_path="/path/to/model",
            )

    @patch("test.evaluation.clients.lm_eval.scripts.run_lm_eval.subprocess.run")
    @patch("test.evaluation.clients.lm_eval.scripts.run_lm_eval.Path.exists")
    def test_environment_variables_set(self, mock_exists, mock_subprocess):
        """Test that environment variables are properly set."""
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock()

        run_lm_eval(
            model="test_model",
            model_path="/path/to/model",
            port=9999,
        )

        assert os.environ.get("OPENAI_API_KEY") == "EMPTY"
        assert os.environ.get("OPENAI_API_BASE") == "http://localhost:9999/v1"
        assert os.environ.get("HF_HUB_OFFLINE") is None
