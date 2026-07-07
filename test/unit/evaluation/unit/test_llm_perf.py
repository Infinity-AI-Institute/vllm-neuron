# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LLMPerfClient class."""

import json
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from test.evaluation.clients.llm_perf.client import LLMPerfClient


class TestLLMPerfClientInit:
    """Test LLMPerfClient initialization."""

    def test_default_initialization(self):
        """Test default client initialization uses llm_perf_brazil."""
        client = LLMPerfClient()

        assert client.client_type == "llm_perf_brazil"
        assert client.scripts_dir.name == "scripts"

    def test_basic_initialization(self):
        """Test basic client initialization."""
        client = LLMPerfClient(client_type="llm_perf")

        assert client.client_type == "llm_perf"
        assert client.scripts_dir.name == "scripts"

    def test_github_patched_client_sets_correct_dir(self):
        """Test github patched client sets correct directory."""
        client = LLMPerfClient(client_type="llm_perf_github_patched")

        assert client.llmperf_dir == Path.home() / "llmperfGithubPatched"

    def test_custom_client_sets_correct_dir(self):
        """Test custom client sets correct directory."""
        client = LLMPerfClient(client_type="custom_llm_perf")

        assert client.llmperf_dir == Path.home() / "CustomLlmPerf"

    def test_unsupported_client_raises_error(self):
        """Test unsupported client type raises error."""
        with pytest.raises(ValueError, match="Unsupported client type"):
            LLMPerfClient(client_type="invalid_client")


class TestLLMPerfClientSetup:
    """Test LLMPerfClient.setup method."""

    @patch("test.evaluation.clients.llm_perf.client.subprocess.run")
    @patch("test.evaluation.clients.llm_perf.client.os.makedirs")
    def test_setup_calls_script(self, mock_makedirs, mock_run):
        """Test setup calls the setup script."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        client = LLMPerfClient(client_type="llm_perf")
        client.setup()

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "python3" in call_args
        assert "setup_llm_perf.py" in str(call_args)

    @patch("test.evaluation.clients.llm_perf.client.subprocess.run")
    @patch("test.evaluation.clients.llm_perf.client.os.makedirs")
    def test_setup_failure_raises_error(self, mock_makedirs, mock_run):
        """Test setup failure raises error."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        client = LLMPerfClient(client_type="llm_perf")

        with pytest.raises(RuntimeError, match="Failed to setup"):
            client.setup()


class TestLLMPerfClientEvaluate:
    """Test LLMPerfClient.evaluate method."""

    @patch("test.evaluation.clients.llm_perf.client.subprocess.Popen")
    @patch.object(LLMPerfClient, "process_results")
    def test_evaluate_basic(self, mock_process, mock_popen):
        """Test basic evaluate call."""
        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.stdout = iter([])  # Empty output
        mock_proc.wait = Mock(return_value=None)
        mock_popen.return_value = mock_proc
        mock_process.return_value = {"metrics": "data", "results_file": "test.json"}

        client = LLMPerfClient(client_type="llm_perf")
        results, results_file = client.evaluate(
            model_path="/path/to/model",
            server_port=8000,
            max_concurrent_requests=4,
            input_size=512,
            output_size=128,
            n_batches=10,
            results_dir="results/",
        )

        assert results["metrics"] == "data"
        assert results_file == "test.json"
        mock_popen.assert_called_once()

    @patch("test.evaluation.clients.llm_perf.client.subprocess.Popen")
    @patch.object(LLMPerfClient, "process_results")
    def test_evaluate_with_tokenizer(self, mock_process, mock_popen):
        """Test evaluate with custom tokenizer."""
        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.stdout = iter([])  # Empty output
        mock_proc.wait = Mock(return_value=None)
        mock_popen.return_value = mock_proc
        mock_process.return_value = {"metrics": "data", "results_file": "test.json"}

        client = LLMPerfClient(client_type="llm_perf")
        client.evaluate(
            model_path="/path/to/model",
            server_port=8000,
            max_concurrent_requests=4,
            input_size=512,
            output_size=128,
            n_batches=10,
            results_dir="results/",
            tokenizer="/path/to/tokenizer",
        )

        # Check that the subprocess was called
        mock_popen.assert_called_once()
        assert mock_process.called


class TestLLMPerfClientProcessResults:
    """Test LLMPerfClient.process_results method."""

    def test_process_results_success(self):
        """Test successful results processing."""
        mock_metrics = {
            "results_end_to_end_latency_s_quantiles_p25": 1.0,
            "results_end_to_end_latency_s_quantiles_p50": 1.5,
            "results_end_to_end_latency_s_quantiles_p75": 2.0,
            "results_end_to_end_latency_s_quantiles_p90": 2.5,
            "results_end_to_end_latency_s_quantiles_p95": 3.0,
            "results_end_to_end_latency_s_quantiles_p99": 3.5,
            "results_end_to_end_latency_s_max": 4.0,
            "results_end_to_end_latency_s_mean": 2.0,
            "results_number_input_tokens_mean": 100,
            "results_number_output_tokens_mean": 50,
            "results_ttft_s_quantiles_p25": 0.1,
            "results_ttft_s_quantiles_p50": 0.15,
            "results_ttft_s_quantiles_p75": 0.2,
            "results_ttft_s_quantiles_p90": 0.25,
            "results_ttft_s_quantiles_p95": 0.3,
            "results_ttft_s_quantiles_p99": 0.35,
            "results_ttft_s_max": 0.4,
            "results_ttft_s_mean": 0.2,
            "results_inter_token_latency_s_quantiles_p25": 0.01,
            "results_inter_token_latency_s_quantiles_p50": 0.015,
            "results_inter_token_latency_s_quantiles_p75": 0.02,
            "results_inter_token_latency_s_quantiles_p90": 0.025,
            "results_inter_token_latency_s_quantiles_p95": 0.03,
            "results_inter_token_latency_s_quantiles_p99": 0.035,
            "results_inter_token_latency_s_max": 0.04,
            "results_request_output_throughput_token_per_s_mean": 50.0,
            "results_mean_output_throughput_token_per_s": 75.0,
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(mock_metrics))):
            with patch("os.path.exists", return_value=True):
                client = LLMPerfClient(client_type="llm_perf")
                results = client.process_results(
                    results_dir="results/",
                    input_size=512,
                    output_size=128,
                    model_path="/path/to/model",
                )

        assert "e2e_model" in results
        assert "context_encoding_model" in results
        assert "token_generation_model" in results
        assert results["e2e_model"]["latency_ms_p50"] == 1500.0  # 1.5s * 1000

    def test_process_results_file_not_found(self):
        """Test error when results file not found."""
        with patch("os.path.exists", return_value=False):
            with patch("os.listdir", return_value=[]):
                client = LLMPerfClient(client_type="llm_perf")

                with pytest.raises(FileNotFoundError, match="Results file not found"):
                    client.process_results(
                        results_dir="results/",
                        input_size=512,
                        output_size=128,
                        model_path="/path/to/model",
                    )
