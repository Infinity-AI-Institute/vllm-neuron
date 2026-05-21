# SPDX-License-Identifier: Apache-2.0
"""Unit tests for VLLMServer class."""

import json
from unittest.mock import Mock, patch

import pytest

from test.evaluation.server.vllm import VLLMServer


class TestVLLMServerInit:
    """Test VLLMServer initialization."""

    def test_basic_initialization(self):
        """Test basic server initialization."""
        server = VLLMServer(
            name="test-model",
            model_path="/path/to/model",
            continuous_batch_size=1,
            max_seq_len=2048,
            tp_degree=32,
            n_vllm_threads=32,
            server_port=8000,
        )

        assert server.name == "test-model"
        assert server.model_path == "/path/to/model"
        assert server.max_seq_len == 2048
        assert server.tp_degree == 32
        assert server.cores == "0-31"

    def test_initialization_with_ctx_output_lengths(self):
        """Test max_seq_len calculation from ctx_output_lengths."""
        server = VLLMServer(
            name="test-model",
            model_path="/path/to/model",
            continuous_batch_size=1,
            ctx_output_lengths=(512, 128),
            tp_degree=32,
            n_vllm_threads=32,
            server_port=8000,
        )

        assert server.max_seq_len == 1280  # 2 * (512 + 128)

    def test_both_max_seq_len_and_ctx_raises_error(self):
        """Test error when both max_seq_len and ctx_output_lengths provided."""
        with pytest.raises(
            ValueError, match="Either max_seq_len or ctx_output_lengths"
        ):
            VLLMServer(
                name="test-model",
                model_path="/path/to/model",
                continuous_batch_size=1,
                max_seq_len=2048,
                ctx_output_lengths=(512, 128),
                tp_degree=32,
                n_vllm_threads=32,
                server_port=8000,
            )


class TestVLLMServerStartVLLMServer:
    """Test VLLMServer.start_vllm_server method."""

    @patch("test.evaluation.server.vllm.subprocess.Popen")
    @patch("test.evaluation.server.vllm.is_port_available")
    @patch("test.evaluation.server.vllm.time.sleep")
    @patch.object(VLLMServer, "check_health_endpoint")
    def test_start_vllm_server_basic(
        self, mock_health, mock_sleep, mock_port_available, mock_popen
    ):
        """Test basic server start."""
        mock_port_available.return_value = True
        mock_process = Mock()
        mock_process.poll.return_value = None
        # Make stdout iterable for the thread that reads it
        mock_process.stdout = iter([])
        mock_popen.return_value = mock_process
        mock_health.return_value = True

        server = VLLMServer(
            name="test-model",
            model_path="/path/to/model",
            continuous_batch_size=1,
            max_seq_len=2048,
            tp_degree=32,
            n_vllm_threads=32,
            server_port=8000,
        )

        port, process, health = server.start_vllm_server()

        assert port == 8000
        assert health is True
        mock_popen.assert_called_once()

    @patch("test.evaluation.server.vllm.subprocess.Popen")
    @patch("test.evaluation.server.vllm.is_port_available")
    @patch("test.evaluation.server.vllm.time.sleep")
    @patch.object(VLLMServer, "check_health_endpoint")
    def test_start_with_override_neuron_config(
        self, mock_health, mock_sleep, mock_port_available, mock_popen
    ):
        """Test server start includes override neuron config in command."""
        mock_port_available.return_value = True
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stdout = iter([])
        mock_popen.return_value = mock_process
        mock_health.return_value = True

        config = {"flash_decoding_enabled": True}
        server = VLLMServer(
            name="test-model",
            model_path="/path/to/model",
            continuous_batch_size=1,
            max_seq_len=2048,
            tp_degree=32,
            n_vllm_threads=32,
            server_port=8000,
            additional_config=config,
        )

        port, process, health = server.start_vllm_server()

        call_args = mock_popen.call_args[0][0]
        assert "--additional-config" in call_args
        assert json.dumps(config) in call_args


class TestVLLMServerCheckHealth:
    """Test VLLMServer.check_health_endpoint method."""

    @patch("test.evaluation.server.vllm.requests.get")
    @patch("test.evaluation.server.vllm.time.sleep")
    def test_check_health_success(self, mock_sleep, mock_get):
        """Test health check succeeds."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        server = VLLMServer(
            name="test-model",
            model_path="/path/to/model",
            continuous_batch_size=1,
            max_seq_len=2048,
            tp_degree=32,
            n_vllm_threads=32,
            server_port=8000,
        )

        result = server.check_health_endpoint(
            "http://localhost:8000/health", num_retries=3
        )

        assert result is True

    @patch("test.evaluation.server.vllm.requests.get")
    @patch("test.evaluation.server.vllm.time.sleep")
    def test_check_health_failure(self, mock_sleep, mock_get):
        """Test health check fails after retries."""
        # Import requests to use the actual ConnectionError from requests module
        import requests

        mock_get.side_effect = requests.ConnectionError()

        server = VLLMServer(
            name="test-model",
            model_path="/path/to/model",
            continuous_batch_size=1,
            max_seq_len=2048,
            tp_degree=32,
            n_vllm_threads=32,
            server_port=8000,
        )

        result = server.check_health_endpoint(
            "http://localhost:8000/health", num_retries=3
        )

        assert result is False


class TestVLLMServerStart:
    """Test VLLMServer.start method (full workflow)."""

    @patch("test.evaluation.server.vllm.check_server_terminated")
    @patch.object(VLLMServer, "start_vllm_server")
    def test_start_success(self, mock_start_vllm, mock_check_terminated):
        """Test successful server start."""
        mock_check_terminated.return_value = True
        mock_start_vllm.return_value = (8000, Mock(), True)

        server = VLLMServer(
            name="test-model",
            model_path="/path/to/model",
            continuous_batch_size=1,
            max_seq_len=2048,
            tp_degree=32,
            n_vllm_threads=32,
            server_port=8000,
        )

        port, process, health = server.start()

        assert port == 8000
        assert health is True

    @patch("test.evaluation.server.vllm.check_server_terminated")
    @patch.object(VLLMServer, "start_vllm_server")
    def test_start_health_check_fails(self, mock_start_vllm, mock_check_terminated):
        """Test start raises error when health check fails."""
        mock_check_terminated.return_value = True
        mock_start_vllm.return_value = (8000, Mock(), False)

        server = VLLMServer(
            name="test-model",
            model_path="/path/to/model",
            continuous_batch_size=1,
            max_seq_len=2048,
            tp_degree=32,
            n_vllm_threads=32,
            server_port=8000,
        )

        with pytest.raises(
            ConnectionRefusedError, match="Server did not start successfully"
        ):
            server.start()


class TestVLLMServerCleanup:
    """Test VLLMServer.cleanup method."""

    @patch.object(VLLMServer, "kill_children_of_process_on_port")
    def test_cleanup(self, mock_kill):
        """Test cleanup calls kill method."""
        server = VLLMServer(
            name="test-model",
            model_path="/path/to/model",
            continuous_batch_size=1,
            max_seq_len=2048,
            tp_degree=32,
            n_vllm_threads=32,
            server_port=8000,
        )

        server.cleanup()

        mock_kill.assert_called_once_with(8000)
