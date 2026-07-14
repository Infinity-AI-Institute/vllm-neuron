# SPDX-License-Identifier: Apache-2.0
"""Unit tests for VLLMServer version detection logic."""

import json
from unittest.mock import Mock, patch

import pytest

from test.evaluation.server.vllm import VLLMServer


def create_version_mock(version_str):
    """Create a mock version object that supports comparison operations."""
    m = Mock()
    m._version_str = version_str

    def compare_versions(v1, v2):
        """Simple version comparison for test purposes."""
        parts1 = [int(x) for x in v1.split(".")]
        parts2 = [int(x) for x in v2.split(".")]
        return (parts1 > parts2) - (parts1 < parts2)

    m.__lt__ = lambda self, other: (
        compare_versions(version_str, getattr(other, "_version_str", "999.999.999")) < 0
    )
    m.__ge__ = lambda self, other: (
        compare_versions(version_str, getattr(other, "_version_str", "0.0.0")) >= 0
    )
    return m


class TestVLLMServerSpeculativeConfigVersionDetection:
    """Test version-specific speculative config handling in VLLMServer."""

    @patch("test.evaluation.server.vllm.subprocess.Popen")
    @patch("test.evaluation.server.vllm.is_port_available")
    @patch("test.evaluation.server.vllm.time.sleep")
    @patch.object(VLLMServer, "check_health_endpoint")
    @patch("test.evaluation.server.vllm.vllm")
    @patch("test.evaluation.server.vllm.version")
    def test_vllm_v0_7_uses_old_speculative_args(
        self,
        mock_version,
        mock_vllm,
        mock_health,
        mock_sleep,
        mock_port_available,
        mock_popen,
    ):
        """Test vLLM < 0.8 uses old --speculative-model arguments."""
        mock_vllm.__version__ = "0.7.0"
        mock_version.parse.side_effect = create_version_mock
        mock_port_available.return_value = True
        mock_process = Mock()
        mock_process.poll.return_value = None
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
            draft_model_path="/path/to/draft",
            spec_len=4,
        )

        port, process, health = server.start_vllm_server()

        call_args = mock_popen.call_args[0][0]
        assert "--speculative-model" in call_args
        assert "/path/to/draft" in call_args
        assert "--num-speculative-tokens" in call_args
        assert "--speculative-config" not in call_args

    @patch("test.evaluation.server.vllm.subprocess.Popen")
    @patch("test.evaluation.server.vllm.is_port_available")
    @patch("test.evaluation.server.vllm.time.sleep")
    @patch.object(VLLMServer, "check_health_endpoint")
    @patch("test.evaluation.server.vllm.vllm")
    @patch("test.evaluation.server.vllm.version")
    def test_vllm_v0_9_uses_speculative_config_without_method(
        self,
        mock_version,
        mock_vllm,
        mock_health,
        mock_sleep,
        mock_port_available,
        mock_popen,
    ):
        """Test vLLM 0.8-0.10.1 uses --speculative-config without method field."""
        mock_vllm.__version__ = "0.9.0"
        mock_version.parse.side_effect = create_version_mock
        mock_port_available.return_value = True
        mock_process = Mock()
        mock_process.poll.return_value = None
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
            draft_model_path="/path/to/draft",
            spec_len=4,
        )

        port, process, health = server.start_vllm_server()

        call_args = mock_popen.call_args[0][0]
        assert "--speculative-config" in call_args

        # Find the config JSON
        config_idx = call_args.index("--speculative-config") + 1
        config = json.loads(call_args[config_idx])

        assert config["model"] == "/path/to/draft"
        assert config["num_speculative_tokens"] == 4
        assert config["max_model_len"] == 2048
        assert "method" not in config

    @patch("test.evaluation.server.vllm.subprocess.Popen")
    @patch("test.evaluation.server.vllm.is_port_available")
    @patch("test.evaluation.server.vllm.time.sleep")
    @patch.object(VLLMServer, "check_health_endpoint")
    @patch("test.evaluation.server.vllm.vllm")
    @patch("test.evaluation.server.vllm.version")
    def test_vllm_v1_requires_eagle_speculation(
        self,
        mock_version,
        mock_vllm,
        mock_health,
        mock_sleep,
        mock_port_available,
        mock_popen,
    ):
        """Test vLLM >= 0.10.2 raises error if speculation_type is not eagle."""
        mock_vllm.__version__ = "0.10.2"
        mock_version.parse.side_effect = create_version_mock
        mock_port_available.return_value = True
        mock_process = Mock()
        mock_process.poll.return_value = None
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
            draft_model_path="/path/to/draft",
            spec_len=4,
            speculation_type="fused",  # Not eagle
        )

        with pytest.raises(
            AssertionError, match="v1 neuron plugin only supports EAGLE speculation"
        ):
            server.start_vllm_server()

    @patch("test.evaluation.server.vllm.subprocess.Popen")
    @patch("test.evaluation.server.vllm.is_port_available")
    @patch("test.evaluation.server.vllm.time.sleep")
    @patch.object(VLLMServer, "check_health_endpoint")
    @patch("test.evaluation.server.vllm.vllm")
    @patch("test.evaluation.server.vllm.version")
    def test_default_spec_len_is_4(
        self,
        mock_version,
        mock_vllm,
        mock_health,
        mock_sleep,
        mock_port_available,
        mock_popen,
    ):
        """Test default num_speculative_tokens is 4 when spec_len is None."""
        mock_vllm.__version__ = "0.9.0"
        mock_version.parse.side_effect = create_version_mock
        mock_port_available.return_value = True
        mock_process = Mock()
        mock_process.poll.return_value = None
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
            draft_model_path="/path/to/draft",
            spec_len=None,  # Not specified
        )

        port, process, health = server.start_vllm_server()

        call_args = mock_popen.call_args[0][0]
        config_idx = call_args.index("--speculative-config") + 1
        config = json.loads(call_args[config_idx])

        assert config["num_speculative_tokens"] == 4
