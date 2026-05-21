# SPDX-License-Identifier: Apache-2.0
"""Unit tests for start_server.py version detection logic."""

import json
from unittest.mock import Mock, patch


from test.evaluation.server.scripts.start_server import build_vllm_command


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


class TestBuildVLLMCommandVersionDetection:
    """Test version-specific argument handling in build_vllm_command."""

    @patch("test.evaluation.server.scripts.start_server.vllm")
    @patch("test.evaluation.server.scripts.start_server.version")
    def test_vllm_v0_9_adds_device_and_block_manager(self, mock_version, mock_vllm):
        """Test vLLM < 0.10.2 adds --device and --use-v2-block-manager."""
        mock_vllm.__version__ = "0.9.0"
        mock_version.parse.side_effect = create_version_mock

        cmd_args = build_vllm_command(
            model_id="test-model",
            port=8000,
            max_seq_len=2048,
            cont_batch_size=32,
            tp_size=32,
        )

        assert "--device" in cmd_args
        assert "neuron" in cmd_args
        assert "--use-v2-block-manager" in cmd_args
        assert "--no-enable-chunked-prefill" not in cmd_args
        assert "--no-enable-prefix-caching" not in cmd_args

    @patch("test.evaluation.server.scripts.start_server.vllm")
    @patch("test.evaluation.server.scripts.start_server.version")
    def test_vllm_v0_7_uses_old_speculative_args(self, mock_version, mock_vllm):
        """Test vLLM < 0.8 uses old speculative decoding arguments."""
        mock_vllm.__version__ = "0.7.0"
        mock_version.parse.side_effect = create_version_mock

        cmd_args = build_vllm_command(
            model_id="test-model",
            port=8000,
            max_seq_len=2048,
            cont_batch_size=32,
            tp_size=32,
            draft_model_id="draft-model",
            num_speculative_tokens=4,
        )

        assert "--speculative-model" in cmd_args
        assert "draft-model" in cmd_args
        assert "--num-speculative-tokens" in cmd_args
        assert "--speculative-max-model-len" in cmd_args
        assert "--speculative-config" not in cmd_args

    @patch("test.evaluation.server.scripts.start_server.vllm")
    @patch("test.evaluation.server.scripts.start_server.version")
    def test_vllm_v0_11_uses_additional_config(self, mock_version, mock_vllm):
        """Test vLLM >= 0.10.2 uses --additional-config."""
        mock_vllm.__version__ = "0.11.0"
        mock_version.parse.side_effect = create_version_mock

        override_config = json.dumps({"flash_decoding_enabled": True})
        cmd_args = build_vllm_command(
            model_id="test-model",
            port=8000,
            max_seq_len=2048,
            cont_batch_size=32,
            tp_size=32,
            additional_config=override_config,
        )

        assert "--additional-config" in cmd_args
        assert override_config in cmd_args
