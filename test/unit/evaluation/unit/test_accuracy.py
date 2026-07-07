# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest

from test.evaluation.accuracy import (
    AccuracyScenario,
    _check_thresholds,
    run_accuracy_test,
)
from test.evaluation.server_config import ServerConfig


class TestCheckThresholds:
    """Test threshold checking functionality"""

    def test_simple_threshold_pass(self):
        """Simple threshold should pass when value meets minimum"""
        _check_thresholds(
            "test_scenario",
            "test_dataset",
            {"metric1": 90.0},
            {"test_dataset": {"metric1": 85.0}},
        )

    def test_simple_threshold_fail(self):
        """Simple threshold should fail when value below minimum"""
        with pytest.raises(AssertionError, match="metric1: 80.0000 < 85.0000"):
            _check_thresholds(
                "test_scenario",
                "test_dataset",
                {"metric1": 80.0},
                {"test_dataset": {"metric1": 85.0}},
            )

    def test_mean_std_threshold_pass_default_num_std(self):
        """Mean/std threshold should pass with default num_std=1"""
        _check_thresholds(
            "test_scenario",
            "test_dataset",
            {"metric1": 89.0},
            {"test_dataset": {"metric1": {"mean": 90.0, "std": 1.0}}},
        )

    def test_mean_std_threshold_fail_default_num_std(self):
        """Mean/std threshold should fail with default num_std=1"""
        with pytest.raises(
            AssertionError, match="metric1: 87.0000 < 88.0000.*mean=90.0000.*std=2.0000"
        ):
            _check_thresholds(
                "test_scenario",
                "test_dataset",
                {"metric1": 87.0},
                {"test_dataset": {"metric1": {"mean": 90.0, "std": 2.0}}},
            )

    def test_mean_std_threshold_pass_custom_num_std(self):
        """Mean/std threshold should pass with custom num_std"""
        _check_thresholds(
            "test_scenario",
            "test_dataset",
            {"metric1": 87.0},
            {"test_dataset": {"metric1": {"mean": 90.0, "std": 2.0, "num_std": 2}}},
        )

    def test_mean_std_threshold_fail_custom_num_std(self):
        """Mean/std threshold should fail with custom num_std"""
        with pytest.raises(
            AssertionError, match="metric1: 85.0000 < 86.0000.*num_std=2"
        ):
            _check_thresholds(
                "test_scenario",
                "test_dataset",
                {"metric1": 85.0},
                {"test_dataset": {"metric1": {"mean": 90.0, "std": 2.0, "num_std": 2}}},
            )

    def test_no_thresholds(self):
        """Should pass when no thresholds defined"""
        _check_thresholds("test_scenario", "test_dataset", {"metric1": 50.0}, {})

    def test_dataset_not_in_thresholds(self):
        """Should pass when dataset not in thresholds"""
        _check_thresholds(
            "test_scenario",
            "test_dataset",
            {"metric1": 50.0},
            {"other_dataset": {"metric1": 90.0}},
        )

    def test_metric_not_in_results(self):
        """Should raise when threshold metric not in results"""
        with pytest.raises(AssertionError, match="metric not found in results"):
            _check_thresholds(
                "test_scenario",
                "test_dataset",
                {"metric1": 90.0},
                {"test_dataset": {"metric2": 85.0}},
            )

    def test_mixed_thresholds_all_pass(self):
        """Multiple metrics with mixed threshold types should all pass"""
        _check_thresholds(
            "test_scenario",
            "test_dataset",
            {"metric1": 90.0, "metric2": 89.0},
            {
                "test_dataset": {
                    "metric1": 85.0,
                    "metric2": {"mean": 90.0, "std": 2.0},
                }
            },
        )

    def test_mixed_thresholds_one_fail(self):
        """Should fail if any metric fails threshold"""
        with pytest.raises(AssertionError, match="metric2"):
            _check_thresholds(
                "test_scenario",
                "test_dataset",
                {"metric1": 90.0, "metric2": 87.0},
                {
                    "test_dataset": {
                        "metric1": 85.0,
                        "metric2": {"mean": 90.0, "std": 2.0},
                    }
                },
            )

    def test_mean_std_missing_mean_key(self):
        """Should raise when mean/std threshold missing mean key"""
        with pytest.raises(AssertionError, match="missing 'mean' key"):
            _check_thresholds(
                "test_scenario",
                "test_dataset",
                {"metric1": 90.0},
                {"test_dataset": {"metric1": {"std": 2.0}}},
            )

    def test_mean_std_zero_std(self):
        """Mean/std threshold with zero std should work like simple threshold"""
        with pytest.raises(AssertionError, match="metric1: 85.0000 < 90.0000"):
            _check_thresholds(
                "test_scenario",
                "test_dataset",
                {"metric1": 85.0},
                {"test_dataset": {"metric1": {"mean": 90.0, "std": 0.0}}},
            )

    def test_multiple_failures_reported(self):
        """Should report all failing metrics in single error"""
        with pytest.raises(AssertionError) as exc_info:
            _check_thresholds(
                "test_scenario",
                "test_dataset",
                {"metric1": 80.0, "metric2": 70.0},
                {"test_dataset": {"metric1": 85.0, "metric2": 75.0}},
            )
        error_msg = str(exc_info.value)
        assert "metric1: 80.0000 < 85.0000" in error_msg
        assert "metric2: 70.0000 < 75.0000" in error_msg


class TestRunAccuracyTestThresholds:
    """Test threshold checking integration in run_accuracy_test"""

    @patch("test.evaluation.accuracy.ArtifactManager")
    @patch("test.evaluation.accuracy.VLLMServer")
    @patch("test.evaluation.accuracy._get_accuracy_client")
    def test_all_datasets_evaluated_before_threshold_check(
        self, mock_get_client, mock_server, mock_artifact_manager
    ):
        """Should evaluate all datasets before checking thresholds"""
        # Setup mocks
        mock_artifact_manager.return_value.setup_model_path.return_value = "/model"
        mock_server_instance = MagicMock()
        mock_server_instance.start.return_value = (8000, None, True)
        mock_server.return_value = mock_server_instance

        mock_client = MagicMock()
        mock_client.evaluate.side_effect = [
            ({"metric1": 90.0}, "results1.json"),
            ({"metric1": 70.0}, "results2.json"),
        ]
        mock_get_client.return_value = mock_client

        server_config = ServerConfig(
            name="test",
            model_path="/model",
            model_s3_path="",
            max_seq_len=2048,
            context_encoding_len=2048,
            tp_degree=1,
            server_port=8000,
            n_vllm_threads=1,
        )
        scenarios = {
            "test_scenario": AccuracyScenario(
                client="lm_eval",
                datasets={
                    "gsm8k_cot": {"thresholds": {"metric1": 85.0}},
                    "mmlu_flan_n_shot_generative": {"thresholds": {"metric1": 85.0}},
                },
            )
        }

        # Should fail on dataset2 but evaluate both
        with pytest.raises(AssertionError, match="mmlu_flan_n_shot_generative"):
            run_accuracy_test(server_config, scenarios)

        # Verify both datasets were evaluated
        assert mock_client.evaluate.call_count == 2

    @patch("test.evaluation.accuracy.ArtifactManager")
    @patch("test.evaluation.accuracy.VLLMServer")
    @patch("test.evaluation.accuracy._get_accuracy_client")
    def test_multiple_threshold_failures_collected(
        self, mock_get_client, mock_server, mock_artifact_manager
    ):
        """Should collect all threshold failures and report together"""
        # Setup mocks
        mock_artifact_manager.return_value.setup_model_path.return_value = "/model"
        mock_server_instance = MagicMock()
        mock_server_instance.start.return_value = (8000, None, True)
        mock_server.return_value = mock_server_instance

        mock_client = MagicMock()
        mock_client.evaluate.side_effect = [
            ({"metric1": 70.0}, "results1.json"),
            ({"metric1": 60.0}, "results2.json"),
        ]
        mock_get_client.return_value = mock_client

        server_config = ServerConfig(
            name="test",
            model_path="/model",
            model_s3_path="",
            max_seq_len=2048,
            context_encoding_len=2048,
            tp_degree=1,
            server_port=8000,
            n_vllm_threads=1,
        )
        scenarios = {
            "test_scenario": AccuracyScenario(
                client="lm_eval",
                datasets={
                    "gsm8k_cot": {"thresholds": {"metric1": 85.0}},
                    "mmlu_flan_n_shot_generative": {"thresholds": {"metric1": 85.0}},
                },
            )
        }

        # Should report both failures
        with pytest.raises(AssertionError) as exc_info:
            run_accuracy_test(server_config, scenarios)

        error_msg = str(exc_info.value)
        assert "gsm8k_cot" in error_msg
        assert "mmlu_flan_n_shot_generative" in error_msg
        assert "70.0000 < 85.0000" in error_msg
        assert "60.0000 < 85.0000" in error_msg

    @patch("test.evaluation.accuracy.ArtifactManager")
    @patch("test.evaluation.accuracy.VLLMServer")
    @patch("test.evaluation.accuracy._get_accuracy_client")
    def test_no_error_when_all_thresholds_pass(
        self, mock_get_client, mock_server, mock_artifact_manager
    ):
        """Should not raise when all thresholds pass"""
        # Setup mocks
        mock_artifact_manager.return_value.setup_model_path.return_value = "/model"
        mock_server_instance = MagicMock()
        mock_server_instance.start.return_value = (8000, None, True)
        mock_server.return_value = mock_server_instance

        mock_client = MagicMock()
        mock_client.evaluate.side_effect = [
            ({"metric1": 90.0}, "results1.json"),
            ({"metric1": 88.0}, "results2.json"),
        ]
        mock_get_client.return_value = mock_client

        server_config = ServerConfig(
            name="test",
            model_path="/model",
            model_s3_path="",
            max_seq_len=2048,
            context_encoding_len=2048,
            tp_degree=1,
            server_port=8000,
            n_vllm_threads=1,
        )
        scenarios = {
            "test_scenario": AccuracyScenario(
                client="lm_eval",
                datasets={
                    "gsm8k_cot": {"thresholds": {"metric1": 85.0}},
                    "mmlu_flan_n_shot_generative": {"thresholds": {"metric1": 85.0}},
                },
            )
        }

        # Should not raise
        result = run_accuracy_test(server_config, scenarios)
        assert result["model_name"] == "test"
        assert "gsm8k_cot" in result["scenarios"]["test_scenario"]
        assert "mmlu_flan_n_shot_generative" in result["scenarios"]["test_scenario"]
