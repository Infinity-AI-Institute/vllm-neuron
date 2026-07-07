# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LMEvalClient class."""

import json
from pathlib import Path
from unittest.mock import Mock, mock_open, patch


from test.evaluation.clients.lm_eval.client import LMEvalClient


class TestLMEvalClientInit:
    """Test LMEvalClient initialization."""

    def test_basic_initialization(self):
        """Test basic client initialization."""
        client = LMEvalClient()

        assert client.scripts_dir.name == "scripts"


class TestLMEvalClientSetup:
    """Test LMEvalClient.setup method."""

    @patch("test.evaluation.clients.lm_eval.client.subprocess.run")
    @patch("test.evaluation.clients.lm_eval.client.os.system")
    @patch.object(LMEvalClient, "check_datasets_downloaded")
    def test_setup_downloads_datasets(self, mock_check, mock_system, mock_run):
        """Test setup downloads datasets when not present."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        mock_check.side_effect = [
            False,
            True,
        ]  # Not present, then present after download
        mock_system.return_value = 0

        client = LMEvalClient()
        client.setup()

        mock_run.assert_called_once()
        mock_system.assert_called_once()

    @patch("test.evaluation.clients.lm_eval.client.subprocess.run")
    @patch.object(LMEvalClient, "check_datasets_downloaded")
    def test_setup_skips_download_when_present(self, mock_check, mock_run):
        """Test setup skips download when datasets already present."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        mock_check.return_value = True

        client = LMEvalClient()
        client.setup()

        mock_run.assert_called_once()
        # os.system should not be called since datasets are present
        assert mock_check.call_count == 1


class TestLMEvalClientEvaluate:
    """Test LMEvalClient.evaluate method."""

    @patch("test.evaluation.clients.lm_eval.client.subprocess.Popen")
    @patch.object(LMEvalClient, "get_latest_results_file")
    @patch("builtins.open", new_callable=mock_open)
    def test_evaluate_basic(self, mock_file, mock_get_results, mock_popen):
        """Test basic evaluate call."""
        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.stdout = iter([])  # Empty output
        mock_proc.wait = Mock(return_value=None)
        mock_popen.return_value = mock_proc
        mock_get_results.return_value = "results.json"

        mock_results = {
            "results": {
                "gsm8k_cot": {
                    "exact_match,none": 0.85,
                    "exact_match_stderr,none": 0.02,
                }
            }
        }
        mock_file.return_value.read.return_value = json.dumps(mock_results)

        with patch("os.path.exists", return_value=True):
            client = LMEvalClient()
            results, results_file = client.evaluate(
                model_path="/path/to/model",
                server_port=8000,
                task_name="gsm8k_cot",
                results_dir="results/",
            )

        assert "gsm8k_cot" in results
        assert results_file == "results.json"
        mock_popen.assert_called()

    @patch("test.evaluation.clients.lm_eval.client.subprocess.Popen")
    @patch.object(LMEvalClient, "get_latest_results_file")
    @patch("builtins.open", new_callable=mock_open)
    def test_evaluate_with_custom_params(self, mock_file, mock_get_results, mock_popen):
        """Test evaluate with custom parameters."""
        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.stdout = iter([])  # Empty output
        mock_proc.wait = Mock(return_value=None)
        mock_popen.return_value = mock_proc
        mock_get_results.return_value = "results.json"

        mock_results = {
            "results": {
                "gsm8k_cot": {
                    "exact_match,none": 0.85,
                    "exact_match_stderr,none": 0.02,
                }
            }
        }
        mock_file.return_value.read.return_value = json.dumps(mock_results)

        with patch("os.path.exists", return_value=True):
            client = LMEvalClient()
            results, results_file = client.evaluate(
                model_path="/path/to/model",
                server_port=8000,
                task_name="gsm8k_cot",
                results_dir="results/",
                limit=100,
                use_chat=False,
                max_length=4096,
            )

        assert results_file == "results.json"
        mock_popen.assert_called()

    @patch("test.evaluation.clients.lm_eval.client.subprocess.Popen")
    @patch.object(LMEvalClient, "get_latest_results_file")
    @patch("builtins.open", new_callable=mock_open)
    def test_evaluate_with_kwargs_extensibility(
        self, mock_file, mock_get_results, mock_popen
    ):
        """Test evaluate with arbitrary kwargs for extensibility."""
        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.stdout = iter([])
        mock_proc.wait = Mock(return_value=None)
        mock_popen.return_value = mock_proc
        mock_get_results.return_value = "results.json"

        mock_results = {
            "results": {
                "test": {"exact_match,none": 0.85, "exact_match_stderr,none": 0.02}
            }
        }
        mock_file.return_value.read.return_value = json.dumps(mock_results)

        with patch("os.path.exists", return_value=True):
            client = LMEvalClient()
            results, results_file = client.evaluate(
                model_path="/path/to/model",
                server_port=8000,
                task_name="test_task",
                results_dir="results",
                # Test arbitrary new arguments
                new_future_arg="test_value",
                another_arg=42,
                boolean_arg=False,
            )

        # Verify new arguments are passed through
        call_args = mock_popen.call_args[0][0]
        assert "--new_future_arg" in call_args
        assert "test_value" in call_args
        assert "--another_arg" in call_args
        assert "42" in call_args
        assert "--boolean_arg" in call_args
        assert "False" in call_args

    def test_get_default_args(self):
        """Test get_default_args static method."""
        defaults = LMEvalClient.get_default_args()

        assert isinstance(defaults, dict)
        assert "max_concurrent_req" in defaults
        assert "timeout" in defaults
        assert "limit" in defaults
        assert "use_chat" in defaults
        assert "gen_kwargs" in defaults
        assert "system_prompt" in defaults
        assert "fewshot_as_multiturn" in defaults

        # Verify default values match expected
        assert defaults["max_concurrent_req"] == 1
        assert defaults["timeout"] == 3600
        assert defaults["limit"] == 200
        assert defaults["use_chat"] is True
        assert defaults["gen_kwargs"] == "{}"
        assert defaults["system_prompt"] is False
        assert defaults["fewshot_as_multiturn"] is True


class TestLMEvalClientProcessResults:
    """Test LMEvalClient._process_results method."""

    def test_process_results_exact_match(self):
        """Test processing results with exact_match metric."""
        client = LMEvalClient()

        results = {
            "results": {
                "gsm8k_cot": {
                    "exact_match,none": 0.85,
                    "exact_match_stderr,none": 0.02,
                }
            }
        }

        processed = client._process_results(results, task_name="gsm8k_cot")

        assert "gsm8k_cot" in processed
        assert processed["gsm8k_cot"]["score"] == 85.0
        assert processed["gsm8k_cot"]["exact_match,none"] == 85.0

    def test_process_results_ifeval(self):
        """Test processing results with IFEval metrics."""
        client = LMEvalClient()

        results = {
            "results": {
                "leaderboard_ifeval": {
                    "prompt_level_strict_acc,none": 0.75,
                    "prompt_level_strict_acc_stderr,none": 0.01,
                    "inst_level_strict_acc,none": 0.80,
                    "prompt_level_loose_acc,none": 0.78,
                    "prompt_level_loose_acc_stderr,none": 0.01,
                    "inst_level_loose_acc,none": 0.82,
                }
            }
        }

        processed = client._process_results(results, task_name="leaderboard_ifeval")

        assert "leaderboard_ifeval" in processed
        # Score should be average of strict accs
        assert processed["leaderboard_ifeval"]["score"] == 77.5


class TestLMEvalClientCheckDatasets:
    """Test LMEvalClient.check_datasets_downloaded method."""

    @patch("pathlib.Path.exists")
    def test_check_datasets_not_exists(self, mock_exists):
        """Test check when cache directory doesn't exist."""
        mock_exists.return_value = False

        client = LMEvalClient()
        result = client.check_datasets_downloaded()

        assert result is False

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rglob")
    def test_check_datasets_empty(self, mock_rglob, mock_exists):
        """Test check when cache directory is empty."""
        mock_exists.return_value = True
        mock_rglob.return_value = []

        client = LMEvalClient()
        result = client.check_datasets_downloaded()

        assert result is False

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rglob")
    @patch("pathlib.Path.iterdir")
    def test_check_datasets_present(self, mock_iterdir, mock_rglob, mock_exists):
        """Test check when datasets are present."""
        mock_exists.return_value = True
        mock_rglob.return_value = [Path("/fake/file")]
        mock_dir = Mock()
        mock_dir.is_dir.return_value = True
        mock_dir.name = "dataset1"
        mock_iterdir.return_value = [mock_dir]

        client = LMEvalClient()
        result = client.check_datasets_downloaded()

        assert result is True
