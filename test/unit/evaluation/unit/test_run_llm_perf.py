# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import patch

from test.evaluation.clients.llm_perf.scripts.run_llm_perf import (
    build_common_args,
    run_llm_perf,
)


class TestRunLlmPerf(unittest.TestCase):
    def test_tokenizer_processing_for_llm_perf_brazil(self):
        """Test that tokenizer is included in args for llm_perf_brazil client type."""
        tokenizer = "test_tokenizer"

        args = build_common_args(
            model="test_model",
            mean_ip_tokens=100,
            stddev_ip_tokens=10,
            mean_op_tokens=50,
            stddev_op_tokens=5,
            max_requests=10,
            max_concurrent_req=2,
            results_dir="/tmp/results",
            tokenizer=tokenizer,
        )

        self.assertIn("--tokenizer", args)
        self.assertIn(tokenizer, args)

    def test_invalid_client_type_exits(self):
        """Test that invalid client type raises exception."""
        with self.assertRaises(Exception) as cm:
            run_llm_perf(
                model="test_model",
                max_concurrent_req=1,
                mean_ip_tokens=100,
                stddev_ip_tokens=10,
                mean_op_tokens=50,
                stddev_op_tokens=5,
                results_dir="/tmp/results",
                n_batches=1,
                port=8000,
                client_type="invalid_client",
            )
        self.assertIn("Invalid client type", str(cm.exception))

    @patch("subprocess.run")
    @patch("builtins.print")
    def test_llm_perf_brazil_subprocess_command(self, mock_print, mock_subprocess):
        """Test subprocess command for llm_perf_brazil client type."""
        run_llm_perf(
            model="test_model",
            max_concurrent_req=2,
            mean_ip_tokens=100,
            stddev_ip_tokens=10,
            mean_op_tokens=50,
            stddev_op_tokens=5,
            results_dir="/tmp/results",
            n_batches=1,
            port=8000,
            client_type="llm_perf_brazil",
            tokenizer="test_tokenizer",
        )

        mock_subprocess.assert_called_once()
        cmd = mock_subprocess.call_args[0][0]
        self.assertIn("-m", cmd)
        self.assertIn("llmperf.token_benchmark_ray", cmd)
        self.assertIn("--tokenizer", cmd)
        self.assertIn("test_tokenizer", cmd)


if __name__ == "__main__":
    unittest.main()
