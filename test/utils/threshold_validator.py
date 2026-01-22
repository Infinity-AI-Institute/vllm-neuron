#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Threshold validation utility for benchmarking tests.

Supports absolute thresholds and regression checks (percentage-based).
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ThresholdValidator:
    """Validates benchmark metrics against configured thresholds."""

    def __init__(self, thresholds_config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator with threshold configuration.

        Args:
            thresholds_config: Dict with threshold specifications. Can include:
                - absolute: Dict of metric_name -> min_value
                - regression: Dict of metric_name -> max_regression_percent
                - baseline_file: Path to baseline metrics JSON file
                - baseline: Inline baseline metrics dict
        """
        self.config = thresholds_config or {}
        self.absolute_thresholds = self.config.get("absolute", {})
        self.regression_thresholds = self.config.get("regression", {})
        self.baseline_file = self.config.get("baseline_file")
        self.baseline_metrics = self.config.get("baseline")

        if self.baseline_file:
            self._load_baseline()

    def _load_baseline(self) -> None:
        """
        Load baseline metrics from JSON file.

        Supports both direct metrics dict and nested structure with "metrics" key.

        Raises:
            FileNotFoundError: If baseline file doesn't exist
        """
        baseline_path = Path(self.baseline_file).expanduser().resolve()
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

        with baseline_path.open() as f:
            data = json.load(f)

        if "metrics" in data:
            self.baseline_metrics = data["metrics"]
        else:
            self.baseline_metrics = data

    def validate(
        self, current_metrics: Dict[str, Any]
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate current metrics against thresholds.

        Args:
            current_metrics: Dict containing current benchmark metrics

        Returns:
            Tuple of (passed, failures, comparison_details)
            - passed: True if all thresholds passed
            - failures: List of failure messages
            - comparison_details: Dict with detailed comparison info
        """
        failures = []
        comparison_details = {}

        # Always check for NaN values first, regardless of threshold configuration
        nan_failures, nan_details = self._validate_no_nan_values(current_metrics)
        failures.extend(nan_failures)
        if nan_details:
            comparison_details["nan_check"] = nan_details

        # Validate absolute thresholds
        if self.absolute_thresholds:
            abs_failures, abs_details = self._validate_absolute(current_metrics)
            failures.extend(abs_failures)
            comparison_details["absolute"] = abs_details

        if self.baseline_metrics:
            reg_failures, reg_details = self._validate_regression(current_metrics)
            failures.extend(reg_failures)
            comparison_details["regression"] = reg_details

        passed = len(failures) == 0
        return passed, failures, comparison_details

    def _validate_no_nan_values(
        self, metrics: Dict[str, Any], prefix: str = ""
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Check all numeric metrics for NaN values.

        This validation runs regardless of whether thresholds are configured.
        NaN values in performance metrics indicate measurement failures and
        should always cause the test to fail.

        Args:
            metrics: Dict containing benchmark metrics (can be nested)
            prefix: Current path prefix for nested metrics

        Returns:
            Tuple of (failures, details)
        """
        failures = []
        details = {}

        for key, value in metrics.items():
            path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively check nested dicts
                nested_failures, nested_details = self._validate_no_nan_values(
                    value, path
                )
                failures.extend(nested_failures)
                details.update(nested_details)
            elif isinstance(value, (int, float)):
                is_nan = math.isnan(value)
                if is_nan:
                    details[path] = {
                        "value": "NaN",
                        "passed": False,
                    }
                    metric_label = self._get_metric_label(path)
                    failures.append(
                        f"NaN VALUE DETECTED: {metric_label} produced NaN - "
                        "this indicates a measurement failure"
                    )

        return failures, details

    def _validate_absolute(
        self, current_metrics: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Validate absolute threshold requirements.

        Latency metrics: current <= threshold (lower is better)
        Throughput/Accuracy metrics: current >= threshold (higher is better)

        Returns:
            Tuple of (failures, details)
        """
        failures = []
        details = {}

        for metric_path, threshold_value in self.absolute_thresholds.items():
            current_value = self._get_nested_value(current_metrics, metric_path)

            if current_value is None:
                failures.append(f"Metric '{metric_path}' not found in current results")
                continue

            is_latency = "latency" in metric_path.lower()

            if is_latency:
                passed = current_value <= threshold_value
                comparison = f"{current_value:.3f} <= {threshold_value:.3f}"
            else:
                passed = current_value >= threshold_value
                comparison = f"{current_value:.3f} >= {threshold_value:.3f}"

            details[metric_path] = {
                "current": current_value,
                "threshold": threshold_value,
                "passed": passed,
                "comparison": comparison,
            }

            if not passed:
                metric_label = self._get_metric_label(metric_path)
                failures.append(
                    f"THRESHOLD VIOLATION: {metric_label} = {current_value:.3f} "
                    f"(threshold: {threshold_value:.3f})"
                )

        return failures, details

    def _validate_regression(
        self, current_metrics: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Validate regression thresholds against baseline.

        Defaults to 5% max regression for TTFT p50, ITL p50, and Throughput
        if no explicit thresholds provided (performance metrics only).

        Regression calculation:
        - Latency: (current - baseline) / baseline * 100 (higher = worse)
        - Throughput/Accuracy: (baseline - current) / baseline * 100 (lower = worse)

        Positive % = regression (worse), Negative % = improvement (better)

        Returns:
            Tuple of (failures, details)
        """
        failures = []
        details = {}

        regression_thresholds = self.regression_thresholds
        if not regression_thresholds and self.baseline_metrics:
            # Default to 5% regression threshold for all metrics with baselines
            regression_thresholds = {}

            # Auto-generate 5% thresholds for all baseline metrics
            def add_thresholds(data, prefix=""):
                for key, value in data.items():
                    path = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        add_thresholds(value, path)
                    elif isinstance(value, (int, float)):
                        regression_thresholds[path] = 5.0

            add_thresholds(self.baseline_metrics)

        for metric_path, max_regression_pct in regression_thresholds.items():
            current_value = self._get_nested_value(current_metrics, metric_path)
            baseline_value = self._get_nested_value(self.baseline_metrics, metric_path)

            if current_value is None:
                failures.append(f"Metric '{metric_path}' not found in current results")
                continue

            if baseline_value is None:
                failures.append(f"Metric '{metric_path}' not found in baseline")
                continue

            is_latency = "latency" in metric_path.lower()

            if is_latency:
                regression_pct = (
                    (current_value - baseline_value) / baseline_value
                ) * 100
            else:
                regression_pct = (
                    (baseline_value - current_value) / baseline_value
                ) * 100

            passed = regression_pct <= max_regression_pct
            comparison_str = (
                f"{regression_pct:+.2f}% (max regression: {max_regression_pct:.2f}%)"
            )

            details[metric_path] = {
                "current": current_value,
                "baseline": baseline_value,
                "regression_pct": regression_pct,
                "max_allowed_pct": max_regression_pct,
                "passed": passed,
                "comparison": comparison_str,
            }

            if not passed:
                metric_label = self._get_metric_label(metric_path)
                failures.append(
                    f"PERFORMANCE REGRESSION: {metric_label} regressed by {regression_pct:+.2f}% "
                    f"(current: {current_value:.3f}, baseline: {baseline_value:.3f}, "
                    f"max allowed: {max_regression_pct:.2f}%)"
                )

        return failures, details

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Optional[float]:
        """
        Get value from nested dict using dot notation.

        Args:
            data: Nested dictionary
            path: Dot-separated path (e.g., "context_encoding_model.latency_ms_p50")

        Returns:
            Float value if found, None otherwise
        """
        keys = path.split(".")
        current = data

        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]

        if isinstance(current, (int, float)):
            return float(current)
        return None

    @staticmethod
    def save_baseline(metrics: Dict[str, Any], output_path: Path) -> None:
        """
        Save metrics as baseline for future regression checks.

        Args:
            metrics: Metrics dict to save
            output_path: Path to save baseline JSON
        """
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Baseline saved to: {output_path}")

    @staticmethod
    def _get_metric_label(metric_path: str) -> str:
        """
        Get friendly label for metric path.

        Args:
            metric_path: Dot-separated metric path

        Returns:
            Human-readable label with abbreviations (TTFT, ITL, etc.)
        """
        labels = {
            "context_encoding_model.latency_ms_p50": "Context Encoding p50 (TTFT)",
            "context_encoding_model.latency_ms_p90": "Context Encoding p90 (TTFT)",
            "context_encoding_model.latency_ms_p95": "Context Encoding p95 (TTFT)",
            "context_encoding_model.latency_ms_p99": "Context Encoding p99 (TTFT)",
            "token_generation_model.latency_ms_p50": "Token Generation p50 (ITL)",
            "token_generation_model.latency_ms_p90": "Token Generation p90 (ITL)",
            "token_generation_model.latency_ms_p95": "Token Generation p95 (ITL)",
            "token_generation_model.latency_ms_p99": "Token Generation p99 (ITL)",
            "token_generation_model.throughput": "Throughput (tok/s)",
            "e2e_model.latency_ms_p50": "End-to-End p50",
            "e2e_model.latency_ms_p90": "End-to-End p90",
            "e2e_model.throughput": "End-to-End Throughput",
        }
        return labels.get(metric_path, metric_path)

    def print_validation_report(
        self, passed: bool, failures: List[str], comparison_details: Dict[str, Any]
    ) -> None:
        """
        Print formatted validation report with pass/fail status and details.

        Args:
            passed: Whether all thresholds passed
            failures: List of failure messages
            comparison_details: Detailed comparison data
        """
        print("\n" + "=" * 80)
        print("THRESHOLD VALIDATION REPORT")
        print("=" * 80)

        if passed:
            print("✅ ALL THRESHOLDS PASSED")
        else:
            print("❌ THRESHOLD VALIDATION FAILED")
            print(f"\nFailures ({len(failures)}):")
            for failure in failures:
                print(f"  - {failure}")

        if "nan_check" in comparison_details and comparison_details["nan_check"]:
            print("\nNaN Value Checks:")
            for metric, details in comparison_details["nan_check"].items():
                status = "✅" if details["passed"] else "❌"
                label = self._get_metric_label(metric)
                print(f"  {status} {label}: NaN detected - FAIL")

        if "absolute" in comparison_details and comparison_details["absolute"]:
            print("\nAbsolute Thresholds:")
            for metric, details in comparison_details["absolute"].items():
                status = "✅" if details["passed"] else "❌"
                label = self._get_metric_label(metric)
                print(
                    f"  {status} {label}: {details['comparison']} {'PASS' if details['passed'] else 'FAIL'}"
                )

        if "regression" in comparison_details and comparison_details["regression"]:
            print("\nRegression Checks:")
            for metric, details in comparison_details["regression"].items():
                status = "✅" if details["passed"] else "❌"
                label = self._get_metric_label(metric)
                print(
                    f"  {status} {label}: {details['comparison']} {'PASS' if details['passed'] else 'FAIL'}"
                )
                print(
                    f"      Current: {details['current']:.3f}, Baseline: {details['baseline']:.3f}"
                )

        print("=" * 80 + "\n")
