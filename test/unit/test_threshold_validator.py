# SPDX-License-Identifier: Apache-2.0
"""Unit tests for threshold validator."""

import json
from pathlib import Path
from test.utils.threshold_validator import ThresholdValidator


def test_absolute_thresholds_pass():
    """Test that absolute thresholds pass when metrics meet requirements."""
    thresholds = {
        "absolute": {
            "context_encoding_model.latency_ms_p50": 5000.0,
            "token_generation_model.latency_ms_p50": 20.0,
        }
    }

    metrics = {
        "context_encoding_model": {
            "latency_ms_p50": 3000.0
        },
        "token_generation_model": {
            "latency_ms_p50": 10.0
        },
    }

    validator = ThresholdValidator(thresholds)
    passed, failures, details = validator.validate(metrics)

    assert passed
    assert len(failures) == 0
    assert details["absolute"]["context_encoding_model.latency_ms_p50"][
        "passed"]
    assert details["absolute"]["token_generation_model.latency_ms_p50"][
        "passed"]


def test_absolute_thresholds_fail():
    """Test that absolute thresholds fail when metrics don't meet requirements."""
    thresholds = {
        "absolute": {
            "context_encoding_model.latency_ms_p50": 5000.0,
            "token_generation_model.latency_ms_p50": 20.0,
        }
    }

    metrics = {
        "context_encoding_model": {
            "latency_ms_p50": 6000.0
        },  # Too high
        "token_generation_model": {
            "latency_ms_p50": 25.0
        },  # Too high
    }

    validator = ThresholdValidator(thresholds)
    passed, failures, details = validator.validate(metrics)

    assert not passed
    assert len(failures) == 2
    assert not details["absolute"]["context_encoding_model.latency_ms_p50"][
        "passed"]
    assert not details["absolute"]["token_generation_model.latency_ms_p50"][
        "passed"]


def test_regression_thresholds_pass(tmp_path: Path):
    """Test regression checks pass when within threshold."""
    baseline_file = tmp_path / "baseline.json"
    baseline = {
        "context_encoding_model": {
            "latency_ms_p50": 3000.0
        },
        "token_generation_model": {
            "latency_ms_p50": 10.0
        },
    }
    with baseline_file.open("w") as f:
        json.dump(baseline, f)

    thresholds = {
        "regression": {
            "context_encoding_model.latency_ms_p50": 5.0,  # Max 5% regression
            "token_generation_model.latency_ms_p50": 5.0,
        },
        "baseline_file": str(baseline_file),
    }

    # Current metrics with 3% regression (within 5% threshold)
    metrics = {
        "context_encoding_model": {
            "latency_ms_p50": 3090.0
        },  # 3% slower
        "token_generation_model": {
            "latency_ms_p50": 10.3
        },  # 3% slower
    }

    validator = ThresholdValidator(thresholds)
    passed, failures, details = validator.validate(metrics)

    assert passed
    assert len(failures) == 0


def test_regression_thresholds_fail(tmp_path: Path):
    """Test regression checks fail when exceeding threshold."""
    baseline_file = tmp_path / "baseline.json"
    baseline = {
        "context_encoding_model": {
            "latency_ms_p50": 3000.0
        },
        "token_generation_model": {
            "latency_ms_p50": 10.0
        },
    }
    with baseline_file.open("w") as f:
        json.dump(baseline, f)

    thresholds = {
        "regression": {
            "context_encoding_model.latency_ms_p50": 5.0,
            "token_generation_model.latency_ms_p50": 5.0,
        },
        "baseline_file": str(baseline_file),
    }

    # Current metrics with 10% regression (exceeds 5% threshold)
    metrics = {
        "context_encoding_model": {
            "latency_ms_p50": 3300.0
        },  # 10% slower
        "token_generation_model": {
            "latency_ms_p50": 11.0
        },  # 10% slower
    }

    validator = ThresholdValidator(thresholds)
    passed, failures, details = validator.validate(metrics)

    assert not passed
    assert len(failures) == 2
    assert "10.00%" in failures[0]
    assert "10.00%" in failures[1]


def test_nested_metric_access():
    """Test accessing nested metrics with dot notation."""
    validator = ThresholdValidator()

    data = {"level1": {"level2": {"metric": 123.45}}}

    value = validator._get_nested_value(data, "level1.level2.metric")
    assert value == 123.45

    value = validator._get_nested_value(data, "nonexistent.path")
    assert value is None


def test_save_baseline(tmp_path: Path):
    """Test saving baseline metrics."""
    metrics = {
        "context_encoding_model": {
            "latency_ms_p50": 3000.0
        },
        "token_generation_model": {
            "latency_ms_p50": 10.0
        },
    }

    baseline_file = tmp_path / "baseline.json"
    ThresholdValidator.save_baseline(metrics, baseline_file)

    assert baseline_file.exists()

    with baseline_file.open() as f:
        loaded = json.load(f)

    assert loaded == metrics
