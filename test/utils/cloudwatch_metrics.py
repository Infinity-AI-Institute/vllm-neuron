# SPDX-License-Identifier: Apache-2.0
import os
from test.utils.server.server import REQUIRED_SERVER_KEYS
from typing import Any, Dict, List


def build_metric_data(
    metrics: Dict[str, Any],
    server_cfg: Dict[str, Any],
    instance_type: str,
    cost_per_million: float,
) -> List[Dict[str, Any]]:
    """Build CloudWatch metric data from performance metrics."""

    ctx_enc_metrics = metrics["context_encoding_model"]
    token_gen_metrics = metrics["token_generation_model"]
    e2e = metrics["e2e_model"]

    # Validate server_cfg fields
    for field in REQUIRED_SERVER_KEYS:
        if field not in server_cfg:
            raise KeyError(f"Missing required server config field: {field}")

    neuron_cfg = (
        server_cfg["override_neuron_config"]
        if "override_neuron_config" in server_cfg
        else {}
    )

    # Build dimensions
    dimensions_perf = [
        {"Name": "ModelName", "Value": server_cfg["model_path"]},
        {"Name": "TestName", "Value": str(server_cfg.get("name", "unknown-test"))},
        {"Name": "TPDegree", "Value": str(server_cfg["tp_degree"])},
        {"Name": "BatchSize", "Value": str(server_cfg["batch_size"])},
        {"Name": "MaxModelLen", "Value": str(server_cfg["max_model_len"])},
        {
            "Name": "ChunkedPrefill",
            "Value": str(
                neuron_cfg["enable_chunked_prefill"]
                if "enable_chunked_prefill" in neuron_cfg
                else (
                    server_cfg["enable_chunked_prefill"]
                    if "enable_chunked_prefill" in server_cfg
                    else False
                )
            ),
        },
        {
            "Name": "PrefixCaching",
            "Value": str(
                neuron_cfg["enable_prefix_caching"]
                if "enable_prefix_caching" in neuron_cfg
                else (
                    server_cfg["enable_prefix_caching"]
                    if "enable_prefix_caching" in server_cfg
                    else False
                )
            ),
        },
        {
            "Name": "Bucketing",
            "Value": str(
                neuron_cfg["enable_bucketing"]
                if "enable_bucketing" in neuron_cfg
                else False
            ),
        },
        {
            "Name": "MaxContextLength",
            "Value": str(
                neuron_cfg["max_context_length"]
                if "max_context_length" in neuron_cfg
                else None
            ),
        },
        {
            "Name": "MaxNewTokens",
            "Value": str(
                neuron_cfg["max_new_tokens"] if "max_new_tokens" in neuron_cfg else None
            ),
        },
        {
            "Name": "Compiled",
            "Value": str(
                bool(
                    server_cfg["compiled_model_path"]
                    if "compiled_model_path" in server_cfg
                    else False
                )
            ),
        },
        {
            "Name": "SpeculationType",
            "Value": str(
                server_cfg["speculation_type"]
                if "speculation_type" in server_cfg
                else None
            ),
        },
        {
            "Name": "QuantizationDtype",
            "Value": str(
                neuron_cfg["quantization_dtype"]
                if "quantization_dtype" in neuron_cfg
                else (
                    server_cfg["quantization_dtype"]
                    if "quantization_dtype" in server_cfg
                    else None
                )
            ),
        },
        {"Name": "Category", "Value": "Performance"},
        {"Name": "InstanceType", "Value": instance_type},
        {"Name": "Source", "Value": os.environ.get("BENCHMARK_SOURCE", "private")},
    ]

    dimensions_cost = dimensions_perf[:-3] + [
        {"Name": "Category", "Value": "Cost"},
        {"Name": "InstanceType", "Value": instance_type},
        {"Name": "Source", "Value": os.environ.get("BENCHMARK_SOURCE", "private")},
    ]

    metric_data = []

    # TTFT percentiles
    for percentile in ["p50", "p90", "p95", "p99"]:
        key = f"latency_ms_{percentile}"
        if key not in ctx_enc_metrics:
            raise KeyError(f"Missing required context encoding metric: {key}")
        metric_data.append(
            {
                "MetricName": f"TTFT_{percentile.upper()}",
                "Value": ctx_enc_metrics[key],
                "Unit": "Milliseconds",
                "Dimensions": dimensions_perf,
            }
        )

    # ITL percentiles
    for percentile in ["p50", "p90", "p95", "p99"]:
        key = f"latency_ms_{percentile}"
        if key not in token_gen_metrics:
            raise KeyError(f"Missing required token generation metric: {key}")
        metric_data.append(
            {
                "MetricName": f"ITL_{percentile.upper()}",
                "Value": token_gen_metrics[key],
                "Unit": "Milliseconds",
                "Dimensions": dimensions_perf,
            }
        )

    # E2E Latency percentiles
    for percentile in ["p50", "p90", "p95", "p99"]:
        key = f"latency_ms_{percentile}"
        if key not in e2e:
            raise KeyError(f"Missing required e2e metric: {key}")
        metric_data.append(
            {
                "MetricName": f"E2E_Latency_{percentile.upper()}",
                "Value": e2e[key],
                "Unit": "Milliseconds",
                "Dimensions": dimensions_perf,
            }
        )

    # Throughput
    if "throughput" not in token_gen_metrics:
        raise KeyError("Missing required token generation metric: throughput")
    metric_data.append(
        {
            "MetricName": "Token_Throughput",
            "Value": token_gen_metrics["throughput"],
            "Unit": "Count/Second",
            "Dimensions": dimensions_perf,
        }
    )

    # Cost
    metric_data.append(
        {
            "MetricName": "CostPerMillion",
            "Value": cost_per_million,
            "Unit": "Count",
            "Dimensions": dimensions_cost,
        }
    )

    return metric_data
