# SPDX-License-Identifier: Apache-2.0
"""Compile/numeric probe for GLM-5.2's exact TP64 dense MLP shape."""

import argparse
import json
import statistics
import time

import torch

import vllm_neuron  # noqa: F401
from examples.vllm_neuron.glm52.shared_mlp_static_fp8_probe import _make_probe
from vllm_neuron.envs import get_compile_backend_name

_HIDDEN_SIZE = 6_144
_INTERMEDIATE_SIZE = 12_288
_TP_DEGREE = 64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(1, 256), default=1)
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="fp8")
    parser.add_argument(
        "--kernel-local-intermediate",
        type=int,
        choices=(192, 256),
        default=192,
    )
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")

    model, hidden, expected = _make_probe(
        _TP_DEGREE,
        args.tokens,
        args.dtype == "fp8",
        intermediate_size=_INTERMEDIATE_SIZE,
        kernel_local_intermediate=args.kernel_local_intermediate,
    )
    device = torch.device("neuron:0")
    model = model.to(device)
    hidden = hidden.to(device)

    started = time.perf_counter()
    compiled = torch.compile(model, backend=get_compile_backend_name())
    output_cpu = compiled(hidden).to("cpu")
    elapsed = time.perf_counter() - started
    abs_error = (output_cpu.float() - expected.float()).abs().reshape(-1)
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()
    p99_abs_error = torch.quantile(abs_error, 0.99).item()
    p999_abs_error = torch.quantile(abs_error, 0.999).item()
    expected_abs_max = expected.float().abs().max().item()
    torch.testing.assert_close(
        output_cpu.float(),
        expected.float(),
        atol=0.75 if args.dtype == "fp8" else 0.15,
        rtol=0.08 if args.dtype == "fp8" else 0.02,
    )

    hot_seconds = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        compiled(hidden).to("cpu")
        hot_seconds.append(time.perf_counter() - started)

    print(
        json.dumps(
            {
                "status": "passed",
                "backend": get_compile_backend_name(),
                "dtype": "static_fp8" if args.dtype == "fp8" else "bf16",
                "tokens": args.tokens,
                "hidden_size": _HIDDEN_SIZE,
                "intermediate_size": _INTERMEDIATE_SIZE,
                "tp_degree": _TP_DEGREE,
                "local_intermediate": _INTERMEDIATE_SIZE // _TP_DEGREE,
                "kernel_local_intermediate": args.kernel_local_intermediate,
                "compile_and_first_run_seconds": elapsed,
                "hot_run_seconds_min": min(hot_seconds),
                "hot_run_seconds_median": statistics.median(hot_seconds),
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "p99_abs_error": p99_abs_error,
                "p999_abs_error": p999_abs_error,
                "expected_abs_max": expected_abs_max,
                "output_finite": bool(torch.isfinite(output_cpu).all()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
