# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest

from vllm_neuron.model.glm53_flash.compile_adapter import (
    Glm53CompileAdapterError,
    assert_emitted_neuron_config,
    assert_joint_both_application,
    compile_kwargs,
    construct_joint_both_application,
)
from vllm_neuron.model.glm53_flash.runtime_config import (
    GLM53_ARCHITECTURE,
    GLM53_CHECKPOINT_REVISION,
    GLM53_RUNTIME_CONFIG_SCHEMA,
    Glm53RuntimeConfig,
)
from vllm_neuron.model.glm53_flash.runtime_factory import GLM53_RUNTIME_ADAPTER

MANIFEST = (
    Path(__file__).parents[3]
    / "vllm_neuron"
    / "model"
    / "glm53_flash"
    / "TRANSPLANT-PROVENANCE.json"
)


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}".encode() + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def _reviewed_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "cat-file", "-p", f"{commit}:{path}"])


def _git_blob(treeish: str, path: str) -> str:
    row = subprocess.check_output(
        ["git", "ls-tree", treeish, "--", path], text=True
    ).strip()
    return row.split()[2]


def _profile(**updates) -> Glm53RuntimeConfig:
    value = {
        "schema": GLM53_RUNTIME_CONFIG_SCHEMA,
        "architecture": GLM53_ARCHITECTURE,
        "checkpoint_revision": GLM53_CHECKPOINT_REVISION,
        "source_commit": "1" * 40,
        "source_tree": "2" * 40,
        "runtime_adapter": GLM53_RUNTIME_ADAPTER,
        "compiler_image_id": "sha256:" + "3" * 64,
        "compiler_image_digest": "example.invalid/neuron@sha256:" + "4" * 64,
        "compiler_version": "pinned",
        "runtime_packages": {"vllm-neuron": "1" * 40},
        "compiler_flags": ["--auto-cast=none"],
        "tensor_parallel_degree": 32,
        "logical_neuron_cores": 2,
        "batch_size": 1,
        "max_sequence_length": 8192,
        "context_encoding_buckets": [128, 512, 8192],
        "token_generation_buckets": [128, 512, 8192],
        "weight_dtype": "bfloat16",
        "cache_dtype": "bfloat16",
        "runtime_quantization": "none",
        "sampling_mode": "greedy",
        "output_logits": True,
        "speculative_decode": False,
    }
    value.update(updates)
    return Glm53RuntimeConfig.from_mapping(value)


def _emitted(profile: Glm53RuntimeConfig) -> dict:
    value = compile_kwargs(profile)
    value["torch_dtype"] = "torch.bfloat16"
    value["blockwise_matmul_config"] = {
        "use_shard_on_intermediate_dynamic_while": True,
        "skip_dma_token": True,
    }
    return value


def _joint_profile(**updates) -> Glm53RuntimeConfig:
    values = {
        "tensor_parallel_degree": 64,
        "batch_size": 1,
        "max_sequence_length": 2560,
        "context_encoding_buckets": [2048],
        "token_generation_buckets": [2560],
    }
    values.update(updates)
    return _profile(**values)


def _phase(tag: str, *, prefill: bool, active: int, buckets: list) -> SimpleNamespace:
    return SimpleNamespace(
        tag=tag,
        neuron_config=SimpleNamespace(
            is_prefill_stage=prefill,
            n_active_tokens=active,
            buckets=buckets,
            seq_len=2560,
            tp_degree=64,
            logical_nc_config=2,
        ),
    )


def _joint_application() -> SimpleNamespace:
    cte = _phase(
        "context_encoding_model", prefill=True, active=2048, buckets=[2048]
    )
    tkg = _phase(
        "token_generation_model", prefill=False, active=1, buckets=[[1, 2560]]
    )
    return SimpleNamespace(
        _emit_phases="BOTH",
        _builder=None,
        models=[cte, tkg],
        context_encoding_model=cte,
        token_generation_model=tkg,
    )


def test_exact_profile_maps_every_shape_field_and_passes_emitted_gate():
    profile = _profile()
    expected = compile_kwargs(profile)
    assert expected == {
        "tp_degree": 32,
        "ctx_batch_size": 1,
        "tkg_batch_size": 1,
        "seq_len": 8192,
        "context_encoding_buckets": [128, 512, 8192],
        "token_generation_buckets": [128, 512, 8192],
        "max_context_length": 8192,
        "logical_nc_config": 2,
        "output_logits": True,
        "skip_sharding": True,
        "save_sharded_checkpoint": True,
    }
    assert_emitted_neuron_config(profile, _emitted(profile))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runtime_adapter", "wrong", "adapter identity"),
        ("logical_neuron_cores", 1, "LNC2"),
        ("weight_dtype", "float8_e4m3fn", "weights must be BF16"),
        ("cache_dtype", "float8_e4m3fn", "cache must be BF16"),
        ("runtime_quantization", "fp8", "quantization must be none"),
        ("output_logits", False, "full-vocabulary correctness gate"),
        ("compiler_flags", ["--auto-cast=matmult"], "--auto-cast=none"),
        (
            "compiler_flags",
            ["--auto-cast=none", "--auto-cast=matmult"],
            "conflicting compiler auto-cast",
        ),
    ],
)
def test_unexpressible_profile_fails_before_compile(field, value, message):
    with pytest.raises(Glm53CompileAdapterError, match=message):
        compile_kwargs(_profile(**{field: value}))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("logical_nc_config"),
        lambda value: value.__setitem__("logical_nc_config", 1),
        lambda value: value.__setitem__("torch_dtype", "float8_e4m3fn"),
        lambda value: value.__setitem__("output_logits", False),
        lambda value: value["blockwise_matmul_config"].__setitem__(
            "use_shard_on_intermediate_dynamic_while", False
        ),
        lambda value: value["blockwise_matmul_config"].pop("skip_dma_token"),
        lambda value: value.__setitem__("kv_cache_quant", "fp8"),
    ],
)
def test_emitted_config_drift_fails_closed(mutation):
    profile = _profile()
    emitted = copy.deepcopy(_emitted(profile))
    mutation(emitted)
    with pytest.raises(Glm53CompileAdapterError):
        assert_emitted_neuron_config(profile, emitted)


def test_joint_application_accepts_exact_both_shape():
    assert_joint_both_application(_joint_profile(), _joint_application())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tensor_parallel_degree", 32, "TP64"),
        ("batch_size", 2, "batch size one"),
        ("max_sequence_length", 3072, "S2560"),
        ("context_encoding_buckets", [1024, 2048], "CTE2048"),
        ("token_generation_buckets", [2048], "TKG position"),
    ],
)
def test_joint_profile_drift_fails_before_construction(field, value, message):
    with pytest.raises(Glm53CompileAdapterError, match=message):
        assert_joint_both_application(_joint_profile(**{field: value}), _joint_application())


@pytest.mark.parametrize("mutation", ["phase", "order", "owner", "cte", "tkg", "builder"])
def test_joint_application_drift_fails_closed(mutation):
    application = _joint_application()
    if mutation == "phase":
        application._emit_phases = "CTE"
    elif mutation == "order":
        application.models.reverse()
    elif mutation == "owner":
        application.token_generation_model = SimpleNamespace()
    elif mutation == "cte":
        application.models[0].neuron_config.n_active_tokens = 1024
    elif mutation == "tkg":
        application.models[1].neuron_config.buckets = [[1, 2048]]
    else:
        application._builder = object()
    with pytest.raises(Glm53CompileAdapterError):
        assert_joint_both_application(_joint_profile(), application)


def test_constructor_pins_both_and_never_builds_or_compiles(monkeypatch):
    calls = []

    class Wrapper:
        @classmethod
        def build_inference_config(cls, source_config, **kwargs):
            calls.append(("config", source_config, kwargs))
            return "inference-config"

        def __new__(cls, model_path, config):
            calls.append(("construct", model_path, config))
            return _joint_application()

    monkeypatch.setenv("NXDI_EMIT_PHASES", "BOTH")
    result = construct_joint_both_application(
        _joint_profile(),
        model_path="/immutable/checkpoint",
        source_config="source-config",
        _wrapper_cls=Wrapper,
    )
    assert result is not None
    assert [row[0] for row in calls] == ["config", "construct"]
    kwargs = calls[0][2]
    assert kwargs["tp_degree"] == 64
    assert kwargs["ctx_batch_size"] == kwargs["tkg_batch_size"] == 1
    assert kwargs["seq_len"] == 2560
    assert kwargs["context_encoding_buckets"] == [2048]
    assert kwargs["token_generation_buckets"] == [2560]
    assert kwargs["max_context_length"] == 2048


@pytest.mark.parametrize("value", [None, "CTE", "TKG", "both"])
def test_constructor_refuses_implicit_or_phase_only_selection(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("NXDI_EMIT_PHASES", raising=False)
    else:
        monkeypatch.setenv("NXDI_EMIT_PHASES", value)
    with pytest.raises(Glm53CompileAdapterError, match="explicitly BOTH"):
        construct_joint_both_application(
            _joint_profile(),
            model_path="/immutable/checkpoint",
            source_config=object(),
            _wrapper_cls=object(),
        )


def test_transplant_manifest_binds_every_reviewed_file_to_current_bytes():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["provenance_status"] == "external_origin_unavailable"
    assert manifest["origin_claim"] is None
    assert manifest["base_commit"] == "1b2e90d1f7fa5296aeaa57420794393958fe566e"
    assert manifest["reviewed_source_commit"] == (
        "55ab7ce771aa6299050e4fb09830831f6ac89375"
    )
    assert manifest["reviewed_source_tree"] == (
        "8a3f86f05fcfeac417072ed1c6fb96c8480be69d"
    )
    rows = manifest["files"]
    assert len(rows) == 30
    reviewed_kernel_paths = {
        "vllm_neuron/kernels/dsa_lightning_indexer.py",
        "vllm_neuron/kernels/glm52_indexer_fp8_scale_fix.py",
        "vllm_neuron/kernels/kda_state_v2.py",
        "vllm_neuron/kernels/moe_dispatch.py",
    }
    reviewed_model_prefix = "vllm_neuron/model/glm53_flash/"
    expected_paths = {
        path
        for path in subprocess.check_output(
            [
                "git",
                "diff",
                "--name-only",
                f"{manifest['base_commit']}..{manifest['reviewed_source_commit']}",
            ],
            text=True,
        ).splitlines()
        if (
            path in reviewed_kernel_paths
            or path == "vllm_neuron/model/registry.py"
            or (
                path.startswith(reviewed_model_prefix)
                and path != f"{reviewed_model_prefix}TRANSPLANT-PROVENANCE.json"
            )
        )
    }
    assert {row["path"] for row in rows} == expected_paths
    root = MANIFEST.parents[3]
    for row in rows:
        path = root / row["path"]
        data = _reviewed_bytes(manifest["reviewed_source_commit"], row["path"])
        assert path.is_file() and not path.is_symlink()
        assert row["mode"] == "100644"
        assert row["git_blob_sha1"] == _git_blob_sha1(data)
        assert row["git_blob_sha1"] == _git_blob(
            manifest["reviewed_source_tree"], row["path"]
        )
        assert row["git_blob_sha1"] == _git_blob("HEAD", row["path"])
        assert row["raw_sha256"] == hashlib.sha256(data).hexdigest()
