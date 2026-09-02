"""CPU-only synthetic contract proof for GLM-5.3 TP64 artifact admission.

No model weights, NxDI modules, Neuron devices, or compiler are imported.
The test makes 64 tiny synthetic files solely to prove the factory/config
contract accepts a TP64 list only when all eleven DSA projections are full
copied `[32,4096]` rank-plan operations.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[3]
PACKAGE = ROOT / "vllm_neuron" / "model" / "glm53_flash"


def _load(name: str):
    qualified = f"vllm_neuron.model.glm53_flash.{name}"
    spec = importlib.util.spec_from_file_location(qualified, PACKAGE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


for package_name in (
    "vllm_neuron",
    "vllm_neuron.model",
    "vllm_neuron.model.glm53_flash",
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(PACKAGE)]
    sys.modules[package_name] = package

converter = types.ModuleType("vllm_neuron.model.glm53_flash.checkpoint_converter")
converter.GLM53_CHECKPOINT_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
converter.GLM53_CONFIG_SHA256 = "a" * 64
converter.GLM53_INDEX_SHA256 = "b" * 64
sys.modules[converter.__name__] = converter

rank_plan = types.ModuleType("vllm_neuron.model.glm53_flash.rank_plan")
rank_plan.build_glm53_rank_plan = None
sys.modules[rank_plan.__name__] = rank_plan

CONFIG = _load("runtime_config")
FACTORY = _load("runtime_factory")
ADAPTER = _load("compile_adapter")


def _profile() -> dict:
    return {
        "schema": CONFIG.GLM53_RUNTIME_CONFIG_SCHEMA,
        "architecture": CONFIG.GLM53_ARCHITECTURE,
        "checkpoint_revision": CONFIG.GLM53_CHECKPOINT_REVISION,
        "source_commit": "1" * 40,
        "source_tree": "2" * 40,
        "runtime_adapter": FACTORY.GLM53_RUNTIME_ADAPTER,
        "compiler_image_id": "sha256:" + "3" * 64,
        "compiler_image_digest": "example.invalid/neuron@sha256:" + "4" * 64,
        "compiler_version": "synthetic-cpu-proof",
        "runtime_packages": {"vllm-neuron": "synthetic-cpu-proof"},
        "compiler_flags": ["--auto-cast=none"],
        "tensor_parallel_degree": 64,
        "logical_neuron_cores": 2,
        "batch_size": 1,
        "max_sequence_length": 2560,
        "context_encoding_buckets": [2048],
        "token_generation_buckets": [2560],
        "weight_dtype": "bfloat16",
        "cache_dtype": "bfloat16",
        "runtime_quantization": "none",
        "sampling_mode": "greedy",
        "output_logits": True,
        "speculative_decode": False,
    }


def _policy() -> dict:
    return {
        "ownership_path": "/mnt/compile/OWNERSHIP.md",
        "active_compile_cap": 2,
        "systemd_unit": "glm53-compile-tp64-static-r3",
        "systemd_nice": 15,
        "systemd_scope": False,
        "network_mode": "none",
        "atomic_staging_suffix": ".partial-<run-id>",
        "compile_permitted": False,
    }


def _plan(rank: int, ownership: str, count: int = 11):
    tensor = SimpleNamespace(
        name=f"layers.{rank}.weight", dtype="torch.bfloat16", shape=(1,), nbytes=2
    )
    inventory = SimpleNamespace(
        contract_sha256=f"{rank + 1:064x}", total_tensor_bytes=2, tensors=(tensor,)
    )
    operations = tuple(
        SimpleNamespace(
            kind=ownership,
            target=SimpleNamespace(
                name=f"layers.{layer}.self_attn.indexer.weights_proj.weight",
                shape=(32, 4096),
            ),
        )
        for layer in range(count)
    )
    return SimpleNamespace(
        inventory=inventory, contract_sha256=f"{rank + 101:064x}", operations=operations
    )


def _materialize(root: Path, plans: dict[int, object]) -> tuple[Path, Path, Path, Path]:
    checkpoint = root / "checkpoint"
    ranks = root / "ranks"
    checkpoint.mkdir()
    ranks.mkdir()
    for rank, plan in plans.items():
        artifact = ranks / f"tp{rank}_sharded_checkpoint.safetensors"
        artifact.write_bytes(f"synthetic-rank-{rank}".encode())
        tensor = plan.inventory.tensors[0]
        manifest = {
            "schema": "glm53-streaming-rank-v1",
            "source": {
                "model": "GLM-5.3-Flash",
                "revision": converter.GLM53_CHECKPOINT_REVISION,
                "config_sha256": converter.GLM53_CONFIG_SHA256,
                "index_sha256": converter.GLM53_INDEX_SHA256,
            },
            "rank": rank,
            "tp_degree": 64,
            "rank_inventory_sha256": plan.inventory.contract_sha256,
            "rank_plan_sha256": plan.contract_sha256,
            "checkpoint": {
                "path": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": FACTORY._sha256_file(artifact),
            },
            "resource_bound": {
                "configured_max_chunk_bytes": FACTORY.GLM53_MAX_CHUNK_BYTES,
                "observed_max_chunk_bytes": 2,
                "full_rank_tensor_bytes": 2,
            },
            "tensors": [
                {
                    "name": tensor.name,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "nbytes": tensor.nbytes,
                    "chunks": 1,
                }
            ],
        }
        artifact.with_suffix(artifact.suffix + ".manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    profile = CONFIG.Glm53RuntimeConfig.from_mapping(_profile())
    requested, emitted = root / "requested.json", root / "emitted.json"
    requested.write_bytes(profile.canonical_bytes())
    emitted.write_bytes(profile.canonical_bytes())
    return checkpoint, ranks, requested, emitted


def _assert_rejected(plans: dict[int, object], message: str) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint, ranks, requested, emitted = _materialize(root, plans)
        FACTORY.build_glm53_rank_plan = lambda _root, *, rank, **_kw: plans[rank]
        try:
            FACTORY.Glm53RuntimeFactory.from_paths(
                checkpoint_dir=checkpoint,
                rank_dir=ranks,
                requested_config=requested,
                emitted_config=emitted,
                compile_policy=_policy(),
            )
        except FACTORY.Glm53RuntimeFactoryError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("misowned TP64 indexer plan was accepted")


def main() -> None:
    plans = {rank: _plan(rank, "copy") for rank in range(64)}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint, ranks, requested, emitted = _materialize(root, plans)
        FACTORY.build_glm53_rank_plan = lambda _root, *, rank, **_kw: plans[rank]
        profile = CONFIG.Glm53RuntimeConfig.load_canonical(requested.read_bytes())
        kwargs = ADAPTER.compile_kwargs(profile)
        assert kwargs["tp_degree"] == 64
        assert kwargs["logical_nc_config"] == 2
        assert kwargs["seq_len"] == 2560
        assert kwargs["output_logits"] is True
        emitted_config = dict(kwargs)
        emitted_config["torch_dtype"] = "torch.bfloat16"
        emitted_config["blockwise_matmul_config"] = {
            "use_shard_on_intermediate_dynamic_while": True,
            "skip_dma_token": True,
        }
        ADAPTER.assert_emitted_neuron_config(profile, emitted_config)
        emitted_config["output_logits"] = False
        try:
            ADAPTER.assert_emitted_neuron_config(profile, emitted_config)
        except ADAPTER.Glm53CompileAdapterError:
            pass
        else:
            raise AssertionError("emitted output_logits=false was accepted")
        bundle = FACTORY.Glm53RuntimeFactory.from_paths(
            checkpoint_dir=checkpoint,
            rank_dir=ranks,
            requested_config=requested,
            emitted_config=emitted,
            compile_policy=_policy(),
        )
        topology = bundle.to_mapping()["topology"]
        assert topology == {
            "tp_degree": 64,
            "rank_count": 64,
            "indexer_weights_proj_ownership": "replicated-copy[32,4096]",
            "output_logits": True,
        }
        assert len(bundle.ranks) == 64
        assert (
            FACTORY.GLM53_TP64_RUNTIME_FACTORY_ABI in bundle.canonical_bytes().decode()
        )

    _assert_rejected(
        {rank: _plan(rank, "copy", 10) for rank in range(64)}, "11 indexer"
    )
    _assert_rejected({rank: _plan(rank, "shard0") for rank in range(64)}, "must copy")
    invalid = _profile()
    invalid["tensor_parallel_degree"] = 48
    try:
        CONFIG.Glm53RuntimeConfig.from_mapping(invalid)
    except CONFIG.Glm53RuntimeConfigError:
        pass
    else:
        raise AssertionError("unsupported TP48 configuration was accepted")
    invalid = _profile()
    invalid["output_logits"] = False
    try:
        CONFIG.Glm53RuntimeConfig.from_mapping(invalid)
    except CONFIG.Glm53RuntimeConfigError:
        pass
    else:
        raise AssertionError("TP64 output_logits=false was accepted")
    print(
        "TP64_FACTORY_CONFIG_MANIFEST_CPU_PASS "
        "tp=64 ranks=64 ownership=copy[32,4096] "
        "prefill_s2048_decode_total_s2560=true output_logits=true"
    )


if __name__ == "__main__":
    main()
