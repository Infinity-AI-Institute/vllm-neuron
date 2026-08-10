"""Allocator contract tests for Kimi K3's hidden static MLA padding sink."""

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[2]
RUNNER_PATH = ROOT / "vllm_neuron/vllm/worker/neuron_model_runner.py"


@dataclass
class FullAttentionSpec:
    block_size: int = 32
    num_kv_heads: int = 1
    head_size: int = 576
    dtype: torch.dtype = torch.bfloat16

    @property
    def page_size_bytes(self) -> int:
        return (
            2
            * self.num_kv_heads
            * self.block_size
            * self.head_size
            * self.dtype.itemsize
        )


def _load_allocator_harness():
    """Load only the reviewed allocator methods without importing vLLM."""
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NeuronModelRunner"
    )
    selected = {
        "_kimi_k3_mla_sink_plan",
        "_allocate_kv_cache_raw_tensors",
        "get_kv_cache_config",
    }
    harness = ast.ClassDef(
        name="AllocatorHarness",
        bases=[],
        keywords=[],
        body=[
            node
            for node in runner.body
            if isinstance(node, ast.FunctionDef) and node.name in selected
        ],
        decorator_list=[],
    )
    constants = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id.startswith("_KIMI_K3_")
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
    ]
    module = ast.fix_missing_locations(
        ast.Module(body=[*constants, harness], type_ignores=[])
    )
    namespace = {
        "FullAttentionSpec": FullAttentionSpec,
        "KVCacheConfig": object,
        "logger": logging.getLogger(__name__),
        "torch": torch,
    }
    exec(  # noqa: S102 - execute the reviewed method-only AST test harness.
        compile(module, str(RUNNER_PATH), "exec"), namespace
    )
    return namespace["AllocatorHarness"]


def _k3_model():
    model_type = type("KimiK3HybridForCausalLM", (), {})
    model_type.__module__ = "neuronx_distributed_inference.models.kimi_k3.serving.model"
    return model_type()


def _config(*, num_blocks: int = 7, spec: FullAttentionSpec | None = None):
    spec = spec or FullAttentionSpec()
    names = [f"kimi_k3.mla_cache.{pair}" for pair in range(12)]
    tensors = [
        SimpleNamespace(
            size=num_blocks * spec.page_size_bytes,
            shared_by=[name],
        )
        for name in names
    ]
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=names)],
        kv_cache_tensors=tensors,
    )


def _runner(*, architecture: str = "KimiK3ForCausalLM", model=None):
    runner = _load_allocator_harness()()
    runner.device = torch.device("cpu")
    runner.model = model if model is not None else _k3_model()
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture,
            get_total_num_kv_heads=lambda: 1,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=64,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )
    return runner


def test_k3_allocates_one_hidden_physical_block_per_paired_cache():
    runner = _runner()
    config = _config(num_blocks=7)
    scheduler_blocks_before = config.num_blocks
    page_size = config.kv_cache_groups[0].kv_cache_spec.page_size_bytes

    roots = runner._allocate_kv_cache_raw_tensors(config)

    assert set(roots) == {f"kimi_k3.mla_cache.{pair}" for pair in range(12)}
    assert len({root.data_ptr() for root in roots.values()}) == 12
    assert all(root.numel() == 8 * page_size for root in roots.values())
    typed_root = roots["kimi_k3.mla_cache.0"].view(torch.bfloat16)
    physical_cache = typed_root.view(2, 8, 1, 32, 576)
    assert physical_cache[:, 7].numel() == page_size // torch.bfloat16.itemsize
    # The sink is physical block N; the public scheduler receipt remains N.
    runner.kv_cache_config = config
    assert runner.get_kv_cache_config()["num_blocks"] == 7
    assert config.num_blocks == scheduler_blocks_before == 7


def test_non_k3_allocation_size_and_aliasing_are_unchanged():
    runner = _runner(architecture="LlamaForCausalLM", model=object())
    tensor = SimpleNamespace(size=41, shared_by=["layer.a", "layer.alias"])
    config = SimpleNamespace(kv_cache_tensors=[tensor])

    roots = runner._allocate_kv_cache_raw_tensors(config)

    assert roots["layer.a"] is roots["layer.alias"]
    assert roots["layer.a"].numel() == 41


def test_non_k3_duplicate_name_preserves_last_allocation_wins_behavior():
    runner = _runner(architecture="LlamaForCausalLM", model=object())
    config = SimpleNamespace(
        kv_cache_tensors=[
            SimpleNamespace(size=17, shared_by=["layer.a"]),
            SimpleNamespace(size=29, shared_by=["layer.a"]),
        ]
    )

    roots = runner._allocate_kv_cache_raw_tensors(config)

    assert roots["layer.a"].numel() == 29


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: setattr(config.kv_cache_tensors[0], "size", 1),
            "allocation size mismatch",
        ),
        (
            lambda config: config.kv_cache_tensors[0].shared_by.append(
                "kimi_k3.mla_cache.1"
            ),
            "singleton authoritative cache allocations",
        ),
        (
            lambda config: setattr(
                config.kv_cache_groups[0].kv_cache_spec, "head_size", 512
            ),
            "shape mismatch",
        ),
    ],
)
def test_k3_shape_or_identity_drift_fails_before_allocation(mutation, message):
    runner = _runner()
    config = _config()
    mutation(config)

    with pytest.raises(RuntimeError, match=message):
        runner._allocate_kv_cache_raw_tensors(config)


def test_k3_architecture_rejects_an_unrecognized_model_implementation():
    runner = _runner(model=object())

    with pytest.raises(RuntimeError, match="requires the frozen NDI model"):
        runner._allocate_kv_cache_raw_tensors(_config())


def test_k3_allocator_unwraps_torch_compile_model_wrapper():
    original = _k3_model()
    compiled = SimpleNamespace(_orig_mod=original)
    runner = _runner(model=compiled)

    roots = runner._allocate_kv_cache_raw_tensors(_config(num_blocks=3))

    assert len(roots) == 12
    page_size = _config(num_blocks=3).kv_cache_groups[0].kv_cache_spec.page_size_bytes
    assert all(root.numel() == 4 * page_size for root in roots.values())
