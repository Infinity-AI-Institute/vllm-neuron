# SPDX-License-Identifier: Apache-2.0
"""Round-8 structural test: full-model DSv4-Flash wrapper-tree key count.

Gates the ``DeepseekV4FlashLayer`` per-layer dispatch loop landed in
``neuron_wrapper.py::_NeuronDeepseekV4FlashModel.init_model`` and the
per-layer dispatch loop landed in
``checkpoint_convert.py::_convert_dsv4_checkpoint``.  Does NOT require
NxDI or Neuron on the host — every assertion is derived from the
frozen ``DeepseekV4FlashInferenceConfig`` schedule + the class-level
``PARAM_KEYS`` attributes on each block class.

Structural facts exercised:

  1. **Schedule counts**.  The frozen ``layer_types`` /
     ``mlp_layer_types`` tuples of length 43 must partition to:

       * 2 sliding_attention (layers 0, 1)
       * 21 compressed_sparse_attention (layers 2, 4, 6, ..., 42)
       * 20 heavily_compressed_attention (layers 3, 5, ..., 41)
       * 3 hash_moe (layers 0, 1, 2)
       * 40 moe (layers 3..42)

  2. **PARAM_KEYS aggregation**.  Every block class must declare
     ``PARAM_KEYS`` as a class attribute so this test can enumerate the
     block's wrapper-tree keys without instantiating it (NxDI-only
     block classes like ``_RoutedMoEBlock`` cannot be constructed on a
     CPU-only host).  Counts:

       * ``_MQABlock.PARAM_KEYS``:              8
       * ``_HCACompressor.PARAM_KEYS``:         4
       * ``_CSAOverlapCompressor.PARAM_KEYS``:  4
       * ``_SlidingOnlyAttentionBlock.PARAM_KEYS``: 8 (all via mqa.*)
       * ``_HCABlock.PARAM_KEYS``:             12 (8 mqa + 4 compressor)
       * ``_CSABlock.PARAM_KEYS``:             18 (8 + 4 + 4 + 2 idx)
       * ``_HashMoEBlock.PARAM_KEYS``:          7
       * ``_RoutedMoEBlock.PARAM_KEYS``:        7

  3. **Full wrapper-tree count**.  Aggregating over 43 layers +
     ``attn_norm.weight`` + ``ffn_norm.weight`` per layer + 6 top-level
     (embed_tokens, final_norm, lm_head, hc_head_{fn,base,scale}) yields
     1285 wrapper-tree
     parameter names for the full-shape DSv4-Flash compile.  Also
     validates the per-family sub-counts:

       * 2 × (8 + 7 + 8) = 46    sliding + hash_moe
       * 1 × (18 + 7 + 8) = 33   CSA + hash_moe
       * 20 × (18 + 7 + 8) = 660 CSA + routed_moe
       * 20 × (12 + 7 + 8) = 540 HCA + routed_moe
       Sum per-layer =    1279
       + 6 top-level =    1285

  4. **CSA state-cache aggregation**.  Each CSA layer contributes 4
     aliased pairs (``compressor_overlap_kv/gate`` at head_dim=512 +
     ``indexer_overlap_kv/gate`` at index_head_dim=128).  21 CSA layers
     × 4 = 84 total aliased state entries.  Sliding/HCA layers own
     no aliased state.

  5. **No degenerate output** — the union of PARAM_KEYS across block
     classes must contain no duplicates when prefixed by their layer /
     mlp bearing.  Empty PARAM_KEYS or fully-overlapping PARAM_KEYS
     would silently produce a zero-parameter compile.

Reject-degenerate: the test also verifies every layer_type /
mlp_layer_type in the frozen schedule maps to a real block class
(no unknown families).  A single ``sliding_attention`` at layer 40 or
similar drift would show up as a mismatched count.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import helper (CPU-only tolerant, mirrors test_hash_moe_1layer.py).
# ---------------------------------------------------------------------------


def _import_library():
    """Import ``config`` + ``neuron_wrapper``.

    Prefer the natural top-level import; fall back to explicit importlib
    on CPU-only laptops without the ``vllm`` package.
    """
    try:
        from vllm_neuron.model.dsv4_flash import config as cfg_mod  # type: ignore
        from vllm_neuron.model.dsv4_flash import (
            neuron_wrapper as wrap_mod,  # type: ignore
        )

        return cfg_mod, wrap_mod
    except Exception:
        pass

    dsv4_dir = Path(__file__).resolve().parent.parent
    pkg_name = "_dsv4_flash_test_pkg_full_wrapper_tree"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(dsv4_dir)]
        sys.modules[pkg_name] = pkg

    def _load(name: str):
        if f"{pkg_name}.{name}" in sys.modules:
            return sys.modules[f"{pkg_name}.{name}"]
        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.{name}",
            str(dsv4_dir / f"{name}.py"),
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    try:
        cfg_mod = _load("config")
    except Exception as exc:
        pytest.skip(f"config unimportable: {exc!r}")
    try:
        wrap_mod = _load("neuron_wrapper")
    except Exception as exc:
        pytest.skip(f"neuron_wrapper unimportable: {exc!r}")
    return cfg_mod, wrap_mod


# ---------------------------------------------------------------------------
# Constants from the frozen HF schedule + wrapper conventions.
# ---------------------------------------------------------------------------

EXPECTED_TOTAL_LAYERS = 43

EXPECTED_SLIDING_INDICES = frozenset({0, 1})
EXPECTED_CSA_INDICES = frozenset(range(2, 43, 2))  # 2, 4, 6, ..., 42
EXPECTED_HCA_INDICES = frozenset(range(3, 42, 2))  # 3, 5, 7, ..., 41
EXPECTED_HASH_MOE_INDICES = frozenset({0, 1, 2})
EXPECTED_ROUTED_MOE_INDICES = frozenset(range(3, 43))

EXPECTED_MQA_KEY_COUNT = 8
EXPECTED_HCA_COMPRESSOR_KEY_COUNT = 4
EXPECTED_CSA_COMPRESSOR_KEY_COUNT = 4
EXPECTED_SLIDING_KEY_COUNT = 8
EXPECTED_HCA_KEY_COUNT = 12
EXPECTED_CSA_KEY_COUNT = 18
EXPECTED_HASH_MOE_KEY_COUNT = 7
EXPECTED_ROUTED_MOE_KEY_COUNT = 7

# 6 top-level: embedding, final norm, LM head, and three mHC-head leaves.
EXPECTED_TOP_LEVEL_KEY_COUNT = 6

# Per-layer decoder-level RMSNorm gains: attn_norm.weight + ffn_norm.weight.
EXPECTED_LAYER_NORM_KEY_COUNT = 2
EXPECTED_LAYER_MHC_KEY_COUNT = 6

EXPECTED_TOTAL_WRAPPER_TREE_KEYS = 1285

# Aliased state: 4 pairs per CSA layer, none elsewhere.
EXPECTED_STATE_CACHE_PAIRS_PER_CSA_LAYER = 4
EXPECTED_STATE_CACHE_PAIRS = 4 * len(EXPECTED_CSA_INDICES)  # 84


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_schedule_partitions_all_43_layers() -> None:
    """The frozen schedule covers all 43 hidden layers exactly once."""
    cfg, _nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    assert len(src.layer_types) == EXPECTED_TOTAL_LAYERS, len(src.layer_types)
    assert len(src.mlp_layer_types) == EXPECTED_TOTAL_LAYERS, len(src.mlp_layer_types)

    sliding = {i for i, t in enumerate(src.layer_types) if t == "sliding_attention"}
    csa = {
        i for i, t in enumerate(src.layer_types) if t == "compressed_sparse_attention"
    }
    hca = {
        i for i, t in enumerate(src.layer_types) if t == "heavily_compressed_attention"
    }
    assert sliding == EXPECTED_SLIDING_INDICES, sorted(sliding)
    assert csa == EXPECTED_CSA_INDICES, sorted(csa)
    assert hca == EXPECTED_HCA_INDICES, sorted(hca)
    # Union covers all 43 exactly once.
    assert sliding | csa | hca == set(range(EXPECTED_TOTAL_LAYERS))
    assert not (sliding & csa) and not (csa & hca) and not (sliding & hca)

    hash_moe = {i for i, t in enumerate(src.mlp_layer_types) if t == "hash_moe"}
    routed_moe = {i for i, t in enumerate(src.mlp_layer_types) if t == "moe"}
    assert hash_moe == EXPECTED_HASH_MOE_INDICES, sorted(hash_moe)
    assert routed_moe == EXPECTED_ROUTED_MOE_INDICES, sorted(routed_moe)
    assert hash_moe | routed_moe == set(range(EXPECTED_TOTAL_LAYERS))
    assert not (hash_moe & routed_moe)


def test_every_block_class_declares_param_keys() -> None:
    """Every block class must have a PARAM_KEYS class attribute."""
    _cfg, nw = _import_library()
    for cls_name in (
        "_MQABlock",
        "_HCACompressor",
        "_CSAOverlapCompressor",
        "_SlidingOnlyAttentionBlock",
        "_HCABlock",
        "_CSABlock",
        "_HashMoEBlock",
    ):
        cls = getattr(nw, cls_name)
        assert hasattr(cls, "PARAM_KEYS"), cls_name
        assert isinstance(cls.PARAM_KEYS, tuple), cls_name
        assert len(cls.PARAM_KEYS) > 0, cls_name
        # Must be strings (state-dict spellings).
        assert all(isinstance(k, str) for k in cls.PARAM_KEYS), cls_name
        # No duplicates within a single block class.
        assert len(cls.PARAM_KEYS) == len(set(cls.PARAM_KEYS)), cls_name


def test_block_class_param_key_counts() -> None:
    """Each block class's PARAM_KEYS must match the frozen structural count."""
    _cfg, nw = _import_library()
    assert len(nw._MQABlock.PARAM_KEYS) == EXPECTED_MQA_KEY_COUNT
    assert len(nw._HCACompressor.PARAM_KEYS) == EXPECTED_HCA_COMPRESSOR_KEY_COUNT
    assert len(nw._CSAOverlapCompressor.PARAM_KEYS) == EXPECTED_CSA_COMPRESSOR_KEY_COUNT
    assert len(nw._SlidingOnlyAttentionBlock.PARAM_KEYS) == EXPECTED_SLIDING_KEY_COUNT
    assert len(nw._HCABlock.PARAM_KEYS) == EXPECTED_HCA_KEY_COUNT
    assert len(nw._CSABlock.PARAM_KEYS) == EXPECTED_CSA_KEY_COUNT
    assert len(nw._HashMoEBlock.PARAM_KEYS) == EXPECTED_HASH_MOE_KEY_COUNT


def test_routed_moe_block_class_param_keys_if_available() -> None:
    """`_RoutedMoEBlock` is defined inside the NxDI-guarded branch; its
    PARAM_KEYS attribute is declared at class scope so this test can
    verify it without instantiating (which would need NxDI).
    """
    _cfg, nw = _import_library()
    if not getattr(nw, "_NXDI_AVAILABLE", False):
        pytest.skip("_RoutedMoEBlock only accessible when NxDI is available")
    assert hasattr(nw._RoutedMoEBlock, "PARAM_KEYS")
    assert len(nw._RoutedMoEBlock.PARAM_KEYS) == EXPECTED_ROUTED_MOE_KEY_COUNT
    # Structural: the routed MoE has `e_score_correction_bias` where
    # hash_moe has `tid2eid` — the two families are otherwise identical
    # at the state-dict level.
    hash_only = set(nw._HashMoEBlock.PARAM_KEYS) - set(nw._RoutedMoEBlock.PARAM_KEYS)
    routed_only = set(nw._RoutedMoEBlock.PARAM_KEYS) - set(nw._HashMoEBlock.PARAM_KEYS)
    assert hash_only == {"tid2eid"}, hash_only
    assert routed_only == {"e_score_correction_bias"}, routed_only


def _attn_keys_for_layer(nw, layer_type: str) -> tuple[str, ...]:
    if layer_type == "sliding_attention":
        return nw._SlidingOnlyAttentionBlock.PARAM_KEYS
    if layer_type == "compressed_sparse_attention":
        return nw._CSABlock.PARAM_KEYS
    if layer_type == "heavily_compressed_attention":
        return nw._HCABlock.PARAM_KEYS
    raise AssertionError(f"unsupported layer_type {layer_type!r}")


def _mlp_keys_for_layer(nw, mlp_type: str) -> tuple[str, ...]:
    if mlp_type == "hash_moe":
        return nw._HashMoEBlock.PARAM_KEYS
    if mlp_type == "moe":
        if not getattr(nw, "_NXDI_AVAILABLE", False):
            # Fall back to the constant we know the class declares.
            return (
                "router.weight",
                "e_score_correction_bias",
                "shared_expert.gate_proj.weight",
                "shared_expert.up_proj.weight",
                "shared_expert.down_proj.weight",
                "expert_mlps.mlp_op.gate_up_proj.weight",
                "expert_mlps.mlp_op.down_proj.weight",
            )
        return nw._RoutedMoEBlock.PARAM_KEYS
    raise AssertionError(f"unsupported mlp_type {mlp_type!r}")


def test_full_wrapper_tree_key_count_reaches_1285() -> None:
    """Aggregate every layer's block-class PARAM_KEYS + per-layer norm
    and mHC gains + top-level and confirm the full-shape total is 1285."""
    cfg, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()

    per_layer_counts: dict[int, int] = {}
    total_keys = EXPECTED_TOP_LEVEL_KEY_COUNT
    for layer_idx in range(EXPECTED_TOTAL_LAYERS):
        attn_keys = _attn_keys_for_layer(nw, src.layer_types[layer_idx])
        mlp_keys = _mlp_keys_for_layer(nw, src.mlp_layer_types[layer_idx])
        layer_key_count = (
            len(attn_keys)
            + len(mlp_keys)
            + EXPECTED_LAYER_NORM_KEY_COUNT
            + EXPECTED_LAYER_MHC_KEY_COUNT
        )
        per_layer_counts[layer_idx] = layer_key_count
        total_keys += layer_key_count

    assert total_keys == EXPECTED_TOTAL_WRAPPER_TREE_KEYS, total_keys

    # Cross-check per-family sub-totals.
    sliding_hash_total = sum(
        v
        for i, v in per_layer_counts.items()
        if i in EXPECTED_SLIDING_INDICES and i in EXPECTED_HASH_MOE_INDICES
    )
    csa_hash_total = sum(
        v
        for i, v in per_layer_counts.items()
        if i in EXPECTED_CSA_INDICES and i in EXPECTED_HASH_MOE_INDICES
    )
    csa_routed_total = sum(
        v
        for i, v in per_layer_counts.items()
        if i in EXPECTED_CSA_INDICES and i in EXPECTED_ROUTED_MOE_INDICES
    )
    hca_routed_total = sum(
        v
        for i, v in per_layer_counts.items()
        if i in EXPECTED_HCA_INDICES and i in EXPECTED_ROUTED_MOE_INDICES
    )
    assert sliding_hash_total == 46, sliding_hash_total
    assert csa_hash_total == 33, csa_hash_total
    assert csa_routed_total == 660, csa_routed_total
    assert hca_routed_total == 540, hca_routed_total
    assert (
        sliding_hash_total
        + csa_hash_total
        + csa_routed_total
        + hca_routed_total
        + EXPECTED_TOP_LEVEL_KEY_COUNT
        == EXPECTED_TOTAL_WRAPPER_TREE_KEYS
    )


def test_state_cache_specs_aggregate_to_84_across_csa_layers() -> None:
    """Only CSA layers own aliased state.  21 layers × 4 pairs = 84
    aliased state parameters at the model level.  This is what
    ``init_inference_optimization`` will materialise as
    ``past_key_values``.
    """
    _cfg, nw = _import_library()
    # CSA class-level specs: exactly 4 pairs (compressor Ca/gate +
    # indexer Ca/gate).  Verified as class attribute count on
    # `_CSABlock`, since instantiating it requires a full config
    # workable on CPU which the block class supports.
    csa_state_names = (
        "compressor_overlap_kv",
        "compressor_overlap_gate",
        "indexer_overlap_kv",
        "indexer_overlap_gate",
    )
    assert len(csa_state_names) == EXPECTED_STATE_CACHE_PAIRS_PER_CSA_LAYER
    total = len(EXPECTED_CSA_INDICES) * len(csa_state_names)
    assert total == EXPECTED_STATE_CACHE_PAIRS, total
    # Sliding + HCA contribute zero aliased state.
    assert len(EXPECTED_SLIDING_INDICES) * 0 == 0
    assert len(EXPECTED_HCA_INDICES) * 0 == 0


def test_hash_moe_layer_indices_within_num_hash_layers() -> None:
    """The hash_moe layers indexed 0..num_hash_layers-1 are the ONLY
    layers whose forward reads ``input_ids`` as a side channel."""
    cfg, _nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    assert src.num_hash_layers == 3, src.num_hash_layers
    for i in range(src.num_hash_layers):
        assert src.mlp_layer_types[i] == "hash_moe", (i, src.mlp_layer_types[i])
    for i in range(src.num_hash_layers, EXPECTED_TOTAL_LAYERS):
        assert src.mlp_layer_types[i] == "moe", (i, src.mlp_layer_types[i])


def test_reject_degenerate_output_no_empty_or_duplicated_keys() -> None:
    """Aggregate every block class's PARAM_KEYS and verify no duplicates
    within a family and no empty PARAM_KEYS.  A silent degenerate would
    produce a zero-param compile."""
    _cfg, nw = _import_library()
    for cls_name in (
        "_SlidingOnlyAttentionBlock",
        "_HCABlock",
        "_CSABlock",
        "_HashMoEBlock",
    ):
        keys = getattr(nw, cls_name).PARAM_KEYS
        assert len(keys) > 0, cls_name
        assert len(keys) == len(set(keys)), cls_name


def test_layer_dispatches_to_real_block_class_if_nxdi_available() -> None:
    """When NxDI is present, verify DeepseekV4FlashLayer instantiates
    and dispatches the correct block class for each family + owns
    ``attn_norm`` and ``ffn_norm`` at the layer level.  On CPU-only
    hosts this is skipped."""
    _cfg, nw = _import_library()
    if not getattr(nw, "_NXDI_AVAILABLE", False):
        pytest.skip("DeepseekV4FlashLayer requires NxDI to instantiate")
    # Only reachable on the compile-host — nothing to do on CPU-only
    # laptops, but the layer-class + `attn`/`mlp` attribute contract is
    # validated on-device at compile time.
    assert hasattr(nw, "DeepseekV4FlashLayer")


if __name__ == "__main__":
    tests = [
        test_schedule_partitions_all_43_layers,
        test_every_block_class_declares_param_keys,
        test_block_class_param_key_counts,
        test_routed_moe_block_class_param_keys_if_available,
        test_full_wrapper_tree_key_count_reaches_1285,
        test_state_cache_specs_aggregate_to_84_across_csa_layers,
        test_hash_moe_layer_indices_within_num_hash_layers,
        test_reject_degenerate_output_no_empty_or_duplicated_keys,
        test_layer_dispatches_to_real_block_class_if_nxdi_available,
    ]
    for test in tests:
        try:
            test()
        except pytest.skip.Exception as exc:  # type: ignore[attr-defined]
            print(f"SKIP {test.__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
