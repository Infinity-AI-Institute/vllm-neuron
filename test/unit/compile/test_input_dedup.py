"""Regression tests for graph-input storage identity."""

import torch

from vllm_neuron.compile.backend import _detect_duplicate_inputs
from vllm_neuron.compile.parallel_trace import _swap_to_meta_no_free


def test_input_dedup_keeps_independent_equal_shape_allocations():
    shared = torch.zeros(8)
    shared_view = shared.view_as(shared)
    independent = torch.zeros(8)

    keep_mask, dupe_map = _detect_duplicate_inputs(
        [shared, shared_view, shared, independent]
    )

    assert keep_mask == [True, False, False, True]
    assert dupe_map == [0, 0, 0, 1]


def test_meta_swap_preserves_real_aliases_without_aliasing_layers():
    class CacheOwner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_0_cache = torch.zeros(4, 2)
            self.layer_1_cache = torch.zeros(4, 2)
            self.layer_0_alias = self.layer_0_cache.view_as(self.layer_0_cache)

    owner = CacheOwner()
    _swap_to_meta_no_free(owner)

    layer_0_storage = owner.layer_0_cache.untyped_storage()._cdata
    layer_1_storage = owner.layer_1_cache.untyped_storage()._cdata
    alias_storage = owner.layer_0_alias.untyped_storage()._cdata
    assert owner.layer_0_cache.device.type == "meta"
    assert owner.layer_1_cache.device.type == "meta"
    assert layer_0_storage != layer_1_storage
    assert alias_storage == layer_0_storage


def test_meta_swap_preserves_distinct_offset_views_of_one_cache():
    class CacheOwner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            full_cache = torch.zeros(2, 4, 2)
            self.k_cache = full_cache[0]
            self.v_cache = full_cache[1]

    owner = CacheOwner()
    _swap_to_meta_no_free(owner)

    assert (
        owner.k_cache.untyped_storage()._cdata == owner.v_cache.untyped_storage()._cdata
    )
    assert owner.k_cache.storage_offset() == 0
    assert owner.v_cache.storage_offset() == owner.k_cache.numel()

    keep_mask, dupe_map = _detect_duplicate_inputs([owner.k_cache, owner.v_cache])
    assert keep_mask == [True, True]
    assert dupe_map == [0, 1]
