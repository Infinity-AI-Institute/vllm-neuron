#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for grammar bitmask logic used in structured outputs.

Tests the bit unpacking and reordering logic from _get_grammar_bitmask
in neuron_model_runner.py.
"""

import pytest
import torch


class TestBitmaskUnpacking:
    """Tests for unpacking packed int32 bitmask to boolean tensor."""

    def test_unpack_single_bit_set(self):
        """Test unpacking with single bit set in each position."""
        # Pack 32 tokens into single int32: bit 0 = token 0, bit 31 = token 31
        # Value 1 means only bit 0 (token 0) is allowed
        packed = torch.tensor([[1]], dtype=torch.int32)

        unpacked = self._unpack_bitmask(packed, vocab_size=32)

        assert unpacked.shape == (1, 32)
        assert unpacked[0, 0]  # Token 0 allowed
        assert unpacked[0, 1:].sum() == 0  # All others disallowed

    def test_unpack_all_ones(self):
        """Test unpacking with all bits set (all tokens allowed)."""
        # -1 in int32 = all 32 bits set
        packed = torch.tensor([[-1]], dtype=torch.int32)

        unpacked = self._unpack_bitmask(packed, vocab_size=32)

        assert unpacked.shape == (1, 32)
        assert unpacked.all()  # All tokens allowed

    def test_unpack_all_zeros(self):
        """Test unpacking with no bits set (no tokens allowed)."""
        packed = torch.tensor([[0]], dtype=torch.int32)

        unpacked = self._unpack_bitmask(packed, vocab_size=32)

        assert unpacked.shape == (1, 32)
        assert not unpacked.any()  # No tokens allowed

    def test_unpack_alternating_bits(self):
        """Test unpacking with alternating bits (even tokens allowed)."""
        # 0x55555555 = 01010101... in binary (bits 0,2,4,6... set)
        packed = torch.tensor([[0x55555555]], dtype=torch.int32)

        unpacked = self._unpack_bitmask(packed, vocab_size=32)

        # Even positions should be True, odd positions should be False
        for i in range(32):
            assert unpacked[0, i] == (i % 2 == 0), f"Position {i} mismatch"

    def test_unpack_multiple_int32_values(self):
        """Test unpacking multiple packed values (vocab > 32)."""
        # vocab_size = 64, need 2 int32 values
        # First 32: all allowed, Second 32: none allowed
        packed = torch.tensor([[-1, 0]], dtype=torch.int32)

        unpacked = self._unpack_bitmask(packed, vocab_size=64)

        assert unpacked.shape == (1, 64)
        assert unpacked[0, :32].all()  # First 32 allowed
        assert not unpacked[0, 32:].any()  # Second 32 disallowed

    def test_unpack_batch_of_requests(self):
        """Test unpacking bitmask for multiple requests."""
        # 2 requests: first allows all, second allows none
        packed = torch.tensor([[-1], [0]], dtype=torch.int32)

        unpacked = self._unpack_bitmask(packed, vocab_size=32)

        assert unpacked.shape == (2, 32)
        assert unpacked[0].all()  # Request 0: all allowed
        assert not unpacked[1].any()  # Request 1: none allowed

    def test_unpack_truncates_to_vocab_size(self):
        """Test that unpacking truncates to actual vocab_size."""
        # 2 packed int32 = 64 bits, but vocab_size = 50
        packed = torch.tensor([[-1, -1]], dtype=torch.int32)

        unpacked = self._unpack_bitmask(packed, vocab_size=50)

        assert unpacked.shape == (1, 50)
        assert unpacked.all()

    def test_unpack_llama_vocab_size(self):
        """Test unpacking for Llama-3 vocab size (128256)."""
        vocab_size = 128256
        packed_size = (vocab_size + 31) // 32  # = 4008

        # All tokens allowed
        packed = torch.full((1, packed_size), -1, dtype=torch.int32)

        unpacked = self._unpack_bitmask(packed, vocab_size=vocab_size)

        assert unpacked.shape == (1, vocab_size)
        assert unpacked.all()

    def _unpack_bitmask(
        self, packed_bitmask: torch.Tensor, vocab_size: int
    ) -> torch.Tensor:
        """
        Unpack packed int32 bitmask to boolean tensor.

        This mirrors the unpacking logic in _get_grammar_bitmask.
        """
        num_rows = packed_bitmask.shape[0]
        packed_size = packed_bitmask.shape[1]

        packed_uint = packed_bitmask.to(torch.int32).view(num_rows, packed_size, 1)
        bit_positions = torch.arange(32, dtype=torch.int32)

        unpacked = ((packed_uint >> bit_positions) & 1).view(num_rows, -1)
        unpacked = unpacked[:, :vocab_size]

        return unpacked.bool()


class TestBitmaskReordering:
    """Tests for reordering bitmask to match batch request order."""

    def test_reorder_single_so_request(self):
        """Test reordering with single SO request in batch."""
        # Batch has 1 request, it has SO
        batch_req_ids = ["req-0"]
        so_req_ids = ["req-0"]
        packed_bitmask = torch.tensor([[1]], dtype=torch.int32)  # Only token 0 allowed

        sorted_bitmask = self._reorder_bitmask(
            packed_bitmask, so_req_ids, batch_req_ids, spec_tokens={}, vocab_size=32
        )

        assert sorted_bitmask.shape == (1, 1)
        assert sorted_bitmask[0, 0] == 1

    def test_reorder_mixed_batch(self):
        """Test reordering with mix of SO and non-SO requests."""
        # Batch has 3 requests, only middle one has SO
        batch_req_ids = ["req-0", "req-1", "req-2"]
        so_req_ids = ["req-1"]
        packed_bitmask = torch.tensor([[0x0F]], dtype=torch.int32)  # Tokens 0-3 allowed

        sorted_bitmask = self._reorder_bitmask(
            packed_bitmask, so_req_ids, batch_req_ids, spec_tokens={}, vocab_size=32
        )

        # Result should be [3, packed_size] with req-0 and req-2 as -1 (all allowed)
        assert sorted_bitmask.shape == (3, 1)
        assert sorted_bitmask[0, 0] == -1  # req-0: no SO, all allowed
        assert sorted_bitmask[1, 0] == 0x0F  # req-1: has SO mask
        assert sorted_bitmask[2, 0] == -1  # req-2: no SO, all allowed

    def test_reorder_multiple_so_requests(self):
        """Test reordering with multiple SO requests."""
        batch_req_ids = ["req-0", "req-1", "req-2"]
        so_req_ids = ["req-0", "req-2"]  # First and third have SO
        packed_bitmask = torch.tensor(
            [
                [0x01],  # req-0: only token 0
                [0x02],  # req-2: only token 1
            ],
            dtype=torch.int32,
        )

        sorted_bitmask = self._reorder_bitmask(
            packed_bitmask, so_req_ids, batch_req_ids, spec_tokens={}, vocab_size=32
        )

        assert sorted_bitmask.shape == (3, 1)
        assert sorted_bitmask[0, 0] == 0x01  # req-0: token 0
        assert sorted_bitmask[1, 0] == -1  # req-1: no SO
        assert sorted_bitmask[2, 0] == 0x02  # req-2: token 1

    def test_reorder_with_spec_decode_tokens(self):
        """Test reordering accounts for speculative decode token offsets."""
        batch_req_ids = ["req-0", "req-1"]
        so_req_ids = ["req-1"]
        spec_tokens = {"req-0": (100, 101, 102)}  # req-0 has 3 spec tokens
        packed_bitmask = torch.tensor(
            [
                [0x0F],  # req-1 mask
            ],
            dtype=torch.int32,
        )

        # num_logit_rows = 2 requests + 3 spec tokens for req-0 = 5
        sorted_bitmask = self._reorder_bitmask(
            packed_bitmask, so_req_ids, batch_req_ids, spec_tokens, vocab_size=32
        )

        # Rows: req-0, spec-0, spec-1, spec-2, req-1
        assert sorted_bitmask.shape == (5, 1)
        assert sorted_bitmask[0, 0] == -1  # req-0: no SO
        assert sorted_bitmask[1, 0] == -1  # spec-0: no SO
        assert sorted_bitmask[2, 0] == -1  # spec-1: no SO
        assert sorted_bitmask[3, 0] == -1  # spec-2: no SO
        assert sorted_bitmask[4, 0] == 0x0F  # req-1: has SO

    def test_reorder_so_request_with_spec_decode(self):
        """Test SO request that also has speculative tokens."""
        batch_req_ids = ["req-0"]
        so_req_ids = ["req-0"]
        spec_tokens = {"req-0": (100, 101)}  # req-0 has 2 spec tokens
        # SO mask applies to req-0 AND its spec tokens
        packed_bitmask = torch.tensor(
            [
                [0x01],  # Main token
                [0x02],  # Spec token 1
                [0x03],  # Spec token 2
            ],
            dtype=torch.int32,
        )

        sorted_bitmask = self._reorder_bitmask(
            packed_bitmask, so_req_ids, batch_req_ids, spec_tokens, vocab_size=32
        )

        # Rows: req-0, spec-0, spec-1
        assert sorted_bitmask.shape == (3, 1)
        assert sorted_bitmask[0, 0] == 0x01  # Main token mask
        assert sorted_bitmask[1, 0] == 0x02  # Spec token 1 mask
        assert sorted_bitmask[2, 0] == 0x03  # Spec token 2 mask

    def test_no_so_requests_returns_all_allowed(self):
        """Test batch with no SO requests gets all -1 (all allowed)."""
        batch_req_ids = ["req-0", "req-1"]
        so_req_ids = []
        packed_bitmask = torch.zeros((0, 1), dtype=torch.int32)  # Empty

        # This should return None in actual implementation since no bitmask
        # But if called, should fill with -1
        sorted_bitmask = self._reorder_bitmask(
            packed_bitmask, so_req_ids, batch_req_ids, spec_tokens={}, vocab_size=32
        )

        assert sorted_bitmask.shape == (2, 1)
        assert (sorted_bitmask == -1).all()

    def _reorder_bitmask(
        self,
        packed_bitmask: torch.Tensor,
        so_req_ids: list[str],
        batch_req_ids: list[str],
        spec_tokens: dict[str, tuple],
        vocab_size: int,
    ) -> torch.Tensor:
        """
        Reorder packed bitmask to match batch request order.

        This mirrors the reordering logic in _get_grammar_bitmask.
        """
        packed_vocab_size = (
            packed_bitmask.shape[1]
            if packed_bitmask.numel() > 0
            else (vocab_size + 31) // 32
        )

        # Build mapping: req_id -> logit_index (accounting for spec tokens)
        struct_out_req_batch_indices: dict[str, int] = {}
        cumulative_offset = 0
        struct_out_req_ids_set = set(so_req_ids)

        for batch_index, req_id in enumerate(batch_req_ids):
            logit_index = batch_index + cumulative_offset
            cumulative_offset += len(spec_tokens.get(req_id, ()))
            if req_id in struct_out_req_ids_set:
                struct_out_req_batch_indices[req_id] = logit_index

        # Calculate total logit rows
        num_logit_rows = len(batch_req_ids) + sum(
            len(spec_tokens.get(r, ())) for r in batch_req_ids
        )

        # Full bitmask: -1 for non-SO rows
        sorted_bitmask = torch.full(
            (num_logit_rows, packed_vocab_size),
            fill_value=-1,
            dtype=torch.int32,
        )

        # Reorder: copy SO masks to correct positions
        cumulative_index = 0
        for req_id in so_req_ids:
            num_spec = len(spec_tokens.get(req_id, ()))
            if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
                for i in range(1 + num_spec):
                    bitmask_index = logit_idx + i
                    if cumulative_index + i < packed_bitmask.shape[0]:
                        sorted_bitmask[bitmask_index] = packed_bitmask[
                            cumulative_index + i
                        ]
            cumulative_index += 1 + num_spec

        return sorted_bitmask


class TestEndToEndBitmask:
    """End-to-end tests combining unpacking and reordering."""

    def test_full_pipeline_single_request(self):
        """Test full pipeline for single SO request."""
        vocab_size = 128
        packed_size = (vocab_size + 31) // 32  # 4

        # Create packed bitmask: only tokens 0-31 allowed
        packed = torch.zeros((1, packed_size), dtype=torch.int32)
        packed[0, 0] = -1  # First 32 tokens allowed

        # Simulate reordering (single request, no change)
        sorted_packed = packed.clone()

        # Unpack
        num_rows = sorted_packed.shape[0]
        packed_uint = sorted_packed.view(num_rows, packed_size, 1)
        bit_positions = torch.arange(32, dtype=torch.int32)
        unpacked = ((packed_uint >> bit_positions) & 1).view(num_rows, -1)
        unpacked = unpacked[:, :vocab_size].bool()

        assert unpacked.shape == (1, vocab_size)
        assert unpacked[0, :32].all()  # First 32 allowed
        assert not unpacked[0, 32:].any()  # Rest disallowed

    def test_full_pipeline_mixed_batch(self):
        """Test full pipeline for mixed batch."""
        vocab_size = 64
        packed_size = 2

        # SO request allows only token 0
        packed_so = torch.tensor([[1, 0]], dtype=torch.int32)

        # Reorder (simulating batch: [non-SO, SO, non-SO])
        sorted_packed = torch.full((3, packed_size), -1, dtype=torch.int32)
        sorted_packed[1] = packed_so[0]  # Request "b" at index 1

        # Unpack
        packed_uint = sorted_packed.view(3, packed_size, 1)
        bit_positions = torch.arange(32, dtype=torch.int32)
        unpacked = ((packed_uint >> bit_positions) & 1).view(3, -1)
        unpacked = unpacked[:, :vocab_size].bool()

        assert unpacked[0].all()  # Request "a": all allowed
        assert unpacked[1, 0]  # Request "b": only token 0
        assert unpacked[1, 1:].sum() == 0  # Request "b": rest disallowed
        assert unpacked[2].all()  # Request "c": all allowed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
