# SPDX-License-Identifier: Apache-2.0
"""
Recurrent state for Qwen3.5's Gated DeltaNet layers
===================================================

vLLM's paged KV cache only covers the 8 full-attention layers. The 24 Gated
DeltaNet layers carry a fixed-size state per request instead:

- conv state ``[slots, conv_dim, K-1]`` in the model dtype (raw inputs), and
- recurrent state ``[slots, num_v_heads, head_k_dim, head_v_dim]`` in F32.

For Qwen3.5-9B that is about 50 MiB per slot (48 MiB recurrent + ~2.3 MiB
conv across 24 layers), independent of prompt length, so the slot count is an
explicit, sized setting rather than inherited from ``max_num_seqs``.

Why slots and not the batch row
-------------------------------
vllm-neuron's runner calls ``input_batch.condense()``, which moves surviving
requests into the rows freed by finished ones. A state keyed on the batch row
would follow the row, and a moved request would continue on another
request's state: fluent, wrong, and invisible to a smoke test. The runner
therefore assigns each request a stable slot for its whole lifetime
(:class:`SlotAllocator`) and passes ``state_slots`` per batch row.

Everything the compiled graph touches is a tensor op indexed by a tensor, so
no Python dict lives inside the traced model.
"""

import torch


class RecurrentStateBuffers:
    """Device buffers for every Gated DeltaNet layer, indexed by slot.

    Slot ``num_slots`` (the last row) is a scratch sink: padded batch rows
    point at it, so their writes never land on a live request.
    """

    def __init__(
        self,
        layer_indices: list[int],
        num_slots: int,
        conv_dim: int,
        conv_width: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_dtype: torch.dtype,
        device: torch.device | str = "cpu",
    ):
        self.layer_indices = list(layer_indices)
        self.num_slots = num_slots
        rows = num_slots + 1
        self.conv = {
            layer: torch.zeros(rows, conv_dim, conv_width, dtype=conv_dtype, device=device)
            for layer in self.layer_indices
        }
        self.recurrent = {
            layer: torch.zeros(
                rows, num_v_heads, head_k_dim, head_v_dim, dtype=torch.float32, device=device
            )
            for layer in self.layer_indices
        }

    @property
    def sink(self) -> int:
        return self.num_slots

    @staticmethod
    def bytes_per_slot(
        num_layers: int, conv_dim: int, conv_width: int, num_v_heads: int,
        head_k_dim: int, head_v_dim: int, conv_itemsize: int,
    ) -> int:
        recurrent = num_v_heads * head_k_dim * head_v_dim * 4
        conv = conv_dim * conv_width * conv_itemsize
        return num_layers * (recurrent + conv)

    def read(
        self, layer: int, slots: torch.Tensor, reset: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather this layer's state for the batch; zeros where ``reset``."""
        conv = self.conv[layer].index_select(0, slots)
        recurrent = self.recurrent[layer].index_select(0, slots)
        conv = torch.where(reset.view(-1, 1, 1), torch.zeros_like(conv), conv)
        recurrent = torch.where(
            reset.view(-1, 1, 1, 1), torch.zeros_like(recurrent), recurrent
        )
        return conv, recurrent

    def write(
        self, layer: int, slots: torch.Tensor, conv: torch.Tensor, recurrent: torch.Tensor
    ) -> None:
        """Scatter this layer's updated state back to its slots."""
        self.conv[layer].index_copy_(0, slots, conv.to(self.conv[layer].dtype))
        self.recurrent[layer].index_copy_(0, slots, recurrent.to(torch.float32))


class SlotAllocator:
    """Host-side ``request_id -> slot`` map owned by the runner.

    Assigned when a request is first scheduled, freed when it finishes, and
    never moved by ``condense()``. ``slots_for_rows`` turns the current batch
    order into the ``state_slots`` tensor the model consumes.
    """

    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self._free = list(range(num_slots - 1, -1, -1))
        self._slot: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._slot)

    def acquire(self, request_id: str) -> int:
        if request_id in self._slot:
            return self._slot[request_id]
        if not self._free:
            raise RuntimeError(
                f"no free Gated DeltaNet state slot for {request_id}: all "
                f"{self.num_slots} are held; raise the slot count or lower "
                f"max_num_seqs"
            )
        slot = self._free.pop()
        self._slot[request_id] = slot
        return slot

    def release(self, request_id: str) -> None:
        slot = self._slot.pop(request_id, None)
        if slot is not None:
            self._free.append(slot)

    def slot(self, request_id: str) -> int:
        return self._slot[request_id]

    def slots_for_rows(self, request_ids: list[str | None], sink: int) -> list[int]:
        """Slot per batch row; ``None`` (padding) rows map to the sink."""
        return [sink if rid is None else self._slot[rid] for rid in request_ids]
