# SPDX-License-Identifier: Apache-2.0
"""Per-layer path counters used by the source-side correctness gates."""

from __future__ import annotations

from dataclasses import dataclass, field


def _zero_layers() -> dict[int, int]:
    return {layer: 0 for layer in range(45)}


@dataclass
class Glm53FlashTelemetry:
    dsa_path_active: dict[int, int] = field(default_factory=_zero_layers)
    kda_path_active: dict[int, int] = field(default_factory=_zero_layers)
    # The lane gate originally called KDA "linear". Keep that name as an ABI
    # alias while exposing the architecture's actual name as the primary key.
    linear_path_active: dict[int, int] = field(default_factory=_zero_layers)
    mla_active: dict[int, int] = field(default_factory=_zero_layers)
    state_buffer_reset_count: dict[int, int] = field(default_factory=_zero_layers)

    def increment(self, counter: str, layer_idx: int) -> None:
        table = getattr(self, counter)
        table[layer_idx] += 1

    def reset(self) -> None:
        for table in (
            self.dsa_path_active,
            self.kda_path_active,
            self.linear_path_active,
            self.mla_active,
            self.state_buffer_reset_count,
        ):
            for layer in table:
                table[layer] = 0

    def snapshot(self) -> dict[str, dict[int, int]]:
        return {
            "dsa_path_active": dict(self.dsa_path_active),
            "kda_path_active": dict(self.kda_path_active),
            "linear_path_active": dict(self.linear_path_active),
            "mla_active": dict(self.mla_active),
            "state_buffer_reset_count": dict(self.state_buffer_reset_count),
        }


__all__ = ["Glm53FlashTelemetry"]
