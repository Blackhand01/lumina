"""Policy-action definitions for KV-cache management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class KVPolicyAction:
    """A retention/precision point in the KV-cache action space."""

    retention: float
    kv_bits: int
    name: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.retention <= 1.0:
            raise ValueError(f"retention must be in (0, 1], got {self.retention}")
        if self.kv_bits <= 0:
            raise ValueError(f"kv_bits must be positive, got {self.kv_bits}")

    @property
    def uses_retention(self) -> bool:
        return self.retention < 1.0

    @property
    def uses_quantization(self) -> bool:
        return self.kv_bits < 16

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        retention_permille = int(round(self.retention * 1000))
        return f"r{retention_permille:03d}_b{self.kv_bits}"


def action_grid(
    *,
    retention_values: Iterable[float],
    bit_widths: Iterable[int],
) -> tuple[KVPolicyAction, ...]:
    """Build the theoretical retention/precision action space."""

    actions = [
        KVPolicyAction(retention=float(retention), kv_bits=int(bits))
        for retention in retention_values
        for bits in bit_widths
    ]
    return tuple(sorted(actions))
