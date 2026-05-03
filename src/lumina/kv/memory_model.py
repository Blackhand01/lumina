"""Analytical KV-cache memory estimates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .actions import KVPolicyAction


BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True)
class KVModelSpec:
    """Model metadata required to estimate KV-cache footprint."""

    layers: int
    kv_heads: int
    head_dim: int
    dtype_bits: int = 16

    @classmethod
    def from_hf_config(cls, path: str | Path) -> "KVModelSpec":
        config_path = Path(path)
        if config_path.is_dir():
            config_path = config_path / "config.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        hidden_size = int(data["hidden_size"])
        attention_heads = int(data["num_attention_heads"])
        head_dim = int(data.get("head_dim") or hidden_size // attention_heads)
        return cls(
            layers=int(data["num_hidden_layers"]),
            kv_heads=int(data.get("num_key_value_heads", attention_heads)),
            head_dim=head_dim,
            dtype_bits=16,
        )

    def full_cache_mib(self, tokens: int) -> float:
        bytes_per_token = self.layers * self.kv_heads * self.head_dim * 2 * (self.dtype_bits / 8)
        return tokens * bytes_per_token / BYTES_PER_MIB

    def action_cache_mib(self, action: KVPolicyAction, tokens: int, *, metadata_multiplier: float = 1.0) -> float:
        return self.full_cache_mib(tokens) * action.retention * (action.kv_bits / self.dtype_bits) * metadata_multiplier
