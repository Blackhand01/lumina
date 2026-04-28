"""Interchangeable KV cache policies for Checkpoint 3 experiments."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def cache_nbytes(prompt_cache: list[Any]) -> int:
    total = 0
    for item in prompt_cache:
        nbytes = getattr(item, "nbytes", 0)
        total += int(nbytes() if callable(nbytes) else nbytes)
    return total


class CachePolicy(abc.ABC):
    """Base interface for all Lumina KV cache compression policies."""

    name: str
    description: str

    @abc.abstractmethod
    def make_cache(self, model: Any) -> list[Any]:
        """Create a prompt cache compatible with MLX generation."""

    def generation_kwargs(self) -> dict[str, Any]:
        """Extra keyword arguments passed to `mlx_lm.generate.generate_step`."""
        return {}

    def cache_nbytes(self, prompt_cache: list[Any]) -> int:
        return cache_nbytes(prompt_cache)


@dataclass
class FullCachePolicy(CachePolicy):
    name: str = "full"
    description: str = "Full FP16 KV cache control."

    def make_cache(self, model: Any) -> list[Any]:
        return model.make_cache()


@dataclass
class QuantizedCachePolicy(CachePolicy):
    bits: int = 8
    group_size: int = 64
    quantized_kv_start: int = 0

    @property
    def name(self) -> str:
        return f"quantized_{self.bits}bit"

    @property
    def description(self) -> str:
        return f"MLX quantized KV cache, {self.bits}-bit, group size {self.group_size}."

    def make_cache(self, model: Any) -> list[Any]:
        return model.make_cache()

    def generation_kwargs(self) -> dict[str, Any]:
        return {
            "kv_bits": self.bits,
            "kv_group_size": self.group_size,
            "quantized_kv_start": self.quantized_kv_start,
        }


@dataclass
class EvictionCachePolicy(CachePolicy):
    max_kv_size: int = 1024
    keep: int = 4
    name: str = "eviction"

    @property
    def description(self) -> str:
        return f"Rotating KV cache, keep first {self.keep} tokens and latest {self.max_kv_size} tokens."

    def make_cache(self, model: Any) -> list[Any]:
        from mlx_lm.models.cache import RotatingKVCache

        num_layers = getattr(getattr(model, "model", model), "num_hidden_layers", None)
        if num_layers is None:
            num_layers = len(getattr(getattr(model, "model", model), "layers"))
        return [RotatingKVCache(max_size=self.max_kv_size, keep=self.keep) for _ in range(num_layers)]


@dataclass
class Projection:
    mean: Any
    basis: Any
    rank: int


class PCAProjectionCache:
    """KV cache that stores projected vectors and reconstructs full vectors on read."""

    def __init__(self, key_projection: Projection, value_projection: Projection) -> None:
        self.key_projection = key_projection
        self.value_projection = value_projection
        self.compressed_keys = None
        self.compressed_values = None
        self.offset = 0

    def _compress(self, values: Any, projection: Projection) -> Any:
        import mlx.core as mx

        centered = values.astype(mx.float32) - projection.mean
        compressed = mx.matmul(centered, projection.basis)
        return compressed.astype(values.dtype)

    def _reconstruct(self, values: Any, projection: Projection) -> Any:
        import mlx.core as mx

        reconstructed = mx.matmul(values.astype(mx.float32), mx.transpose(projection.basis)) + projection.mean
        return reconstructed.astype(mx.float16)

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        compressed_keys = self._compress(keys, self.key_projection)
        compressed_values = self._compress(values, self.value_projection)
        if self.compressed_keys is None:
            self.compressed_keys = compressed_keys
            self.compressed_values = compressed_values
        else:
            self.compressed_keys = mx.concatenate([self.compressed_keys, compressed_keys], axis=2)
            self.compressed_values = mx.concatenate([self.compressed_values, compressed_values], axis=2)
        self.offset = self.compressed_keys.shape[2]
        return self.keys, self.values

    @property
    def keys(self) -> Any:
        return self._reconstruct(self.compressed_keys, self.key_projection)

    @property
    def values(self) -> Any:
        return self._reconstruct(self.compressed_values, self.value_projection)

    @property
    def state(self) -> tuple[Any, Any]:
        return self.keys, self.values

    @state.setter
    def state(self, value: tuple[Any, Any]) -> None:
        keys, values = value
        self.compressed_keys = self._compress(keys, self.key_projection)
        self.compressed_values = self._compress(values, self.value_projection)
        self.offset = self.compressed_keys.shape[2]

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.compressed_keys is None

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        if self.compressed_keys is not None:
            self.compressed_keys = self.compressed_keys[..., : self.offset, :]
            self.compressed_values = self.compressed_values[..., : self.offset, :]
        return n

    def make_mask(self, n_tokens: int, window_size: int | None = None, return_array: bool = False) -> Any:
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(n_tokens, offset=self.offset, return_array=return_array, window_size=window_size)

    @property
    def nbytes(self) -> int:
        if self.compressed_keys is None:
            cache_bytes = 0
        else:
            cache_bytes = self.compressed_keys.nbytes + self.compressed_values.nbytes
        return (
            cache_bytes
            + self.key_projection.mean.nbytes
            + self.key_projection.basis.nbytes
            + self.value_projection.mean.nbytes
            + self.value_projection.basis.nbytes
        )


@dataclass
class PCACachePolicy(CachePolicy):
    samples_dir: Path
    rank: int = 35
    max_fit_samples: int = 20000
    name: str = "pca_35"

    @property
    def description(self) -> str:
        return f"PCA projection cache with rank {self.rank}; reconstructs full KV on read."

    def make_cache(self, model: Any) -> list[Any]:
        if not hasattr(self, "_projectors"):
            self._projectors = fit_pca_projectors(self.samples_dir, self.rank, self.max_fit_samples)
        projectors = self._projectors
        num_layers = getattr(getattr(model, "model", model), "num_hidden_layers", None)
        if num_layers is None:
            num_layers = len(getattr(getattr(model, "model", model), "layers"))
        caches: list[Any] = []
        for layer_index in range(num_layers):
            key_projection = projectors.get((layer_index, "key"))
            value_projection = projectors.get((layer_index, "value"))
            if key_projection is None or value_projection is None:
                raise ValueError(f"missing PCA projector for layer {layer_index}")
            caches.append(PCAProjectionCache(key_projection, value_projection))
        return caches


def _matrix_from_cache(array: np.ndarray) -> np.ndarray:
    if array.ndim != 4:
        raise ValueError(f"expected 4D KV tensor, got {array.shape}")
    return array.astype(np.float32, copy=False).transpose(0, 2, 1, 3).reshape(-1, array.shape[-1])


def _sample_rows(matrix: np.ndarray, max_rows: int) -> np.ndarray:
    if len(matrix) <= max_rows:
        return matrix
    indices = np.linspace(0, len(matrix) - 1, max_rows).astype(np.int64)
    return matrix[indices]


def fit_pca_projectors(samples_dir: Path, rank: int, max_fit_samples: int) -> dict[tuple[int, str], Projection]:
    import mlx.core as mx

    groups: dict[tuple[int, str], list[Path]] = {}
    for path in sorted(samples_dir.glob("*/layer_*_*.npy")):
        parts = path.stem.split("_")
        if len(parts) != 3:
            continue
        layer = int(parts[1])
        kind = parts[2]
        groups.setdefault((layer, kind), []).append(path)

    projectors: dict[tuple[int, str], Projection] = {}
    for key, paths in groups.items():
        per_file = max(max_fit_samples // max(len(paths), 1), 512)
        matrices = [_sample_rows(_matrix_from_cache(np.load(path, mmap_mode="r")), per_file) for path in paths]
        matrix = np.concatenate(matrices, axis=0)
        matrix = _sample_rows(matrix, max_fit_samples)
        mean = matrix.mean(axis=0, keepdims=False)
        centered = matrix - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        effective_rank = min(rank, vt.shape[0])
        basis = vt[:effective_rank].T
        projectors[key] = Projection(
            mean=mx.array(mean.astype(np.float32)),
            basis=mx.array(basis.astype(np.float32)),
            rank=effective_rank,
        )
    return projectors


def policy_from_name(name: str, *, samples_dir: Path, pca_rank: int, eviction_size: int) -> CachePolicy:
    if name == "full":
        return FullCachePolicy()
    if name in {"quantized", "quantized_8bit", "q8"}:
        return QuantizedCachePolicy(bits=8)
    if name in {"quantized_4bit", "q4"}:
        return QuantizedCachePolicy(bits=4)
    if name == "eviction":
        return EvictionCachePolicy(max_kv_size=eviction_size)
    if name in {"pca", "pca_35"}:
        return PCACachePolicy(samples_dir=samples_dir, rank=pca_rank, name=f"pca_{pca_rank}")
    raise ValueError(f"unknown cache policy: {name}")
