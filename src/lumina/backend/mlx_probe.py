"""Best-effort MLX-LM cache capability probe."""

from __future__ import annotations

import importlib.metadata as metadata
import inspect

from .feasible_set import BackendCapabilities


def probe_mlx_lm() -> BackendCapabilities:
    """Inspect an installed MLX-LM package and return cache capabilities."""

    try:
        version = metadata.version("mlx-lm")
    except metadata.PackageNotFoundError:
        return BackendCapabilities(name="mlx-lm", version="not-installed")

    try:
        from mlx_lm.models import cache
    except Exception:
        return BackendCapabilities(name="mlx-lm", version=version)

    supports_retention = hasattr(cache, "RotatingKVCache")
    quantized_bits: tuple[int, ...] = (8,) if hasattr(cache, "QuantizedKVCache") else ()
    supports_quantized_retention = False

    rotating = getattr(cache, "RotatingKVCache", None)
    if rotating is not None and hasattr(rotating, "to_quantized"):
        try:
            source = inspect.getsource(rotating.to_quantized)
        except (OSError, TypeError):
            source = ""
        supports_quantized_retention = "NotImplementedError" not in source

    return BackendCapabilities(
        name="mlx-lm",
        version=version,
        quantized_kv_bits=quantized_bits,
        supports_retention=supports_retention,
        supports_quantized_retention=supports_quantized_retention,
    )


def main() -> int:
    import json
    from dataclasses import asdict

    print(json.dumps(asdict(probe_mlx_lm()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
