"""OS telemetry used by the runtime cost model."""

from .macos import TelemetrySnapshot, snapshot

__all__ = ["TelemetrySnapshot", "snapshot"]
