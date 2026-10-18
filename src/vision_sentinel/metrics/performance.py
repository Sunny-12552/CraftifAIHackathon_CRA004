# SPDX-License-Identifier: Apache-2.0
"""Runtime performance measurement (FR-11, AT-12).

Every number produced here is measured on the machine that is running the
application. Nothing in this module contains a baked-in figure, and the report
writer refuses to add any. If a claim about Intel-GPU or TensorRT performance is
going to be made, it has to come out of one of these reports.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Mapping

import numpy as np

RUNTIME_MEASURED_NOTE = (
    "All values were measured at runtime on the machine that produced this "
    "report. No figure in this file is a design target, a specification or an "
    "estimate."
)


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Distribution of a latency series, in milliseconds."""

    count: int
    mean_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


class LatencyTracker:
    """Rolling latency window with percentile summaries."""

    def __init__(self, window: int = 120) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = window
        self._samples: Deque[float] = deque(maxlen=window)
        self._total = 0

    def record(self, milliseconds: float) -> None:
        if milliseconds < 0:
            raise ValueError("Latency cannot be negative")
        self._samples.append(float(milliseconds))
        self._total += 1

    @property
    def total_recorded(self) -> int:
        return self._total

    def __len__(self) -> int:
        return len(self._samples)

    def summary(self) -> LatencySummary:
        if not self._samples:
            return LatencySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        values = np.asarray(self._samples, dtype=np.float64)
        return LatencySummary(
            count=int(values.size),
            mean_ms=float(values.mean()),
            min_ms=float(values.min()),
            p50_ms=float(np.percentile(values, 50)),
            p95_ms=float(np.percentile(values, 95)),
            max_ms=float(values.max()),
        )

    def reset(self) -> None:
        self._samples.clear()
        self._total = 0


class FpsMeter:
    """Frame-rate meter with an instantaneous and a smoothed reading."""

    def __init__(
        self,
        window: int = 60,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if window < 2:
            raise ValueError("window must be at least 2 to compute a rate")
        self.window = window
        self._clock = clock
        self._timestamps: Deque[float] = deque(maxlen=window)
        self._last: float | None = None
        self._instantaneous = 0.0

    def tick(self, now: float | None = None) -> float:
        """Record a frame and return the instantaneous FPS."""
        moment = self._clock() if now is None else now
        if self._last is not None:
            delta = moment - self._last
            if delta > 0:
                self._instantaneous = 1.0 / delta
        self._last = moment
        self._timestamps.append(moment)
        return self._instantaneous

    @property
    def instantaneous(self) -> float:
        return self._instantaneous

    @property
    def value(self) -> float:
        """FPS averaged across the retained window."""
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / span

    def reset(self) -> None:
        self._timestamps.clear()
        self._last = None
        self._instantaneous = 0.0


@dataclass
class _StageSeries:
    """Accumulated timings for one pipeline stage."""

    latency: LatencyTracker
    total_ms: float = 0.0

    def record(self, milliseconds: float) -> None:
        self.latency.record(milliseconds)
        self.total_ms += milliseconds


@dataclass
class PerformanceReport:
    """Collects per-frame stage timings and writes an evidence file."""

    window: int = 120
    report_path: Path | None = None
    fps: FpsMeter = field(default_factory=FpsMeter)
    stages: dict[str, _StageSeries] = field(default_factory=dict)
    frames: int = 0
    frames_dropped: int = 0
    started_utc: str = ""
    started_monotonic: float = field(default_factory=time.monotonic)
    _stage_order: list[str] = field(default_factory=list)

    def record_frame(self, **timings_ms: float) -> None:
        """Record one frame's stage timings, for example ``inference_ms=12.4``."""
        self.frames += 1
        for stage, milliseconds in timings_ms.items():
            series = self.stages.get(stage)
            if series is None:
                series = _StageSeries(LatencyTracker(self.window))
                self.stages[stage] = series
                self._stage_order.append(stage)
            series.record(float(milliseconds))

    def record_drop(self) -> None:
        self.frames_dropped += 1

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_monotonic

    def throughput_fps(self) -> float:
        elapsed = self.elapsed_seconds()
        return self.frames / elapsed if elapsed > 0 else 0.0

    def snapshot(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Build the report document."""
        from ..events import utc_now_iso

        elapsed = self.elapsed_seconds()
        document: dict[str, Any] = {
            "application": "VisionSentinel AI",
            "note": RUNTIME_MEASURED_NOTE,
            "generated_utc": utc_now_iso(),
            "measurement": "runtime-measured",
            "window": self.window,
            "totals": {
                "frames_processed": self.frames,
                "frames_dropped": self.frames_dropped,
                "elapsed_s": round(elapsed, 3),
                "throughput_fps": round(self.throughput_fps(), 3),
                "rolling_fps": round(self.fps.value, 3),
            },
            "stages": {
                stage: self.stages[stage].latency.summary().to_dict()
                for stage in self._stage_order
            },
            "stage_share_ms": {
                stage: round(self.stages[stage].total_ms, 3) for stage in self._stage_order
            },
            "environment": environment_info(),
        }
        if extra:
            document["context"] = dict(extra)
        return document

    def write(
        self, path: str | Path | None = None, extra: Mapping[str, Any] | None = None
    ) -> Path:
        """Write the report to disk atomically and return its path."""
        target = Path(path) if path is not None else self.report_path
        if target is None:
            raise ValueError("No report path configured")
        target.parent.mkdir(parents=True, exist_ok=True)
        document = self.snapshot(extra)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=target.parent, suffix=".tmp", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(document, handle, indent=2)
                handle.write("\n")
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return target

    def reset(self) -> None:
        self.fps.reset()
        self.stages.clear()
        self._stage_order.clear()
        self.frames = 0
        self.frames_dropped = 0
        self.started_monotonic = time.monotonic()


def environment_info() -> dict[str, Any]:
    """Describe the machine the measurements were taken on."""
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import openvino as ov

        info["openvino_version"] = ov.__version__
        info["openvino_devices"] = list(ov.Core().available_devices)
    except Exception:
        info["openvino_version"] = None
        info["openvino_devices"] = []
    try:
        import cv2

        info["opencv_version"] = cv2.__version__
    except Exception:
        info["opencv_version"] = None
    return info


def snapshot_to_json(document: Mapping[str, Any], path: str | Path) -> Path:
    """Convenience writer for a report document that already exists."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return target


def dataclass_to_dict(instance: Any) -> dict[str, Any]:
    """Shallow dataclass serialisation helper."""
    return asdict(instance)
