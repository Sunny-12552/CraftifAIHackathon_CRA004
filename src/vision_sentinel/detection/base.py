# SPDX-License-Identifier: Apache-2.0
"""Common detector contract shared by every inference backend.

A backend may run on an Intel GPU, a CPU, an NVIDIA GPU through ONNX Runtime,
or on the real CraftifAI PipeGen / DeepStream pipeline. All of them return the
same :class:`~vision_sentinel.events.Detection` records, so nothing downstream
needs to know which engine produced the numbers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..events import Detection


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    """Detections for one frame, plus any track identifiers.

    Upstream track ids only exist when the inference engine also tracks. The
    CraftifAI PipeGen / DeepStream pipeline supplies NvDCF ids, whereas the
    Intel and CPU backends leave ``track_ids`` empty and let the local IoU
    tracker assign stable ids instead.
    """

    detections: tuple[Detection, ...] = ()
    track_ids: tuple[str | None, ...] = ()
    frame_timestamp_utc: str | None = None
    sequence: int = 0

    def __post_init__(self) -> None:
        if self.track_ids and len(self.track_ids) != len(self.detections):
            raise ValueError(
                "track_ids must be the same length as detections "
                f"({len(self.track_ids)} != {len(self.detections)})"
            )

    def __len__(self) -> int:
        return len(self.detections)

    def ids(self) -> tuple[str | None, ...]:
        """Track ids padded with ``None`` when the backend does not track."""
        if self.track_ids:
            return self.track_ids
        return tuple(None for _ in self.detections)


class DetectorError(RuntimeError):
    """Raised when a detector cannot be initialised or used.

    Messages from this exception are surfaced directly in the dashboard and on
    the console, so they must be actionable (FR-10).
    """


class Detector(ABC):
    """Abstract inference backend."""

    #: Stable identifier recorded in every event for later evidence auditing.
    backend_id: str = "unknown"

    #: Human-readable description of the compute actually in use.
    device_name: str = "uninitialised"

    #: False when the backend's own pipeline owns the camera, meaning any local
    #: capture is for display only. The PipeGen backend reports False.
    owns_video_source: bool = True

    #: True when the backend can be trusted for a hackathon/PipeGen compliance
    #: claim. Only the genuine PipeGen pipeline sets this.
    compliance_eligible: bool = False

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for one BGR frame."""

    def detect_tracked(self, frame: np.ndarray) -> DetectionBatch:
        """Return detections plus upstream track ids when the engine provides them.

        Backends that do not track inherit this default, which simply wraps
        :meth:`detect` and reports no ids.
        """
        detections = self.detect(frame)
        return DetectionBatch(detections=tuple(detections))

    def warmup(self) -> float:
        """Run one inference pass on a blank frame and return its latency in ms.

        Backends override this to pay first-call compilation cost before the
        live loop starts, so the first measured FPS is not misleading.
        """
        import time

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        started = time.perf_counter()
        self.detect(frame)
        return (time.perf_counter() - started) * 1000.0

    def close(self) -> None:
        """Release any backend resources. Safe to call more than once."""

    def describe(self) -> dict[str, str]:
        return {"backend": self.backend_id, "device": self.device_name}


def filter_by_confidence(
    detections: Sequence[Detection], minimum: float
) -> list[Detection]:
    """Drop detections below ``minimum`` confidence."""
    return [item for item in detections if item.confidence >= minimum]


def filter_by_label(
    detections: Sequence[Detection], labels: Sequence[str] | None
) -> list[Detection]:
    """Keep only detections whose label is in ``labels`` (``None`` keeps all)."""
    if labels is None:
        return list(detections)
    wanted = {label.lower() for label in labels}
    return [item for item in detections if item.label.lower() in wanted]
