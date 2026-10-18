# SPDX-License-Identifier: Apache-2.0
"""Tracker contract.

Two modes exist in the architecture:

* The Intel/CPU path uses :class:`~vision_sentinel.tracking.iou_tracker.IouTracker`
  to assign stable ids locally.
* The PipeGen path receives ids assigned upstream by the DeepStream tracker
  (for example NvDCF), and
  :class:`~vision_sentinel.tracking.base.UpstreamTracker` simply passes them
  through.

Both present the same interface, so the zone monitor never needs to know which
mode is active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from ..events import Detection


@dataclass(frozen=True, slots=True)
class TrackInfo:
    """A snapshot of one live track."""

    track_id: str
    label: str
    bbox_xywh: tuple[int, int, int, int]
    hits: int
    age: int
    confirmed: bool


class Tracker(ABC):
    """Assigns a stable identity to each detection in a frame."""

    @abstractmethod
    def update(
        self, detections: Sequence[Detection], frame_id: int = 0
    ) -> list[str | None]:
        """Return one track id per detection, in the same order.

        ``None`` means "this detection has no confirmed identity yet" and the
        zone monitor will not treat it as an occupant.
        """

    def forget(self, track_ids: Sequence[str]) -> None:
        """Drop state for tracks the zone monitor has finished with."""

    def tracks(self) -> tuple[TrackInfo, ...]:
        return ()

    def reset(self) -> None:
        """Clear all tracking state, for example after a stream reconnects."""


class UpstreamTracker(Tracker):
    """Passes through track ids already assigned by an upstream pipeline.

    Used with the PipeGen/DeepStream backend, where the generated pipeline
    performs tracking itself and its ids are authoritative.
    """

    def update(
        self, detections: Sequence[Detection], frame_id: int = 0
    ) -> list[str | None]:
        raise NotImplementedError(
            "UpstreamTracker does not compute ids; the engine passes the ids "
            "published by the pipeline instead."
        )
