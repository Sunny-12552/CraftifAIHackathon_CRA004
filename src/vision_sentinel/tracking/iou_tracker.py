# SPDX-License-Identifier: Apache-2.0
"""A small, dependency-free IoU tracker.

Why not ByteTrack/boxmot?
-------------------------
ByteTrack would add a heavyweight dependency to the Windows install and, on the
NVIDIA path, would duplicate tracking the generated pipeline already performs.
The requirement here is narrower: keep one identity per person so that a single
person standing in a restricted zone cannot generate a stream of duplicate
incidents. A greedy IoU association with a maximum age does that, is
deterministic, and is fully unit-testable without a camera.

Association details
-------------------
* Detections are only associated with tracks of the same label, so a person is
  never matched onto a vehicle.
* Candidate pairs are considered in descending IoU with the track id and frame
  index as tie-breakers, making the result independent of dictionary order.
* A track survives ``max_age`` frames without a match; an id is never reused
  after its track expires.
* A detection is only reported with an id once its track has ``min_hits``
  observations, which suppresses one-frame false positives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..events import Detection
from ..geometry import iou_xywh
from .base import TrackInfo, Tracker


@dataclass(slots=True)
class _Track:
    track_id: int
    label: str
    bbox_xywh: tuple[int, int, int, int]
    confidence: float
    hits: int
    age: int
    created_frame: int
    last_frame: int

    @property
    def confirmed(self) -> bool:
        return self.hits >= self.required_hits

    #: Injected by the tracker so tracks know their own confirmation rule.
    required_hits: int = 1


class IouTracker(Tracker):
    """Greedy IoU association with age-based expiry."""

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_age: int = 15,
        min_hits: int = 1,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        if max_age < 0:
            raise ValueError("max_age must be zero or greater")
        if min_hits < 1:
            raise ValueError("min_hits must be one or greater")

        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._expired: list[str] = []
        self._frame_id = 0

    # -- public API ------------------------------------------------------
    def update(
        self, detections: Sequence[Detection], frame_id: int = 0
    ) -> list[str | None]:
        self._frame_id = frame_id
        self._expired = []

        matches = self._associate(detections)
        assigned: list[str | None] = [None] * len(detections)

        matched_tracks: set[int] = set()
        for detection_index, track in matches.items():
            track.bbox_xywh = detections[detection_index].bbox_xywh
            track.confidence = detections[detection_index].confidence
            track.hits += 1
            track.age = 0
            track.last_frame = frame_id
            matched_tracks.add(track.track_id)
            if track.confirmed:
                assigned[detection_index] = str(track.track_id)

        # Age unmatched tracks and retire the ones that have been missing too long.
        for track_id in list(self._tracks):
            if track_id in matched_tracks:
                continue
            track = self._tracks[track_id]
            track.age += 1
            if track.age > self.max_age:
                del self._tracks[track_id]
                self._expired.append(str(track_id))

        # New identities for unmatched detections.
        for detection_index, detection in enumerate(detections):
            if detection_index in matches:
                continue
            track = _Track(
                track_id=self._next_id,
                label=detection.label,
                bbox_xywh=detection.bbox_xywh,
                confidence=detection.confidence,
                hits=1,
                age=0,
                created_frame=frame_id,
                last_frame=frame_id,
                required_hits=self.min_hits,
            )
            self._next_id += 1
            self._tracks[track.track_id] = track
            if track.confirmed:
                assigned[detection_index] = str(track.track_id)

        return assigned

    def _associate(self, detections: Sequence[Detection]) -> dict[int, _Track]:
        """Return ``{detection_index: track}`` using greedy IoU matching."""
        candidates: list[tuple[float, int, int]] = []
        track_ids = sorted(self._tracks)
        for track_id in track_ids:
            track = self._tracks[track_id]
            for detection_index, detection in enumerate(detections):
                if detection.label != track.label:
                    continue
                overlap = iou_xywh(track.bbox_xywh, detection.bbox_xywh)
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, track_id, detection_index))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        matches: dict[int, _Track] = {}
        for _, track_id, detection_index in candidates:
            if track_id in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(detection_index)
            matches[detection_index] = self._tracks[track_id]
        return matches

    # -- inspection ------------------------------------------------------
    def forget(self, track_ids: Sequence[str]) -> None:
        for track_id in track_ids:
            try:
                numeric = int(track_id)
            except (TypeError, ValueError):
                continue
            self._tracks.pop(numeric, None)

    def tracks(self) -> tuple[TrackInfo, ...]:
        return tuple(
            TrackInfo(
                track_id=str(track.track_id),
                label=track.label,
                bbox_xywh=track.bbox_xywh,
                hits=track.hits,
                age=track.age,
                confirmed=track.confirmed,
            )
            for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
        )

    def expired_last_update(self) -> tuple[str, ...]:
        """Ids retired during the most recent :meth:`update`."""
        return tuple(self._expired)

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._expired.clear()
        self._next_id = 1
