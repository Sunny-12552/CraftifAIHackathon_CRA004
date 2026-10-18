# SPDX-License-Identifier: Apache-2.0
"""The frozen detection/zone/incident schema shared by every backend.

Every inference backend -- Intel OpenVINO GPU, OpenVINO CPU, optional ONNX
Runtime CUDA, and the CraftifAI PipeGen / DeepStream pipeline -- normalises its
output into :class:`Detection`. The ``DetectionEvent`` / ``ZoneEvent`` /
``Incident`` records below are the only vocabulary the tracking, zone, alert,
storage and metrics layers understand.

Because these records are frozen and backend-agnostic, the same restricted-zone
logic provably runs against a webcam on an Intel laptop and against a
TensorRT-accelerated pipeline on NVIDIA hardware.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

#: Bumped whenever the persisted/serialised shape changes.
SCHEMA_VERSION = 1


def utc_now() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


def to_utc_iso(moment: datetime) -> str:
    """Render ``moment`` as ISO-8601 UTC with a trailing ``Z``.

    All persisted timestamps are UTC. Local time is a presentation concern and
    is applied in the dashboard, never in storage.
    """
    if moment.tzinfo is None:
        raise ValueError("to_utc_iso requires a timezone-aware datetime")
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return to_utc_iso(utc_now())


def make_incident_id(moment: datetime | None = None) -> str:
    """Build a readable, collision-resistant incident identifier.

    Format: ``INC-<UTC compact>-<6 hex>`` e.g. ``INC-20261017T143005207-a1b2c3``.
    """
    moment = moment or utc_now()
    stamp = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
    return f"INC-{stamp}-{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object in one frame, in pixel coordinates.

    ``bbox_xywh`` is ``(x, y, width, height)`` with the origin at the top-left
    of the frame, matching the convention already used by the verified Intel
    baseline.
    """

    class_id: int
    label: str
    confidence: float
    bbox_xywh: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[float, float]:
        """Geometric centre of the box."""
        x, y, width, height = self.bbox_xywh
        return x + width / 2.0, y + height / 2.0

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box.

        This is the anchor used for restricted-zone occupancy. A foot point is
        far more stable than a box centre for a standing person: it stays put
        while the arms and head move, and it matches where the person actually
        stands in the scene.
        """
        x, y, width, height = self.bbox_xywh
        return x + width / 2.0, float(y + height)

    def clamped_foot_point(
        self, frame_width: int, frame_height: int
    ) -> tuple[float, float]:
        """Foot point clipped into the frame, for points on the frame edge."""
        x, y = self.foot_point
        max_x = max(0.0, float(frame_width - 1))
        max_y = max(0.0, float(frame_height - 1))
        return min(max(x, 0.0), max_x), min(max(y, 0.0), max_y)


@dataclass(frozen=True, slots=True)
class DetectionEvent:
    """Normalised per-detection record handed to the tracking layer.

    ``source`` identifies the engine that produced the numbers so that evidence
    can be audited afterwards. No downstream logic may branch on it.
    """

    camera_id: str
    frame_id: int
    timestamp_utc: str
    label: str
    confidence: float
    bbox_xywh: tuple[int, int, int, int]
    track_id: str | None = None
    source: str = "unknown"
    class_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "timestamp_utc": self.timestamp_utc,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox_xywh": list(self.bbox_xywh),
            "track_id": self.track_id,
            "source": self.source,
            "class_id": self.class_id,
        }


class ZoneEventType(str, Enum):
    """Zone transitions that produce an incident.

    A zone-level "cleared" rollup is deliberately *not* an event: the per-track
    exit already records the end of an occupancy period, and a second record
    would double every incident in the log. Live occupancy is exposed as state
    by the zone monitor and rendered by the dashboard instead.
    """

    ENTERED = "person_entered_zone"
    EXITED = "person_exited_zone"


@dataclass(frozen=True, slots=True)
class ZoneEvent:
    """A confirmed, de-duplicated zone transition."""

    event_type: ZoneEventType
    timestamp_utc: str
    camera_id: str
    zone_id: str
    zone_name: str
    track_id: str | None
    label: str
    confidence: float
    bbox_xywh: tuple[int, int, int, int]
    foot_point: tuple[float, float]
    people_count: int
    source: str
    frame_id: int
    suppressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp_utc": self.timestamp_utc,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "track_id": self.track_id,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox_xywh": list(self.bbox_xywh),
            "foot_point": [round(float(self.foot_point[0]), 2), round(float(self.foot_point[1]), 2)],
            "people_count": int(self.people_count),
            "source": self.source,
            "frame_id": int(self.frame_id),
            "suppressed": bool(self.suppressed),
        }


@dataclass(frozen=True, slots=True)
class Incident:
    """A persisted incident record (one row in SQLite, one JSON export entry)."""

    incident_id: str
    created_utc: str
    event_type: str
    camera_id: str
    zone_id: str
    zone_name: str
    track_id: str | None
    label: str
    confidence: float
    bbox_xywh: tuple[int, int, int, int]
    foot_point: tuple[float, float]
    source: str
    frame_id: int
    snapshot_path: str | None = None
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_zone_event(
        cls,
        event: ZoneEvent,
        id_factory: Callable[[], str] = make_incident_id,
        snapshot_path: str | None = None,
    ) -> "Incident":
        return cls(
            incident_id=id_factory(),
            created_utc=event.timestamp_utc,
            event_type=event.event_type.value,
            camera_id=event.camera_id,
            zone_id=event.zone_id,
            zone_name=event.zone_name,
            track_id=event.track_id,
            label=event.label,
            confidence=float(event.confidence),
            bbox_xywh=event.bbox_xywh,
            foot_point=event.foot_point,
            source=event.source,
            frame_id=event.frame_id,
            snapshot_path=snapshot_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "created_utc": self.created_utc,
            "event_type": self.event_type,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "track_id": self.track_id,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox_xywh": list(self.bbox_xywh),
            "foot_point": [round(float(self.foot_point[0]), 2), round(float(self.foot_point[1]), 2)],
            "source": self.source,
            "frame_id": int(self.frame_id),
            "snapshot_path": self.snapshot_path,
            "schema_version": int(self.schema_version),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Incident":
        """Rebuild an incident from a SQLite row."""
        import json

        bbox = json.loads(row["bbox_json"])
        foot = json.loads(row["foot_point_json"]) if row["foot_point_json"] else [0.0, 0.0]
        return cls(
            incident_id=row["incident_id"],
            created_utc=row["created_utc"],
            event_type=row["event_type"],
            camera_id=row["camera_id"],
            zone_id=row["zone_id"],
            zone_name=row["zone_name"] or "",
            track_id=row["track_id"],
            label=row["class_name"],
            confidence=float(row["confidence"]),
            bbox_xywh=tuple(int(value) for value in bbox),  # type: ignore[arg-type]
            foot_point=(float(foot[0]), float(foot[1])),
            source=row["source"],
            frame_id=int(row["frame_id"]),
            snapshot_path=row["snapshot_path"],
            schema_version=int(row["schema_version"]),
        )


def detections_to_events(
    detections: Iterable[Detection],
    camera_id: str,
    frame_id: int,
    source: str,
    timestamp_utc: str | None = None,
    track_ids: Mapping[int, str] | None = None,
) -> list[DetectionEvent]:
    """Normalise raw backend detections into the shared event schema."""
    stamp = timestamp_utc or utc_now_iso()
    events: list[DetectionEvent] = []
    for index, detection in enumerate(detections):
        track_id = None
        if track_ids is not None:
            track_id = track_ids.get(index)
        events.append(
            DetectionEvent(
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp_utc=stamp,
                label=detection.label,
                confidence=float(detection.confidence),
                bbox_xywh=detection.bbox_xywh,
                track_id=track_id,
                source=source,
                class_id=int(detection.class_id),
            )
        )
    return events
