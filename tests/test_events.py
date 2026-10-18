# SPDX-License-Identifier: Apache-2.0
"""The shared event schema: timestamps, anchors and serialisation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from vision_sentinel.events import (
    SCHEMA_VERSION,
    Detection,
    DetectionEvent,
    Incident,
    ZoneEvent,
    ZoneEventType,
    detections_to_events,
    make_incident_id,
    to_utc_iso,
    utc_now,
    utc_now_iso,
)


def test_utc_timestamps_are_timezone_aware_and_suffixed() -> None:
    stamp = utc_now_iso()
    assert stamp.endswith("Z")

    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


def test_naive_datetimes_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        to_utc_iso(datetime(2026, 10, 17, 10, 30, 0))


def test_local_times_are_converted_to_utc() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    moment = datetime(2026, 10, 17, 15, 0, 0, tzinfo=ist)
    assert to_utc_iso(moment) == "2026-10-17T09:30:00.000Z"


def test_foot_point_is_the_bottom_centre_of_the_box() -> None:
    detection = Detection(0, "person", 0.9, (100, 40, 60, 200))
    assert detection.center == (130.0, 140.0)
    assert detection.foot_point == (130.0, 240.0)


def test_foot_point_is_clamped_into_the_frame() -> None:
    detection = Detection(0, "person", 0.9, (300, 200, 100, 100))
    assert detection.foot_point == (350.0, 300.0)
    assert detection.clamped_foot_point(320, 240) == (319.0, 239.0)


def test_incident_ids_are_readable_and_unique() -> None:
    moment = datetime(2026, 10, 17, 14, 30, 5, 207000, tzinfo=timezone.utc)
    identifier = make_incident_id(moment)

    assert identifier.startswith("INC-20261017T143005207-")
    assert len({make_incident_id(moment) for _ in range(500)}) == 500


def test_detection_events_carry_the_shared_schema() -> None:
    detections = [Detection(0, "person", 0.91234, (10, 20, 30, 40))]
    events = detections_to_events(
        detections,
        camera_id="cam-1",
        frame_id=7,
        source="intel_openvino",
        timestamp_utc="2026-10-17T09:30:00.000Z",
        track_ids={0: "3"},
    )

    assert len(events) == 1
    event = events[0]
    assert event.camera_id == "cam-1"
    assert event.frame_id == 7
    assert event.track_id == "3"
    assert event.source == "intel_openvino"
    assert event.timestamp_utc == "2026-10-17T09:30:00.000Z"

    payload = event.to_dict()
    assert payload["confidence"] == 0.9123
    assert payload["bbox_xywh"] == [10, 20, 30, 40]


def test_detection_event_defaults_to_no_track() -> None:
    event = DetectionEvent(
        camera_id="c", frame_id=1, timestamp_utc=utc_now_iso(), label="person",
        confidence=0.5, bbox_xywh=(0, 0, 10, 10),
    )
    assert event.track_id is None
    assert event.source == "unknown"


def make_zone_event(event_type: ZoneEventType = ZoneEventType.ENTERED) -> ZoneEvent:
    return ZoneEvent(
        event_type=event_type,
        timestamp_utc="2026-10-17T09:30:00.000Z",
        camera_id="cam-1",
        zone_id="zone-1",
        zone_name="Zone A",
        track_id="3",
        label="person",
        confidence=0.87654,
        bbox_xywh=(100, 40, 60, 200),
        foot_point=(130.0, 240.0),
        people_count=1,
        source="intel_openvino",
        frame_id=7,
    )


def test_zone_event_serialisation() -> None:
    payload = make_zone_event().to_dict()

    assert payload["event_type"] == "person_entered_zone"
    assert payload["zone_name"] == "Zone A"
    assert payload["confidence"] == 0.8765
    assert payload["foot_point"] == [130.0, 240.0]
    assert payload["suppressed"] is False


def test_incident_is_built_from_a_zone_event() -> None:
    event = make_zone_event()
    incident = Incident.from_zone_event(event, id_factory=lambda: "INC-TEST-1")

    assert incident.incident_id == "INC-TEST-1"
    assert incident.created_utc == event.timestamp_utc
    assert incident.event_type == "person_entered_zone"
    assert incident.schema_version == SCHEMA_VERSION
    assert incident.to_dict()["track_id"] == "3"


def test_incident_round_trips_through_a_database_row() -> None:
    incident = Incident.from_zone_event(make_zone_event(), id_factory=lambda: "INC-TEST-2")
    row = {
        "incident_id": incident.incident_id,
        "created_utc": incident.created_utc,
        "event_type": incident.event_type,
        "camera_id": incident.camera_id,
        "zone_id": incident.zone_id,
        "zone_name": incident.zone_name,
        "track_id": incident.track_id,
        "class_name": incident.label,
        "confidence": incident.confidence,
        "bbox_json": "[100, 40, 60, 200]",
        "foot_point_json": "[130.0, 240.0]",
        "source": incident.source,
        "frame_id": incident.frame_id,
        "snapshot_path": None,
        "schema_version": incident.schema_version,
    }
    restored = Incident.from_row(row)
    assert restored == incident


def test_utc_now_is_aware() -> None:
    assert utc_now().tzinfo is not None


def test_zone_event_types_are_the_documented_values() -> None:
    assert ZoneEventType.ENTERED.value == "person_entered_zone"
    assert ZoneEventType.EXITED.value == "person_exited_zone"
    # A redundant zone-level "cleared" event is deliberately not part of the
    # vocabulary; per-track exits already record the end of an occupancy period.
    assert not hasattr(ZoneEventType, "CLEARED")
