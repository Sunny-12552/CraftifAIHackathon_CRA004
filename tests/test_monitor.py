# SPDX-License-Identifier: Apache-2.0
"""Multi-zone monitoring: foot-point anchoring, filtering and lifecycle."""

from __future__ import annotations

from vision_sentinel.events import Detection, ZoneEventType
from vision_sentinel.zones.monitor import AGGREGATE_TRACK, ZoneMonitor
from vision_sentinel.zones.polygon import PolygonZone, resolve_cached

from conftest import person

WIDTH, HEIGHT = 320, 240
ZONE_POINTS = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]

# Foot point inside the zone.
INSIDE = (140, 60, 40, 100)
# Foot point below the zone, while the box centre is still inside it.
STRADDLING = (140, 100, 40, 120)
# Entirely outside.
OUTSIDE = (10, 60, 40, 100)


def zone(zone_id: str = "zone-1", name: str = "Zone A", enabled: bool = True) -> PolygonZone:
    return PolygonZone(
        zone_id=zone_id,
        name=name,
        points_normalized=tuple((x, y) for x, y in ZONE_POINTS),
        enabled=enabled,
    )


def make_monitor(*zones: PolygonZone, **kwargs) -> ZoneMonitor:
    return ZoneMonitor(list(zones) or [zone()], **kwargs)


def run(monitor: ZoneMonitor, boxes, frames: int, track_ids=None) -> list:
    events = []
    for frame_id in range(1, frames + 1):
        detections = [person(box) for box in boxes]
        events.extend(
            monitor.update(
                detections,
                track_ids if track_ids is not None else [str(i + 1) for i in range(len(boxes))],
                frame_id=frame_id,
                frame_width=WIDTH,
                frame_height=HEIGHT,
                timestamp_utc="2026-10-17T09:30:00.000Z",
                camera_id="cam-test",
                source="fake",
            )
        )
    return events


def test_the_straddling_case_proves_the_foot_point_is_the_anchor() -> None:
    """A box centre inside the zone with feet outside must not count.

    The replaced baseline used the box centre, so a person standing just below
    a zone in front of the camera would have been reported as inside it.
    """
    pixel_zone = resolve_cached(zone(), WIDTH, HEIGHT)
    centre = person(STRADDLING).center
    foot = person(STRADDLING).foot_point

    assert pixel_zone.contains(centre) is True
    assert pixel_zone.contains(foot) is False

    events = run(make_monitor(enter_frames=3), [STRADDLING], frames=10)
    assert events == []


def test_a_person_inside_produces_exactly_one_entry() -> None:
    events = run(make_monitor(enter_frames=3), [INSIDE], frames=30)

    assert len(events) == 1
    assert events[0].event_type is ZoneEventType.ENTERED
    assert events[0].zone_id == "zone-1"
    assert events[0].zone_name == "Zone A"
    assert events[0].track_id == "1"
    assert events[0].people_count == 1
    assert events[0].source == "fake"
    assert events[0].camera_id == "cam-test"
    assert events[0].foot_point == person(INSIDE).foot_point


def test_a_person_outside_produces_nothing() -> None:
    assert run(make_monitor(enter_frames=3), [OUTSIDE], frames=30) == []


def test_labels_other_than_person_are_ignored() -> None:
    monitor = make_monitor(enter_frames=1)
    bicycle = Detection(1, "bicycle", 0.99, INSIDE)

    for frame_id in range(1, 11):
        assert (
            monitor.update(
                [bicycle], ["1"], frame_id=frame_id, frame_width=WIDTH, frame_height=HEIGHT
            )
            == []
        )


def test_low_confidence_detections_never_inform_occupancy() -> None:
    monitor = make_monitor(enter_frames=1, min_confidence=0.8)

    for frame_id in range(1, 11):
        events = monitor.update(
            [person(INSIDE, confidence=0.4)],
            ["1"],
            frame_id=frame_id,
            frame_width=WIDTH,
            frame_height=HEIGHT,
        )
        assert events == []


def test_each_zone_is_evaluated_independently() -> None:
    # A zone in the lower-left that contains OUTSIDE's foot point (30, 160) but
    # not its box centre (30, 110). Zone A does not contain it, so exactly one
    # entry is reported, and it must belong to zone B.
    second = PolygonZone(
        zone_id="zone-2",
        name="Zone B",
        points_normalized=((0.02, 0.60), (0.20, 0.60), (0.20, 0.90)),
    )
    monitor = make_monitor(zone(), second, enter_frames=2)

    events = run(monitor, [OUTSIDE], frames=5)
    assert len(events) == 1
    assert events[0].zone_id == "zone-2"


def test_occupancy_counts_confirmed_occupants_only() -> None:
    monitor = make_monitor(enter_frames=3)

    monitor.update([person(INSIDE)], ["1"], frame_id=1, frame_width=WIDTH, frame_height=HEIGHT)
    assert monitor.occupancy() == {"zone-1": 0}  # still filling the dwell counter

    for frame_id in (2, 3):
        monitor.update([person(INSIDE)], ["1"], frame_id=frame_id, frame_width=WIDTH, frame_height=HEIGHT)
    assert monitor.occupancy() == {"zone-1": 1}
    assert monitor.is_zone_occupied("zone-1") is True


def test_exit_is_reported_after_the_dwell_period() -> None:
    monitor = make_monitor(enter_frames=1, exit_frames=2)

    monitor.update([person(INSIDE)], ["1"], frame_id=1, frame_width=WIDTH, frame_height=HEIGHT)
    assert monitor.occupancy()["zone-1"] == 1

    monitor.update([person(OUTSIDE)], ["1"], frame_id=2, frame_width=WIDTH, frame_height=HEIGHT)
    events = monitor.update(
        [person(OUTSIDE)], ["1"], frame_id=3, frame_width=WIDTH, frame_height=HEIGHT
    )

    assert [event.event_type for event in events] == [ZoneEventType.EXITED]
    assert monitor.occupancy()["zone-1"] == 0


def test_active_occupants_are_reported_for_the_overlay() -> None:
    monitor = make_monitor(enter_frames=1)
    monitor.update([person(INSIDE)], ["7"], frame_id=1, frame_width=WIDTH, frame_height=HEIGHT)

    occupants = monitor.active_occupants([person(INSIDE)], ["7"])
    assert len(occupants) == 1
    assert occupants[0].track_id == "7"
    assert occupants[0].zone_name == "Zone A"


def test_disabled_zones_are_not_evaluated() -> None:
    monitor = make_monitor(zone(enabled=False), enter_frames=1)
    assert run(monitor, [INSIDE], frames=5) == []
    assert monitor.occupancy() == {"zone-1": 0}


def test_forget_drops_state_and_counts_tracks_lost_inside() -> None:
    monitor = make_monitor(enter_frames=1)
    monitor.update([person(INSIDE)], ["1"], frame_id=1, frame_width=WIDTH, frame_height=HEIGHT)

    assert monitor.forget(["1"]) == ("1",)
    assert monitor.occupancy()["zone-1"] == 0
    assert monitor.dropped_inside_count == 1


def test_without_a_tracker_detections_share_one_aggregate_state() -> None:
    """Aggregate mode measures "is anybody in the zone", not who."""
    monitor = make_monitor(enter_frames=1)

    events = monitor.update(
        [person(INSIDE), person(OUTSIDE)],
        [None, None],
        frame_id=1,
        frame_width=WIDTH,
        frame_height=HEIGHT,
    )
    assert len(events) == 1
    assert events[0].track_id is None
    assert monitor.track_state("zone-1", AGGREGATE_TRACK) == "inside"


def test_a_second_person_entering_adds_no_extra_entry_event() -> None:
    """Only the first occupant of a zone creates the entry incident."""
    monitor = make_monitor(enter_frames=1)

    first = monitor.update([person(INSIDE)], ["1"], frame_id=1, frame_width=WIDTH, frame_height=HEIGHT)
    assert len(first) == 1

    second = monitor.update(
        [person(INSIDE), person((150, 60, 40, 100))],
        ["1", "2"],
        frame_id=2,
        frame_width=WIDTH,
        frame_height=HEIGHT,
    )
    # Track 2 is new to the zone, so it does report its own entry; track 1 does not
    # report again. That is one event, not two.
    assert len(second) == 1
    assert second[0].track_id == "2"
    assert monitor.occupancy()["zone-1"] == 2


def test_set_zones_discards_state_for_removed_zones() -> None:
    monitor = make_monitor(enter_frames=1)
    monitor.update([person(INSIDE)], ["1"], frame_id=1, frame_width=WIDTH, frame_height=HEIGHT)
    assert monitor.occupancy()["zone-1"] == 1

    monitor.set_zones([zone("zone-2", "Zone B")])
    assert monitor.occupancy() == {"zone-2": 0}


def test_invalid_frame_dimensions_are_rejected() -> None:
    import pytest

    monitor = make_monitor(enter_frames=1)
    with pytest.raises(ValueError, match="positive"):
        monitor.update([person(INSIDE)], ["1"], frame_width=0, frame_height=240)


def test_mismatched_track_id_lengths_are_rejected() -> None:
    import pytest

    monitor = make_monitor(enter_frames=1)
    with pytest.raises(ValueError, match="parallel"):
        monitor.update([person(INSIDE)], ["1", "2"], frame_width=WIDTH, frame_height=HEIGHT)
