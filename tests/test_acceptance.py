# SPDX-License-Identifier: Apache-2.0
"""End-to-end acceptance tests for the complete pipeline (AT-01 - AT-12).

    video source -> detection -> tracking -> zones -> alerts -> SQLite -> report

Nothing here needs a camera, a GPU, the network or a real model: the detector and
the video source are injected, and the clock is exercised through the ordinary
code path. What is being asserted is the behaviour the acceptance criteria name:
one incident per genuine entry, no incident for a flicker, no incident for a
boundary-crossing torso, no fabricated incident when detection is unavailable,
a valid export, and a runtime-measured performance report.

The PipeGen/NVIDIA criteria (AT-13 - AT-17) are deliberately absent. They cannot
be satisfied on this machine; see README.md and evidence/pipegen/README.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import (
    INSIDE_BOX,
    OUTSIDE_BOX,
    FakeDetector,
    ScriptedSource,
    blank_frame,
    frames,
    person,
)

from vision_sentinel.engine import EngineStatus, SentinelEngine
from vision_sentinel.events import ZoneEventType
from vision_sentinel.storage import IncidentStore, export_incidents, load_export

#: A person whose box centre is inside the zone but whose foot point is not.
TORSO_LEANING_IN = (140, 100, 40, 120)
#: A second identity that is genuinely inside the zone.
SECOND_INSIDE_BOX = (170, 60, 40, 100)


@pytest.fixture
def store(tmp_path: Path):
    incident_store = IncidentStore(tmp_path / "incidents.db")
    yield incident_store
    incident_store.close()


def build_engine(settings, store, source, detector, **kwargs) -> SentinelEngine:
    """Assemble an engine with injected parts and no rendering by default."""
    kwargs.setdefault("render", False)
    return SentinelEngine(
        settings, store=store, source=source, detector=detector, **kwargs
    )


def run(engine: SentinelEngine) -> list:
    """Run the engine to completion, collecting every frame result."""
    results: list = []
    engine.run_forever(
        on_frame=lambda result: (results.append(result), None)[1]
    )
    return results


# --- one incident per genuine entry -----------------------------------------


def test_a_person_who_enters_a_zone_raises_exactly_one_incident(
    settings_factory, store: IncidentStore
) -> None:
    settings = settings_factory()
    detector = FakeDetector(script=[[person()]], upstream_track_ids=["t1"])
    engine = build_engine(settings, store, ScriptedSource(frames(6)), detector)

    run(engine)

    history = engine.alerts.history()
    assert engine.status is EngineStatus.STOPPED
    assert len(history) == 1

    incident = history[0]
    assert incident.event_type == ZoneEventType.ENTERED.value
    assert incident.zone_id == "zone-1"
    assert incident.track_id == "t1"
    assert incident.label == "person"
    assert incident.bbox_xywh == INSIDE_BOX
    assert incident.created_utc.endswith("Z")

    # One confirmed entry means exactly one persisted row and one occupant.
    assert store.count() == 1
    assert store.query()[0] == incident
    assert engine.monitor.occupancy() == {"zone-1": 1}
    assert len(engine.alerts.active_alerts()) == 1


def test_two_identities_in_the_same_zone_raise_two_independent_incidents(
    settings_factory, store: IncidentStore
) -> None:
    settings = settings_factory()
    detector = FakeDetector(
        script=[[person(INSIDE_BOX), person(SECOND_INSIDE_BOX)]],
        upstream_track_ids=["a", "b"],
    )
    engine = build_engine(settings, store, ScriptedSource(frames(4)), detector)

    run(engine)

    history = engine.alerts.history()
    assert len(history) == 2
    assert {incident.track_id for incident in history} == {"a", "b"}
    assert engine.monitor.occupancy() == {"zone-1": 2}
    assert store.count() == 2


# --- the anti-churn behaviour the baseline got wrong (FR-05) ----------------


def test_a_one_frame_flicker_at_the_boundary_raises_nothing(
    settings_factory, store: IncidentStore
) -> None:
    settings = settings_factory()  # enter_frames = 3
    script = [
        [person(INSIDE_BOX)],
        [person(OUTSIDE_BOX)],
        [person(OUTSIDE_BOX)],
        [person(OUTSIDE_BOX)],
    ]
    detector = FakeDetector(script=script, upstream_track_ids=["t1"])
    engine = build_engine(settings, store, ScriptedSource(frames(4)), detector)

    results = run(engine)

    assert [result.events for result in results] == [(), (), (), ()]
    assert engine.alerts.history() == ()
    assert store.count() == 0
    assert engine.monitor.occupancy() == {"zone-1": 0}


def test_a_torso_leaning_into_the_zone_does_not_count_as_an_entry(
    settings_factory, store: IncidentStore
) -> None:
    """Occupancy is anchored on the foot point, not the box centre (FR-02)."""
    settings = settings_factory()
    detector = FakeDetector(
        script=[[person(TORSO_LEANING_IN)]], upstream_track_ids=["t1"]
    )
    engine = build_engine(settings, store, ScriptedSource(frames(5)), detector)

    # The box centre (160, 160) is inside the zone; the foot point (160, 220) is
    # not, so the person is standing outside and leaning over the boundary.
    run(engine)

    assert engine.alerts.history() == ()
    assert store.count() == 0
    assert engine.monitor.occupancy() == {"zone-1": 0}


def test_a_brief_exit_then_re_entry_is_suppressed_by_the_cooldown(
    settings_factory, store: IncidentStore
) -> None:
    settings = settings_factory()  # enter_frames 3, exit_frames 3, cooldown 10s
    script = (
        [[person(INSIDE_BOX)]] * 3   # entry confirmed on frame 3
        + [[person(OUTSIDE_BOX)]] * 3  # exit recorded on frame 6
        + [[person(INSIDE_BOX)]] * 3   # re-entry on frame 9, still inside cooldown
    )
    detector = FakeDetector(script=script, upstream_track_ids=["t1"])
    engine = build_engine(settings, store, ScriptedSource(frames(9)), detector)

    run(engine)

    history = engine.alerts.history()
    assert [incident.event_type for incident in history] == [
        ZoneEventType.ENTERED.value,
        ZoneEventType.EXITED.value,
    ]
    # The third transition is the same event in progress, so it is suppressed.
    assert engine.alerts.suppressed_total == 1
    assert store.count() == 2


def test_an_abrupt_disappearance_does_not_fabricate_an_exit(
    settings_factory, store: IncidentStore
) -> None:
    """A track that vanishes inside a zone is dropped, never reported as leaving."""
    settings = settings_factory()  # tracking.max_age = 5
    # The entry needs three consecutive inside frames to confirm; only then does
    # the person vanish, mid-zone, before the track's max age elapses.
    script = [[person()]] * 3 + [[]] * 22
    detector = FakeDetector(script=script)  # local IoU tracker assigns the id
    engine = build_engine(settings, store, ScriptedSource(frames(25)), detector)

    run(engine)

    history = engine.alerts.history()
    assert len(history) == 1
    assert history[0].event_type == ZoneEventType.ENTERED.value
    assert engine.monitor.dropped_inside_count == 1
    assert engine.monitor.occupancy() == {"zone-1": 0}


# --- failure handling -------------------------------------------------------


def test_an_unavailable_detector_never_fabricates_zone_activity(
    settings_factory, store: IncidentStore
) -> None:
    settings = settings_factory()
    detector = FakeDetector(
        error="PipeGen endpoint 127.0.0.1:8554 is unreachable", backend_id="pipegen"
    )
    engine = build_engine(settings, store, ScriptedSource(frames(4)), detector)

    results = run(engine)

    assert len(results) == 4
    assert all(result.error for result in results)
    assert all(result.detections == () for result in results)
    assert all(result.events == () for result in results)
    assert engine.alerts.history() == ()
    assert engine.monitor.occupancy() == {"zone-1": 0}
    assert store.count() == 0

    snapshot = engine.snapshot()
    assert "unreachable" in (snapshot.error or "")


# --- evidence: export and performance report --------------------------------


def test_the_incident_history_exports_as_a_valid_document(
    settings_factory, store: IncidentStore, tmp_path: Path
) -> None:
    settings = settings_factory()
    detector = FakeDetector(script=[[person()]], upstream_track_ids=["t1"])
    engine = build_engine(settings, store, ScriptedSource(frames(5)), detector)
    run(engine)

    result = export_incidents(store, tmp_path / "exports" / "incidents.json")
    payload = load_export(result.path)  # validates while reading

    assert result.count == 1
    assert payload["application"] == "VisionSentinel AI"
    assert payload["count"] == 1
    incident = payload["incidents"][0]
    assert incident["zone_id"] == "zone-1"
    assert incident["track_id"] == "t1"
    assert isinstance(incident["bbox_xywh"], list)

    # The dashboard offers the in-process history as JSON too; it must be
    # serialisable with no Python-specific types.
    bridge = json.loads(engine.incidents_json())
    assert bridge[0]["incident_id"] == incident["incident_id"]


def test_the_performance_report_is_runtime_measured_and_names_the_backend(
    settings_factory, store: IncidentStore
) -> None:
    settings = settings_factory()
    detector = FakeDetector(script=[[person()]], upstream_track_ids=["t1"])
    engine = build_engine(settings, store, ScriptedSource(frames(5)), detector)
    run(engine)

    path = engine.write_performance_report()
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["measurement"] == "runtime-measured"
    assert "measured at runtime" in document["note"]
    assert document["totals"]["frames_processed"] == 5
    assert document["stages"]["inference_ms"]["count"] == 5
    assert document["stages"]["total_ms"]["count"] == 5

    context = document["context"]
    assert context["backend"] == "fake"
    assert context["compliance_eligible"] is False
    assert context["camera_id"] == "cam-test"
    assert context["zones"] == ["zone-1"]
    assert context["incidents"] == 1


# --- the snapshot the UI reads ----------------------------------------------


def test_the_snapshot_is_serialisable_and_free_of_compliance_claims(
    settings_factory, store: IncidentStore
) -> None:
    settings = settings_factory()
    detector = FakeDetector(script=[[person()]], upstream_track_ids=["t1"])
    engine = build_engine(settings, store, ScriptedSource(frames(4)), detector)

    run(engine)

    snapshot = engine.snapshot()
    payload = snapshot.to_dict()

    assert payload["status"] == EngineStatus.STOPPED.value
    assert payload["backend"] == "fake"
    assert payload["frames_processed"] == 4
    assert payload["incident_count"] == 1
    assert payload["occupancy"] == {"zone-1": 1}
    assert payload["compliance_eligible"] is False
    assert snapshot.compliance_eligible is False
    assert engine.compliance_eligible is False
    # A local backend must not raise the "this is not PipeGen" warning.
    assert engine.notes == []

    json.dumps(payload)  # raises if anything Python-specific slipped in


def test_the_engine_hands_the_dashboard_an_encoded_frame(
    settings_factory, store: IncidentStore
) -> None:
    settings = settings_factory()
    detector = FakeDetector(script=[[]])
    engine = build_engine(
        settings, store, ScriptedSource([blank_frame()]), detector, render=True
    )

    run(engine)

    snapshot = engine.snapshot()
    assert snapshot.frame is not None
    encoded = snapshot.jpeg(quality=70)
    assert isinstance(encoded, bytes)
    assert encoded[:2] == b"\xff\xd8"  # JPEG SOI marker
