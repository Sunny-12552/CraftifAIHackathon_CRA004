# SPDX-License-Identifier: Apache-2.0
"""Alert policy: cooldown suppression, incident recording and active alerts."""

from __future__ import annotations

import pytest

from vision_sentinel.alerts.cooldown import CooldownGate, cooldown_key
from vision_sentinel.alerts.manager import AlertManager
from vision_sentinel.events import Incident, ZoneEvent, ZoneEventType, utc_now_iso


class FakeClock:
    """Deterministic monotonic clock, so cooldown tests never sleep."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingSink:
    """Stands in for the SQLite store."""

    def __init__(self, fail: bool = False) -> None:
        self.incidents: list[Incident] = []
        self.fail = fail

    def append(self, incident: Incident) -> Incident:
        if self.fail:
            raise RuntimeError("storage unavailable")
        self.incidents.append(incident)
        return incident


def zone_event(
    event_type: ZoneEventType = ZoneEventType.ENTERED,
    zone_id: str = "zone-1",
    track_id: str | None = "1",
) -> ZoneEvent:
    return ZoneEvent(
        event_type=event_type,
        timestamp_utc=utc_now_iso(),
        camera_id="cam-test",
        zone_id=zone_id,
        zone_name="Zone A",
        track_id=track_id,
        label="person",
        confidence=0.91,
        bbox_xywh=(140, 60, 40, 100),
        foot_point=(160.0, 160.0),
        people_count=1,
        source="fake",
        frame_id=1,
    )


def make_manager(clock: FakeClock | None = None, **kwargs):
    clock = clock or FakeClock()
    sink = kwargs.pop("sink", RecordingSink())
    counter = {"n": 0}

    def id_factory() -> str:
        counter["n"] += 1
        return f"INC-TEST-{counter['n']}"

    manager = AlertManager(
        sink=sink, clock=clock, id_factory=id_factory, **kwargs
    )
    return manager, sink, clock


def test_a_normal_entry_raises_and_persists_an_incident() -> None:
    manager, sink, _ = make_manager(cooldown_seconds=20.0)

    outcome = manager.handle([zone_event()])

    assert outcome.raised == 1
    assert outcome.duplicate_count == 0
    assert len(sink.incidents) == 1
    assert sink.incidents[0].incident_id == "INC-TEST-1"
    assert manager.active_count == 1


def test_repeated_events_inside_the_cooldown_are_suppressed() -> None:
    """AT-05 at the policy layer: no second incident inside the window."""
    manager, sink, clock = make_manager(cooldown_seconds=20.0)

    manager.handle([zone_event()])
    clock.advance(5.0)
    outcome = manager.handle([zone_event()])
    clock.advance(5.0)
    outcome_again = manager.handle([zone_event()])

    assert outcome.duplicate_count == 1
    assert outcome_again.duplicate_count == 1
    assert len(sink.incidents) == 1
    assert manager.suppressed_total == 2


def test_reentry_after_the_cooldown_creates_a_new_incident() -> None:
    """AT-06: a genuine re-entry past the cooldown is a new event."""
    manager, sink, clock = make_manager(cooldown_seconds=20.0)

    manager.handle([zone_event()])
    clock.advance(21.0)
    outcome = manager.handle([zone_event()])

    assert outcome.raised == 1
    assert len(sink.incidents) == 2
    assert {incident.incident_id for incident in sink.incidents} == {
        "INC-TEST-1",
        "INC-TEST-2",
    }


def test_the_cooldown_is_not_reset_by_an_exit() -> None:
    """Otherwise the gate would be meaningless: every re-entry would qualify."""
    manager, _, clock = make_manager(cooldown_seconds=20.0)

    manager.handle([zone_event()])
    clock.advance(2.0)
    manager.handle([zone_event(ZoneEventType.EXITED)])
    outcome = manager.handle([zone_event()])

    assert outcome.duplicate_count == 1


def test_cooldown_keys_are_independent_per_zone_and_track() -> None:
    manager, sink, _ = make_manager(cooldown_seconds=60.0)

    manager.handle(
        [
            zone_event(track_id="1"),
            zone_event(track_id="2"),
            zone_event(track_id="1", zone_id="zone-2"),
        ]
    )

    assert len(sink.incidents) == 3
    assert manager.active_count == 3


def test_active_alerts_clear_when_the_person_leaves() -> None:
    manager, _, _ = make_manager()

    manager.handle([zone_event()])
    assert manager.active_count == 1

    manager.handle([zone_event(ZoneEventType.EXITED)])
    assert manager.active_count == 0


def test_exits_are_recorded_by_default() -> None:
    manager, sink, _ = make_manager(record_exits=True)

    manager.handle([zone_event()])
    manager.handle([zone_event(ZoneEventType.EXITED)])

    assert [incident.event_type for incident in sink.incidents] == [
        "person_entered_zone",
        "person_exited_zone",
    ]


def test_exits_can_be_omitted_from_the_record() -> None:
    manager, sink, _ = make_manager(record_exits=False)

    outcome = manager.handle([zone_event()])
    exit_outcome = manager.handle([zone_event(ZoneEventType.EXITED)])

    assert outcome.raised == 1
    assert exit_outcome.raised == 0
    assert exit_outcome.ignored
    assert len(sink.incidents) == 1
    assert manager.exit_count == 1


def test_a_storage_failure_propagates_instead_of_being_swallowed() -> None:
    """A persistence failure must surface rather than silently drop an incident."""
    manager, _, _ = make_manager(sink=RecordingSink(fail=True))

    with pytest.raises(RuntimeError, match="storage unavailable"):
        manager.handle([zone_event()])


def test_stats_report_the_policy_outcome() -> None:
    manager, _, clock = make_manager(cooldown_seconds=10.0)

    manager.handle([zone_event()])
    clock.advance(1.0)
    manager.handle([zone_event()])  # suppressed
    manager.handle([zone_event(ZoneEventType.EXITED)])

    stats = manager.stats()
    assert stats["entries"] == 1
    assert stats["exits"] == 1
    assert stats["suppressed_duplicates"] == 1
    assert stats["active_alerts"] == 0
    assert stats["incidents"] == 2


def test_history_is_ordered_and_resettable() -> None:
    manager, _, clock = make_manager(cooldown_seconds=1.0)

    manager.handle([zone_event()])
    clock.advance(2.0)
    manager.handle([zone_event()])

    history = manager.history()
    assert [incident.incident_id for incident in history] == ["INC-TEST-1", "INC-TEST-2"]

    manager.reset()
    assert manager.history() == ()
    assert manager.stats()["incidents"] == 0


# --- cooldown gate in isolation --------------------------------------------


def test_gate_allows_the_first_call_then_blocks() -> None:
    clock = FakeClock()
    gate = CooldownGate(10.0, clock=clock)

    assert gate.evaluate("k").allowed is True
    blocked = gate.evaluate("k")
    assert blocked.allowed is False
    assert blocked.remaining_s == pytest.approx(10.0)


def test_gate_peek_does_not_open_a_window() -> None:
    clock = FakeClock()
    gate = CooldownGate(10.0, clock=clock)

    assert gate.peek("k").allowed is True
    assert gate.peek("k").allowed is True  # still allowed, nothing recorded
    assert gate.evaluate("k").allowed is True


def test_gate_reopens_after_the_window() -> None:
    clock = FakeClock()
    gate = CooldownGate(10.0, clock=clock)

    gate.evaluate("k")
    clock.advance(10.5)
    assert gate.evaluate("k").allowed is True


def test_a_zero_window_disables_suppression() -> None:
    gate = CooldownGate(0.0)
    for _ in range(5):
        assert gate.evaluate("k").allowed is True


def test_gate_prunes_old_entries() -> None:
    clock = FakeClock()
    gate = CooldownGate(10.0, clock=clock)

    gate.evaluate("old")
    clock.advance(120.0)
    gate.evaluate("trigger-prune")

    assert "old" not in gate._last_accepted


def test_cooldown_key_is_stable() -> None:
    assert cooldown_key("zone-1", "7") == "zone-1::7"
    assert cooldown_key("zone-1", None) == "zone-1::__anonymous__"


def test_negative_cooldown_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero or greater"):
        CooldownGate(-1.0)
