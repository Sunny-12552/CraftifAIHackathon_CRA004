# SPDX-License-Identifier: Apache-2.0
"""Turns confirmed zone transitions into persisted incidents.

Responsibilities
----------------
* Apply the per ``(zone, track)`` cooldown to entry events.
* Build :class:`~vision_sentinel.events.Incident` records with UTC timestamps.
* Optionally snapshot the frame that triggered an incident (off by default:
  incident imagery can contain identifiable people).
* Hand incidents to the storage layer and expose the live alert state for the
  dashboard.

The manager never invents an incident. Every incident it produces traces back to
one confirmed state-machine transition and carries the detection that caused it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

from ..events import Incident, ZoneEvent, ZoneEventType, make_incident_id
from .cooldown import CooldownGate, cooldown_key


class IncidentSink(Protocol):
    """Anything that can persist an incident (normally the SQLite store)."""

    def append(self, incident: Incident) -> Incident:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True, slots=True)
class AlertOutcome:
    """What the manager did with one frame's worth of zone events."""

    incidents: tuple[Incident, ...] = ()
    suppressed: tuple[ZoneEvent, ...] = ()
    ignored: tuple[ZoneEvent, ...] = ()

    @property
    def raised(self) -> int:
        return len(self.incidents)

    @property
    def duplicate_count(self) -> int:
        return len(self.suppressed)


class AlertManager:
    """Applies alert policy and records incidents."""

    def __init__(
        self,
        cooldown_seconds: float = 20.0,
        record_exits: bool = True,
        sink: IncidentSink | None = None,
        id_factory: Callable[[], str] = make_incident_id,
        clock: Callable[[], float] | None = None,
        snapshot_writer: Callable[[ZoneEvent], str | None] | None = None,
    ) -> None:
        self.record_exits = record_exits
        self._sink = sink
        self._id_factory = id_factory
        self._snapshot_writer = snapshot_writer

        kwargs = {} if clock is None else {"clock": clock}
        self.gate = CooldownGate(cooldown_seconds, **kwargs)

        self._active: dict[str, Incident] = {}
        self._history: list[Incident] = []
        self._suppressed_total = 0
        self._entries = 0
        self._exits = 0

    # -- policy ----------------------------------------------------------
    def handle(self, events: Sequence[ZoneEvent]) -> AlertOutcome:
        """Apply alert policy to the zone events produced for one frame."""
        incidents: list[Incident] = []
        suppressed: list[ZoneEvent] = []
        ignored: list[ZoneEvent] = []

        for event in events:
            if event.event_type is ZoneEventType.ENTERED:
                decision = self.gate.evaluate(cooldown_key(event.zone_id, event.track_id))
                if not decision.allowed:
                    suppressed.append(event)
                    self._suppressed_total += 1
                    continue
                self._entries += 1
                incident = self._build(event)
                self._record(incident)
                key = cooldown_key(event.zone_id, event.track_id)
                self._active[key] = incident
                incidents.append(incident)
                continue

            if event.event_type is ZoneEventType.EXITED:
                self._exits += 1
                self._active.pop(cooldown_key(event.zone_id, event.track_id), None)
                if not self.record_exits:
                    ignored.append(event)
                    continue
                incident = self._build(event)
                self._record(incident)
                incidents.append(incident)
                continue

            ignored.append(event)

        return AlertOutcome(
            incidents=tuple(incidents),
            suppressed=tuple(suppressed),
            ignored=tuple(ignored),
        )

    # -- internals -------------------------------------------------------
    def _build(self, event: ZoneEvent) -> Incident:
        snapshot_path: str | None = None
        if self._snapshot_writer is not None and event.event_type is ZoneEventType.ENTERED:
            try:
                snapshot_path = self._snapshot_writer(event)
            except Exception:
                # A failed snapshot must never lose the incident itself.
                snapshot_path = None
        return Incident.from_zone_event(
            event, id_factory=self._id_factory, snapshot_path=snapshot_path
        )

    def _record(self, incident: Incident) -> None:
        self._history.append(incident)
        if self._sink is not None:
            self._sink.append(incident)

    # -- inspection ------------------------------------------------------
    def active_alerts(self) -> tuple[Incident, ...]:
        """Entries whose subject has not yet been recorded as having left."""
        return tuple(self._active.values())

    @property
    def active_count(self) -> int:
        return len(self._active)

    def history(self) -> tuple[Incident, ...]:
        """Incidents raised during this process, newest last."""
        return tuple(self._history)

    @property
    def suppressed_total(self) -> int:
        return self._suppressed_total

    @property
    def entry_count(self) -> int:
        return self._entries

    @property
    def exit_count(self) -> int:
        return self._exits

    def stats(self) -> dict[str, int]:
        return {
            "incidents": len(self._history),
            "entries": self._entries,
            "exits": self._exits,
            "suppressed_duplicates": self._suppressed_total,
            "active_alerts": len(self._active),
        }

    def reset(self) -> None:
        self.gate.reset()
        self._active.clear()
        self._history.clear()
        self._suppressed_total = 0
        self._entries = 0
        self._exits = 0


def merge_outcomes(outcomes: Iterable[AlertOutcome]) -> AlertOutcome:
    """Combine several outcomes, mainly for tests and batch reporting."""
    incidents: list[Incident] = []
    suppressed: list[ZoneEvent] = []
    ignored: list[ZoneEvent] = []
    for outcome in outcomes:
        incidents.extend(outcome.incidents)
        suppressed.extend(outcome.suppressed)
        ignored.extend(outcome.ignored)
    return AlertOutcome(
        incidents=tuple(incidents),
        suppressed=tuple(suppressed),
        ignored=tuple(ignored),
    )
