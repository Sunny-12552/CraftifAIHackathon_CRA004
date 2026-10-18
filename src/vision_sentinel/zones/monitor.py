# SPDX-License-Identifier: Apache-2.0
"""Multi-zone occupancy monitoring built on the shared detection schema.

The monitor owns one :class:`ZoneStateMachine` per ``(zone, track)`` pair. It
consumes plain :class:`~vision_sentinel.events.Detection` values plus whatever
track ids the active backend produced, and it has no idea whether those came
from OpenVINO on an Intel GPU or from a PipeGen/TensorRT pipeline. That is the
property that lets one restricted-zone implementation serve both execution
paths.

Occupancy is anchored on each detection's **foot point** (bottom-centre of the
box), not its centre, so a person's swaying upper body cannot move them in and
out of a zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..events import Detection, ZoneEvent, ZoneEventType, utc_now_iso
from .polygon import PixelZone, PolygonZone, resolve_cached
from .state_machine import ZoneStateMachine

#: Internal key used when no tracker identity is available.
AGGREGATE_TRACK = "__aggregate__"


@dataclass(frozen=True, slots=True)
class ZoneOccupant:
    """A track currently believed to be inside a zone."""

    zone_id: str
    zone_name: str
    track_id: str | None
    label: str
    confidence: float
    bbox_xywh: tuple[int, int, int, int]
    foot_point: tuple[float, float]


class ZoneMonitor:
    """Evaluates configured restricted zones against normalised detections."""

    def __init__(
        self,
        zones: Sequence[PolygonZone],
        enter_frames: int = 3,
        exit_frames: int = 5,
        min_confidence: float = 0.0,
        monitored_labels: Sequence[str] = ("person",),
    ) -> None:
        if min_confidence < 0.0 or min_confidence > 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        self.zones: tuple[PolygonZone, ...] = tuple(zones)
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.min_confidence = min_confidence
        self.monitored_labels = tuple(label.lower() for label in monitored_labels)

        self._states: dict[tuple[str, str], ZoneStateMachine] = {}
        self._dropped_inside = 0
        self._last_frame_size: tuple[int, int] | None = None
        self._resolved: dict[str, PixelZone] = {}

    # -- configuration ---------------------------------------------------
    def set_zones(self, zones: Sequence[PolygonZone]) -> None:
        """Replace the zone set, discarding state for zones that disappeared."""
        self.zones = tuple(zones)
        valid = {zone.zone_id for zone in self.zones}
        for key in [key for key in self._states if key[0] not in valid]:
            del self._states[key]
        self._resolved.clear()

    def enabled_zones(self) -> tuple[PolygonZone, ...]:
        return tuple(zone for zone in self.zones if zone.enabled)

    def zone_by_id(self, zone_id: str) -> PolygonZone | None:
        return next((zone for zone in self.zones if zone.zone_id == zone_id), None)

    # -- frame evaluation ------------------------------------------------
    def update(
        self,
        detections: Sequence[Detection],
        track_ids: Sequence[str | None] | None = None,
        frame_id: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        timestamp_utc: str | None = None,
        camera_id: str = "cam-0",
        source: str = "unknown",
    ) -> list[ZoneEvent]:
        """Evaluate every zone for this frame and return confirmed events.

        Events are de-duplicated by construction: each one corresponds to a
        state-machine transition, which by definition cannot recur while the
        track stays on the same side of the boundary.
        """
        if frame_width < 1 or frame_height < 1:
            raise ValueError("Frame dimensions must be positive")
        if track_ids is not None and len(track_ids) != len(detections):
            raise ValueError(
                "track_ids must be parallel to detections "
                f"({len(track_ids)} != {len(detections)})"
            )

        stamp = timestamp_utc or utc_now_iso()
        zones = self.enabled_zones()
        if not zones:
            return []

        self._refresh_resolved(frame_width, frame_height)

        eligible = self._eligible(detections, track_ids)
        events: list[ZoneEvent] = []

        for zone in zones:
            pixel_zone = self._resolved[zone.zone_id]
            grouped = self._group_for_zone(eligible, pixel_zone)

            pending: list[tuple[int, ZoneEventType, str, str | None, Detection]] = []
            for order, (track_key, group) in enumerate(sorted(grouped.items())):
                inside = any(
                    pixel_zone.contains(detection.clamped_foot_point(frame_width, frame_height))
                    for detection in group
                )
                machine = self._machine(zone.zone_id, track_key)
                transition = machine.update(inside, frame_id)
                if transition is None:
                    continue
                representative = max(group, key=lambda item: item.confidence)
                pending.append(
                    (
                        order,
                        transition.kind,
                        zone.zone_id,
                        None if track_key == AGGREGATE_TRACK else track_key,
                        representative,
                    )
                )

            if not pending:
                continue

            people_count = self._count_inside(zone.zone_id)
            for _, kind, zone_id, track_id, detection in pending:
                events.append(
                    ZoneEvent(
                        event_type=kind,
                        timestamp_utc=stamp,
                        camera_id=camera_id,
                        zone_id=zone_id,
                        zone_name=zone.name,
                        track_id=track_id,
                        label=detection.label,
                        confidence=detection.confidence,
                        bbox_xywh=detection.bbox_xywh,
                        foot_point=detection.clamped_foot_point(frame_width, frame_height),
                        people_count=people_count,
                        source=source,
                        frame_id=frame_id,
                        suppressed=False,
                    )
                )

        return events

    # -- internals -------------------------------------------------------
    def _eligible(
        self, detections: Sequence[Detection], track_ids: Sequence[str | None] | None
    ) -> list[tuple[Detection, str]]:
        selected: list[tuple[Detection, str]] = []
        for index, detection in enumerate(detections):
            if detection.label.lower() not in self.monitored_labels:
                continue
            if detection.confidence < self.min_confidence:
                continue
            raw_track = track_ids[index] if track_ids is not None else None
            track_key = raw_track or AGGREGATE_TRACK
            selected.append((detection, track_key))
        return selected

    @staticmethod
    def _group_for_zone(
        eligible: Sequence[tuple[Detection, str]], pixel_zone: PixelZone
    ) -> dict[str, list[Detection]]:
        """Group detections by track for one zone.

        Grouping is done for every track rather than only for tracks already
        inside, so the state machine also observes the frames where a track is
        outside -- which is what makes the exit debounce work.
        """
        grouped: dict[str, list[Detection]] = {}
        for detection, track_key in eligible:
            grouped.setdefault(track_key, []).append(detection)
        return grouped

    def _refresh_resolved(self, frame_width: int, frame_height: int) -> None:
        if self._last_frame_size == (frame_width, frame_height) and self._resolved:
            return
        self._last_frame_size = (frame_width, frame_height)
        self._resolved = {
            zone.zone_id: resolve_cached(zone, frame_width, frame_height)
            for zone in self.enabled_zones()
        }

    def _machine(self, zone_id: str, track_key: str) -> ZoneStateMachine:
        key = (zone_id, track_key)
        machine = self._states.get(key)
        if machine is None:
            machine = ZoneStateMachine(
                enter_frames=self.enter_frames, exit_frames=self.exit_frames
            )
            self._states[key] = machine
        return machine

    def _count_inside(self, zone_id: str) -> int:
        return sum(
            1
            for (state_zone_id, _), machine in self._states.items()
            if state_zone_id == zone_id and machine.is_inside
        )

    # -- state inspection ------------------------------------------------
    def occupancy(self) -> dict[str, int]:
        """Number of confirmed occupants per zone id."""
        result = {zone.zone_id: 0 for zone in self.zones}
        for (zone_id, _), machine in self._states.items():
            if machine.is_inside and zone_id in result:
                result[zone_id] += 1
        return result

    def is_zone_occupied(self, zone_id: str) -> bool:
        return self._count_inside(zone_id) > 0

    def active_occupants(
        self, detections: Sequence[Detection], track_ids: Sequence[str | None] | None
    ) -> tuple[ZoneOccupant, ...]:
        """Occupants believed to be inside a zone, for overlay rendering."""
        occupants: list[ZoneOccupant] = []
        for detection, track_key in self._eligible(detections, track_ids):
            for zone in self.enabled_zones():
                if not self._machine(zone.zone_id, track_key).is_inside:
                    continue
                occupants.append(
                    ZoneOccupant(
                        zone_id=zone.zone_id,
                        zone_name=zone.name,
                        track_id=None if track_key == AGGREGATE_TRACK else track_key,
                        label=detection.label,
                        confidence=detection.confidence,
                        bbox_xywh=detection.bbox_xywh,
                        foot_point=detection.foot_point,
                    )
                )
        return tuple(occupants)

    def track_state(self, zone_id: str, track_key: str) -> str | None:
        machine = self._states.get((zone_id, track_key))
        return machine.state.value if machine else None

    @property
    def dropped_inside_count(self) -> int:
        """Tracks discarded while still marked inside (abrupt disappearance)."""
        return self._dropped_inside

    def forget(self, track_ids: Iterable[str]) -> tuple[str, ...]:
        """Drop state for tracks that have expired in the tracker.

        A track that vanishes mid-zone can never emit a natural EXITED, so its
        state would otherwise pin the zone as occupied forever. Such tracks are
        dropped and counted. Emitting a synthetic EXITED would be a false claim
        that the person left, so nothing is emitted.
        """
        dropped: list[str] = []
        for track_id in track_ids:
            for key in [key for key in self._states if key[1] == track_id]:
                machine = self._states.pop(key)
                if machine.is_inside:
                    self._dropped_inside += 1
                dropped.append(track_id)
        return tuple(dict.fromkeys(dropped))

    def reset(self) -> None:
        self._states.clear()
        self._resolved.clear()
        self._last_frame_size = None
