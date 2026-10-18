# SPDX-License-Identifier: Apache-2.0
"""Per-track zone occupancy state machine.

This is the component that fixes duplicate-alert behaviour. The baseline it
replaces stored one boolean per zone, so a detection that flickered at the zone
boundary for a single frame flipped the zone from clear to occupied and back.
Over a 2h17m session that produced 576 entry records and 572 clear records --
almost all of them the same person crossing the same edge.

The machine here is per ``(zone, track)`` and requires a transition to persist
before it is believed:

    OUTSIDE --inside--> ENTERING --enter_frames--> INSIDE   (emits ENTERED)
    INSIDE --outside--> EXITING --exit_frames----> OUTSIDE  (emits EXITED)
    ENTERING --outside--> OUTSIDE        (flicker absorbed, nothing emitted)
    EXITING --inside--> INSIDE           (brief occlusion absorbed)

With ``enter_frames = 3`` a one- or two-frame spurious detection can never
produce an incident, while a person who genuinely walks in is reported after
roughly 100 ms at 30 FPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..events import ZoneEventType


class TrackState(str, Enum):
    """Where one track stands relative to one zone."""

    OUTSIDE = "outside"
    ENTERING = "entering"
    INSIDE = "inside"
    EXITING = "exiting"

    @property
    def occupied(self) -> bool:
        """Whether the track currently counts as being in the zone."""
        return self is TrackState.INSIDE


@dataclass(frozen=True, slots=True)
class Transition:
    """A confirmed transition worth reporting."""

    kind: ZoneEventType
    frame_id: int


class ZoneStateMachine:
    """Debounced inside/outside tracking for a single track in a single zone."""

    def __init__(self, enter_frames: int = 3, exit_frames: int = 5) -> None:
        if enter_frames < 1:
            raise ValueError("enter_frames must be one or greater")
        if exit_frames < 1:
            raise ValueError("exit_frames must be one or greater")
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.state = TrackState.OUTSIDE
        self.inside_streak = 0
        self.outside_streak = 0
        self.last_change_frame: int | None = None
        self.transitions = 0

    def update(self, is_inside: bool, frame_id: int = 0) -> Transition | None:
        """Feed one observation and return a confirmed transition, if any."""
        if self.state is TrackState.OUTSIDE:
            if is_inside:
                self.state = TrackState.ENTERING
                self.inside_streak = 1
                self.outside_streak = 0
                if self.inside_streak >= self.enter_frames:
                    return self._confirm(TrackState.INSIDE, ZoneEventType.ENTERED, frame_id)
            return None

        if self.state is TrackState.ENTERING:
            if is_inside:
                self.inside_streak += 1
                self.outside_streak = 0
                if self.inside_streak >= self.enter_frames:
                    return self._confirm(TrackState.INSIDE, ZoneEventType.ENTERED, frame_id)
            else:
                # A flicker: not enough evidence, so treat it as never having
                # entered and emit nothing.
                self.state = TrackState.OUTSIDE
                self.inside_streak = 0
                self.outside_streak = 0
            return None

        if self.state is TrackState.INSIDE:
            if not is_inside:
                self.state = TrackState.EXITING
                self.outside_streak = 1
                self.inside_streak = 0
                if self.outside_streak >= self.exit_frames:
                    return self._confirm(TrackState.OUTSIDE, ZoneEventType.EXITED, frame_id)
            else:
                self.outside_streak = 0
            return None

        # EXITING
        if is_inside:
            # Person is still there; cancel the pending exit.
            self.state = TrackState.INSIDE
            self.outside_streak = 0
            return None
        self.outside_streak += 1
        if self.outside_streak >= self.exit_frames:
            return self._confirm(TrackState.OUTSIDE, ZoneEventType.EXITED, frame_id)
        return None

    def _confirm(
        self, state: TrackState, kind: ZoneEventType, frame_id: int
    ) -> Transition:
        self.state = state
        self.last_change_frame = frame_id
        self.transitions += 1
        if state is TrackState.INSIDE:
            self.outside_streak = 0
        else:
            self.inside_streak = 0
        return Transition(kind=kind, frame_id=frame_id)

    @property
    def is_inside(self) -> bool:
        return self.state is TrackState.INSIDE

    def reset(self) -> None:
        self.state = TrackState.OUTSIDE
        self.inside_streak = 0
        self.outside_streak = 0

    def describe(self) -> str:
        return (
            f"{self.state.value} (in={self.inside_streak}, out={self.outside_streak})"
        )
