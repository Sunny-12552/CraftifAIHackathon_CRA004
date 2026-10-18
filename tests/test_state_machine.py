# SPDX-License-Identifier: Apache-2.0
"""Zone occupancy debouncing -- the core of duplicate-alert prevention.

These tests are the evidence for FR-05, AT-05 and AT-06. They need no camera:
they drive the state machine directly, which is exactly why the behaviour can be
trusted before anyone points a webcam at a doorway.
"""

from __future__ import annotations

import pytest

from vision_sentinel.events import ZoneEventType
from vision_sentinel.zones.state_machine import TrackState, ZoneStateMachine


def test_single_frame_blip_never_becomes_an_entry() -> None:
    """A one-frame spurious detection must produce no transition at all."""
    machine = ZoneStateMachine(enter_frames=3, exit_frames=3)

    assert machine.update(True, frame_id=1) is None
    assert machine.update(False, frame_id=2) is None
    assert machine.transitions == 0
    assert machine.state is TrackState.OUTSIDE


def test_two_frame_blip_never_becomes_an_entry() -> None:
    machine = ZoneStateMachine(enter_frames=3, exit_frames=3)

    machine.update(True, frame_id=1)
    machine.update(True, frame_id=2)
    machine.update(False, frame_id=3)

    assert machine.transitions == 0
    assert machine.state is TrackState.OUTSIDE


def test_sustained_presence_emits_exactly_one_entry() -> None:
    machine = ZoneStateMachine(enter_frames=3, exit_frames=3)

    assert machine.update(True, frame_id=1) is None
    assert machine.update(True, frame_id=2) is None

    transition = machine.update(True, frame_id=3)
    assert transition is not None
    assert transition.kind is ZoneEventType.ENTERED
    assert transition.frame_id == 3
    assert machine.state is TrackState.INSIDE
    assert machine.transitions == 1


def test_staying_inside_never_repeats_the_entry() -> None:
    """AT-05: remaining in the zone must not alert on every frame."""
    machine = ZoneStateMachine(enter_frames=3, exit_frames=3)
    for frame_id in range(1, 4):
        machine.update(True, frame_id=frame_id)

    for frame_id in range(4, 400):
        assert machine.update(True, frame_id=frame_id) is None

    assert machine.transitions == 1


def test_exit_requires_the_configured_number_of_frames() -> None:
    machine = ZoneStateMachine(enter_frames=1, exit_frames=3)
    machine.update(True, frame_id=1)
    assert machine.state is TrackState.INSIDE

    assert machine.update(False, frame_id=2) is None
    assert machine.update(False, frame_id=3) is None
    assert machine.state is TrackState.EXITING

    transition = machine.update(False, frame_id=4)
    assert transition is not None
    assert transition.kind is ZoneEventType.EXITED
    assert machine.state is TrackState.OUTSIDE


def test_brief_occlusion_does_not_end_the_visit() -> None:
    """A person briefly hidden behind an object must not generate an exit."""
    machine = ZoneStateMachine(enter_frames=2, exit_frames=5)
    machine.update(True, frame_id=1)
    machine.update(True, frame_id=2)
    assert machine.state is TrackState.INSIDE

    # Three frames of nothing, then visible again.
    for frame_id in (3, 4, 5):
        assert machine.update(False, frame_id=frame_id) is None
    assert machine.state is TrackState.EXITING

    assert machine.update(True, frame_id=6) is None
    assert machine.state is TrackState.INSIDE
    assert machine.transitions == 1


def test_reentry_after_a_confirmed_exit_emits_again() -> None:
    """AT-06: leaving and coming back must be able to raise a new event."""
    machine = ZoneStateMachine(enter_frames=2, exit_frames=2)

    machine.update(True, frame_id=1)
    assert machine.update(True, frame_id=2) is not None  # entry

    assert machine.update(False, frame_id=3) is None
    exit_transition = machine.update(False, frame_id=4)
    assert exit_transition is not None
    assert exit_transition.kind is ZoneEventType.EXITED

    machine.update(True, frame_id=5)
    second_entry = machine.update(True, frame_id=6)
    assert second_entry is not None
    assert second_entry.kind is ZoneEventType.ENTERED
    assert machine.transitions == 3


def test_enter_frames_of_one_reacts_immediately() -> None:
    machine = ZoneStateMachine(enter_frames=1, exit_frames=1)

    transition = machine.update(True, frame_id=1)
    assert transition is not None
    assert transition.kind is ZoneEventType.ENTERED

    exit_transition = machine.update(False, frame_id=2)
    assert exit_transition is not None
    assert exit_transition.kind is ZoneEventType.EXITED


def test_flapping_at_the_boundary_produces_no_incident_storm() -> None:
    """The regression that motivated this component.

    The replaced baseline stored one boolean per zone, so this alternating
    sequence produced an entry and a clear on almost every frame -- 576 entries
    in a single 2h17m session.
    """
    machine = ZoneStateMachine(enter_frames=3, exit_frames=5)

    for frame_id in range(1, 601):
        machine.update(is_inside=bool(frame_id % 2), frame_id=frame_id)

    # Alternating one frame at a time never satisfies either dwell requirement.
    assert machine.transitions == 0


@pytest.mark.parametrize(
    ("enter_frames", "exit_frames"),
    [(0, 1), (1, 0), (-1, 2)],
)
def test_invalid_frame_counts_are_rejected(enter_frames: int, exit_frames: int) -> None:
    with pytest.raises(ValueError):
        ZoneStateMachine(enter_frames=enter_frames, exit_frames=exit_frames)


def test_reset_returns_to_a_clean_state() -> None:
    machine = ZoneStateMachine(enter_frames=1, exit_frames=1)
    machine.update(True, frame_id=1)
    machine.reset()

    assert machine.state is TrackState.OUTSIDE
    assert machine.inside_streak == 0
    assert machine.outside_streak == 0
