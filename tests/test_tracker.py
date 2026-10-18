# SPDX-License-Identifier: Apache-2.0
"""IoU tracking: identity stability, label gating and expiry."""

from __future__ import annotations

import pytest

from vision_sentinel.tracking.iou_tracker import IouTracker

from conftest import person

# Two boxes that overlap enough to associate across frames.
BOX_A = (100, 100, 40, 100)
BOX_A_SHIFTED = (106, 104, 40, 100)
BOX_B = (400, 100, 40, 100)


def test_first_detection_receives_an_identifier() -> None:
    tracker = IouTracker()
    ids = tracker.update([person(BOX_A)], frame_id=1)
    assert ids == ["1"]


def test_identity_is_stable_across_small_movements() -> None:
    tracker = IouTracker(iou_threshold=0.3)
    first = tracker.update([person(BOX_A)], frame_id=1)
    second = tracker.update([person(BOX_A_SHIFTED)], frame_id=2)
    third = tracker.update([person(BOX_A)], frame_id=3)

    assert first == second == third == [first[0]]


def test_two_people_keep_separate_identities() -> None:
    tracker = IouTracker()
    ids = tracker.update([person(BOX_A), person(BOX_B)], frame_id=1)
    assert len(set(ids)) == 2

    ids_again = tracker.update([person(BOX_B), person(BOX_A)], frame_id=2)
    # The ids follow the boxes, not the list order.
    assert ids_again[0] == ids[1]
    assert ids_again[1] == ids[0]


def test_a_person_is_never_matched_onto_a_vehicle() -> None:
    from vision_sentinel.events import Detection

    tracker = IouTracker()
    tracker.update([person(BOX_A)], frame_id=1)

    bicycle = Detection(1, "bicycle", 0.9, BOX_A)
    ids = tracker.update([bicycle], frame_id=2)

    assert ids == ["2"]  # a new identity, not the person's
    assert tracker.active_count == 2


def test_tracks_expire_after_max_age_and_report_it() -> None:
    tracker = IouTracker(max_age=2)
    tracker.update([person(BOX_A)], frame_id=1)

    tracker.update([], frame_id=2)
    assert tracker.expired_last_update() == ()

    tracker.update([], frame_id=3)
    assert tracker.expired_last_update() == ()

    tracker.update([], frame_id=4)
    assert tracker.expired_last_update() == ("1",)
    assert tracker.active_count == 0


def test_an_expired_identifier_is_never_reused() -> None:
    tracker = IouTracker(max_age=0)
    tracker.update([person(BOX_A)], frame_id=1)
    tracker.update([], frame_id=2)  # expires id 1

    ids = tracker.update([person(BOX_A)], frame_id=3)
    assert ids == ["2"]


def test_min_hits_withholds_identity_until_confirmed() -> None:
    tracker = IouTracker(min_hits=3)
    assert tracker.update([person(BOX_A)], frame_id=1) == [None]
    assert tracker.update([person(BOX_A)], frame_id=2) == [None]
    assert tracker.update([person(BOX_A)], frame_id=3) == ["1"]


def test_forget_drops_a_track_immediately() -> None:
    tracker = IouTracker(max_age=100)
    ids = tracker.update([person(BOX_A)], frame_id=1)
    assert tracker.active_count == 1

    tracker.forget(ids)
    assert tracker.active_count == 0


def test_reset_clears_everything() -> None:
    tracker = IouTracker()
    tracker.update([person(BOX_A), person(BOX_B)], frame_id=1)
    tracker.reset()

    assert tracker.active_count == 0
    assert tracker.tracks() == ()
    assert tracker.update([person(BOX_A)], frame_id=2) == ["1"]


def test_track_snapshot_reports_hits_and_age() -> None:
    tracker = IouTracker(max_age=5)
    tracker.update([person(BOX_A)], frame_id=1)
    tracker.update([person(BOX_A)], frame_id=2)
    info = tracker.tracks()[0]

    assert info.track_id == "1"
    assert info.label == "person"
    assert info.hits == 2
    assert info.age == 0
    assert info.confirmed is True


@pytest.mark.parametrize(
    ("iou_threshold", "max_age", "min_hits"),
    [(-0.1, 5, 1), (1.1, 5, 1), (0.3, -1, 1), (0.3, 5, 0)],
)
def test_invalid_parameters_are_rejected(
    iou_threshold: float, max_age: int, min_hits: int
) -> None:
    with pytest.raises(ValueError):
        IouTracker(
            iou_threshold=iou_threshold, max_age=max_age, min_hits=min_hits
        )
