# SPDX-License-Identifier: Apache-2.0
"""Object tracking: local IoU tracking, or upstream pipeline ids."""

from __future__ import annotations

from ..config import Settings
from .base import TrackInfo, Tracker, UpstreamTracker
from .iou_tracker import IouTracker

__all__ = [
    "IouTracker",
    "TrackInfo",
    "Tracker",
    "UpstreamTracker",
    "create_tracker",
]


def create_tracker(settings: Settings) -> Tracker:
    """Build the tracker implied by the configuration.

    When tracking is disabled the zone monitor falls back to a single aggregate
    pseudo-track per zone, which preserves the original occupancy semantics but
    still applies dwell and cooldown. That mode cannot distinguish two people,
    so it is documented as a reduced-capability fallback rather than the norm.
    """
    tracking = settings.tracking
    if not tracking.enabled:
        return UpstreamTracker()
    return IouTracker(
        iou_threshold=tracking.iou_threshold,
        max_age=tracking.max_age,
        min_hits=tracking.min_hits,
    )
