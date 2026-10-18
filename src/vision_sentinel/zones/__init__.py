# SPDX-License-Identifier: Apache-2.0
"""Restricted-zone geometry, debounced occupancy state and multi-zone monitoring."""

from __future__ import annotations

from .monitor import AGGREGATE_TRACK, ZoneMonitor, ZoneOccupant
from .polygon import PixelZone, PolygonZone, build_zones, resolve_cached
from .state_machine import TrackState, Transition, ZoneStateMachine

__all__ = [
    "AGGREGATE_TRACK",
    "PixelZone",
    "PolygonZone",
    "TrackState",
    "Transition",
    "ZoneMonitor",
    "ZoneOccupant",
    "ZoneStateMachine",
    "build_zones",
    "resolve_cached",
]
