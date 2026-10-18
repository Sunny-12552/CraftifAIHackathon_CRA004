# SPDX-License-Identifier: Apache-2.0
"""Polygonal restricted zones.

Zones are stored as normalised coordinates so a zone defined on a 640x480 webcam
still means the same physical area on a 1920x1080 capture. Resolution only
enters the picture in :meth:`PolygonZone.resolve`, which converts to pixels once
per frame size rather than once per detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

from ..config import ZoneConfig
from ..geometry import polygon_area_magnitude, validate_polygon_shape


@dataclass(frozen=True, slots=True)
class PolygonZone:
    """A restricted area described by a normalised simple polygon."""

    zone_id: str
    name: str
    points_normalized: tuple[tuple[float, float], ...]
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise ValueError("zone_id must not be empty")
        if len(self.points_normalized) < 3:
            raise ValueError(
                f"Zone {self.zone_id!r} needs at least three points, "
                f"got {len(self.points_normalized)}"
            )
        for index, (x, y) in enumerate(self.points_normalized):
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"Zone {self.zone_id!r} point {index} is not finite")
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"Zone {self.zone_id!r} point {index} = ({x}, {y}) is outside "
                    "the frame; normalised coordinates must be within 0.0 - 1.0"
                )
        # A zone that self-intersects or encloses no area can never be used, so
        # construction fails rather than producing silently useless occupancy.
        validate_polygon_shape(self.points_normalized, f"Zone {self.zone_id!r}")

    # -- construction ---------------------------------------------------
    @classmethod
    def from_config(cls, zone: ZoneConfig) -> "PolygonZone":
        return cls(
            zone_id=zone.zone_id,
            name=zone.name,
            points_normalized=tuple(zone.polygon_normalized),
            enabled=zone.enabled,
        )

    @classmethod
    def from_points(
        cls, zone_id: str, name: str, points: Iterable[Sequence[float]]
    ) -> "PolygonZone":
        return cls(
            zone_id=zone_id,
            name=name,
            points_normalized=tuple((float(x), float(y)) for x, y in points),
        )

    # -- geometry -------------------------------------------------------
    def resolve(self, frame_width: int, frame_height: int) -> "PixelZone":
        """Convert to pixel coordinates for a frame size."""
        if frame_width < 1 or frame_height < 1:
            raise ValueError("Frame dimensions must be positive")
        points = self.points_normalized
        pixels = np.rint(
            np.asarray(points, dtype=np.float64)
            * np.array([frame_width - 1, frame_height - 1], dtype=np.float64)
        ).astype(np.int32)
        return PixelZone(zone=self, points=pixels)

    def area_normalized(self) -> float:
        """Area magnitude of the polygon in normalised units."""
        return polygon_area_magnitude(self.points_normalized)

    def to_config(self) -> ZoneConfig:
        return ZoneConfig(
            zone_id=self.zone_id,
            name=self.name,
            polygon_normalized=self.points_normalized,
            enabled=self.enabled,
        )


@dataclass(frozen=True, slots=True)
class PixelZone:
    """A zone resolved to pixel coordinates for the current frame size."""

    zone: PolygonZone
    points: np.ndarray

    @property
    def zone_id(self) -> str:
        return self.zone.zone_id

    @property
    def name(self) -> str:
        return self.zone.name

    def contains(self, point: Sequence[float]) -> bool:
        """True when ``point`` lies inside or on the boundary of the polygon.

        Boundary-inclusive matching preserves the behaviour of the verified
        baseline and avoids a dead band exactly on the zone edge.
        """
        import cv2

        return (
            cv2.pointPolygonTest(
                self.points, (float(point[0]), float(point[1])), False
            )
            >= 0
        )

    def bounds(self) -> tuple[int, int, int, int]:
        """Bounding box of the polygon as ``(x, y, w, h)``."""
        xs = self.points[:, 0]
        ys = self.points[:, 1]
        left, top = int(xs.min()), int(ys.min())
        return left, top, int(xs.max()) - left, int(ys.max()) - top

    def area_pixels(self) -> float:
        return float(abs(_shoelace(self.points)))


def _shoelace(points: np.ndarray) -> float:
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def build_zones(zones: Sequence[ZoneConfig]) -> tuple[PolygonZone, ...]:
    """Convert validated configuration entries into zone objects."""
    return tuple(PolygonZone.from_config(zone) for zone in zones)


@lru_cache(maxsize=32)
def _cached_points(
    zone_id: str, points: tuple[tuple[float, float], ...], width: int, height: int
) -> tuple[tuple[int, int], ...]:
    """Memoise pixel conversion for a given zone definition and frame size."""
    pixels = (
        np.rint(
            np.asarray(points, dtype=np.float64)
            * np.array([width - 1, height - 1], dtype=np.float64)
        )
        .astype(np.int32)
    )
    return tuple((int(x), int(y)) for x, y in pixels)


def resolve_cached(
    zone: PolygonZone, frame_width: int, frame_height: int
) -> PixelZone:
    """Resolve a zone, reusing the pixel conversion across frames."""
    pairs = _cached_points(
        zone.zone_id, zone.points_normalized, frame_width, frame_height
    )
    return PixelZone(zone=zone, points=np.asarray(pairs, dtype=np.int32))
