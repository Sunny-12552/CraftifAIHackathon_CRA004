# SPDX-License-Identifier: Apache-2.0
"""Small geometric helpers shared by detection NMS and tracking."""

from __future__ import annotations

from typing import Sequence


def iou_xywh(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection-over-union of two ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = (float(value) for value in first)
    bx, by, bw, bh = (float(value) for value in second)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def footprint_overlap(first: Sequence[float], second: Sequence[float]) -> float:
    """Fraction of the smaller box's area covered by the intersection.

    Useful as a fallback association metric for boxes that barely overlap but
    clearly describe the same object.
    """
    ax, ay, aw, ah = (float(value) for value in first)
    bx, by, bw, bh = (float(value) for value in second)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    smaller = min(aw * ah, bw * bh)
    return intersection / smaller if smaller > 0 else 0.0


# ---------------------------------------------------------------------------
# Polygon validation
# ---------------------------------------------------------------------------
# Kept here rather than in the config layer so that a PolygonZone constructed
# directly in code is validated exactly as strictly as one loaded from YAML.

#: Minimum magnitude of the shoelace area for a polygon to enclose anything.
MINIMUM_POLYGON_AREA = 1e-9


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Signed area of a polygon via the shoelace formula."""
    total = 0.0
    count = len(points)
    for index in range(count):
        x1, y1 = float(points[index][0]), float(points[index][1])
        x2, y2 = float(points[(index + 1) % count][0]), float(points[(index + 1) % count][1])
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _orientation(
    a: Sequence[float], b: Sequence[float], c: Sequence[float]
) -> float:
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (
        float(b[1]) - float(a[1])
    ) * (float(c[0]) - float(a[0]))


def segments_intersect(
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
    p4: Sequence[float],
) -> bool:
    """True when segments ``p1-p2`` and ``p3-p4`` cross or touch collinearly."""
    d1 = _orientation(p3, p4, p1)
    d2 = _orientation(p3, p4, p2)
    d3 = _orientation(p1, p2, p3)
    d4 = _orientation(p1, p2, p4)

    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    # Treat collinear overlap as an intersection as well.
    return d1 == 0 or d2 == 0 or d3 == 0 or d4 == 0


def is_self_intersecting(points: Sequence[Sequence[float]]) -> bool:
    """True when a polygon's edges cross each other."""
    count = len(points)
    for i in range(count):
        first_start = points[i]
        first_end = points[(i + 1) % count]
        for j in range(i + 1, count):
            # Skip adjacent edges, which always share a vertex.
            if j == i or (j + 1) % count == i or (i + 1) % count == j:
                continue
            second_start = points[j]
            second_end = points[(j + 1) % count]
            if segments_intersect(first_start, first_end, second_start, second_end):
                return True
    return False


def validate_polygon_shape(
    points: Sequence[Sequence[float]], where: str
) -> None:
    """Raise :class:`ValueError` when ``points`` cannot describe a usable zone.

    The self-intersection test runs before the area test so that a bow-tie
    polygon reports the more specific problem.
    """
    if is_self_intersecting(points):
        raise ValueError(f"{where} is self-intersecting; use a simple polygon")
    if abs(polygon_area(points)) <= MINIMUM_POLYGON_AREA:
        raise ValueError(f"{where} encloses no area; the points are collinear")


def polygon_area_magnitude(points: Sequence[Sequence[float]]) -> float:
    """Absolute area of a polygon."""
    return abs(polygon_area(points))
