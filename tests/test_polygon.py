# SPDX-License-Identifier: Apache-2.0
"""Zone polygon geometry: scaling, validation, containment and caching."""

from __future__ import annotations

import pytest

from vision_sentinel.config import ConfigError, parse_settings
from vision_sentinel.zones.polygon import PolygonZone, build_zones, resolve_cached


def make_zone(points=None) -> PolygonZone:
    return PolygonZone.from_points(
        "zone-1",
        "Zone A",
        points
        or [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
    )


def test_normalised_coordinates_scale_to_pixels() -> None:
    zone = make_zone()
    pixels = zone.resolve(101, 201)
    assert pixels.points.tolist() == [[25, 50], [75, 50], [75, 150], [25, 150]]


def test_containment_is_inclusive_on_the_boundary() -> None:
    zone = make_zone()
    pixels = zone.resolve(101, 201)

    assert pixels.contains((50, 100))  # dead centre
    assert pixels.contains((25, 100))  # left edge
    assert pixels.contains((75, 150))  # corner
    assert not pixels.contains((10, 100))  # left of the zone
    assert not pixels.contains((50, 190))  # below the zone


def test_coordinate_scaling_matches_the_verified_baseline_convention() -> None:
    """``rint(p * (size - 1))`` must be preserved for zone compatibility."""
    zone = make_zone([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    pixels = zone.resolve(640, 480)
    assert pixels.points.tolist() == [[0, 0], [639, 0], [639, 479]]


def test_area_and_bounds() -> None:
    zone = make_zone()
    assert zone.area_normalized() == pytest.approx(0.25)

    pixels = zone.resolve(101, 201)
    assert pixels.bounds() == (25, 50, 50, 100)


def test_cached_resolution_matches_the_direct_path() -> None:
    zone = make_zone()
    direct = zone.resolve(640, 480)
    cached = resolve_cached(zone, 640, 480)
    assert cached.points.tolist() == direct.points.tolist()

    # A different frame size must not be served from the cache incorrectly.
    other = resolve_cached(zone, 320, 240)
    assert other.points.tolist() == zone.resolve(320, 240).points.tolist()


def test_disabled_zones_are_preserved_but_not_returned() -> None:
    from vision_sentinel.config import ZoneConfig

    zones = build_zones(
        [
            ZoneConfig("z1", "One", ((0.1, 0.1), (0.5, 0.1), (0.5, 0.5)), True),
            ZoneConfig("z2", "Two", ((0.1, 0.1), (0.5, 0.1), (0.5, 0.5)), False),
        ]
    )
    assert [zone.zone_id for zone in zones] == ["z1", "z2"]
    assert [zone.zone_id for zone in zones if zone.enabled] == ["z1"]


@pytest.mark.parametrize(
    "points",
    [
        [[0.1, 0.1], [0.2, 0.2]],  # too few points
        [[0.1, 0.1], [1.4, 0.2], [0.5, 0.5]],  # outside the frame
        [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]],  # collinear, zero area
    ],
)
def test_invalid_zone_definitions_are_rejected(points) -> None:
    with pytest.raises(ValueError):
        PolygonZone.from_points("bad", "Bad", points)


def test_self_intersecting_polygon_is_rejected_by_configuration() -> None:
    raw = {
        "zones": [
            {
                "id": "bowtie",
                "name": "Bowtie",
                "polygon_normalized": [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            }
        ]
    }
    with pytest.raises(ConfigError, match="self-intersecting"):
        parse_settings(raw)


def test_non_finite_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError):
        PolygonZone.from_points(
            "bad", "Bad", [[0.1, 0.1], [float("nan"), 0.5], [0.5, 0.5]]
        )


def test_to_config_round_trips() -> None:
    zone = make_zone()
    config = zone.to_config()
    assert config.zone_id == "zone-1"
    assert config.name == "Zone A"
    assert config.polygon_normalized == zone.points_normalized
