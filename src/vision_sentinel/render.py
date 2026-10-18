# SPDX-License-Identifier: Apache-2.0
"""Frame overlay rendering, shared by the native viewer and the dashboard.

The overlay only colours a detection as an intrusion when the zone monitor has
*confirmed* occupancy. A box that merely overlaps the zone boundary while the
dwell counter is still filling is drawn in the neutral colour, so what the
operator sees always matches what the alerting logic believes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .events import Detection
from .zones.polygon import PixelZone

# BGR colours.
COLOUR_ZONE_CLEAR = (0, 200, 0)
COLOUR_ZONE_OCCUPIED = (0, 0, 255)
COLOUR_DETECTION = (50, 220, 50)
COLOUR_INTRUDER = (0, 0, 255)
COLOUR_FOOTPOINT = (0, 215, 255)
COLOUR_TEXT = (255, 255, 255)
COLOUR_BANNER_OK = (40, 220, 40)
COLOUR_BANNER_ALERT = (0, 0, 255)
COLOUR_BANNER_ERROR = (0, 140, 255)
COLOUR_EDITOR = (255, 180, 0)


@dataclass(frozen=True, slots=True)
class OverlayOptions:
    """Toggles for what appears in the rendered frame."""

    show_track_ids: bool = True
    show_confidence: bool = True
    show_foot_points: bool = True
    show_zone_labels: bool = True
    zone_alpha: float = 0.12
    banner_height: int = 34


def render_frame(
    frame: np.ndarray,
    *,
    zones: Sequence[PixelZone] = (),
    detections: Sequence[Detection] = (),
    track_ids: Sequence[str | None] = (),
    occupied_zone_ids: Iterable[str] = (),
    intruder_track_ids: Iterable[str | None] = (),
    alerts: int = 0,
    fps: float = 0.0,
    inference_ms: float = 0.0,
    device: str = "",
    status: str = "running",
    error: str | None = None,
    editor_points: Sequence[tuple[int, int]] = (),
    options: OverlayOptions | None = None,
) -> np.ndarray:
    """Return a copy of ``frame`` with zones, detections and a status banner."""
    import cv2

    settings = options or OverlayOptions()
    output = frame.copy()
    occupied = set(occupied_zone_ids)
    intruders = {track for track in intruder_track_ids if track is not None}

    # --- zones ---------------------------------------------------------
    if zones:
        overlay = output.copy()
        for pixel_zone in zones:
            colour = (
                COLOUR_ZONE_OCCUPIED
                if pixel_zone.zone_id in occupied
                else COLOUR_ZONE_CLEAR
            )
            cv2.fillPoly(overlay, [pixel_zone.points], colour)
        output = cv2.addWeighted(overlay, settings.zone_alpha, output, 1 - settings.zone_alpha, 0)
        for pixel_zone in zones:
            colour = (
                COLOUR_ZONE_OCCUPIED
                if pixel_zone.zone_id in occupied
                else COLOUR_ZONE_CLEAR
            )
            cv2.polylines(output, [pixel_zone.points], True, colour, 2)
            if settings.show_zone_labels:
                x, y, _, _ = pixel_zone.bounds()
                label = pixel_zone.name
                if pixel_zone.zone_id in occupied:
                    label = f"{label} - OCCUPIED"
                _label(output, label, x, max(16, y - 6), colour)

    # --- detections ----------------------------------------------------
    ids = list(track_ids) + [None] * max(0, len(detections) - len(track_ids))
    for index, detection in enumerate(detections):
        track_id = ids[index] if index < len(ids) else None
        is_intruder = track_id is not None and track_id in intruders
        colour = COLOUR_INTRUDER if is_intruder else COLOUR_DETECTION
        x, y, width, height = detection.bbox_xywh
        cv2.rectangle(output, (x, y), (x + width, y + height), colour, 2)

        parts = [detection.label]
        if settings.show_confidence:
            parts.append(f"{detection.confidence:.0%}")
        if settings.show_track_ids and track_id is not None:
            parts.append(f"#{track_id}")
        _label(output, " ".join(parts), x, max(16, y - 6), colour)

        if settings.show_foot_points:
            foot = detection.clamped_foot_point(output.shape[1], output.shape[0])
            cv2.circle(output, (int(foot[0]), int(foot[1])), 4, COLOUR_FOOTPOINT, -1)

    # --- zone editor ---------------------------------------------------
    if editor_points:
        for point in editor_points:
            cv2.circle(output, point, 5, COLOUR_EDITOR, -1)
        if len(editor_points) > 1:
            cv2.polylines(
                output,
                [np.asarray(editor_points, dtype=np.int32)],
                False,
                COLOUR_EDITOR,
                2,
            )

    # --- banner --------------------------------------------------------
    if error:
        banner = f"ERROR: {error}"
        colour = COLOUR_BANNER_ERROR
    elif alerts > 0:
        banner = f"ALERT: {alerts} PERSON(S) IN RESTRICTED ZONE"
        colour = COLOUR_BANNER_ALERT
    elif editor_points is not None and status == "editing":
        banner = "ZONE EDIT: click vertices | Enter save | Backspace undo | Esc cancel"
        colour = COLOUR_EDITOR
    else:
        banner = f"{device or status} | {fps:.1f} FPS | {inference_ms:.1f} ms"
        colour = COLOUR_BANNER_OK

    height, width = output.shape[:2]
    cv2.rectangle(output, (0, 0), (width, settings.banner_height), (0, 0, 0), -1)
    _label(
        output,
        _truncate(banner, width // 9),
        10,
        int(settings.banner_height * 0.72),
        colour,
        scale=0.55,
    )
    return output


def _label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    colour: tuple[int, int, int],
    scale: float = 0.55,
) -> None:
    import cv2

    cv2.putText(
        frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2
    )


def _truncate(text: str, maximum: int) -> str:
    if maximum <= 3 or len(text) <= maximum:
        return text
    return text[: maximum - 3] + "..."


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode a frame as JPEG bytes (used by the dashboard and snapshots)."""
    import cv2

    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("Could not encode the frame as JPEG")
    return bytes(buffer)
