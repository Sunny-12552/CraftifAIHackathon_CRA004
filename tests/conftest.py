# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures.

Nothing here needs a camera, a GPU or the network. The detector, the video
source and the clock are all injectable, which is what allows the acceptance
tests to exercise the complete pipeline deterministically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_sentinel.config import Settings, parse_settings  # noqa: E402
from vision_sentinel.detection.base import Detector, DetectorError, DetectionBatch  # noqa: E402
from vision_sentinel.events import Detection  # noqa: E402
from vision_sentinel.video.source import Frame, VideoSource  # noqa: E402
from vision_sentinel.events import utc_now_iso  # noqa: E402

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

#: A person whose foot point lands inside the default test zone.
INSIDE_BOX = (140, 60, 40, 100)
#: A person whose foot point lands outside the default test zone.
OUTSIDE_BOX = (10, 60, 40, 100)


def person(bbox_xywh: tuple[int, int, int, int] = INSIDE_BOX, confidence: float = 0.9) -> Detection:
    """Build a person detection for the given box."""
    return Detection(class_id=0, label="person", confidence=confidence, bbox_xywh=bbox_xywh)


def blank_frame(width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@pytest.fixture
def settings_factory(tmp_path: Path) -> Callable[..., Settings]:
    """Build validated settings with test-friendly defaults."""

    def factory(**overrides: object) -> Settings:
        raw = {
            "video": {
                "source_type": "file",
                "file_path": "clip.mp4",
                "camera_id": "cam-test",
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
            },
            "detection": {
                "backend": "cpu",
                "model_path": "dummy.onnx",
                "confidence_threshold": 0.35,
                "nms_threshold": 0.45,
            },
            "tracking": {
                "enabled": True,
                "iou_threshold": 0.3,
                "max_age": 5,
                "min_hits": 1,
            },
            "alerts": {
                "enter_frames": 3,
                "exit_frames": 3,
                "cooldown_seconds": 10.0,
                "min_confidence": 0.0,
                "record_exits": True,
            },
            "zones": [
                {
                    "id": "zone-1",
                    "name": "Zone A",
                    "enabled": True,
                    "polygon_normalized": [
                        [0.25, 0.25],
                        [0.75, 0.25],
                        [0.75, 0.75],
                        [0.25, 0.75],
                    ],
                }
            ],
            "storage": {
                "db_path": str(tmp_path / "incidents.db"),
                "export_dir": str(tmp_path / "exports"),
            },
            "metrics": {"window": 30, "report_path": str(tmp_path / "report.json")},
        }
        return parse_settings(_deep_merge(raw, overrides))

    return factory


class FakeDetector(Detector):
    """A detector driven by a scripted sequence instead of a model.

    ``script`` is one list of detections per call. The last entry repeats once
    the script is exhausted, so a test can describe only the interesting frames.
    """

    def __init__(
        self,
        script: Sequence[Sequence[Detection]] | None = None,
        *,
        device_name: str = "Fake detector",
        backend_id: str = "fake",
        error: str | None = None,
        owns_video_source: bool = True,
        compliance_eligible: bool = False,
        upstream_track_ids: Sequence[str | None] | None = None,
    ) -> None:
        self.script = [list(frame) for frame in (script or [[]])]
        self.device_name = device_name
        self.backend_id = backend_id
        self.error = error
        self.owns_video_source = owns_video_source
        self.compliance_eligible = compliance_eligible
        self.upstream_track_ids = upstream_track_ids
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:  # noqa: ARG002
        if self.error is not None:
            raise DetectorError(self.error)
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return list(self.script[index])

    def detect_tracked(self, frame: np.ndarray) -> DetectionBatch:
        if self.error is not None:
            raise DetectorError(self.error)
        if self.upstream_track_ids is not None:
            index = min(self.calls, len(self.script) - 1)
            self.calls += 1
            return DetectionBatch(
                detections=tuple(self.script[index]),
                track_ids=tuple(self.upstream_track_ids)[: len(self.script[index])],
            )
        return DetectionBatch(detections=tuple(self.detect(frame)))


class ScriptedSource(VideoSource):
    """A finite video source that replays a list of frames."""

    finite = True

    def __init__(self, frames: Iterable[np.ndarray], camera_id: str = "cam-test") -> None:
        super().__init__(camera_id)
        self.frames = list(frames)
        self._index = 0
        self.opened_count = 0

    def open(self) -> None:
        self.opened_count += 1
        self._opened = True

    def read(self) -> Frame | None:
        if self._index >= len(self.frames):
            return None
        image = self.frames[self._index]
        self._index += 1
        self._frame_id += 1
        return Frame(
            image=image,
            frame_id=self._frame_id,
            timestamp_utc=utc_now_iso(),
            is_last=self._index >= len(self.frames),
        )

    def close(self) -> None:
        self._opened = False

    def describe(self) -> dict:
        return {"camera_id": self.camera_id, "type": "scripted"}


def frames(count: int, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> list[np.ndarray]:
    """A list of identical blank frames."""
    return [blank_frame(width, height) for _ in range(count)]


def repeated(source: np.ndarray, count: int) -> list[np.ndarray]:
    """Repeat one frame ``count`` times."""
    return [source.copy() for _ in range(count)]
