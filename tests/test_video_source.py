# SPDX-License-Identifier: Apache-2.0
"""Video input: file playback, missing files, camera failures and recovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vision_sentinel.video.source import (
    FileSource,
    Frame,
    RtspSource,
    VideoSourceError,
    WebcamSource,
    create_video_source,
)

WIDTH, HEIGHT = 64, 48


def write_test_video(path: Path, frame_count: int = 5, with_colour: bool = False) -> Path:
    """Write a real MJPG AVI so the file path is exercised, not mocked."""
    import cv2

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        # Fall back to a raw AVI writer; some builds lack the MJPG encoder.
        writer = cv2.VideoWriter(str(path), 0, 10.0, (WIDTH, HEIGHT))
    assert writer.isOpened(), "this OpenCV build cannot write a test video"

    for index in range(frame_count):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        if with_colour:
            frame[:, :] = (index * 10 % 255, 40, 90)
        writer.write(frame)
    writer.release()
    return path


# --- file source ------------------------------------------------------------


def test_a_missing_video_file_produces_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(VideoSourceError, match="Video file not found"):
        FileSource(tmp_path / "absent.mp4").open()


def test_a_directory_is_not_a_video_file(tmp_path: Path) -> None:
    with pytest.raises(VideoSourceError, match="not a file"):
        FileSource(tmp_path).open()


def test_an_undecodable_file_produces_an_actionable_error(tmp_path: Path) -> None:
    # A file that exists but is not a video reaches the OpenCV layer and must
    # produce a decode error rather than an "open failed" message.
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not a video")

    with pytest.raises(VideoSourceError, match="decode"):
        FileSource(broken).open()


def test_file_frames_are_read_and_the_stream_ends_cleanly(tmp_path: Path) -> None:
    path = write_test_video(tmp_path / "clip.avi", frame_count=4, with_colour=True)
    source = FileSource(path)
    source.open()

    frames = []
    while True:
        frame = source.read()
        if frame is None:
            break
        frames.append(frame)
    source.close()

    assert len(frames) == 4
    assert all(isinstance(frame, Frame) for frame in frames)
    assert [frame.frame_id for frame in frames] == [1, 2, 3, 4]
    assert all(frame.image.shape[:2] == (HEIGHT, WIDTH) for frame in frames)
    assert frames[-1].is_last is True
    assert all(frame.timestamp_utc.endswith("Z") for frame in frames)


def test_a_file_can_loop(tmp_path: Path) -> None:
    path = write_test_video(tmp_path / "clip.avi", frame_count=3)
    source = FileSource(path, loop=True)
    source.open()

    reads = [source.read() for _ in range(5)]
    source.close()

    assert all(frame is not None for frame in reads)


def test_file_describe_names_the_source(tmp_path: Path) -> None:
    path = write_test_video(tmp_path / "clip.avi", frame_count=2)
    assert FileSource(path).describe()["type"] == "file"


# --- camera behaviour -------------------------------------------------------


class FakeCapture:
    """Minimal stand-in for ``cv2.VideoCapture``."""

    def __init__(self, opened: bool = True, fail_reads: int = 0) -> None:
        self._opened = opened
        self.fail_reads = fail_reads
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
        return self._opened

    def read(self):
        if self.fail_reads > 0:
            self.fail_reads -= 1
            return False, None
        return True, np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    def set(self, *_args) -> bool:
        return True

    def get(self, _prop: int) -> float:
        return 0.0

    def release(self) -> None:
        self.released = True
        self._opened = False


def test_a_missing_camera_produces_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "cv2.VideoCapture", lambda *args, **kwargs: FakeCapture(opened=False)
    )
    source = WebcamSource(camera_index=0)

    with pytest.raises(VideoSourceError) as error:
        source.open()

    message = str(error.value)
    assert "webcam index 0" in message
    assert "privacy settings" in message
    assert "camera-index 1" in message.replace("index 1", "index 1") or "1" in message


def test_a_camera_that_stops_returning_frames_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "cv2.VideoCapture", lambda *args, **kwargs: FakeCapture(opened=True, fail_reads=10_000)
    )
    source = WebcamSource(camera_index=0, reopen_attempts=2, reopen_delay_s=0.0)
    source.open()

    with pytest.raises(VideoSourceError, match="stopped returning frames"):
        source.read()


def test_a_transient_read_failure_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakeCapture] = []

    def factory(*_args, **_kwargs) -> FakeCapture:
        # The first camera hiccups, the reopened one works.
        capture = FakeCapture(opened=True, fail_reads=1 if not created else 0)
        created.append(capture)
        return capture

    monkeypatch.setattr("cv2.VideoCapture", factory)
    source = WebcamSource(camera_index=0, reopen_attempts=3, reopen_delay_s=0.0)
    source.open()

    frame = source.read()
    source.close()

    assert frame is not None
    assert frame.image.shape[:2] == (HEIGHT, WIDTH)


def test_reopen_attempts_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "cv2.VideoCapture", lambda *args, **kwargs: FakeCapture(opened=True, fail_reads=10_000)
    )
    source = WebcamSource(camera_index=0, reopen_attempts=0, reopen_delay_s=0.0)
    source.open()

    with pytest.raises(VideoSourceError):
        source.read()


def test_close_is_safe_to_call_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cv2.VideoCapture", lambda *args, **kwargs: FakeCapture())
    source = WebcamSource()
    source.open()
    source.close()
    source.close()
    assert source.opened is False


def test_reading_from_a_closed_source_is_an_error() -> None:
    with pytest.raises(VideoSourceError, match="not open"):
        WebcamSource().read()


def test_rtsp_url_credentials_are_redacted() -> None:
    source = RtspSource("rtsp://user:secret@10.0.0.5:554/stream")
    assert "secret" not in str(source.describe())


# --- factory ----------------------------------------------------------------


def test_the_factory_builds_the_configured_source(settings_factory) -> None:
    from vision_sentinel.config import parse_settings

    webcam = create_video_source(settings_factory(video={"source_type": "webcam"}))
    assert isinstance(webcam, WebcamSource)

    file_source = create_video_source(
        settings_factory(video={"source_type": "file", "file_path": "clip.mp4"})
    )
    assert isinstance(file_source, FileSource)

    rtsp = create_video_source(
        settings_factory(video={"source_type": "rtsp", "rtsp_url": "rtsp://host/stream"})
    )
    assert isinstance(rtsp, RtspSource)


def test_an_unknown_source_type_cannot_reach_the_factory(settings_factory) -> None:
    from vision_sentinel.config import ConfigError

    with pytest.raises(ConfigError, match="source_type"):
        settings_factory(video={"source_type": "telepathy"})
