# SPDX-License-Identifier: Apache-2.0
"""Video input: laptop webcam, USB camera, video file, and RTSP.

All sources present the same ``open()/read()/close()`` interface. Read failures
are retried with a bounded number of reopen attempts before raising, so a
temporarily busy camera does not kill the session and a genuinely disconnected
one produces an actionable message instead of a crash (AT-09).

Every frame is stamped with a UTC wall-clock time and a monotonic frame id, so
detections from a live webcam and from a PipeGen pipeline are directly
comparable in the incident log.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Settings
from ..events import utc_now_iso


class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be opened or has failed terminally."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One captured frame."""

    image: np.ndarray
    frame_id: int
    timestamp_utc: str
    media_time_s: float = 0.0
    is_last: bool = False

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


class VideoSource(ABC):
    """Common contract for every video input."""

    def __init__(self, camera_id: str = "cam-0") -> None:
        self.camera_id = camera_id
        self._frame_id = 0
        self._opened = False

    @property
    def opened(self) -> bool:
        return self._opened

    @property
    def frame_id(self) -> int:
        return self._frame_id

    #: True for finite sources such as a video file.
    finite: bool = False

    @abstractmethod
    def open(self) -> None:
        """Open the source or raise :class:`VideoSourceError`."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or ``None`` when a finite source ends."""

    @abstractmethod
    def close(self) -> None:
        """Release the source. Safe to call more than once."""

    def describe(self) -> dict[str, Any]:
        return {"camera_id": self.camera_id, "type": type(self).__name__}

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _resolve_file_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        from ..paths import PROJECT_ROOT

        path = PROJECT_ROOT / path
    return path


class _OpenCVSource(VideoSource):
    """Shared OpenCV capture handling, including bounded reopen retries."""

    def __init__(
        self,
        camera_id: str = "cam-0",
        width: int = 640,
        height: int = 480,
        reopen_attempts: int = 3,
        reopen_delay_s: float = 1.0,
    ) -> None:
        super().__init__(camera_id)
        self.width = width
        self.height = height
        self.reopen_attempts = reopen_attempts
        self.reopen_delay_s = reopen_delay_s
        self._capture: Any = None
        self._failures = 0

    # -- to override -----------------------------------------------------
    def _create_capture(self) -> Any:
        raise NotImplementedError

    def _configure(self, capture: Any) -> None:
        if self.width > 0:
            capture.set(3, self.width)  # CAP_PROP_FRAME_WIDTH
        if self.height > 0:
            capture.set(4, self.height)  # CAP_PROP_FRAME_HEIGHT

    def _open_error(self) -> VideoSourceError:
        return VideoSourceError(f"Could not open {self.describe()}")

    # -- lifecycle -------------------------------------------------------
    def open(self) -> None:
        if self._opened:
            return
        capture = self._create_capture()
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            raise self._open_error()
        self._capture = capture
        self._configure(capture)
        # ``_failures`` is deliberately NOT reset here. ``read`` reopens the
        # source itself, so a counter that ``open`` cleared would reset on every
        # retry and the attempt limit below could never be reached. It is reset
        # only when a frame is actually read successfully.
        self._opened = True

    def read(self) -> Frame | None:
        if not self._opened or self._capture is None:
            raise VideoSourceError("The video source is not open")

        while True:
            ok, image = self._capture.read()
            if ok and image is not None and getattr(image, "size", 0):
                self._failures = 0
                self._frame_id += 1
                return Frame(
                    image=image,
                    frame_id=self._frame_id,
                    timestamp_utc=utc_now_iso(),
                    media_time_s=self._media_time(),
                )

            # A read failure on a finite source means end of stream.
            if self.finite:
                self._frame_id += 1
                return None

            self._failures += 1
            if self._failures > self.reopen_attempts:
                raise VideoSourceError(
                    f"{self.describe()} stopped returning frames after "
                    f"{self.reopen_attempts} reopen attempts. Check that the "
                    "camera is still connected and not held by another "
                    "application."
                )
            self.close()
            time.sleep(max(self.reopen_delay_s, 0.0))
            try:
                self.open()
            except VideoSourceError:
                # Keep counting failures; the loop will either recover or hit
                # the attempt limit and raise the message above.
                continue

    def _media_time(self) -> float:
        try:
            value = self._capture.get(5)  # CAP_PROP_POS_MSEC
            return float(value) / 1000.0 if value and value > 0 else 0.0
        except Exception:
            return 0.0

    def close(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
        self._opened = False

    def _configure_common_properties(self, capture: Any) -> None:  # pragma: no cover
        capture.set(38, 1)  # CAP_PROP_BUFFERSIZE


class WebcamSource(_OpenCVSource):
    """A laptop or USB camera."""

    def __init__(
        self,
        camera_index: int = 0,
        camera_id: str | None = None,
        width: int = 640,
        height: int = 480,
        reopen_attempts: int = 3,
        reopen_delay_s: float = 1.0,
        prefer_directshow: bool = True,
    ) -> None:
        super().__init__(
            camera_id or f"cam-{camera_index}",
            width,
            height,
            reopen_attempts,
            reopen_delay_s,
        )
        self.camera_index = camera_index
        self.prefer_directshow = prefer_directshow

    def _create_capture(self) -> Any:
        import cv2

        if self.prefer_directshow:
            backend = getattr(cv2, "CAP_DSHOW", 0)
            if backend:
                capture = cv2.VideoCapture(self.camera_index, backend)
                if capture.isOpened():
                    return capture
                capture.release()
        return cv2.VideoCapture(self.camera_index)

    def _open_error(self) -> VideoSourceError:
        return VideoSourceError(
            f"Could not open webcam index {self.camera_index}. Check the Windows "
            "camera privacy settings, make sure no other application is using the "
            f"camera, or try another index "
            f"(currently {self.camera_index}); other indices to try: "
            f"{self.camera_index + 1}."
        )

    def describe(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "type": "webcam",
            "index": self.camera_index,
        }


class FileSource(_OpenCVSource):
    """A video file. Ends instead of failing when the file is exhausted."""

    finite = True

    def __init__(
        self,
        path: str | Path,
        camera_id: str = "file-0",
        width: int = 0,
        height: int = 0,
        loop: bool = False,
    ) -> None:
        super().__init__(camera_id, width, height, reopen_attempts=0, reopen_delay_s=0.0)
        self.path = _resolve_file_path(str(path))
        self.loop = loop

    def open(self) -> None:
        if not self.path.exists():
            raise VideoSourceError(
                f"Video file not found: {self.path}\nCheck the path, or set "
                "video.file_path in config/settings.yaml."
            )
        if not self.path.is_file():
            raise VideoSourceError(f"Video path is not a file: {self.path}")
        super().open()

    def _create_capture(self) -> Any:
        import cv2

        return cv2.VideoCapture(str(self.path))

    def _configure(self, capture: Any) -> None:
        # A file must not be resized by the capture layer; its own resolution is
        # what the zone coordinates were authored against.
        return

    def _open_error(self) -> VideoSourceError:
        return VideoSourceError(
            f"OpenCV could not decode the video file {self.path}. Check the "
            "format and that the file is not corrupt."
        )

    def read(self) -> Frame | None:
        frame = super().read()
        if frame is None and self.loop:
            self.close()
            self.open()
            return super().read()
        if frame is not None and self.finite:
            # Mark the final frame so the caller can stop cleanly.
            position = self._capture.get(1) if self._capture else 0  # CAP_PROP_POS_FRAMES
            total = self._capture.get(7) if self._capture else 0  # CAP_PROP_FRAME_COUNT
            if total and position >= total:
                return Frame(
                    image=frame.image,
                    frame_id=frame.frame_id,
                    timestamp_utc=frame.timestamp_utc,
                    media_time_s=frame.media_time_s,
                    is_last=True,
                )
        return frame

    def describe(self) -> dict[str, Any]:
        return {"camera_id": self.camera_id, "type": "file", "path": str(self.path)}


class RtspSource(_OpenCVSource):
    """An RTSP stream. Requires the backend to support the URL."""

    def __init__(
        self,
        url: str,
        camera_id: str = "rtsp-0",
        width: int = 1280,
        height: int = 720,
        reopen_attempts: int = 5,
        reopen_delay_s: float = 2.0,
    ) -> None:
        super().__init__(camera_id, width, height, reopen_attempts, reopen_delay_s)
        self.url = url

    def _create_capture(self) -> Any:
        import cv2

        capture = cv2.VideoCapture(self.url, getattr(cv2, "CAP_FFMPEG", 0))
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.url)
        try:
            capture.set(38, 1)  # CAP_PROP_BUFFERSIZE, keeps latency low
        except Exception:
            pass
        return capture

    def _open_error(self) -> VideoSourceError:
        redacted = self.url.split("@")[-1] if "@" in self.url else self.url
        return VideoSourceError(
            f"Could not open the RTSP stream {redacted}. Check the URL, "
            "credentials and that the camera is reachable from this machine."
        )

    def describe(self) -> dict[str, Any]:
        redacted = self.url.split("@")[-1] if "@" in self.url else self.url
        return {"camera_id": self.camera_id, "type": "rtsp", "url": redacted}


def create_video_source(settings: Settings) -> VideoSource:
    """Build the configured video source."""
    video = settings.video
    if video.source_type == "webcam":
        return WebcamSource(
            camera_index=video.camera_index,
            camera_id=video.camera_id,
            width=video.width,
            height=video.height,
            reopen_attempts=video.reopen_attempts,
            reopen_delay_s=video.reopen_delay_s,
        )
    if video.source_type == "file":
        return FileSource(video.file_path, camera_id=video.camera_id)
    if video.source_type == "rtsp":
        return RtspSource(
            video.rtsp_url,
            camera_id=video.camera_id,
            reopen_attempts=video.reopen_attempts,
            reopen_delay_s=video.reopen_delay_s,
        )
    raise VideoSourceError(
        f"Unsupported video source type {video.source_type!r}; expected one of "
        "webcam, file, rtsp"
    )
