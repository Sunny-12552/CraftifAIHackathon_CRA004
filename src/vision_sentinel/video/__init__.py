# SPDX-License-Identifier: Apache-2.0
"""Video capture: webcam, USB camera, video file and RTSP."""

from __future__ import annotations

from .source import (
    FileSource,
    Frame,
    RtspSource,
    VideoSource,
    VideoSourceError,
    WebcamSource,
    create_video_source,
)

__all__ = [
    "FileSource",
    "Frame",
    "RtspSource",
    "VideoSource",
    "VideoSourceError",
    "WebcamSource",
    "create_video_source",
]
