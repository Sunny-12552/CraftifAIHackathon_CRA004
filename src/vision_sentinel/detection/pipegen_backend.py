# SPDX-License-Identifier: Apache-2.0
"""PipeGen / DeepStream backend -- the only compliance-eligible NVIDIA path.

Unlike every other backend in this package, this class contains **no detector**.
It does not load a model, does not run inference, and cannot invent a detection.
It consumes the detections that the genuine CraftifAI PipeGen-generated
DeepStream/TensorRT pipeline publishes, and it fails loudly when that pipeline
is not running.

Why it is written this way
--------------------------
The hackathon rule is explicit: on NVIDIA the real PipeGen pipeline must produce
the perception results, and "a separate YOLO detector cannot be presented as
proof that PipeGen is running". A backend that silently fell back to a local
detector whenever the pipeline was unreachable would let a non-compliant run be
mistaken for a compliant one. So there is no fallback here at all.

Current status: BLOCKED
-----------------------
No PipeGen tooling access and no compatible NVIDIA GPU were available while
building this project, so this backend is **unverified and must not be presented
as working**. See ``evidence/pipegen/README.md`` for the exact outstanding steps.

Wire protocol
-------------
Newline-delimited JSON over TCP. The pipeline side is expected to publish one
object per processed batch::

    {"frame_id": 4171,
     "timestamp_utc": "2026-10-17T09:31:04.221Z",
     "source": "pipegen",
     "objects": [{"label": "person", "confidence": 0.91,
                  "bbox_xywh": [120, 80, 140, 310], "track_id": 7}]}

``track_id`` carries the ids assigned upstream by the DeepStream tracker (for
example NvDCF), which is why no local tracking is needed in this mode.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

import numpy as np

from ..events import Detection, utc_now_iso
from .base import Detector, DetectorError, DetectionBatch


class PipeGenUnavailableError(DetectorError):
    """Raised when the PipeGen pipeline cannot supply detections."""


#: The steps that must be completed on compatible NVIDIA hardware before this
#: backend may be enabled, or before any compliance claim can be made.
PIPEGEN_CHECKLIST = (
    "1. Install the official CraftifAI PipeGen version and its confirmed "
    "prerequisites on a compatible NVIDIA Linux machine.",
    "2. Sign in to the official CraftifAI tool.",
    "3. Complete the Requirement stage from the project specification.",
    "4. Generate and export the Pipeline IR graph.",
    "5. Configure the model in the Model stage.",
    "6. Compile through PipeGen to a genuine TensorRT engine for the target GPU.",
    "7. Run the generated pipeline live and publish detection JSON to "
    "detection.pipegen_endpoint.",
    "8. Save the PipeGen export, Pipeline IR graph, speed report, TensorRT "
    "engine details and runtime log under evidence/pipegen/.",
)

#: How long a published batch stays fresh. Older data is treated as a stalled
#: pipeline rather than as an empty frame, because "no fresh detections" must
#: never be silently interpreted as "the zone is empty".
DEFAULT_STALE_AFTER_S = 5.0


class PipeGenBackend(Detector):
    """Consumes detections published by the real PipeGen pipeline."""

    backend_id = "pipegen"
    compliance_eligible = True
    #: The generated pipeline owns the camera; a local capture is display-only.
    owns_video_source = False

    def __init__(
        self,
        endpoint: str,
        confidence_threshold: float = 0.35,
        stale_after_s: float = DEFAULT_STALE_AFTER_S,
        connect_timeout_s: float = 5.0,
    ) -> None:
        if not endpoint.strip():
            raise PipeGenUnavailableError(
                "No PipeGen endpoint is configured (detection.pipegen_endpoint "
                "or VS_PIPEGEN_ENDPOINT).\n"
                "The PipeGen backend reads detections from a real generated "
                "pipeline; it cannot run without one.\nOutstanding steps:\n  "
                + "\n  ".join(PIPEGEN_CHECKLIST)
            )

        host, _, port_text = endpoint.rpartition(":")
        if not host or not port_text.isdigit():
            raise PipeGenUnavailableError(
                f"PipeGen endpoint {endpoint!r} must be in 'host:port' form."
            )

        self.endpoint = endpoint
        self.host = host
        self.port = int(port_text)
        self.confidence_threshold = confidence_threshold
        self.stale_after_s = stale_after_s
        self.device_name = f"PipeGen pipeline at {endpoint}"

        self._lock = threading.Lock()
        self._latest: DetectionBatch | None = None
        self._latest_at = 0.0
        self._socket: socket.socket | None = None
        self._buffer = b""
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._connect(connect_timeout_s)

        self._reader = threading.Thread(
            target=self._read_loop, name="pipegen-reader", daemon=True
        )
        self._reader.start()

    # -- lifecycle ------------------------------------------------------
    def _connect(self, timeout_s: float) -> None:
        try:
            sock = socket.create_connection((self.host, self.port), timeout=timeout_s)
        except OSError as error:
            raise PipeGenUnavailableError(
                f"Could not connect to the PipeGen pipeline at {self.endpoint}: "
                f"{error}\n"
                "Start the generated pipeline first, then retry.\n"
                "Outstanding steps:\n  " + "\n  ".join(PIPEGEN_CHECKLIST)
            ) from error
        sock.setblocking(False)
        self._socket = sock

    def close(self) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=2.0)
            self._reader = None
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

    # -- ingestion ------------------------------------------------------
    def _read_loop(self) -> None:
        while not self._stop.is_set():
            sock = self._socket
            if sock is None:
                return
            try:
                chunk = sock.recv(65536)
            except BlockingIOError:
                time.sleep(0.01)
                continue
            except OSError:
                return
            if not chunk:
                time.sleep(0.05)
                continue
            self._buffer += chunk
            while b"\n" in self._buffer:
                line, _, self._buffer = self._buffer.partition(b"\n")
                if line.strip():
                    self._ingest(line)

    def _ingest(self, line: bytes) -> None:
        try:
            payload: Any = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return  # Ignore malformed lines rather than corrupt the state.
        if not isinstance(payload, dict):
            return
        objects = payload.get("objects")
        if not isinstance(objects, list):
            return

        detections: list[Detection] = []
        track_ids: list[str | None] = []
        for entry in objects:
            if not isinstance(entry, dict):
                continue
            try:
                confidence = float(entry["confidence"])
                bbox = entry["bbox_xywh"]
                label = str(entry["label"])
            except (KeyError, TypeError, ValueError):
                continue
            if confidence < self.confidence_threshold:
                continue
            try:
                box = tuple(int(value) for value in bbox)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
                continue
            raw_track = entry.get("track_id")
            detections.append(
                Detection(
                    class_id=int(entry.get("class_id", 0) or 0),
                    label=label,
                    confidence=confidence,
                    bbox_xywh=box,  # type: ignore[arg-type]
                )
            )
            track_ids.append(None if raw_track is None else str(raw_track))

        batch = DetectionBatch(
            detections=tuple(detections),
            track_ids=tuple(track_ids),
            frame_timestamp_utc=payload.get("timestamp_utc"),
            sequence=int(payload.get("frame_id", 0) or 0),
        )
        with self._lock:
            self._latest = batch
            self._latest_at = time.monotonic()

    # -- detector interface ---------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return list(self.detect_tracked(frame).detections)

    def detect_tracked(self, frame: np.ndarray) -> DetectionBatch:
        """Return the newest detection batch published by the pipeline.

        ``frame`` is deliberately unused: in this mode the pipeline owns the
        camera and a local capture exists only so the operator can see what the
        pipeline is looking at.
        """
        with self._lock:
            batch = self._latest
            age = time.monotonic() - self._latest_at if batch is not None else None

        if batch is None:
            raise PipeGenUnavailableError(
                "Connected to the PipeGen pipeline but no detection batch has "
                "been published yet. Confirm the generated pipeline is running "
                "and that its output is wired to this socket."
            )
        if age is not None and age > self.stale_after_s:
            raise PipeGenUnavailableError(
                f"The PipeGen pipeline has published nothing for {age:.1f}s "
                f"(limit {self.stale_after_s:.1f}s). The pipeline has probably "
                "stopped; zone evaluation is paused rather than being reported "
                "as an empty zone."
            )
        return batch

    def describe(self) -> dict[str, str]:
        return {
            "backend": self.backend_id,
            "device": self.device_name,
            "endpoint": self.endpoint,
            "compliance": "PipeGen pipeline output (unverified in this build)",
        }


def publish_help() -> str:
    """Return the human-readable list of outstanding PipeGen steps."""
    return "\n".join(f"  {step}" for step in PIPEGEN_CHECKLIST)


def latest_timestamp() -> str:
    """Timestamp used when a batch carries none of its own."""
    return utc_now_iso()
