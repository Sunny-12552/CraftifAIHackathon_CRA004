# SPDX-License-Identifier: Apache-2.0
"""The orchestration engine: one pipeline, two execution paths.

    video source -> detection backend -> shared detection events -> tracking
    -> zone monitoring -> alert manager -> SQLite -> JSON/MQTT -> UI

The engine holds no UI code and no hardware assumptions. It can be driven
synchronously (the native OpenCV viewer does this in the main thread) or on a
background thread with the dashboard reading :meth:`SentinelEngine.snapshot`.

Everything the engine touches -- the detector, the video source, the tracker,
the zone monitor, the alert manager and the store -- can be injected, which is
what makes the end-to-end acceptance test possible without a camera or a GPU.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .alerts import AlertManager
from .config import Settings
from .detection import Detector, DetectorError, DetectorResolution, create_detector
from .events import Detection, DetectionEvent, Incident, ZoneEvent, utc_now_iso
from .metrics import PerformanceReport
from .render import OverlayOptions, encode_jpeg, render_frame
from .storage import IncidentStore, StorageError
from .tracking import IouTracker, Tracker, create_tracker
from .video import Frame, VideoSource, VideoSourceError, create_video_source
from .zones import PolygonZone, ZoneMonitor, build_zones, resolve_cached


class EngineStatus(str, Enum):
    """Lifecycle state reported to the UI."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True, slots=True, eq=False)
class FrameResult:
    """Everything that happened for one processed frame."""

    frame_id: int
    timestamp_utc: str
    detections: tuple[Detection, ...] = ()
    track_ids: tuple[str | None, ...] = ()
    events: tuple[ZoneEvent, ...] = ()
    incidents: tuple[Incident, ...] = ()
    suppressed: tuple[ZoneEvent, ...] = ()
    occupancy: Mapping[str, int] = field(default_factory=dict)
    inference_ms: float = 0.0
    zone_ms: float = 0.0
    total_ms: float = 0.0
    annotated: np.ndarray | None = None
    error: str | None = None

    @property
    def detection_events(self) -> tuple[DetectionEvent, ...]:
        """Detections in the shared event schema, with track ids attached."""
        stamp = self.timestamp_utc
        return tuple(
            DetectionEvent(
                camera_id="",
                frame_id=self.frame_id,
                timestamp_utc=stamp,
                label=detection.label,
                confidence=detection.confidence,
                bbox_xywh=detection.bbox_xywh,
                track_id=self.track_ids[index] if index < len(self.track_ids) else None,
                class_id=detection.class_id,
            )
            for index, detection in enumerate(self.detections)
        )


@dataclass(frozen=True, slots=True, eq=False)
class EngineSnapshot:
    """Thread-safe view of the engine for the UI layer."""

    status: str
    timestamp_utc: str
    backend: str = ""
    device: str = ""
    source: str = ""
    compliance_eligible: bool = False
    frame_id: int = 0
    frame: np.ndarray | None = None
    fps: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0
    latency: Mapping[str, Any] = field(default_factory=dict)
    detections: tuple[DetectionEvent, ...] = ()
    occupancy: Mapping[str, int] = field(default_factory=dict)
    active_alerts: tuple[Incident, ...] = ()
    incident_count: int = 0
    suppressed_duplicates: int = 0
    frames_processed: int = 0
    error: str | None = None
    notes: tuple[str, ...] = ()

    def jpeg(self, quality: int = 80) -> bytes | None:
        """Encode the current frame for the dashboard."""
        if self.frame is None:
            return None
        return encode_jpeg(self.frame, quality=quality)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view, excluding the pixel buffer."""
        return {
            "status": self.status,
            "timestamp_utc": self.timestamp_utc,
            "backend": self.backend,
            "device": self.device,
            "source": self.source,
            "compliance_eligible": self.compliance_eligible,
            "frame_id": self.frame_id,
            "fps": round(self.fps, 2),
            "inference_ms": round(self.inference_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "latency": dict(self.latency),
            "detections": [event.to_dict() for event in self.detections],
            "occupancy": dict(self.occupancy),
            "active_alerts": [incident.to_dict() for incident in self.active_alerts],
            "incident_count": self.incident_count,
            "suppressed_duplicates": self.suppressed_duplicates,
            "frames_processed": self.frames_processed,
            "error": self.error,
            "notes": list(self.notes),
        }


class SentinelEngine:
    """Owns the perception-to-incident pipeline for one camera."""

    def __init__(
        self,
        settings: Settings,
        *,
        store: IncidentStore | None = None,
        detector: Detector | DetectorResolution | None = None,
        source: VideoSource | None = None,
        tracker: Tracker | None = None,
        monitor: ZoneMonitor | None = None,
        alerts: AlertManager | None = None,
        allow_download: bool = True,
        render: bool = True,
        overlay_options: OverlayOptions | None = None,
    ) -> None:
        self.settings = settings
        self.render_enabled = render
        self.overlay_options = overlay_options or OverlayOptions()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = EngineStatus.STARTING
        self._error: str | None = None
        self._snapshot: EngineSnapshot | None = None
        self._snapshot_frame: np.ndarray | None = None
        self._pending_frame: np.ndarray | None = None

        # --- detection -------------------------------------------------
        self.notes: list[str] = []
        self._attempts: tuple[Any, ...] = ()
        if detector is None:
            resolution = create_detector(settings, allow_download=allow_download)
            self.detector = resolution.detector
            self._attempts = resolution.attempts
            self.notes.extend(resolution.warnings)
        elif isinstance(detector, DetectorResolution):
            self.detector = detector.detector
            self._attempts = detector.attempts
            self.notes.extend(detector.warnings)
        else:
            self.detector = detector

        # --- video -----------------------------------------------------
        self.source = source if source is not None else create_video_source(settings)

        # --- tracking --------------------------------------------------
        self.tracker: Tracker = tracker if tracker is not None else create_tracker(settings)

        # --- zones -----------------------------------------------------
        self.zones: tuple[PolygonZone, ...] = build_zones(settings.zones)
        self.monitor = (
            monitor
            if monitor is not None
            else ZoneMonitor(
                self.zones,
                enter_frames=settings.alerts.enter_frames,
                exit_frames=settings.alerts.exit_frames,
                min_confidence=settings.alerts.min_confidence,
            )
        )

        # --- storage ---------------------------------------------------
        self._owns_store = store is None
        self.store: IncidentStore | None = (
            store
            if store is not None
            else IncidentStore(
                settings.storage.resolved_db_path(),
                retention_days=settings.storage.retention_days,
            )
        )

        # --- alerts ----------------------------------------------------
        self.alerts = (
            alerts
            if alerts is not None
            else AlertManager(
                cooldown_seconds=settings.alerts.cooldown_seconds,
                record_exits=settings.alerts.record_exits,
                sink=self.store,
                snapshot_writer=self._write_snapshot if settings.storage.save_snapshots else None,
            )
        )

        # --- metrics ---------------------------------------------------
        self.metrics = PerformanceReport(
            window=settings.metrics.window,
            report_path=settings.metrics.resolved_report_path(),
        )
        self.metrics.started_utc = utc_now_iso()

        if self.store is not None and settings.storage.retention_days:
            try:
                self.store.prune()
            except StorageError as error:
                self.notes.append(f"Retention pruning failed: {error}")

        if not getattr(self.detector, "compliance_eligible", False) and self.detector.backend_id == "nvidia_ort":
            self.notes.append(
                "onnxruntime CUDA inference is a development convenience and is "
                "not CraftifAI PipeGen. It does not satisfy the Track A "
                "requirements."
            )

    # -- properties ------------------------------------------------------
    @property
    def status(self) -> EngineStatus:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def backend_id(self) -> str:
        return getattr(self.detector, "backend_id", "unknown")

    @property
    def device_name(self) -> str:
        return getattr(self.detector, "device_name", "unknown")

    @property
    def compliance_eligible(self) -> bool:
        return bool(getattr(self.detector, "compliance_eligible", False))

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        """Open the source (if the backend needs one) and start the worker."""
        if self.is_running:
            return
        if getattr(self.detector, "owns_video_source", True):
            self.source.open()
        self._stop_event.clear()
        self._status = EngineStatus.RUNNING
        self._thread = threading.Thread(target=self._loop, name="sentinel-engine", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and wait for it."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        if self._status is not EngineStatus.ERROR:
            self._set_status(EngineStatus.STOPPED)

    def _loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                frame = self.source.read()
                if frame is None:
                    # A finite source (video file) has ended.
                    self._set_status(EngineStatus.STOPPED)
                    break
                self.process_frame(frame)
                limit = self.settings.video.fps_limit
                if limit and limit > 0:
                    time.sleep(max(0.0, 1.0 / limit - 0.001))
        except VideoSourceError as error:
            self._set_error(str(error))
        except Exception as error:  # pragma: no cover - safety net
            self._set_error(f"Unexpected engine failure: {error}")
        finally:
            if self._status is not EngineStatus.ERROR:
                self._set_status(EngineStatus.STOPPED)

    def run_forever(
        self,
        on_frame: Callable[[FrameResult], bool | None] | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> None:
        """Run synchronously in the calling thread.

        Used by the native OpenCV viewer, which needs to draw and handle keys in
        the main thread. ``on_frame`` may return ``False`` to stop the loop.
        """
        self.source.open()
        self._status = EngineStatus.RUNNING
        try:
            while True:
                if stop_check is not None and stop_check():
                    break
                frame = self.source.read()
                if frame is None:
                    break
                result = self.process_frame(frame)
                if on_frame is not None and on_frame(result) is False:
                    break
        finally:
            self.source.close()
            if self._status is not EngineStatus.ERROR:
                self._set_status(EngineStatus.STOPPED)

    # -- the pipeline ----------------------------------------------------
    def process_frame(self, frame: Frame) -> FrameResult:
        """Run one frame through detection, tracking, zones and alerts."""
        started = time.perf_counter()
        self.metrics.fps.tick()

        inference_ms = 0.0
        detections: tuple[Detection, ...] = ()
        track_ids: list[str | None] = []
        error: str | None = None

        # --- detection --------------------------------------------------
        detection_started = time.perf_counter()
        try:
            batch = self.detector.detect_tracked(frame.image)
        except DetectorError as detector_error:
            # A paused or unreachable backend must not be reported as an empty
            # scene: that would fabricate a "zone cleared" transition. Detection
            # is skipped and zone evaluation is paused for this frame.
            error = str(detector_error)
            batch = None
        inference_ms = (time.perf_counter() - detection_started) * 1000.0

        if batch is not None:
            detections = batch.detections
            if batch.track_ids:
                track_ids = list(batch.ids())
            elif isinstance(self.tracker, IouTracker):
                track_ids = self.tracker.update(detections, frame.frame_id)
            else:
                track_ids = [None] * len(detections)

            expired = (
                self.tracker.expired_last_update()
                if isinstance(self.tracker, IouTracker)
                else ()
            )
            if expired:
                self.monitor.forget(expired)

        # --- zones and alerts ------------------------------------------
        zone_started = time.perf_counter()
        events: tuple[ZoneEvent, ...] = ()
        incidents: tuple[Incident, ...] = ()
        suppressed: tuple[ZoneEvent, ...] = ()
        occupancy = self.monitor.occupancy()

        if batch is not None and self.monitor.enabled_zones():
            events = tuple(
                self.monitor.update(
                    detections,
                    track_ids,
                    frame_id=frame.frame_id,
                    frame_width=frame.width,
                    frame_height=frame.height,
                    timestamp_utc=frame.timestamp_utc,
                    camera_id=self.settings.video.camera_id,
                    source=self.detector.backend_id,
                )
            )
            if events:
                self._pending_frame = frame.image
                outcome = self.alerts.handle(events)
                incidents = outcome.incidents
                suppressed = outcome.suppressed
            occupancy = self.monitor.occupancy()
        zone_ms = (time.perf_counter() - zone_started) * 1000.0

        # --- rendering --------------------------------------------------
        annotated: np.ndarray | None = None
        if self.render_enabled:
            resolved = [
                resolve_cached(zone, frame.width, frame.height)
                for zone in self.monitor.enabled_zones()
            ]
            intruders = {incident.track_id for incident in self.alerts.active_alerts()}
            annotated = render_frame(
                frame.image,
                zones=resolved,
                detections=detections,
                track_ids=track_ids,
                occupied_zone_ids={
                    zone_id for zone_id, count in occupancy.items() if count > 0
                },
                intruder_track_ids=intruders,
                alerts=self.alerts.active_count,
                fps=self.metrics.fps.value,
                inference_ms=inference_ms,
                device=self.device_name,
                error=error,
                options=self.overlay_options,
            )
            self._snapshot_frame = annotated

        total_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.record_frame(
            inference_ms=inference_ms, zone_ms=zone_ms, total_ms=total_ms
        )

        result = FrameResult(
            frame_id=frame.frame_id,
            timestamp_utc=frame.timestamp_utc,
            detections=detections,
            track_ids=tuple(track_ids),
            events=events,
            incidents=incidents,
            suppressed=suppressed,
            occupancy=occupancy,
            inference_ms=inference_ms,
            zone_ms=zone_ms,
            total_ms=total_ms,
            annotated=annotated,
            error=error,
        )
        self._publish(result, frame)
        return result

    # -- snapshot --------------------------------------------------------
    def _publish(self, result: FrameResult, frame: Frame) -> None:
        settings = self.settings
        detection_events = tuple(
            DetectionEvent(
                camera_id=settings.video.camera_id,
                frame_id=result.frame_id,
                timestamp_utc=result.timestamp_utc,
                label=detection.label,
                confidence=detection.confidence,
                bbox_xywh=detection.bbox_xywh,
                track_id=result.track_ids[index] if index < len(result.track_ids) else None,
                source=self.detector.backend_id,
                class_id=detection.class_id,
            )
            for index, detection in enumerate(result.detections)
        )
        summary = self.metrics.stages.get("inference_ms")
        latency = summary.latency.summary().to_dict() if summary else {}

        snapshot = EngineSnapshot(
            status=self._status.value,
            timestamp_utc=utc_now_iso(),
            backend=self.detector.backend_id,
            device=self.device_name,
            source=self._source_description(),
            compliance_eligible=self.compliance_eligible,
            frame_id=result.frame_id,
            frame=result.annotated,
            fps=self.metrics.fps.value,
            inference_ms=result.inference_ms,
            total_ms=result.total_ms,
            latency=latency,
            detections=detection_events,
            occupancy=dict(result.occupancy),
            active_alerts=self.alerts.active_alerts(),
            incident_count=len(self.alerts.history()),
            suppressed_duplicates=self.alerts.suppressed_total,
            frames_processed=self.metrics.frames,
            error=result.error,
            notes=tuple(self.notes),
        )
        with self._lock:
            self._snapshot = snapshot

    def snapshot(self) -> EngineSnapshot:
        """Return the most recent engine state."""
        with self._lock:
            if self._snapshot is not None:
                return self._snapshot
        return EngineSnapshot(
            status=self._status.value,
            timestamp_utc=utc_now_iso(),
            backend=self.backend_id,
            device=self.device_name,
            source=self._source_description(),
            compliance_eligible=self.compliance_eligible,
            notes=tuple(self.notes),
            error=self._error,
        )

    # -- reporting -------------------------------------------------------
    def write_performance_report(self, path: str | Path | None = None) -> Path:
        """Write a runtime-measured performance report (AT-12)."""
        context = {
            "backend": self.detector.backend_id,
            "device": self.device_name,
            "camera_id": self.settings.video.camera_id,
            "source": self.source.describe(),
            "original_frame_size": [
                self.settings.video.width,
                self.settings.video.height,
            ],
            "compliance_eligible": self.compliance_eligible,
            "zones": [zone.zone_id for zone in self.zones],
            "enter_frames": self.settings.alerts.enter_frames,
            "exit_frames": self.settings.alerts.exit_frames,
            "cooldown_seconds": self.settings.alerts.cooldown_seconds,
            "incidents": len(self.alerts.history()),
            "suppressed_duplicates": self.alerts.suppressed_total,
            "notes": list(self.notes),
        }
        return self.metrics.write(path, context)

    def incidents_json(self, limit: int | None = None) -> str:
        """Serialise the in-process incident history (dashboard download helper)."""
        history = self.alerts.history()
        if limit is not None:
            history = history[-limit:]
        return json.dumps([incident.to_dict() for incident in history], indent=2)

    # -- internals -------------------------------------------------------
    def _source_description(self) -> str:
        """Short, human-readable description of the active video input."""
        info = self.source.describe()
        kind = info.get("type", "source")
        if kind == "webcam":
            return f"webcam index {info.get('index')}"
        if kind == "file":
            return f"video file {Path(str(info.get('path'))).name}"
        if kind == "rtsp":
            return f"RTSP {info.get('url')}"
        return kind

    def _write_snapshot(self, event: ZoneEvent) -> str | None:
        """Save the frame that triggered an entry, when snapshots are enabled."""
        frame = self._pending_frame
        if frame is None:
            return None
        directory = self.settings.storage.resolved_snapshot_dir()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = event.timestamp_utc.replace(":", "").replace("-", "")
        target = directory / f"{stamp}_{event.zone_id}_{event.track_id or 'anon'}.jpg"
        try:
            import cv2

            ok = cv2.imwrite(str(target), frame)
        except Exception:
            return None
        return str(target) if ok else None

    def _set_status(self, status: EngineStatus) -> None:
        """Update the lifecycle state and reflect it in the published snapshot.

        The dashboard reads the cached snapshot, so a status change that only
        updated ``self._status`` would leave it showing a stale state after a
        finite source ended or the worker stopped.
        """
        self._status = status
        with self._lock:
            base = self._snapshot
            if base is not None:
                self._snapshot = replace(
                    base, status=status.value, timestamp_utc=utc_now_iso()
                )

    def _set_error(self, message: str) -> None:
        self._error = message
        self._status = EngineStatus.ERROR
        with self._lock:
            base = self._snapshot
        self._snapshot = EngineSnapshot(
            status=EngineStatus.ERROR.value,
            timestamp_utc=utc_now_iso(),
            backend=self.backend_id,
            device=self.device_name,
            source=self._source_description(),
            compliance_eligible=self.compliance_eligible,
            frame=base.frame if base else None,
            frame_id=base.frame_id if base else 0,
            fps=self.metrics.fps.value,
            latency=base.latency if base else {},
            detections=base.detections if base else (),
            occupancy=base.occupancy if base else {},
            active_alerts=self.alerts.active_alerts(),
            incident_count=len(self.alerts.history()),
            suppressed_duplicates=self.alerts.suppressed_total,
            frames_processed=self.metrics.frames,
            error=message,
            notes=tuple(self.notes),
        )

    def set_zones(self, zones: Sequence[PolygonZone]) -> None:
        """Replace the restricted zones at runtime (dashboard zone editor)."""
        self.zones = tuple(zones)
        self.monitor.set_zones(self.zones)

    def close(self) -> None:
        """Stop the engine and release the source and store."""
        self.stop()
        self.source.close()
        try:
            self.detector.close()
        except Exception:
            pass
        if self._owns_store and self.store is not None:
            self.store.close()
