# SPDX-License-Identifier: Apache-2.0
"""Configuration loading, validation and persistence.

Configuration lives in ``config/settings.yaml`` and is validated eagerly: an
invalid threshold, an unusable video source selection or a degenerate polygon
must fail with a precise message *before* a camera is opened (FR-10).

A small set of environment variables can override the values that most often
differ between the Intel laptop and the NVIDIA machine; see ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .geometry import validate_polygon_shape
from .paths import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_SETTINGS_PATH,
    EVIDENCE_DIR,
    resolve_path,
)

DEFAULT_MODEL_FILENAME = "object_detection_yolox_2022nov.onnx"

VIDEO_SOURCE_TYPES = ("webcam", "file", "rtsp")
#: ``nvidia_ort`` is intentionally absent -- it must be requested explicitly.
AUTO_BACKENDS = ("intel", "cpu")
DETECTION_BACKENDS = ("auto", "intel", "cpu", "nvidia_ort", "pipegen")


class ConfigError(ValueError):
    """Raised when the configuration is missing, malformed or contradictory."""


@dataclass(frozen=True, slots=True)
class VideoConfig:
    source_type: str = "webcam"
    camera_id: str = "cam-0"
    camera_index: int = 0
    file_path: str = ""
    rtsp_url: str = ""
    width: int = 640
    height: int = 480
    fps_limit: float = 0.0
    reopen_attempts: int = 3
    reopen_delay_s: float = 1.0


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    backend: str = "auto"
    model_path: str = ""
    model_url: str = ""
    confidence_threshold: float = 0.35
    nms_threshold: float = 0.45
    model_size: int = 640
    strides: tuple[int, ...] = (8, 16, 32)
    pipegen_endpoint: str = ""  # host:port of the real PipeGen/DeepStream sink

    def resolved_model_path(self) -> Path:
        default = DEFAULT_MODELS_DIR / DEFAULT_MODEL_FILENAME
        return resolve_path(self.model_path, default)


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    enabled: bool = True
    iou_threshold: float = 0.30
    max_age: int = 15
    min_hits: int = 1


@dataclass(frozen=True, slots=True)
class ZoneConfig:
    zone_id: str
    name: str
    polygon_normalized: tuple[tuple[float, float], ...]
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.zone_id,
            "name": self.name,
            "enabled": self.enabled,
            "polygon_normalized": [[x, y] for x, y in self.polygon_normalized],
        }


@dataclass(frozen=True, slots=True)
class AlertConfig:
    enter_frames: int = 3
    exit_frames: int = 5
    cooldown_seconds: float = 20.0
    min_confidence: float = 0.35
    record_exits: bool = True


@dataclass(frozen=True, slots=True)
class StorageConfig:
    db_path: str = ""
    export_dir: str = ""
    save_snapshots: bool = False
    snapshot_dir: str = ""
    retention_days: int = 0

    def resolved_db_path(self) -> Path:
        return resolve_path(self.db_path, DEFAULT_DATA_DIR / "incidents.db")

    def resolved_export_dir(self) -> Path:
        return resolve_path(self.export_dir, DEFAULT_DATA_DIR / "exports")

    def resolved_snapshot_dir(self) -> Path:
        return resolve_path(self.snapshot_dir, DEFAULT_DATA_DIR / "snapshots")


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    window: int = 120
    report_path: str = ""

    def resolved_report_path(self) -> Path:
        return resolve_path(self.report_path, EVIDENCE_DIR / "performance_report.json")


@dataclass(frozen=True, slots=True)
class Settings:
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    zones: tuple[ZoneConfig, ...] = ()
    alerts: AlertConfig = field(default_factory=AlertConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    source_path: Path | None = None

    # -- derived ---------------------------------------------------------
    def resolved_backend(self) -> tuple[str, ...]:
        """Backend preference order derived from ``detection.backend``.

        ``auto`` means "Intel GPU, else CPU". It deliberately never includes
        ``nvidia_ort`` or ``pipegen``.
        """
        if self.detection.backend == "auto":
            return AUTO_BACKENDS
        return (self.detection.backend,)

    def enabled_zones(self) -> tuple[ZoneConfig, ...]:
        return tuple(zone for zone in self.zones if zone.enabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video.__dict__ if hasattr(self.video, "__dict__") else _asdict(self.video),
            "detection": _asdict(self.detection),
            "tracking": _asdict(self.tracking),
            "alerts": _asdict(self.alerts),
            "storage": _asdict(self.storage),
            "metrics": _asdict(self.metrics),
            "zones": [zone.to_dict() for zone in self.zones],
        }

    def with_changes(self, **kwargs: Any) -> "Settings":
        return replace(self, **kwargs)


def _asdict(instance: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(instance)


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _as_float(value: Any, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{where} must be a number, got {value!r}") from error


def _as_int(value: Any, where: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{where} must be an integer, got {value!r}") from error


def _as_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "on", "1"}:
            return True
        if lowered in {"false", "no", "off", "0"}:
            return False
    raise ConfigError(f"{where} must be a boolean, got {value!r}")


def _validate_polygon(points: Sequence[Any], where: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise ConfigError(f"{where} must be a list of [x, y] points")
    if len(points) < 3:
        raise ConfigError(f"{where} must contain at least three points")

    parsed: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            raise ConfigError(f"{where}[{index}] must be an [x, y] pair")
        x = _as_float(point[0], f"{where}[{index}][0]")
        y = _as_float(point[1], f"{where}[{index}][1]")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ConfigError(
                f"{where}[{index}] = [{x}, {y}] is outside the frame; "
                "normalised coordinates must be between 0.0 and 1.0"
            )
        parsed.append((x, y))

    try:
        validate_polygon_shape(parsed, where)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    return tuple(parsed)


def _env_override(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value is not None and value.strip() != "" else None


def load_settings(path: str | Path | None = None) -> Settings:
    """Load and validate ``settings.yaml``.

    Missing file means "use defaults", which keeps a fresh clone runnable.
    A file that exists but is malformed is a hard error.
    """
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    raw: Mapping[str, Any] = {}
    if settings_path.exists():
        try:
            loaded = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ConfigError(f"{settings_path} is not valid YAML: {error}") from error
        raw = _require_mapping(loaded, str(settings_path))
    elif path is not None:
        raise ConfigError(f"Configuration file not found: {settings_path}")

    return parse_settings(raw, source_path=settings_path if settings_path.exists() else None)


def parse_settings(raw: Mapping[str, Any], source_path: Path | None = None) -> Settings:
    """Validate an already-parsed configuration mapping."""
    video_raw = _require_mapping(raw.get("video"), "video")
    detection_raw = _require_mapping(raw.get("detection"), "detection")
    tracking_raw = _require_mapping(raw.get("tracking"), "tracking")
    alerts_raw = _require_mapping(raw.get("alerts"), "alerts")
    storage_raw = _require_mapping(raw.get("storage"), "storage")
    metrics_raw = _require_mapping(raw.get("metrics"), "metrics")

    # --- video ---------------------------------------------------------
    source_type = str(video_raw.get("source_type", "webcam")).strip().lower()
    if source_type not in VIDEO_SOURCE_TYPES:
        raise ConfigError(
            f"video.source_type must be one of {VIDEO_SOURCE_TYPES}, got {source_type!r}"
        )
    width = _as_int(video_raw.get("width", 640), "video.width")
    height = _as_int(video_raw.get("height", 480), "video.height")
    if width <= 0 or height <= 0:
        raise ConfigError("video.width and video.height must be positive")

    file_path = str(video_raw.get("file_path", "") or "")
    rtsp_url = str(video_raw.get("rtsp_url", "") or "")
    if source_type == "file" and not file_path.strip():
        raise ConfigError("video.file_path is required when source_type is 'file'")
    if source_type == "rtsp" and not rtsp_url.strip():
        raise ConfigError("video.rtsp_url is required when source_type is 'rtsp'")

    camera_index = _as_int(video_raw.get("camera_index", 0), "video.camera_index")
    if camera_index < 0:
        raise ConfigError("video.camera_index must be zero or greater")

    video = VideoConfig(
        source_type=source_type,
        camera_id=str(video_raw.get("camera_id", f"cam-{camera_index}") or f"cam-{camera_index}"),
        camera_index=camera_index,
        file_path=file_path,
        rtsp_url=rtsp_url,
        width=width,
        height=height,
        fps_limit=_as_float(video_raw.get("fps_limit", 0.0), "video.fps_limit"),
        reopen_attempts=_as_int(video_raw.get("reopen_attempts", 3), "video.reopen_attempts"),
        reopen_delay_s=_as_float(video_raw.get("reopen_delay_s", 1.0), "video.reopen_delay_s"),
    )
    if video.reopen_attempts < 0:
        raise ConfigError("video.reopen_attempts must be zero or greater")
    if video.reopen_delay_s < 0:
        raise ConfigError("video.reopen_delay_s must be zero or greater")

    # --- detection -----------------------------------------------------
    backend = str(detection_raw.get("backend", "auto")).strip().lower()
    if backend not in DETECTION_BACKENDS:
        raise ConfigError(
            f"detection.backend must be one of {DETECTION_BACKENDS}, got {backend!r}"
        )
    confidence = _as_float(
        detection_raw.get("confidence_threshold", 0.35), "detection.confidence_threshold"
    )
    nms = _as_float(detection_raw.get("nms_threshold", 0.45), "detection.nms_threshold")
    if not 0.0 < confidence <= 1.0:
        raise ConfigError("detection.confidence_threshold must be in (0.0, 1.0]")
    if not 0.0 < nms <= 1.0:
        raise ConfigError("detection.nms_threshold must be in (0.0, 1.0]")

    model_size = _as_int(detection_raw.get("model_size", 640), "detection.model_size")
    if model_size <= 0:
        raise ConfigError("detection.model_size must be positive")
    strides_raw = detection_raw.get("strides", (8, 16, 32))
    if not isinstance(strides_raw, Sequence) or isinstance(strides_raw, (str, bytes)):
        raise ConfigError("detection.strides must be a list of integers")
    strides = tuple(_as_int(item, "detection.strides[]") for item in strides_raw)
    if not strides or any(stride <= 0 for stride in strides):
        raise ConfigError("detection.strides must contain positive integers")
    for stride in strides:
        if model_size % stride != 0:
            raise ConfigError(
                f"detection.model_size ({model_size}) must be divisible by each stride ({stride})"
            )

    detection = DetectionConfig(
        backend=backend,
        model_path=str(detection_raw.get("model_path", "") or ""),
        model_url=str(detection_raw.get("model_url", "") or ""),
        confidence_threshold=confidence,
        nms_threshold=nms,
        model_size=model_size,
        strides=strides,
        pipegen_endpoint=str(detection_raw.get("pipegen_endpoint", "") or ""),
    )

    # --- tracking ------------------------------------------------------
    tracking = TrackingConfig(
        enabled=_as_bool(tracking_raw.get("enabled", True), "tracking.enabled"),
        iou_threshold=_as_float(tracking_raw.get("iou_threshold", 0.30), "tracking.iou_threshold"),
        max_age=_as_int(tracking_raw.get("max_age", 15), "tracking.max_age"),
        min_hits=_as_int(tracking_raw.get("min_hits", 1), "tracking.min_hits"),
    )
    if not 0.0 <= tracking.iou_threshold <= 1.0:
        raise ConfigError("tracking.iou_threshold must be between 0.0 and 1.0")
    if tracking.max_age < 0:
        raise ConfigError("tracking.max_age must be zero or greater")
    if tracking.min_hits < 1:
        raise ConfigError("tracking.min_hits must be one or greater")

    # --- alerts --------------------------------------------------------
    alerts = AlertConfig(
        enter_frames=_as_int(alerts_raw.get("enter_frames", 3), "alerts.enter_frames"),
        exit_frames=_as_int(alerts_raw.get("exit_frames", 5), "alerts.exit_frames"),
        cooldown_seconds=_as_float(alerts_raw.get("cooldown_seconds", 20.0), "alerts.cooldown_seconds"),
        min_confidence=_as_float(alerts_raw.get("min_confidence", 0.35), "alerts.min_confidence"),
        record_exits=_as_bool(alerts_raw.get("record_exits", True), "alerts.record_exits"),
    )
    if alerts.enter_frames < 1:
        raise ConfigError("alerts.enter_frames must be one or greater")
    if alerts.exit_frames < 1:
        raise ConfigError("alerts.exit_frames must be one or greater")
    if alerts.cooldown_seconds < 0:
        raise ConfigError("alerts.cooldown_seconds must be zero or greater")
    if not 0.0 <= alerts.min_confidence <= 1.0:
        raise ConfigError("alerts.min_confidence must be between 0.0 and 1.0")

    # --- zones ---------------------------------------------------------
    zones_raw = raw.get("zones") or []
    if not isinstance(zones_raw, Sequence) or isinstance(zones_raw, (str, bytes)):
        raise ConfigError("zones must be a list")
    zones: list[ZoneConfig] = []
    seen_ids: set[str] = set()
    for index, zone_raw in enumerate(zones_raw):
        zone_map = _require_mapping(zone_raw, f"zones[{index}]")
        zone_id = str(zone_map.get("id", f"zone-{index + 1}")).strip()
        if not zone_id:
            raise ConfigError(f"zones[{index}].id must not be empty")
        if zone_id in seen_ids:
            raise ConfigError(f"Duplicate zone id {zone_id!r}")
        seen_ids.add(zone_id)
        zones.append(
            ZoneConfig(
                zone_id=zone_id,
                name=str(zone_map.get("name", zone_id) or zone_id),
                enabled=_as_bool(zone_map.get("enabled", True), f"zones[{index}].enabled"),
                polygon_normalized=_validate_polygon(
                    zone_map.get("polygon_normalized"), f"zones[{index}].polygon_normalized"
                ),
            )
        )

    # --- storage -------------------------------------------------------
    storage = StorageConfig(
        db_path=str(storage_raw.get("db_path", "") or ""),
        export_dir=str(storage_raw.get("export_dir", "") or ""),
        save_snapshots=_as_bool(storage_raw.get("save_snapshots", False), "storage.save_snapshots"),
        snapshot_dir=str(storage_raw.get("snapshot_dir", "") or ""),
        retention_days=_as_int(storage_raw.get("retention_days", 0), "storage.retention_days"),
    )
    if storage.retention_days < 0:
        raise ConfigError("storage.retention_days must be zero or greater")

    # --- metrics -------------------------------------------------------
    metrics = MetricsConfig(
        window=_as_int(metrics_raw.get("window", 120), "metrics.window"),
        report_path=str(metrics_raw.get("report_path", "") or ""),
    )
    if metrics.window < 1:
        raise ConfigError("metrics.window must be at least 1")

    settings = Settings(
        video=video,
        detection=detection,
        tracking=tracking,
        zones=tuple(zones),
        alerts=alerts,
        storage=storage,
        metrics=metrics,
        source_path=source_path,
    )
    return apply_env_overrides(settings)


def apply_env_overrides(settings: Settings) -> Settings:
    """Apply the small set of supported environment overrides."""
    detection = settings.detection
    video = settings.video
    storage = settings.storage

    if (backend := _env_override("VS_DEVICE")) is not None:
        normalised = backend.strip().lower()
        if normalised not in DETECTION_BACKENDS:
            raise ConfigError(
                f"VS_DEVICE must be one of {DETECTION_BACKENDS}, got {backend!r}"
            )
        detection = replace(detection, backend=normalised)

    if (model_path := _env_override("VS_MODEL_PATH")) is not None:
        detection = replace(detection, model_path=model_path)

    if (endpoint := _env_override("VS_PIPEGEN_ENDPOINT")) is not None:
        detection = replace(detection, pipegen_endpoint=endpoint)

    if (camera_index := _env_override("VS_CAMERA_INDEX")) is not None:
        video = replace(video, camera_index=_as_int(camera_index, "VS_CAMERA_INDEX"))

    if (source_type := _env_override("VS_SOURCE_TYPE")) is not None:
        normalised = source_type.strip().lower()
        if normalised not in VIDEO_SOURCE_TYPES:
            raise ConfigError(f"VS_SOURCE_TYPE must be one of {VIDEO_SOURCE_TYPES}")
        video = replace(video, source_type=normalised)

    if (file_path := _env_override("VS_VIDEO_FILE")) is not None:
        video = replace(video, file_path=file_path)

    if (db_path := _env_override("VS_DB_PATH")) is not None:
        storage = replace(storage, db_path=db_path)

    return replace(settings, detection=detection, video=video, storage=storage)


def save_zones(path: str | Path, zones: Sequence[ZoneConfig]) -> Path:
    """Persist the zone list back into the YAML file, preserving other settings.

    Every polygon is re-validated first, so an editing mistake in the dashboard
    cannot produce a configuration file that fails to load on the next start.
    """
    for zone in zones:
        _validate_polygon(
            [[x, y] for x, y in zone.polygon_normalized],
            f"zones[{zone.zone_id}].polygon_normalized",
        )

    target = Path(path)
    if target.exists():
        try:
            payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as error:
            raise ConfigError(f"{target} is not valid YAML: {error}") from error
        if not isinstance(payload, dict):
            raise ConfigError(f"{target} must contain a mapping at the top level")
    else:
        payload = {}

    payload["zones"] = [zone.to_dict() for zone in zones]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return target
