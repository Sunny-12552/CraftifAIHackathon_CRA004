# SPDX-License-Identifier: Apache-2.0
"""Detector construction: resolve the configured backend into a live detector.

Resolution rules
----------------
* ``auto`` tries the Intel GPU first and then the CPU. It never selects
  ``nvidia_ort`` or ``pipegen``: those must be requested by name, so an
  NVIDIA-related claim can never be made by accident.
* An explicitly requested backend that fails raises. Nothing silently swaps the
  compute device underneath a result, because that would invalidate any
  performance or compliance statement derived from it.
* Every attempt is recorded in :class:`DetectorResolution` so the dashboard can
  explain exactly why a given backend is running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from .base import Detector, DetectorError
from .model_manager import ensure_model
from .nvidia_ort_backend import NvidiaOrtBackend
from .openvino_backend import CPUBackend, IntelBackend, describe_environment
from .pipegen_backend import PIPEGEN_CHECKLIST, PipeGenBackend, publish_help

__all__ = [
    "BackendAttempt",
    "Detector",
    "DetectorError",
    "DetectorResolution",
    "CPUBackend",
    "IntelBackend",
    "NvidiaOrtBackend",
    "PIPEGEN_CHECKLIST",
    "PipeGenBackend",
    "create_detector",
    "describe_environment",
    "publish_help",
]


@dataclass(frozen=True, slots=True)
class BackendAttempt:
    """The outcome of trying one backend."""

    backend: str
    succeeded: bool
    message: str = ""


@dataclass(frozen=True, slots=True)
class DetectorResolution:
    """The live detector plus the audit trail of how it was chosen."""

    detector: Detector
    attempts: tuple[BackendAttempt, ...] = ()
    warnings: tuple[str, ...] = field(default=())

    def explanation(self) -> str:
        lines = [f"Backend: {self.detector.backend_id} on {self.detector.device_name}"]
        for warning in self.warnings:
            lines.append(f"Warning: {warning}")
        for attempt in self.attempts:
            status = "ok" if attempt.succeeded else "failed"
            detail = f" - {attempt.message}" if attempt.message else ""
            lines.append(f"  [{status}] {attempt.backend}{detail}")
        return "\n".join(lines)


def _build(
    backend: str,
    settings: Settings,
    model_path: Path,
    allow_download: bool,
) -> Detector:
    detection = settings.detection
    common = {
        "confidence_threshold": detection.confidence_threshold,
        "nms_threshold": detection.nms_threshold,
        "model_size": detection.model_size,
        "strides": detection.strides,
    }

    if backend == "pipegen":
        return PipeGenBackend(
            endpoint=detection.pipegen_endpoint,
            confidence_threshold=detection.confidence_threshold,
        )

    ensure_model(model_path, detection.model_url, allow_download=allow_download)

    if backend == "intel":
        return IntelBackend(model_path, **common)
    if backend == "cpu":
        return CPUBackend(model_path, **common)
    if backend == "nvidia_ort":
        return NvidiaOrtBackend(model_path, **common)
    raise DetectorError(f"Unknown detection backend {backend!r}")


def create_detector(
    settings: Settings, allow_download: bool = True
) -> DetectorResolution:
    """Construct the detector described by ``settings``.

    ``allow_download=False`` is used by the test suite so that tests never touch
    the network.
    """
    model_path = settings.detection.resolved_model_path()
    order = settings.resolved_backend()
    explicit = settings.detection.backend != "auto"

    attempts: list[BackendAttempt] = []
    warnings: list[str] = []
    last_error: DetectorError | None = None

    for backend in order:
        try:
            detector = _build(backend, settings, model_path, allow_download)
        except DetectorError as error:
            attempts.append(BackendAttempt(backend, False, str(error).splitlines()[0]))
            last_error = error
            if explicit:
                raise
            continue
        attempts.append(BackendAttempt(backend, True))
        if backend == "cpu" and len(order) > 1:
            warnings.append(
                "The Intel GPU backend was unavailable, so inference is running "
                "on the CPU. Performance figures from this run describe CPU "
                "inference, not Intel GPU inference."
            )
        return DetectorResolution(
            detector=detector, attempts=tuple(attempts), warnings=tuple(warnings)
        )

    detail = "\n".join(
        f"  - {attempt.backend}: {attempt.message}" for attempt in attempts
    )
    raise DetectorError(
        "No usable detection backend could be initialised.\n"
        f"Attempts:\n{detail}\n"
        f"OpenVINO environment: {describe_environment()}"
    ) from last_error
