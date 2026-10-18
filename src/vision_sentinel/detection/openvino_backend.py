# SPDX-License-Identifier: Apache-2.0
"""OpenVINO inference backends: Intel GPU and CPU.

Both derive from the same code path, which is the verified decode from
``intel_webcam_zone_monitor/app.py`` with the box maths vectorised (see
``yolox.py``). Only the compile target and the failure messages differ.

The Intel GPU backend never silently degrades to CPU: if the GPU is requested
and unusable it raises, so an Intel-GPU performance claim can never be made on
the strength of a CPU run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..events import Detection
from .base import Detector, DetectorError
from .yolox import decode_yolox, make_anchors, preprocess


def openvino_devices() -> list[str]:
    """Return the OpenVINO devices visible on this machine, or ``[]``."""
    try:
        import openvino as ov

        return list(ov.Core().available_devices)
    except Exception:
        return []


def first_intel_gpu(devices: list[str]) -> str | None:
    """Pick the first device that belongs to the Intel GPU family."""
    for device in devices:
        if device.split(".")[0] == "GPU":
            return device
    return None


class _OpenVINOBackend(Detector):
    """Shared implementation for the OpenVINO targets."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float,
        nms_threshold: float,
        model_size: int,
        strides: tuple[int, ...],
        target_device: str,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise DetectorError(
                f"Detection model not found at {path}. Pass --model-path or let "
                "the application download it."
            )
        if not 0.0 < confidence_threshold <= 1.0:
            raise DetectorError("confidence_threshold must be in (0.0, 1.0]")
        if not 0.0 < nms_threshold <= 1.0:
            raise DetectorError("nms_threshold must be in (0.0, 1.0]")

        self.model_path = path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model_size = model_size
        self.grids, self.expanded_strides = make_anchors(model_size, strides)

        try:
            import openvino as ov
        except ImportError as error:  # pragma: no cover - import guard
            raise DetectorError(
                "OpenVINO is not installed. Install it with "
                "'python -m pip install openvino'."
            ) from error

        self._ov = ov
        self._core = ov.Core()
        try:
            ov_model = self._core.read_model(str(path))
        except Exception as error:
            raise DetectorError(
                f"OpenVINO could not read the model at {path}: {error}"
            ) from error

        self.target_device = target_device
        try:
            self._compiled = self._core.compile_model(ov_model, target_device)
        except Exception as error:
            raise self._compile_error(target_device, error) from error

        self._input = self._compiled.input(0)
        self._output = self._compiled.output(0)
        self._validate_output_shape()

        if target_device.upper().startswith("GPU"):
            self.device_name = f"Intel GPU ({target_device})"
        else:
            self.device_name = f"CPU ({target_device})"

    def _compile_error(self, target_device: str, error: Exception) -> DetectorError:
        return DetectorError(
            f"OpenVINO could not compile the model for '{target_device}': {error}"
        )

    def _validate_output_shape(self) -> None:
        """Fail early if the model is not the YOLOX export we expect."""
        shape = self._output.get_partial_shape() if hasattr(self._output, "get_partial_shape") else None
        if shape is None:
            return
        try:
            rank = len(shape)
            last = shape[rank - 1]
            if last.is_static and int(last) not in (85, -1):
                raise DetectorError(
                    f"The model at {self.model_path} outputs width {int(last)} "
                    "per anchor; this build expects a YOLOX COCO export with 85 "
                    "(4 box + 1 objectness + 80 classes)."
                )
        except DetectorError:
            raise
        except Exception:
            # Shape introspection is best-effort; the decode step validates too.
            return

    def detect(self, frame: np.ndarray) -> list[Detection]:
        blob, scale = preprocess(frame, self.model_size)
        result = self._compiled([blob])[self._output]
        height, width = frame.shape[:2]
        return decode_yolox(
            result,
            self.grids,
            self.expanded_strides,
            scale,
            width,
            height,
            self.confidence_threshold,
            self.nms_threshold,
        )

    def close(self) -> None:
        self._compiled = None  # type: ignore[assignment]
        self._core = None  # type: ignore[assignment]


class IntelBackend(_OpenVINOBackend):
    """OpenVINO on the Intel integrated GPU, with no silent CPU fallback."""

    backend_id = "intel_openvino"

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float,
        nms_threshold: float,
        model_size: int,
        strides: tuple[int, ...],
        device: str | None = None,
    ) -> None:
        devices = openvino_devices()
        target = device or first_intel_gpu(devices)
        if target is None:
            raise DetectorError(
                "OpenVINO did not find an Intel GPU on this machine. "
                f"Devices reported: {devices or 'none'}. Install or update the "
                "Intel graphics driver, or run with --device cpu."
            )
        try:
            super().__init__(
                model_path,
                confidence_threshold,
                nms_threshold,
                model_size,
                strides,
                target,
            )
        except DetectorError as error:
            raise DetectorError(
                f"{error}\nIf the Intel GPU is unavailable, run with --device cpu."
            ) from error


class CPUBackend(_OpenVINOBackend):
    """OpenVINO on the CPU. Always available."""

    backend_id = "cpu_openvino"

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float,
        nms_threshold: float,
        model_size: int,
        strides: tuple[int, ...],
        device: str = "CPU",
    ) -> None:
        super().__init__(
            model_path,
            confidence_threshold,
            nms_threshold,
            model_size,
            strides,
            device,
        )


def describe_environment() -> dict[str, Any]:
    """Summarise the OpenVINO environment for the dashboard and evidence."""
    devices = openvino_devices()
    version = "not installed"
    try:
        import openvino as ov

        version = ov.__version__
    except Exception:
        pass
    return {
        "openvino_version": version,
        "available_devices": devices,
        "intel_gpu": first_intel_gpu(devices),
    }
