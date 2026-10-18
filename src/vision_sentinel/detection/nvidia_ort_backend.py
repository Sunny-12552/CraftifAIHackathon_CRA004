# SPDX-License-Identifier: Apache-2.0
"""Optional ONNX Runtime CUDA backend -- explicitly NOT PipeGen.

This backend exists for developer convenience on an NVIDIA workstation: it runs
the *same* YOLOX ONNX export through ONNX Runtime's CUDA execution provider.

It is **not** the CraftifAI PipeGen workflow. It does not use a PipeGen-built
TensorRT engine, does not produce a PipeGen speed report, and does not satisfy
Track A. It is therefore excluded from ``--device auto`` and reports
``compliance_eligible = False`` so that no evidence path can mistake it for the
compliant NVIDIA branch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..events import Detection
from .base import Detector, DetectorError
from .yolox import decode_yolox, make_anchors, preprocess

NON_COMPLIANCE_NOTICE = (
    "onnxruntime CUDA inference is a developer convenience only. It is NOT "
    "CraftifAI PipeGen, does NOT use a PipeGen-generated TensorRT engine, and "
    "does NOT satisfy the Track A / PipeGen requirements."
)


def cuda_provider_available() -> bool:
    """True when ONNX Runtime exposes its CUDA execution provider."""
    try:
        import onnxruntime
    except ImportError:
        return False
    return "CUDAExecutionProvider" in onnxruntime.get_available_providers()


class NvidiaOrtBackend(Detector):
    """YOLOX on an NVIDIA GPU through ONNX Runtime's CUDA provider."""

    backend_id = "nvidia_ort"
    compliance_eligible = False

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float,
        nms_threshold: float,
        model_size: int,
        strides: tuple[int, ...],
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise DetectorError(
                f"Detection model not found at {path}. Pass --model-path or let "
                "the application download it."
            )

        try:
            import onnxruntime as ort
        except ImportError as error:
            raise DetectorError(
                "onnxruntime is not installed. Install the CUDA build with "
                "'python -m pip install onnxruntime-gpu'.\n" + NON_COMPLIANCE_NOTICE
            ) from error

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            raise DetectorError(
                "ONNX Runtime did not report a CUDA execution provider. "
                f"Available providers: {providers}. Install 'onnxruntime-gpu' "
                "together with the matching CUDA and cuDNN runtimes.\n"
                + NON_COMPLIANCE_NOTICE
            )

        self.model_path = path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model_size = model_size
        self.grids, self.expanded_strides = make_anchors(model_size, strides)

        try:
            self._session = ort.InferenceSession(
                str(path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception as error:
            raise DetectorError(
                f"ONNX Runtime could not create a CUDA session for {path}: {error}\n"
                + NON_COMPLIANCE_NOTICE
            ) from error

        active = self._session.get_providers()
        if not active or active[0] != "CUDAExecutionProvider":
            raise DetectorError(
                "ONNX Runtime did not activate its CUDA provider; active "
                f"providers: {active}.\n" + NON_COMPLIANCE_NOTICE
            )

        self._input_name = self._session.get_inputs()[0].name
        self.device_name = "NVIDIA GPU (ONNX Runtime CUDA) - not PipeGen"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        blob, scale = preprocess(frame, self.model_size)
        result = self._session.run(None, {self._input_name: blob})[0]
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
        self._session = None  # type: ignore[assignment]
