# SPDX-License-Identifier: Apache-2.0
"""YOLOX pre-processing, anchor generation and output decoding.

The decode maths here is ported unchanged from the verified Intel baseline
(``intel_webcam_zone_monitor/app.py``), which is known to produce correct boxes
on this machine. Two things are deliberately different:

* The per-class box computation is vectorised and converts model coordinates to
  pixels with round-to-nearest (``np.rint``). The baseline truncated toward zero
  with ``int()``; because a decoded width or height comes back through
  ``exp(log(w))`` it is rarely exact, so truncation pushed every box up and to
  the left by up to one pixel and reported a one-pixel off-by-one on any box
  whose edges land on integer coordinates.
* Non-maximum suppression is implemented here (``nms_xywh``) instead of being
  delegated to ``cv2.dnn.NMSBoxes``. It is deterministic, unit-tested, and its
  return shape does not vary between OpenCV releases.
"""

from __future__ import annotations

import numpy as np

from ..events import Detection
from ..geometry import iou_xywh
from .coco import label_for

__all__ = [
    "decode_yolox",
    "iou_xywh",
    "make_anchors",
    "nms_xywh",
    "preprocess",
]


def make_anchors(
    model_size: int, strides: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Build the YOLOX grid and expanded-stride tensors."""
    if model_size <= 0:
        raise ValueError("model_size must be positive")
    grids: list[np.ndarray] = []
    expanded_strides: list[np.ndarray] = []
    for stride in strides:
        if stride <= 0:
            raise ValueError("strides must be positive")
        if model_size % stride != 0:
            raise ValueError(f"model_size {model_size} is not divisible by stride {stride}")
        cells = model_size // stride
        grid_x, grid_y = np.meshgrid(np.arange(cells), np.arange(cells))
        grids.append(np.stack((grid_x, grid_y), axis=2).reshape(1, -1, 2))
        expanded_strides.append(np.full((1, cells * cells, 1), stride))
    return np.concatenate(grids, axis=1), np.concatenate(expanded_strides, axis=1)


def preprocess(frame: np.ndarray, model_size: int) -> tuple[np.ndarray, float]:
    """Letterbox a BGR frame into a normalised NCHW float blob.

    Returns the blob and the scale factor needed to map boxes back to pixels.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 BGR frame, got shape {frame.shape}")
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Frame dimensions must be positive")

    scale = min(model_size / height, model_size / width)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))

    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)

    canvas = np.full((model_size, model_size, 3), 114, dtype=np.float32)
    canvas[:resized_height, :resized_width] = resized
    blob = np.transpose(canvas, (2, 0, 1))[np.newaxis, ...].copy()
    return blob, scale


def nms_xywh(
    boxes: np.ndarray, scores: np.ndarray, threshold: float
) -> list[int]:
    """Greedy non-maximum suppression over ``(x, y, w, h)`` boxes.

    Boxes are considered in descending score order and any box overlapping an
    already-kept box by more than ``threshold`` is suppressed. Ties are broken
    by original index so the output is fully deterministic.
    """
    if len(boxes) == 0:
        return []
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    order = sorted(range(len(boxes)), key=lambda index: (-float(scores[index]), index))
    kept: list[int] = []
    while order:
        current = order.pop(0)
        kept.append(current)
        order = [
            index
            for index in order
            if iou_xywh(boxes[current], boxes[index]) <= threshold
        ]
    return kept


def decode_yolox(
    output: np.ndarray,
    grids: np.ndarray,
    expanded_strides: np.ndarray,
    scale: float,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float,
    nms_threshold: float,
) -> list[Detection]:
    """Decode raw YOLOX output into pixel-space detections.

    ``output`` is the model tensor with shape ``(1, N, 85)`` or ``(N, 85)``.
    """
    if not 0.0 < confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in (0.0, 1.0]")
    if not 0.0 < nms_threshold <= 1.0:
        raise ValueError("nms_threshold must be in (0.0, 1.0]")
    if scale <= 0:
        raise ValueError("scale must be positive")

    predictions = np.asarray(output)
    if predictions.ndim == 3:
        if predictions.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {predictions.shape[0]}")
        predictions = predictions[0]
    if predictions.ndim != 2 or predictions.shape[1] != 85:
        raise ValueError(
            "Unexpected YOLOX output shape "
            f"{np.asarray(output).shape}; expected (1, N, 85) or (N, 85)"
        )
    if predictions.shape[0] != grids.shape[1]:
        raise ValueError(
            f"Model returned {predictions.shape[0]} predictions but "
            f"{grids.shape[1]} anchors were generated; the configured "
            "model_size/strides do not match the model"
        )

    decoded = predictions.astype(np.float32, copy=True)
    decoded[:, :2] = (decoded[:, :2] + grids[0]) * expanded_strides[0]
    decoded[:, 2:4] = (
        np.exp(np.clip(decoded[:, 2:4], -20, 20)) * expanded_strides[0]
    )

    class_scores = decoded[:, 4:5] * decoded[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]

    candidate_indices = np.flatnonzero(confidences >= confidence_threshold)
    if candidate_indices.size == 0:
        return []

    centers_x = decoded[candidate_indices, 0]
    centers_y = decoded[candidate_indices, 1]
    widths = decoded[candidate_indices, 2]
    heights = decoded[candidate_indices, 3]

    # Round to the nearest pixel rather than truncate: the decoded width and
    # height have already travelled through ``exp(log(w))``, so a box whose
    # edges are on integer coordinates arrives a few ULPs either side of them.
    left = np.rint((centers_x - widths / 2) / scale).astype(np.int64)
    top = np.rint((centers_y - heights / 2) / scale).astype(np.int64)
    right = np.rint((centers_x + widths / 2) / scale).astype(np.int64)
    bottom = np.rint((centers_y + heights / 2) / scale).astype(np.int64)

    np.clip(left, 0, max(frame_width - 1, 0), out=left)
    np.clip(top, 0, max(frame_height - 1, 0), out=top)
    right = np.maximum(left + 1, np.minimum(frame_width, right))
    bottom = np.maximum(top + 1, np.minimum(frame_height, bottom))

    boxes = np.column_stack(
        (left, top, right - left, bottom - top)
    ).astype(np.int64)
    candidate_class_ids = class_ids[candidate_indices]
    candidate_scores = confidences[candidate_indices]

    detections: list[Detection] = []
    for class_id in np.unique(candidate_class_ids):
        class_mask = candidate_class_ids == class_id
        class_indices = np.flatnonzero(class_mask)
        kept = nms_xywh(
            boxes[class_indices].astype(np.float64),
            candidate_scores[class_indices],
            nms_threshold,
        )
        for local_index in kept:
            position = int(class_indices[local_index])
            detections.append(
                Detection(
                    class_id=int(class_id),
                    label=label_for(int(class_id)),
                    confidence=float(candidate_scores[position]),
                    bbox_xywh=(
                        int(boxes[position, 0]),
                        int(boxes[position, 1]),
                        int(boxes[position, 2]),
                        int(boxes[position, 3]),
                    ),
                )
            )
    return detections
