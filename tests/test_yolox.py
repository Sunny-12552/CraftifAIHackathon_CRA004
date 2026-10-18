# SPDX-License-Identifier: Apache-2.0
"""YOLOX decoding, anchor generation and non-maximum suppression."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vision_sentinel.detection.yolox import (
    decode_yolox,
    iou_xywh,
    make_anchors,
    nms_xywh,
    preprocess,
)

MODEL_SIZE = 640
STRIDES = (8, 16, 32)


def test_anchor_count_matches_the_yolox_grids() -> None:
    grids, expanded_strides = make_anchors(MODEL_SIZE, STRIDES)

    expected = sum((MODEL_SIZE // stride) ** 2 for stride in STRIDES)
    assert grids.shape == (1, expected, 2)
    assert expanded_strides.shape == (1, expected, 1)
    assert expected == 8400


def test_anchor_values_are_laid_out_row_major() -> None:
    grids, expanded_strides = make_anchors(16, (8, 16))
    # 16/8 = 2 cells per side -> 4 anchors, then 16/16 = 1 -> 1 anchor.
    assert grids[0, :4].tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]
    assert grids[0, 4].tolist() == [0, 0]
    assert expanded_strides[0, :5, 0].tolist() == [8, 8, 8, 8, 16]


def test_model_size_must_be_divisible_by_strides() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        make_anchors(100, (8, 16, 32))


def test_preprocess_letterboxes_and_preserves_aspect_ratio() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    blob, scale = preprocess(frame, MODEL_SIZE)

    assert blob.shape == (1, 3, MODEL_SIZE, MODEL_SIZE)
    assert blob.dtype == np.float32
    # 640 wide into a 640 box is unscaled.
    assert scale == pytest.approx(1.0)
    # The padding region keeps the YOLOX grey value of 114.
    assert blob[0, 0, -1, -1] == pytest.approx(114.0)


def test_preprocess_scales_a_tall_frame_to_fit() -> None:
    frame = np.zeros((1280, 640, 3), dtype=np.uint8)
    _, scale = preprocess(frame, MODEL_SIZE)
    assert scale == pytest.approx(0.5)


def test_preprocess_rejects_a_non_bgr_array() -> None:
    with pytest.raises(ValueError, match="BGR frame"):
        preprocess(np.zeros((10, 10), dtype=np.uint8), MODEL_SIZE)


def test_iou_matches_hand_calculation() -> None:
    assert iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou_xywh((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    # Half-overlap: intersection 50, union 150.
    assert iou_xywh((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_nms_suppresses_overlapping_boxes_and_is_deterministic() -> None:
    boxes = np.array(
        [
            [0, 0, 10, 10],  # score 0.9, kept
            [1, 1, 10, 10],  # heavy overlap, suppressed
            [50, 50, 10, 10],  # no overlap, kept
        ],
        dtype=np.float64,
    )
    scores = np.array([0.9, 0.8, 0.7])

    kept = nms_xywh(boxes, scores, 0.5)
    assert kept == [0, 2]
    # Repeated calls must agree.
    assert nms_xywh(boxes, scores, 0.5) == kept


def test_nms_handles_empty_input_and_validation() -> None:
    assert nms_xywh(np.zeros((0, 4)), np.zeros(0), 0.5) == []
    with pytest.raises(ValueError, match="same length"):
        nms_xywh(np.zeros((2, 4)), np.zeros(1), 0.5)
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        nms_xywh(np.zeros((1, 4)), np.zeros(1), 2.0)


def build_output(
    predictions: list[tuple[float, float, float, float, int, float, str]],
    model_size: int = MODEL_SIZE,
    strides: tuple[int, ...] = STRIDES,
) -> np.ndarray:
    """Synthesise a YOLOX output tensor from decoded-space boxes.

    Each entry is ``(center_x, center_y, width, height, class_id, confidence,
    stride)`` in model pixel space. The output is written in the encoded form
    the ONNX export produces, so ``decode_yolox`` has to do real work.
    """
    _, expanded = make_anchors(model_size, strides)
    total = expanded.shape[1]
    output = np.zeros((1, total, 85), dtype=np.float32)

    for center_x, center_y, width, height, class_id, confidence, stride in predictions:
        cells = model_size // stride
        grid_x = min(max(int(center_x // stride), 0), cells - 1)
        grid_y = min(max(int(center_y // stride), 0), cells - 1)
        offset = _anchor_offset(model_size, strides, stride, grid_x, grid_y)

        output[0, offset, 0] = center_x / stride - grid_x
        output[0, offset, 1] = center_y / stride - grid_y
        output[0, offset, 2] = math.log(width / stride)
        output[0, offset, 3] = math.log(height / stride)
        output[0, offset, 4] = 0.99  # objectness
        output[0, offset, 5 + class_id] = confidence
    return output


def _anchor_offset(
    model_size: int, strides: tuple[int, ...], stride: int, grid_x: int, grid_y: int
) -> int:
    offset = 0
    for current in strides:
        cells = model_size // current
        if current == stride:
            return offset + grid_y * cells + grid_x
        offset += cells * cells
    raise AssertionError("stride not in strides")


def test_decode_recovers_the_original_box() -> None:
    grids, expanded = make_anchors(MODEL_SIZE, STRIDES)
    output = build_output([(200.0, 70.0, 80.0, 80.0, 0, 0.9, 8)])

    detections = decode_yolox(
        output, grids, expanded, scale=1.0, frame_width=640, frame_height=640,
        confidence_threshold=0.35, nms_threshold=0.45,
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.label == "person"
    assert detection.confidence == pytest.approx(0.99 * 0.9, abs=1e-6)
    assert detection.bbox_xywh == (160, 30, 80, 80)


def test_decode_applies_the_letterbox_scale() -> None:
    grids, expanded = make_anchors(MODEL_SIZE, STRIDES)
    # Same model-space box, but the source frame was twice the model size, so
    # the letterbox scale is 0.5 (preprocess uses scale = model_size / frame).
    output = build_output([(200.0, 70.0, 80.0, 80.0, 0, 0.9, 8)])

    detections = decode_yolox(
        output, grids, expanded, scale=0.5, frame_width=1280, frame_height=1280,
        confidence_threshold=0.35, nms_threshold=0.45,
    )

    assert detections[0].bbox_xywh == (320, 60, 160, 160)


def test_low_confidence_predictions_are_dropped() -> None:
    grids, expanded = make_anchors(MODEL_SIZE, STRIDES)
    output = build_output([(200.0, 70.0, 80.0, 80.0, 0, 0.10, 8)])

    detections = decode_yolox(
        output, grids, expanded, scale=1.0, frame_width=640, frame_height=640,
        confidence_threshold=0.35, nms_threshold=0.45,
    )
    assert detections == []


def test_boxes_are_clamped_into_the_frame() -> None:
    grids, expanded = make_anchors(MODEL_SIZE, STRIDES)
    # A huge box centred near the top-left corner.
    output = build_output([(20.0, 20.0, 600.0, 600.0, 0, 0.95, 8)])

    detections = decode_yolox(
        output, grids, expanded, scale=1.0, frame_width=320, frame_height=240,
        confidence_threshold=0.35, nms_threshold=0.45,
    )

    x, y, width, height = detections[0].bbox_xywh
    assert x >= 0 and y >= 0
    assert x + width <= 320
    assert y + height <= 240


def test_class_scores_pick_the_right_class() -> None:
    grids, expanded = make_anchors(MODEL_SIZE, STRIDES)
    # class_id 0 is person, class_id 1 is bicycle.
    output = build_output([(200.0, 70.0, 80.0, 80.0, 1, 0.95, 8)])

    detections = decode_yolox(
        output, grids, expanded, scale=1.0, frame_width=640, frame_height=640,
        confidence_threshold=0.35, nms_threshold=0.45,
    )
    assert detections[0].label == "bicycle"


def test_predictions_are_accepted_without_a_batch_dimension() -> None:
    grids, expanded = make_anchors(MODEL_SIZE, STRIDES)
    output = build_output([(200.0, 70.0, 80.0, 80.0, 0, 0.9, 8)])[0]

    detections = decode_yolox(
        output, grids, expanded, scale=1.0, frame_width=640, frame_height=640,
        confidence_threshold=0.35, nms_threshold=0.45,
    )
    assert len(detections) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda array: array[:, :, :80],  # wrong feature width
        lambda array: array[:, :10, :],  # wrong anchor count
        lambda array: np.repeat(array, 2, axis=0),  # batch size greater than one
    ],
)
def test_malformed_model_output_is_reported_clearly(mutate) -> None:
    grids, expanded = make_anchors(MODEL_SIZE, STRIDES)
    output = mutate(build_output([(200.0, 70.0, 80.0, 80.0, 0, 0.9, 8)]))

    with pytest.raises(ValueError):
        decode_yolox(
            output, grids, expanded, scale=1.0, frame_width=640, frame_height=640,
            confidence_threshold=0.35, nms_threshold=0.45,
        )


def test_mismatched_anchor_count_is_reported() -> None:
    grids, expanded = make_anchors(320, (8, 16, 32))
    output = build_output([(200.0, 70.0, 80.0, 80.0, 0, 0.9, 8)])

    with pytest.raises(ValueError, match="anchors were generated"):
        decode_yolox(
            output, grids, expanded, scale=1.0, frame_width=640, frame_height=640,
            confidence_threshold=0.35, nms_threshold=0.45,
        )


@pytest.mark.parametrize(
    ("confidence", "nms", "scale"),
    [(0.0, 0.45, 1.0), (1.5, 0.45, 1.0), (0.35, 0.0, 1.0), (0.35, 0.45, 0.0)],
)
def test_invalid_decode_parameters_are_rejected(
    confidence: float, nms: float, scale: float
) -> None:
    grids, expanded = make_anchors(MODEL_SIZE, STRIDES)
    output = build_output([(200.0, 70.0, 80.0, 80.0, 0, 0.9, 8)])
    with pytest.raises(ValueError):
        decode_yolox(
            output, grids, expanded, scale=scale, frame_width=640, frame_height=640,
            confidence_threshold=confidence, nms_threshold=nms,
        )
