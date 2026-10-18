# SPDX-License-Identifier: Apache-2.0
"""COCO class labels used by the pretrained YOLOX detector."""

from __future__ import annotations

#: The 80 COCO object classes, in the model's output order.
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

#: The class monitored for restricted-zone occupancy.
PERSON_CLASS = "person"


def label_for(class_id: int) -> str:
    """Map a model class index to a label."""
    if not 0 <= class_id < len(COCO_CLASSES):
        raise ValueError(f"Unknown COCO class id {class_id}")
    return COCO_CLASSES[class_id]
