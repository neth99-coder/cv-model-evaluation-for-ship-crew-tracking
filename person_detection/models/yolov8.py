"""
YOLOv8 person detector via the Ultralytics library.

Model variants (controlled by ``model_size``):
    n  → yolov8n.pt  (nano,   fastest)
    s  → yolov8s.pt  (small)
    m  → yolov8m.pt  (medium)
    l  → yolov8l.pt  (large)
    x  → yolov8x.pt  (xlarge, most accurate)

The COCO class index for "person" is 0.
"""
from typing import Any

import numpy as np

from .base import BasePersonDetector

PERSON_CLASS = 0  # COCO index


class YOLOv8Detector(BasePersonDetector):

    def __init__(
        self,
        model_size: str = "n",
        confidence_threshold: float = 0.5,
    ) -> None:
        from ultralytics import YOLO  # lazy import – heavy dependency

        self.model_name = f"YOLOv8{model_size.upper()}"
        self.confidence_threshold = confidence_threshold
        self._model = YOLO(f"yolov8{model_size}.pt")

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self._model(
            frame,
            classes=[PERSON_CLASS],
            conf=self.confidence_threshold,
            verbose=False,
        )
        detections: list[dict[str, Any]] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf = round(float(box.conf[0]), 4)
                detections.append({"bbox": (x1, y1, x2, y2), "confidence": conf})
        return detections
