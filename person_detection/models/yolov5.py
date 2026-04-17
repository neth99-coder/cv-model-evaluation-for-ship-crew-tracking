"""
YOLOv5 person detector via the Ultralytics library.

Model variants (controlled by ``model_size``):
    s  → yolov5su.pt  (small, fastest)
    m  → yolov5mu.pt  (medium)
    l  → yolov5lu.pt  (large)
    x  → yolov5xu.pt  (xlarge, most accurate)

The COCO class index for "person" is 0.

Using Ultralytics directly avoids a torch.hub dependency on cloning the
``ultralytics/yolov5`` GitHub repo metadata at runtime.
"""
from typing import Any

import numpy as np

from .base import BasePersonDetector

PERSON_CLASS = 0  # COCO index


class YOLOv5Detector(BasePersonDetector):

    def __init__(
        self,
        model_size: str = "s",
        confidence_threshold: float = 0.5,
    ) -> None:
        from ultralytics import YOLO  # lazy import

        self.model_name = f"YOLOv5{model_size.upper()}"
        self.confidence_threshold = confidence_threshold

        weights_map = {
            "s": "yolov5su.pt",
            "m": "yolov5mu.pt",
            "l": "yolov5lu.pt",
            "x": "yolov5xu.pt",
        }
        weights = weights_map.get(model_size, "yolov5su.pt")
        self._model = YOLO(weights)

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
