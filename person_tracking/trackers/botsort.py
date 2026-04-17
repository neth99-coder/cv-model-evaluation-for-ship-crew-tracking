"""BoT-SORT person tracker using Ultralytics YOLO track mode."""

from typing import Any

import numpy as np
from ultralytics import YOLO

from .base import BasePersonTracker

PERSON_CLASS = 0


class BoTSORTTracker(BasePersonTracker):
    model_name = "BoT-SORT"

    def __init__(self, detector_weights: str = "yolov8n.pt", conf: float = 0.25, imgsz: int = 640) -> None:
        self.detector_weights = detector_weights
        self.conf = conf
        self.imgsz = imgsz
        self._model = YOLO(detector_weights)

    def reset(self) -> None:
        self._model.predictor = None

    def update(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self._model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[PERSON_CLASS],
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None or boxes.id is None:
            return []

        ids = boxes.id.int().cpu().tolist()
        coords = boxes.xyxy.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(ids)

        tracks: list[dict[str, Any]] = []
        for i, track_id in enumerate(ids):
            x1, y1, x2, y2 = coords[i]
            tracks.append(
                {
                    "track_id": int(track_id),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": float(confs[i]),
                }
            )
        return tracks
