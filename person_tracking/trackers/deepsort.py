"""DeepSORT person tracker using YOLO detections + deep-sort-realtime."""

from typing import Any

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from .base import BasePersonTracker

PERSON_CLASS = 0


class DeepSORTTracker(BasePersonTracker):
    model_name = "DeepSORT"

    def __init__(
        self,
        detector_weights: str = "yolov8n.pt",
        conf: float = 0.25,
        imgsz: int = 640,
        max_age: int = 30,
        n_init: int = 2,
    ) -> None:
        self.detector_weights = detector_weights
        self.conf = conf
        self.imgsz = imgsz
        self.max_age = max_age
        self.n_init = n_init

        self._detector = YOLO(detector_weights)
        self._tracker = self._build_tracker()

    def _build_tracker(self) -> DeepSort:
        return DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            max_iou_distance=0.7,
            embedder="mobilenet",
            half=True,
            bgr=True,
        )

    def reset(self) -> None:
        self._tracker = self._build_tracker()

    def update(self, frame: np.ndarray) -> list[dict[str, Any]]:
        det_results = self._detector(
            frame,
            classes=[PERSON_CLASS],
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections = []
        if det_results:
            result = det_results[0]
            boxes = result.boxes
            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(xyxy)
                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = box
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    detections.append(([x1, y1, w, h], float(confs[i]), "person"))

        raw_tracks = self._tracker.update_tracks(detections, frame=frame)

        tracks: list[dict[str, Any]] = []
        for track in raw_tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            tracks.append(
                {
                    "track_id": int(track.track_id),
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(getattr(track, "det_conf", 0.0) or 0.0),
                }
            )
        return tracks
