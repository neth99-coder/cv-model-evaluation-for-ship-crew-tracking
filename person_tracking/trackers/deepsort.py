"""Custom DeepSORT tracker using pluggable external detections."""

from __future__ import annotations

from typing import Any

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort, EMBEDDER_CHOICES

from .base import BasePersonTracker


class DeepSORTTracker(BasePersonTracker):
    model_name = "DeepSORT"
    backend = "custom"
    tracker_backend = "deepsort"

    def __init__(
        self,
        *,
        max_age: int = 30,
        n_init: int = 2,
        embedder: str = "mobilenet",
        embedder_model_name: str | None = None,
        embedder_weights: str | None = None,
        half: bool = True,
        embedder_gpu: bool | None = None,
    ) -> None:
        if embedder not in EMBEDDER_CHOICES:
            raise ValueError(
                f"Unsupported DeepSORT embedder '{embedder}'. Supported values: {', '.join(EMBEDDER_CHOICES)}"
            )

        self.max_age = max_age
        self.n_init = n_init
        self.embedder = embedder
        self.embedder_model_name = embedder_model_name
        self.embedder_weights = embedder_weights
        self.half = half
        self.reid_name = embedder_model_name or embedder_weights or embedder
        self._embedder_gpu = _resolve_embedder_gpu(embedder_gpu)
        self._tracker = self._build_tracker()

    def _build_tracker(self) -> DeepSort:
        return DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            max_iou_distance=0.7,
            embedder=self.embedder,
            half=self.half,
            bgr=True,
            embedder_gpu=self._embedder_gpu,
            embedder_model_name=self.embedder_model_name,
            embedder_wts=self.embedder_weights,
        )

    def reset(self) -> None:
        self._tracker = self._build_tracker()

    def update(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        formatted = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            width = max(0.0, float(x2 - x1))
            height = max(0.0, float(y2 - y1))
            if width <= 0 or height <= 0:
                continue
            formatted.append(([float(x1), float(y1), width, height], float(det.get("confidence", 0.0)), "person"))

        raw_tracks = self._tracker.update_tracks(formatted, frame=frame)

        tracks: list[dict[str, Any]] = []
        for track in raw_tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = [int(v) for v in track.to_ltrb()]
            tracks.append(
                {
                    "track_id": int(track.track_id),
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(getattr(track, "det_conf", 0.0) or 0.0),
                }
            )
        return tracks


def _resolve_embedder_gpu(explicit_value: bool | None) -> bool:
    if explicit_value is not None:
        return explicit_value
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
