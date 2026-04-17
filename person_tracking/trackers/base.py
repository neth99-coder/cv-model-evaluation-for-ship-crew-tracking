"""Shared tracker interfaces and helpers for person tracking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePersonTracker(ABC):
    """Abstract interface for trackers that consume external person detections."""

    model_name: str = "BaseTracker"
    backend: str = "custom"
    tracker_backend: str = "base"
    reid_name: str | None = None

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state before processing a new video stream."""

    @abstractmethod
    def update(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Update tracker state from a frame and person detections."""


def detections_to_boxmot_array(detections: list[dict[str, Any]]) -> np.ndarray:
    """Convert repo detection dicts into the Nx6 layout expected by BoxMOT."""
    if not detections:
        return np.empty((0, 6), dtype=np.float32)

    rows: list[list[float]] = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        rows.append(
            [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(det.get("confidence", 0.0)),
                0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def format_boxmot_tracks(raw_tracks: np.ndarray | list[Any] | tuple[Any, ...]) -> list[dict[str, Any]]:
    """Normalize BoxMOT tracker outputs into the repo's track dict format."""
    tracks_arr = np.asarray(raw_tracks, dtype=np.float32)
    if tracks_arr.size == 0:
        return []
    if tracks_arr.ndim == 1:
        tracks_arr = tracks_arr.reshape(1, -1)

    tracks: list[dict[str, Any]] = []
    for row in tracks_arr:
        if len(row) < 6:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in row[:4]]
        tracks.append(
            {
                "track_id": int(row[4]),
                "bbox": (x1, y1, x2, y2),
                "confidence": float(row[5]),
            }
        )
    return tracks
