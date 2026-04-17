"""Shared runtime helpers for tracking evaluation and live playback."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from detectors import build_detector
from trackers import build_tracker

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class TrackingFrameResult:
    detections: list[dict[str, Any]]
    tracks: list[dict[str, Any]]
    detector_ms: float
    tracker_ms: float

    @property
    def total_ms(self) -> float:
        return self.detector_ms + self.tracker_ms


class TrackingRuntime:
    """Pair one detector with one tracker and process frames end to end."""

    def __init__(self, detector: Any, tracker: Any) -> None:
        self.detector = detector
        self.tracker = tracker

    @property
    def tracker_name(self) -> str:
        return self.tracker.model_name

    @property
    def tracker_backend(self) -> str:
        return self.tracker.tracker_backend

    @property
    def detector_name(self) -> str:
        return self.detector.model_name

    @property
    def reid_name(self) -> str | None:
        return self.tracker.reid_name

    @property
    def backend(self) -> str:
        return self.tracker.backend

    @property
    def run_name(self) -> str:
        parts = [self.tracker_name, self.detector_name]
        if self.reid_name:
            parts.append(str(self.reid_name))
        return "__".join(_slugify(part) for part in parts)

    def reset(self) -> None:
        self.tracker.reset()

    def process_frame(self, frame: np.ndarray) -> TrackingFrameResult:
        detections, detector_ms = self.detector.detect_timed(frame)
        t0 = time.perf_counter()
        tracks = self.tracker.update(frame, detections)
        tracker_ms = (time.perf_counter() - t0) * 1_000.0
        return TrackingFrameResult(
            detections=detections,
            tracks=tracks,
            detector_ms=detector_ms,
            tracker_ms=tracker_ms,
        )


def build_runtime(
    detector_spec: str,
    tracker_spec: Any,
    *,
    confidence_threshold: float,
    reid_model: str | None,
    device: str,
    half: bool,
    deepsort_embedder: str,
    deepsort_embedder_model: str | None,
    deepsort_embedder_weights: str | None,
) -> TrackingRuntime:
    detector = build_detector(detector_spec, confidence_threshold=confidence_threshold)
    tracker = build_tracker(
        tracker_spec,
        reid_model=reid_model,
        device=device,
        half=half,
        deepsort_embedder=deepsort_embedder,
        deepsort_embedder_model=deepsort_embedder_model,
        deepsort_embedder_weights=deepsort_embedder_weights,
    )
    return TrackingRuntime(detector=detector, tracker=tracker)


def annotate_tracks(
    frame: np.ndarray,
    tracks: list[dict[str, Any]],
    header_lines: Iterable[str] | None = None,
) -> np.ndarray:
    out = frame.copy()

    if header_lines:
        y = 24
        for line in header_lines:
            if not line:
                continue
            cv2.rectangle(out, (10, y - 16), (330, y + 8), (20, 20, 20), -1)
            cv2.putText(out, str(line), (14, y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24

    for track in tracks:
        x1, y1, x2, y2 = track["bbox"]
        track_id = int(track["track_id"])
        confidence = float(track.get("confidence", 0.0))
        colour = _track_colour(track_id)

        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = f"ID {track_id} {confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        ly = max(th + baseline + 4, y1)
        cv2.rectangle(out, (x1, ly - th - baseline - 4), (x1 + tw + 6, ly), colour, -1)
        cv2.putText(out, label, (x1 + 3, ly - baseline - 2), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return out


def append_mot_rows(handle: Any, frame_index: int, tracks: list[dict[str, Any]]) -> None:
    for track in tracks:
        x1, y1, x2, y2 = track["bbox"]
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        handle.write(
            f"{frame_index},{int(track['track_id'])},{x1},{y1},{width},{height},{float(track.get('confidence', 0.0)):.6f},-1,-1,-1\n"
        )


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "run"


def _track_colour(track_id: int) -> tuple[int, int, int]:
    base = (track_id * 123457) % 255
    return int(base), int((base * 3) % 255), int((base * 7) % 255)