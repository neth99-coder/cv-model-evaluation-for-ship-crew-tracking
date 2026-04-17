"""Detector registry for person-tracking pipelines."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person_detection.models.base import BasePersonDetector
from person_detection.models.faster_rcnn import FasterRCNNDetector
from person_detection.models.ssd_mobilenet import SSDMobileNetDetector
from person_detection.models.yolov5 import YOLOv5Detector
from person_detection.models.yolov8 import YOLOv8Detector

DEFAULT_DETECTORS = (
    "yolov8n",
    "yolov5s",
    "faster_rcnn",
    "ssd_mobilenet",
)

SUPPORTED_DETECTORS = (
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolov5s",
    "yolov5m",
    "yolov5l",
    "yolov5x",
    "faster_rcnn",
    "ssd_mobilenet",
)

_ALIASES = {
    "fasterrcnn": "faster_rcnn",
    "faster_rcnn": "faster_rcnn",
    "ssd": "ssd_mobilenet",
    "ssd_mobilenet": "ssd_mobilenet",
    "ssdlite": "ssd_mobilenet",
}


def normalize_detector_spec(spec: str) -> str:
    normalized = spec.strip().lower()
    normalized = _ALIASES.get(normalized, normalized)
    if normalized in SUPPORTED_DETECTORS:
        return normalized
    raise ValueError(
        f"Unsupported detector '{spec}'. Supported values: {', '.join(SUPPORTED_DETECTORS)}"
    )


def expand_detector_specs(specs: list[str]) -> list[str]:
    normalized: list[str] = []
    for spec in specs:
        if spec.lower() == "all":
            normalized.extend(DEFAULT_DETECTORS)
            continue
        normalized.append(normalize_detector_spec(spec))
    return list(dict.fromkeys(normalized))


def build_detector(spec: str, confidence_threshold: float = 0.25) -> BasePersonDetector:
    normalized = normalize_detector_spec(spec)
    if normalized.startswith("yolov8"):
        return YOLOv8Detector(
            model_size=normalized.removeprefix("yolov8") or "n",
            confidence_threshold=confidence_threshold,
        )
    if normalized.startswith("yolov5"):
        return YOLOv5Detector(
            model_size=normalized.removeprefix("yolov5") or "s",
            confidence_threshold=confidence_threshold,
        )
    if normalized == "faster_rcnn":
        return FasterRCNNDetector(confidence_threshold=confidence_threshold)
    if normalized == "ssd_mobilenet":
        return SSDMobileNetDetector(confidence_threshold=confidence_threshold)
    raise ValueError(f"Unsupported detector '{spec}'")