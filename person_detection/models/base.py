"""
Abstract base class for all person detection models.
"""
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePersonDetector(ABC):
    """
    Abstract base for person detectors.

    Subclasses must implement :meth:`detect`, which runs inference on a
    single BGR ``numpy`` frame and returns a list of detection dicts:

        [{"bbox": (x1, y1, x2, y2), "confidence": float}, ...]

    All coordinates are integer pixel values in the input frame's space.
    """

    model_name: str = "Base"
    confidence_threshold: float = 0.5

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
        Run person detection on a single BGR frame.

        Returns
        -------
        List of dicts with keys:
            ``bbox``        – (x1, y1, x2, y2) in pixel coordinates
            ``confidence``  – float in [0, 1]
        """

    # ------------------------------------------------------------------
    # Timing wrapper
    # ------------------------------------------------------------------

    def detect_timed(self, frame: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        """
        Call :meth:`detect` and return ``(detections, inference_time_ms)``.

        The inference time covers only the model forward pass; frame
        decoding / video I/O is excluded.
        """
        t0 = time.perf_counter()
        detections = self.detect(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1_000.0
        return detections, elapsed_ms
