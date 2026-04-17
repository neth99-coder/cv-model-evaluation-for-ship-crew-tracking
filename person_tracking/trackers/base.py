"""Shared tracker interface for person tracking evaluation."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePersonTracker(ABC):
    """Abstract interface for per-frame person trackers."""

    model_name: str = "BaseTracker"

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracker state before processing a new video."""

    @abstractmethod
    def update(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
        Track persons in a BGR frame.

        Returns a list of dicts:
            {
                "track_id": int,
                "bbox": (x1, y1, x2, y2),
                "confidence": float,
            }
        """
