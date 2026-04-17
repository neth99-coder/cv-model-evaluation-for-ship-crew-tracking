"""Generic BoxMOT tracker wrapper that accepts external detector outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
from boxmot.engine.workflow_support import REID_TRACKERS, reid_path_from_spec
from boxmot.trackers.tracker_zoo import TRACKER_MAPPING, create_tracker, get_tracker_config
from boxmot.utils.torch_utils import select_device

from .base import BasePersonTracker, detections_to_boxmot_array, format_boxmot_tracks

DISPLAY_NAMES = {
    "botsort": "BoT-SORT",
    "bytetrack": "ByteTrack",
    "strongsort": "StrongSORT",
    "deepocsort": "DeepOCSORT",
    "ocsort": "OCSORT",
    "sfsort": "SFSORT",
    "hybridsort": "HybridSORT",
    "boosttrack": "BoostTrack",
}


class BoxMOTTrackerClient(BasePersonTracker):
    """Detector-agnostic BoxMOT tracker runtime."""

    def __init__(
        self,
        tracker_backend: str,
        *,
        reid_model: str | None = None,
        device: str = "cpu",
        half: bool = False,
        backend_label: str = "boxmot",
    ) -> None:
        normalized = tracker_backend.strip().lower()
        if normalized not in TRACKER_MAPPING:
            raise ValueError(
                f"Unsupported BoxMOT tracker '{tracker_backend}'. Supported values: {', '.join(sorted(TRACKER_MAPPING))}"
            )

        self.tracker_backend = normalized
        self.model_name = DISPLAY_NAMES.get(normalized, normalized.title())
        self.backend = backend_label
        self.half = half
        self._device = select_device(device)
        self.reid_name = None
        if normalized in REID_TRACKERS:
            self.reid_name = str(reid_model or "osnet_x0_25_msmt17")

        self._tracker = self._build_tracker()

    def _build_tracker(self) -> Any:
        reid_weights = None
        if self.reid_name is not None:
            reid_weights = reid_path_from_spec(self.reid_name, required=True)
        return create_tracker(
            tracker_type=self.tracker_backend,
            tracker_config=get_tracker_config(self.tracker_backend),
            reid_weights=reid_weights,
            device=self._device,
            half=self.half,
            per_class=False,
        )

    def reset(self) -> None:
        self._tracker = self._build_tracker()

    def update(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        raw_tracks = self._tracker.update(detections_to_boxmot_array(detections), frame)
        return format_boxmot_tracks(raw_tracks)