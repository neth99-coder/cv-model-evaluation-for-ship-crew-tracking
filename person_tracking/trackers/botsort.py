"""Custom BoT-SORT wrapper with pluggable detector and ReID backends."""

from .boxmot_client import BoxMOTTrackerClient


class BoTSORTTracker(BoxMOTTrackerClient):
    model_name = "BoT-SORT"
    tracker_backend = "botsort"

    def __init__(
        self,
        *,
        reid_model: str | None = "osnet_x0_25_msmt17",
        device: str = "cpu",
        half: bool = False,
    ) -> None:
        super().__init__(
            tracker_backend="botsort",
            reid_model=reid_model,
            device=device,
            half=half,
            backend_label="custom",
        )
        self.model_name = "BoT-SORT"
