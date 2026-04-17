"""Custom ByteTrack wrapper with pluggable detector backends."""

from .boxmot_client import BoxMOTTrackerClient


class ByteTrackTracker(BoxMOTTrackerClient):
    model_name = "ByteTrack"
    tracker_backend = "bytetrack"

    def __init__(self, *, device: str = "cpu", half: bool = False) -> None:
        super().__init__(
            tracker_backend="bytetrack",
            reid_model=None,
            device=device,
            half=half,
            backend_label="custom",
        )
        self.model_name = "ByteTrack"
