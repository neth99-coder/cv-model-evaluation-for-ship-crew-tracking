"""
Tracking metrics collector.
"""
import time
import numpy as np
from collections import defaultdict


class TrackingMetrics:
    def __init__(self):
        self.frame_times: list[float] = []
        self.track_counts: list[int] = []
        self.unique_ids: set[int] = set()
        self.id_lifetimes: dict[int, int] = defaultdict(int)
        self.start_time = time.time()
        self.total_frames = 0

    def update(self, tracks: np.ndarray, elapsed: float, frame_idx: int):
        self.frame_times.append(elapsed)
        self.track_counts.append(len(tracks))
        self.total_frames += 1
        for t in tracks:
            tid = int(t[4])
            self.unique_ids.add(tid)
            self.id_lifetimes[tid] += 1

    def summary(self) -> dict:
        if not self.frame_times:
            return {}

        fps_list = [1.0 / t for t in self.frame_times if t > 0]
        avg_fps = float(np.mean(fps_list)) if fps_list else 0.0
        avg_tracks = float(np.mean(self.track_counts)) if self.track_counts else 0.0
        max_tracks = int(max(self.track_counts)) if self.track_counts else 0
        avg_lifetime = float(np.mean(list(self.id_lifetimes.values()))) if self.id_lifetimes else 0.0

        return {
            "avg_fps": round(avg_fps, 2),
            "min_fps": round(min(fps_list), 2) if fps_list else 0,
            "max_fps": round(max(fps_list), 2) if fps_list else 0,
            "total_frames_processed": self.total_frames,
            "unique_tracks": len(self.unique_ids),
            "avg_tracks_per_frame": round(avg_tracks, 2),
            "max_simultaneous_tracks": max_tracks,
            "avg_track_lifetime_frames": round(avg_lifetime, 2),
            "total_wall_time_s": round(time.time() - self.start_time, 2),
        }
