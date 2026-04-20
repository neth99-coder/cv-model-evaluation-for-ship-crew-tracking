"""
FairMOT Tracker Wrapper
Pluggable backbone: dla34 | hrnet | resnet50
FairMOT jointly learns detection + Re-ID in a single network.
Installs missing dependencies on first run.
"""

import numpy as np
import subprocess
import sys


def _install_packages(packages: list[str]) -> None:
    """Install required Python packages using the active interpreter."""
    print(f"[FairMOT] Installing missing packages: {packages}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])


def _is_missing_dependency_error(err: Exception, module_tokens: list[str]) -> bool:
    """Return True only when the exception points to missing Python modules."""
    if isinstance(err, ModuleNotFoundError):
        missing = (getattr(err, "name", "") or "").lower()
        return any(m in missing for m in module_tokens)
    if isinstance(err, ImportError):
        msg = str(err).lower()
        return any(m in msg for m in module_tokens)
    return False


class FairMOTTracker:
    """
    Wraps FairMOT for person tracking.
    FairMOT has built-in Re-ID — no external Re-ID model needed.
    Supported backbones: dla34, hrnet, resnet50
    """

    def __init__(self, backbone: str = "dla34", conf: float = 0.4):
        self.backbone = backbone
        self.conf = conf
        self._tracker = None
        self._next_id = 1
        self._tracks: dict[int, dict] = {}
        self._install_attempted = False
        self._init()

    def _init(self):
        try:
            # Try to import FairMOT (requires the fairmot package to be installed)
            from fairmot.tracker.multitracker import JDETracker
            from fairmot.opts import opts as FairMOTOpts

            opt = FairMOTOpts().parse([])
            opt.conf_thres = self.conf
            if self.backbone == "hrnet":
                opt.arch = "hrnet_32"
            elif self.backbone == "resnet50":
                opt.arch = "resdcn_50"
            else:
                opt.arch = "dla_34"

            self._tracker = JDETracker(opt, frame_rate=30)
            print(f"[FairMOT] Initialized with backbone={self.backbone}")
        except Exception as e:
            if (not self._install_attempted
                    and _is_missing_dependency_error(e, ["fairmot"])):
                self._install_attempted = True
                _install_packages(["fairmot"])
                return self._init()
            raise RuntimeError(f"FairMOT init failed: {e}") from e

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one frame, return tracks as [x1,y1,x2,y2,track_id,score].
        """
        if self._tracker is None:
            raise RuntimeError("FairMOT tracker is not initialized")
        try:
            import torch, cv2
            blob = self._preprocess(frame)
            online_targets = self._tracker.update(blob, frame.shape[:2], (frame.shape[0], frame.shape[1]))
            tracks = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                x1, y1, w, h = tlwh
                # FairMOT wrapper does not expose per-track confidence here.
                tracks.append([x1, y1, x1 + w, y1 + h, tid, 1.0])
            return np.array(tracks) if tracks else np.empty((0, 6))
        except Exception as e:
            raise RuntimeError(f"FairMOT update error: {e}") from e

    def _preprocess(self, frame: np.ndarray) -> "torch.Tensor":
        import torch, cv2
        img = cv2.resize(frame, (1088, 608))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0)

    def runtime_config(self) -> dict:
        return {
            "framework": "fairmot",
            "tracker": "fairmot",
            "detector": self.backbone,
            "reid_model": "built-in",
        }

    # ── Mock ──────────────────────────────────────────────────────────────────

    def _mock_update(self, frame: np.ndarray) -> np.ndarray:
        print("[FairMOT] Mock update called.")
        if not hasattr(self, '_mock_tracks'):
            self._mock_tracks: dict = {}
            self._mock_next_id = 1

        h, w = frame.shape[:2]
        n = np.random.randint(1, 4)
        new_boxes = []
        for _ in range(n):
            x1 = np.random.randint(0, w // 2)
            y1 = np.random.randint(0, h // 2)
            x2 = min(x1 + np.random.randint(60, 160), w)
            y2 = min(y1 + np.random.randint(120, 280), h)
            new_boxes.append([x1, y1, x2, y2])
        new_boxes = np.array(new_boxes, dtype=float)

        matched_ids = [None] * len(new_boxes)
        if self._mock_tracks:
            track_ids = list(self._mock_tracks.keys())
            track_boxes = np.array([self._mock_tracks[tid] for tid in track_ids])
            iou_mat = _iou_matrix(track_boxes, new_boxes)
            used_tracks, used_dets = set(), set()
            pairs = sorted(
                [(iou_mat[ti, di], ti, di)
                 for ti in range(len(track_ids))
                 for di in range(len(new_boxes))],
                reverse=True
            )
            for iou_val, ti, di in pairs:
                if iou_val < 0.3:
                    break
                if ti in used_tracks or di in used_dets:
                    continue
                matched_ids[di] = track_ids[ti]
                used_tracks.add(ti)
                used_dets.add(di)

        new_tracks = {}
        result = []
        for di, box in enumerate(new_boxes):
            tid = matched_ids[di]
            if tid is None:
                tid = self._mock_next_id
                self._mock_next_id += 1
            new_tracks[tid] = box.tolist()
            result.append([box[0], box[1], box[2], box[3], tid])

        self._mock_tracks = new_tracks
        return np.array(result, dtype=float)


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)
