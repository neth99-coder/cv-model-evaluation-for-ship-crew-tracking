"""
BOXMOT Tracker Wrapper
Pluggable: tracker (bytetrack/botsort/ocsort/deepocsort/strongsort) + detector (yolov8/yolov5) + optional RE-ID
"""

import numpy as np
from pathlib import Path
import subprocess
import sys

from trackers.reid_helpers import AppearanceEncoder, StableIdentityMemory, normalize_embeddings

REID_SUPPORTED = {"botsort", "deepocsort", "strongsort"}

REID_TRACKER_KWARGS = {
    # Default library settings are conservative for short MOT benchmarks.
    # Ship-crew tracking needs longer memory and looser appearance matching
    # so an earlier identity can be recovered after brief misses/occlusions.
    "botsort": {
        "track_buffer": 90,
        "match_thresh": 0.85,
        "proximity_thresh": 0.2,
        "appearance_thresh": 0.35,
        "fuse_first_associate": True,
        "max_age": 300,
        "min_hits": 1,
    },
    "deepocsort": {
        "w_association_emb": 0.75,
        "aw_param": 0.3,
        "max_age": 300,
        "min_hits": 1,
    },
    "strongsort": {
        "max_cos_dist": 0.35,
        "max_iou_dist": 0.85,
        "n_init": 1,
        "nn_budget": 200,
        "max_age": 300,
    },
}

YOLO_WEIGHTS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo11x": "yolo11x.pt",
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26m": "yolo26m.pt",
    "yolo26l": "yolo26l.pt",
    "yolo26x": "yolo26x.pt",
    # Map legacy YOLOv5 names to Ultralytics v8-compatible YOLOv5u checkpoints.
    "yolov5n": "yolov5nu.pt",
    "yolov5s": "yolov5su.pt",
    "yolov5m": "yolov5mu.pt",
    "yolov5l": "yolov5lu.pt",
    "yolov5x": "yolov5xu.pt",
}


def _install_packages(packages: list[str]) -> None:
    """Install required Python packages using the active interpreter."""
    print(f"[BoxMOT] Installing missing packages: {packages}")
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


class BoxMOTTracker:
    """
    Wraps the BOXMOT library to provide a unified tracking interface.
    Installs missing dependencies on first run.
    """

    def __init__(self, tracker_type: str = "bytetrack", detector: str = "yolov8n",
                 reid_model: str | None = None, conf: float = 0.4,
                 manual_reid: bool = False):
        self.tracker_type = tracker_type
        self.detector_name = detector
        self.reid_model = reid_model if tracker_type in REID_SUPPORTED else None
        self.conf = conf
        self.manual_reid = bool(manual_reid)
        self._manual_reid_model = reid_model or ("osnet_x0_25" if self.manual_reid else None)
        self._tracker = None
        self._detector = None
        self._detector_install_attempted = False
        self._tracker_install_attempted = False
        self._appearance_encoder = (
            AppearanceEncoder(self._manual_reid_model, "BoxMOT")
            if self._manual_reid_model else None
        )
        self._stable_memory = StableIdentityMemory()
        self._init_detector()
        self._init_tracker()

    # ── Detector ────────────────────────────────────────────────────────────

    def _init_detector(self):
        try:
            if self.detector_name.startswith("yolo"):
                from ultralytics import YOLO
                weights = YOLO_WEIGHTS.get(self.detector_name, f"{self.detector_name}.pt")
                self._detector = YOLO(weights)
            elif self.detector_name == "fasterrcnn":
                import torch
                from torchvision.models.detection import (
                    fasterrcnn_resnet50_fpn_v2,
                    FasterRCNN_ResNet50_FPN_V2_Weights,
                )
                self._detector = fasterrcnn_resnet50_fpn_v2(
                    weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                )
                self._detector.eval()
            elif self.detector_name == "ssd_mobilenet":
                import torch
                from torchvision.models.detection import (
                    ssdlite320_mobilenet_v3_large,
                    SSDLite320_MobileNet_V3_Large_Weights,
                )
                self._detector = ssdlite320_mobilenet_v3_large(
                    weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
                )
                self._detector.eval()
            else:
                raise ValueError(f"Unsupported detector: {self.detector_name}")
        except Exception as e:
            if (not self._detector_install_attempted
                    and _is_missing_dependency_error(e, ["ultralytics", "torch", "torchvision"])):
                self._detector_install_attempted = True
                _install_packages(["ultralytics", "torch", "torchvision"])
                return self._init_detector()
            raise RuntimeError(f"BoxMOT detector init failed: {e}") from e

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        """Returns [x1,y1,x2,y2,conf,cls] detections for class 'person' (cls=0)."""
        if self._detector is None:
            raise RuntimeError("BoxMOT detector is not initialized")
        try:
            if self.detector_name.startswith("yolo"):
                results = self._detector.predict(frame, conf=self.conf, classes=[0], verbose=False)
                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    return np.empty((0, 6))
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy().reshape(-1, 1)
                cls = boxes.cls.cpu().numpy().reshape(-1, 1)
                return np.hstack([xyxy, confs, cls])
            elif self.detector_name in ("fasterrcnn", "ssd_mobilenet"):
                return self._torchvision_detect(frame)
        except Exception as e:
            raise RuntimeError(f"BoxMOT detection error: {e}") from e

    def _torchvision_detect(self, frame: np.ndarray) -> np.ndarray:
        """Shared torchvision detection path (FasterRCNN / SSD MobileNet)."""
        import torch
        import torchvision.transforms.functional as F
        rgb = frame[:, :, ::-1].copy()          # BGR → RGB
        img_t = F.to_tensor(rgb)
        with torch.no_grad():
            out = self._detector([img_t])
        boxes  = out[0]["boxes"].cpu().numpy()   # [N, 4] xyxy
        scores = out[0]["scores"].cpu().numpy()
        labels = out[0]["labels"].cpu().numpy()
        mask = (labels == 1) & (scores >= self.conf)  # COCO person = 1
        if not mask.any():
            return np.empty((0, 6))
        cls_col = np.zeros((mask.sum(), 1))
        return np.hstack([boxes[mask], scores[mask].reshape(-1, 1), cls_col])

    # ── Tracker ─────────────────────────────────────────────────────────────

    def _init_tracker(self):
        try:
            import torch
            from boxmot.trackers.bytetrack.bytetrack import ByteTrack
            from boxmot.trackers.botsort.botsort import BotSort
            from boxmot.trackers.ocsort.ocsort import OcSort
            from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
            from boxmot.trackers.strongsort.strongsort import StrongSort

            device = torch.device("cpu")

            if self.tracker_type in REID_SUPPORTED and not self.reid_model:
                raise ValueError(f"Tracker '{self.tracker_type}' requires a reid_model")

            if self.tracker_type in REID_SUPPORTED:
                reid_path = self._get_reid_path()
                reid_trackers = {
                    "botsort": BotSort,
                    "deepocsort": DeepOcSort,
                    "strongsort": StrongSort,
                }
                cls = reid_trackers[self.tracker_type]
                self._tracker = cls(
                    reid_weights=reid_path,
                    device=device,
                    half=False,
                    **REID_TRACKER_KWARGS.get(self.tracker_type, {}),
                )
            else:
                no_reid_map = {
                    "bytetrack": ByteTrack,
                    "ocsort": OcSort,
                }
                if self.tracker_type not in no_reid_map:
                    supported = ", ".join(sorted([*no_reid_map.keys(), *REID_SUPPORTED]))
                    raise ValueError(f"Unsupported BoxMOT tracker '{self.tracker_type}'. Supported: {supported}")
                cls = no_reid_map[self.tracker_type]
                self._tracker = cls()

            print(f"[BoxMOT] {self.tracker_type} initialized OK.")
        except Exception as e:
            if (not self._tracker_install_attempted
                    and _is_missing_dependency_error(e, ["boxmot"])):
                self._tracker_install_attempted = True
                _install_packages(["boxmot"])
                return self._init_tracker()
            raise RuntimeError(f"BoxMOT tracker init failed: {e}") from e

    def _get_reid_path(self) -> Path:
        """Return path to reid weights, downloading via boxmot if needed."""
        # Map short model names to the msmt17-trained filenames boxmot auto-downloads
        name_map = {
            "osnet_x0_25": "osnet_x0_25_msmt17.pt",
            "osnet_x1_0": "osnet_x1_0_msmt17.pt",
            "resnet50": "resnet50_msmt17.pt",
            "mlfn": "mlfn_msmt17.pt",
        }
        filename = name_map.get(self.reid_model, f"{self.reid_model}.pt")
        try:
            from boxmot.utils import WEIGHTS
            return WEIGHTS / filename
        except Exception:
            return Path(filename)

    def runtime_config(self) -> dict:
        return {
            "framework": "boxmot",
            "tracker": self.tracker_type,
            "detector": self.detector_name,
            "reid_model": self.reid_model,
            "manual_reid": self.manual_reid,
            "manual_reid_model": self._manual_reid_model,
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one frame and return tracks.
        Returns:
        - [N, 6] -> [x1, y1, x2, y2, track_id, score] when score is available
        - [N, 5] -> [x1, y1, x2, y2, track_id] otherwise
        """
        dets = self._detect(frame)
        if self._tracker is None:
            raise RuntimeError("BoxMOT tracker is not initialized")
        try:
            if len(dets) == 0:
                dets_in = np.empty((0, 6))
                embs = None
            else:
                dets_in = dets[:, :6]
                embs = self._extract_detection_embeddings(frame, dets_in[:, :4])
            if self.tracker_type in REID_SUPPORTED:
                tracks = self._tracker.update(dets_in, frame, embs=embs)
            else:
                tracks = self._tracker.update(dets_in, frame)
            self._stable_memory.step()
            if tracks is None or len(tracks) == 0:
                self._stable_memory.clear_active_mapping()
                return np.empty((0, 5))
            if self.manual_reid:
                tracks = self._stabilize_track_ids(frame, tracks)
            # Common boxmot output includes confidence at index 5.
            if tracks.shape[1] > 5:
                return tracks[:, [0, 1, 2, 3, 4, 5]].astype(float)
            return tracks[:, :5].astype(float)
        except Exception as e:
            raise RuntimeError(f"BoxMOT tracker update error: {e}") from e

    def _stabilize_track_ids(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        boxes = np.asarray(tracks[:, :4], dtype=float)
        internal_ids = [int(float(track_id)) for track_id in tracks[:, 4]]
        embeddings = self._extract_detection_embeddings(frame, boxes)
        return self._stable_memory.reassign(
            tracks=tracks,
            boxes=boxes,
            internal_ids=internal_ids,
            embeddings=embeddings,
            stringify_ids=False,
        )

    def _extract_detection_embeddings(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray | None:
        if boxes is None or len(boxes) == 0:
            return None

        native_embeddings = None
        model = getattr(self._tracker, "model", None)
        if model is not None and hasattr(model, "get_features"):
            try:
                native_embeddings = normalize_embeddings(model.get_features(boxes, frame))
            except Exception:
                native_embeddings = None

        if native_embeddings is not None and len(native_embeddings) == len(boxes):
            return native_embeddings
        if self._appearance_encoder is None:
            return None
        return self._appearance_encoder.extract(frame, boxes)

    # ── Mock helpers (no deps) ───────────────────────────────────────────────

    def _mock_detect(self, frame: np.ndarray) -> np.ndarray:
        print("[BoxMOT] Mock detect called.")
        h, w = frame.shape[:2]
        n = np.random.randint(1, 4)
        dets = []
        for _ in range(n):
            x1 = np.random.randint(0, w // 2)
            y1 = np.random.randint(0, h // 2)
            x2 = x1 + np.random.randint(50, 150)
            y2 = y1 + np.random.randint(100, 250)
            dets.append([min(x1, w), min(y1, h), min(x2, w), min(y2, h),
                          np.random.uniform(0.5, 0.95), 0])
        return np.array(dets)

    def __init_mock_state(self):
        """Lazy-init per-instance mock state."""
        if not hasattr(self, '_mock_next_id'):
            self._mock_next_id = 1
            self._mock_tracks: dict = {}  # track_id -> last [x1,y1,x2,y2]

    def _mock_track(self, dets: np.ndarray) -> np.ndarray:
        """IoU-based greedy matching to keep stable IDs across frames."""
        self.__init_mock_state()
        if len(dets) == 0:
            self._mock_tracks.clear()
            return np.empty((0, 5))

        boxes = dets[:, :4]
        matched_ids = [None] * len(boxes)

        if self._mock_tracks:
            track_ids = list(self._mock_tracks.keys())
            track_boxes = np.array([self._mock_tracks[tid] for tid in track_ids])
            iou_mat = _iou_matrix(track_boxes, boxes)

            used_tracks, used_dets = set(), set()
            # greedy: best IoU first
            pairs = sorted(
                [(iou_mat[ti, di], ti, di)
                 for ti in range(len(track_ids))
                 for di in range(len(boxes))],
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
        for di, box in enumerate(boxes):
            tid = matched_ids[di]
            if tid is None:
                tid = self._mock_next_id
                self._mock_next_id += 1
            new_tracks[tid] = box.tolist()
            result.append([box[0], box[1], box[2], box[3], tid])

        self._mock_tracks = new_tracks
        return np.array(result, dtype=float)


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of [x1,y1,x2,y2] boxes."""
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


def _center_distance_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax = float(box_a[0] + box_a[2]) / 2.0
    ay = float(box_a[1] + box_a[3]) / 2.0
    bx = float(box_b[0] + box_b[2]) / 2.0
    by = float(box_b[1] + box_b[3]) / 2.0
    dist = float(np.hypot(ax - bx, ay - by))

    aw = max(float(box_a[2] - box_a[0]), 1.0)
    ah = max(float(box_a[3] - box_a[1]), 1.0)
    bw = max(float(box_b[2] - box_b[0]), 1.0)
    bh = max(float(box_b[3] - box_b[1]), 1.0)
    scale = max(np.hypot(aw, ah), np.hypot(bw, bh), 1.0)
    return dist / scale
