"""
DeepSORT Tracker Wrapper
Pluggable: detector (yolov8/yolov5) + RE-ID model (osnet/resnet)
Installs missing dependencies on first run.
"""

import numpy as np
import subprocess
import sys

from trackers.reid_helpers import AppearanceEncoder, StableIdentityMemory, normalize_embeddings


DEEPSORT_KWARGS = {
    # Keep tracks alive longer and allow appearance matching across larger motion gaps.
    "max_iou_distance": 0.85,
    "max_age": 120,
    "n_init": 1,
    "max_cosine_distance": 0.30,
    "nn_budget": 300,
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
    print(f"[DeepSORT] Installing missing packages: {packages}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])


def _is_missing_dependency_error(err: Exception, module_tokens: list[str]) -> bool:
    """Return True only when the exception points to missing Python modules."""
    msg = str(err).lower()
    if isinstance(err, ModuleNotFoundError):
        missing = (getattr(err, "name", "") or "").lower()
        return any(m in missing for m in module_tokens)
    if isinstance(err, ImportError):
        return any(m in msg for m in module_tokens)
    return any(m in msg for m in module_tokens)


class DeepSORTTracker:
    """
    Wraps DeepSORT with a pluggable detector and RE-ID model.
    Supported detectors: YOLOv8, YOLO11, YOLO26, YOLOv5, Faster R-CNN, SSD MobileNet
    Supported RE-ID: osnet_x0_25, osnet_x1_0, resnet50
    """

    def __init__(self, detector: str = "yolov8n", reid_model: str = "osnet_x0_25",
                 conf: float = 0.4, manual_reid: bool = False):
        self.detector_name = detector
        self.reid_model = reid_model
        self.conf = conf
        self.manual_reid = bool(manual_reid)
        self._detector = None
        self._tracker = None
        self._detector_install_attempted = False
        self._tracker_install_attempted = False
        self._appearance_encoder = AppearanceEncoder(reid_model, "DeepSORT")
        self._stable_memory = StableIdentityMemory()
        self._init_detector()
        self._init_tracker()

    # ── Detector ─────────────────────────────────────────────────────────────

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
            raise RuntimeError(f"DeepSORT detector init failed: {e}") from e

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        """Returns [x1,y1,x2,y2,conf] for persons only."""
        if self._detector is None:
            raise RuntimeError("DeepSORT detector is not initialized")
        try:
            if self.detector_name.startswith("yolo"):
                results = self._detector.predict(frame, conf=self.conf, classes=[0], verbose=False)
                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    return np.empty((0, 5))
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy().reshape(-1, 1)
                return np.hstack([xyxy, confs])
            elif self.detector_name in ("fasterrcnn", "ssd_mobilenet"):
                return self._torchvision_detect(frame)
        except Exception as e:
            raise RuntimeError(f"DeepSORT detection error: {e}") from e

    def _torchvision_detect(self, frame: np.ndarray) -> np.ndarray:
        """Shared torchvision detection path (FasterRCNN / SSD MobileNet)."""
        import torch
        import torchvision.transforms.functional as F
        rgb = frame[:, :, ::-1].copy()           # BGR -> RGB
        img_t = F.to_tensor(rgb)
        with torch.no_grad():
            out = self._detector([img_t])
        boxes  = out[0]["boxes"].cpu().numpy()    # [N,4] xyxy
        scores = out[0]["scores"].cpu().numpy()
        labels = out[0]["labels"].cpu().numpy()
        mask = (labels == 1) & (scores >= self.conf)  # COCO person = 1
        if not mask.any():
            return np.empty((0, 5))
        return np.hstack([boxes[mask], scores[mask].reshape(-1, 1)])

    # ── RE-ID Encoder ─────────────────────────────────────────────────────────

    def _init_tracker(self):
        try:
            # deep_sort_realtime package
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._tracker = DeepSort(
                embedder=None,
                half=False,
                **DEEPSORT_KWARGS,
            )
            print(
                f"[DeepSORT] Initialized with external embeddings, reid={self.reid_model}"
            )
        except Exception as e:
            if not self._tracker_install_attempted:
                if _is_missing_dependency_error(e, ["deep_sort_realtime", "deep-sort-realtime"]):
                    self._tracker_install_attempted = True
                    _install_packages(["deep-sort-realtime"])
                    return self._init_tracker()
            raise RuntimeError(f"DeepSORT tracker init failed: {e}") from e

    def runtime_config(self) -> dict:
        return {
            "framework": "deepsort",
            "tracker": "deepsort",
            "detector": self.detector_name,
            "reid_model": self.reid_model,
            "embedder": "external",
            "embedder_model_name": self.reid_model,
            "manual_reid": self.manual_reid,
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> np.ndarray:
        dets = self._detect(frame)
        if self._tracker is None:
            raise RuntimeError("DeepSORT tracker is not initialized")

        try:
            # deep_sort_realtime format: [[x1,y1,w,h], conf, class]
            ds_dets = []
            det_boxes = []
            for d in dets:
                x1, y1, x2, y2, conf = d[:5]
                # Use exact detector boxes for appearance extraction and tracker updates.
                ds_dets.append(([x1, y1, x2 - x1, y2 - y1], float(conf), "person"))
                det_boxes.append([x1, y1, x2, y2])
            embeds = self._appearance_encoder.extract(frame, np.asarray(det_boxes, dtype=np.float32))
            tracks = self._tracker.update_tracks(ds_dets, embeds=embeds)
            result = []
            track_embeddings = []
            for t in tracks:
                if not t.is_confirmed() or t.time_since_update > 0:
                    continue
                ltrb = t.to_ltrb(orig=True)
                if ltrb is None:
                    ltrb = t.to_ltrb()
                # deep_sort_realtime does not always expose stable detection confidence per track.
                score = t.get_det_conf()
                if score is None:
                    score = 1.0
                result.append([float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3]), str(t.track_id), float(score)])
                try:
                    track_embeddings.append(np.asarray(t.get_feature(), dtype=np.float32))
                except Exception:
                    track_embeddings.append(None)
            self._stable_memory.step()
            if not result:
                self._stable_memory.clear_active_mapping()
                return np.empty((0, 6), dtype=object)

            out = np.array(result, dtype=object)
            if self.manual_reid:
                out = self._stabilize_track_ids(frame, out, track_embeddings)
            return out
        except Exception as e:
            raise RuntimeError(f"DeepSORT update error: {e}") from e

    def _stabilize_track_ids(
        self,
        frame: np.ndarray,
        tracks: np.ndarray,
        track_embeddings: list[np.ndarray | None],
    ) -> np.ndarray:
        boxes = np.asarray(tracks[:, :4], dtype=float)
        internal_ids = []
        for track_id in tracks[:, 4]:
            try:
                internal_ids.append(int(float(track_id)))
            except Exception:
                internal_ids.append(abs(hash(str(track_id))) % 1000000)
        embeddings = self._resolve_track_embeddings(frame, boxes, track_embeddings)
        return self._stable_memory.reassign(
            tracks=tracks,
            boxes=boxes,
            internal_ids=internal_ids,
            embeddings=embeddings,
            stringify_ids=True,
        )

    def _resolve_track_embeddings(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        track_embeddings: list[np.ndarray | None],
    ) -> np.ndarray | None:
        if track_embeddings and any(embedding is not None for embedding in track_embeddings):
            reference = next((embedding for embedding in track_embeddings if embedding is not None), None)
            dim = int(reference.shape[0]) if reference is not None else 0
            filled = []
            missing_indices = []
            for idx, embedding in enumerate(track_embeddings):
                if embedding is None:
                    filled.append(np.zeros((dim,), dtype=np.float32) if dim else None)
                    missing_indices.append(idx)
                else:
                    filled.append(np.asarray(embedding, dtype=np.float32))
            if missing_indices:
                fallback = self._appearance_encoder.extract(frame, boxes[missing_indices])
                if fallback is not None:
                    for pos, emb in zip(missing_indices, fallback):
                        filled[pos] = emb
            if all(embedding is not None for embedding in filled):
                return normalize_embeddings(np.asarray(filled, dtype=np.float32))
        return self._appearance_encoder.extract(frame, boxes)

    # ── Mock helpers ─────────────────────────────────────────────────────────

    def _mock_detect(self, frame: np.ndarray) -> np.ndarray:
        print("[DeepSORT] Mock detect called.")
        h, w = frame.shape[:2]
        n = np.random.randint(1, 4)
        dets = []
        for _ in range(n):
            x1 = np.random.randint(0, w // 2)
            y1 = np.random.randint(0, h // 2)
            x2 = min(x1 + np.random.randint(50, 150), w)
            y2 = min(y1 + np.random.randint(100, 250), h)
            dets.append([x1, y1, x2, y2, np.random.uniform(0.5, 0.95)])
        return np.array(dets)

    def __init_mock_state(self):
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
