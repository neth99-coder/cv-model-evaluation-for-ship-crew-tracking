"""
DeepSORT Tracker Wrapper
Pluggable: detector (yolov8/yolov5) + RE-ID model (osnet/resnet)
Installs missing dependencies on first run.
"""

import numpy as np
import subprocess
import sys


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
    Supported detectors: yolov8n, yolov8s, yolov8m, yolov5n, yolov5s
    Supported RE-ID: osnet_x0_25, osnet_x1_0, resnet50
    """

    def __init__(self, detector: str = "yolov8n", reid_model: str = "osnet_x0_25",
                 conf: float = 0.4):
        self.detector_name = detector
        self.reid_model = reid_model
        self.conf = conf
        self._detector = None
        self._tracker = None
        self._encoder = None
        self._next_id = 1
        self._embedder = None
        self._embedder_model_name = None
        self._detector_install_attempted = False
        self._tracker_install_attempted = False
        self._init_detector()
        self._init_tracker()

    # ── Detector ─────────────────────────────────────────────────────────────

    def _init_detector(self):
        try:
            if self.detector_name.startswith("yolov8"):
                from ultralytics import YOLO
                self._detector = YOLO(f"{self.detector_name}.pt")
            elif self.detector_name.startswith("yolov5"):
                import torch
                self._detector = torch.hub.load("ultralytics/yolov5", self.detector_name, pretrained=True)
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
            if self.detector_name.startswith("yolov8"):
                results = self._detector.predict(frame, conf=self.conf, classes=[0], verbose=False)
                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    return np.empty((0, 5))
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy().reshape(-1, 1)
                return np.hstack([xyxy, confs])
            elif self.detector_name.startswith("yolov5"):
                results = self._detector(frame)
                dets = results.xyxy[0].cpu().numpy()
                persons = dets[dets[:, 5] == 0]
                return persons[:, :5]
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
        embedder, embedder_model_name = self._map_reid_config()
        self._embedder = embedder
        self._embedder_model_name = embedder_model_name
        try:
            # deep_sort_realtime package
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._tracker = DeepSort(
                max_age=30,
                n_init=3,
                embedder=embedder,
                embedder_model_name=embedder_model_name,
                half=False,
            )
            print(
                f"[DeepSORT] Initialized with reid={self.reid_model}, "
                f"embedder={embedder}, embedder_model_name={embedder_model_name}"
            )
        except Exception as e:
            if not self._tracker_install_attempted:
                if _is_missing_dependency_error(e, ["deep_sort_realtime", "deep-sort-realtime"]):
                    self._tracker_install_attempted = True
                    _install_packages(["deep-sort-realtime"])
                    return self._init_tracker()
                if _is_missing_dependency_error(e, ["torchreid"]):
                    self._tracker_install_attempted = True
                    _install_packages([
                        "git+https://github.com/KaiyangZhou/deep-person-reid.git",
                        "tensorboard",
                    ])
                    return self._init_tracker()
            raise RuntimeError(f"DeepSORT tracker init failed: {e}") from e

    def runtime_config(self) -> dict:
        return {
            "framework": "deepsort",
            "tracker": "deepsort",
            "detector": self.detector_name,
            "reid_model": self.reid_model,
            "embedder": self._embedder,
            "embedder_model_name": self._embedder_model_name,
        }

    def _map_reid_config(self) -> tuple[str, str]:
        """Map selected Re-ID model to DeepSORT embedder and exact model name."""
        mapping = {
            "osnet_x0_25": ("torchreid", "osnet_x0_25"),
            "osnet_x1_0": ("torchreid", "osnet_x1_0"),
            "resnet50": ("torchreid", "resnet50"),
            "mlfn": ("torchreid", "mlfn"),
        }
        if self.reid_model not in mapping:
            supported = ", ".join(sorted(mapping.keys()))
            raise ValueError(f"Unsupported DeepSORT reid_model '{self.reid_model}'. Supported: {supported}")
        return mapping[self.reid_model]

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> np.ndarray:
        dets = self._detect(frame)
        if self._tracker is None:
            raise RuntimeError("DeepSORT tracker is not initialized")

        if len(dets) == 0:
            return np.empty((0, 5))

        try:
            # deep_sort_realtime format: [[x1,y1,w,h], conf, class]
            ds_dets = []
            for d in dets:
                x1, y1, x2, y2, conf = d[:5]
                ds_dets.append(([x1, y1, x2 - x1, y2 - y1], float(conf), "person"))
            tracks = self._tracker.update_tracks(ds_dets, frame=frame)
            result = []
            for t in tracks:
                if not t.is_confirmed():
                    continue
                ltrb = t.to_ltrb()
                # deep_sort_realtime does not always expose stable detection confidence per track.
                # Use score=1.0 as a fallback for downstream AP computation.
                result.append([float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3]), str(t.track_id), 1.0])
            return np.array(result, dtype=object) if result else np.empty((0, 6), dtype=object)
        except Exception as e:
            raise RuntimeError(f"DeepSORT update error: {e}") from e

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
