"""
╔══════════════════════════════════════════════════════════════════════════╗
║     Person Re-Identification Pipeline                                    ║
║     Detector : YOLOv8n (CrowdHuman)  —  'pedestrian' → 'person'        ║
║     Tracker  : ByteTrack  |  BoT-SORT                                   ║
║     ReID     : OSNet-x1_0 (torchreid)                                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Modes                                                                   ║
║    eval      — metrics on test/images  (+ labels if available)          ║
║    compare   — side-by-side det grid saved as PNG                       ║
║    video     — annotated output video  (--source path/to/video.mp4)     ║
║    webcam    — live feed  (--cam 0)                                      ║
║    benchmark — latency / FPS distribution plot                           ║
║    reid_gallery — build gallery from test/images, query any image       ║
║    all       — eval + compare + benchmark + video                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Usage examples                                                          ║
║    python reid_pipeline.py --mode eval                                   ║
║    python reid_pipeline.py --mode video --source test/videos/cam1.mp4   ║
║    python reid_pipeline.py --mode webcam --cam 0 --tracker botsort      ║
║    python reid_pipeline.py --mode reid_gallery                           ║
║    python reid_pipeline.py --mode all --source test/videos/cam1.mp4     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── optional matplotlib (headless-safe) ───────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── ultralytics ───────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("❌  Run:  pip install ultralytics")

# ── torchreid ─────────────────────────────────────────────────────────────
try:
    import torchreid
    import torchvision.transforms as T
    HAS_REID = True
except ImportError:
    HAS_REID = False
    print("⚠️   torchreid not found — ReID features disabled.")
    print("     Install: pip install torchreid  OR")
    print("     git clone https://github.com/KaiyangZhou/deep-person-reid && "
          "cd deep-person-reid && python setup.py develop")

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = "pretrained_models/yolov8n_crowdhuman.pt"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# CrowdHuman model uses 'pedestrian' internally — we display 'person'
LABEL_REMAP  = {"pedestrian": "person", "person": "person"}

# Conservative ReID recovery settings for visually similar ship crews.
REID_MIN_INACTIVE_FRAMES = 45
REID_MAX_CENTER_DIST_RATIO = 0.18
REID_MIN_SIM_MARGIN = 0.08
REID_DIRECTION_MIN_COS = -0.2
REID_MOTION_WEIGHT = 0.20
REID_OCCLUSION_RECOVERY_FRAMES = 120
REID_OCCLUSION_ZONE_EXPAND = 0.10
REID_OCCLUSION_SIM_THRESH = 0.78
REID_OCCLUSION_MIN_MARGIN = 0.04
DEFAULT_SHIP_BARRIER_ZONE = (0.25, 0.36, 0.67, 0.78)

# Colour palette (BGR)
COLOURS = {
    "person"    : (0,   200, 100),   # green
    "head"      : (255, 165,   0),   # orange (some CrowdHuman variants detect heads)
    "rider"     : (200,   0, 200),
    "default"   : (180, 180, 180),
}

REID_INPUT_SIZE = (256, 128)         # OSNet standard input
REID_FEAT_DIM   = 512

# ─────────────────────────────────────────────────────────────────────────────
#  ReID helpers
# ─────────────────────────────────────────────────────────────────────────────
class OSNetExtractor:
    """Wraps OSNet-x1_0 for appearance embedding extraction."""

    def __init__(self, device: str = "cpu"):
        if not HAS_REID:
            self.model = None
            return
        print("  Loading OSNet-x1_0 …")
        self.model = torchreid.models.build_model(
            name        = "osnet_x1_0",
            num_classes = 1000,
            pretrained  = True,
        ).to(device).eval()
        self.device = device
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(REID_INPUT_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])
        print(f"  OSNet ready  ({sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params)")

    def extract(self, frame_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Args:
            frame_bgr : (H,W,3) BGR frame
            boxes     : (N,4)  int   [x1,y1,x2,y2]
        Returns:
            features  : (N, 512) float32  L2-normalised
        """
        if self.model is None or len(boxes) == 0:
            return np.zeros((len(boxes), REID_FEAT_DIM), dtype=np.float32)

        H, W = frame_bgr.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes.astype(int):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                crops.append(torch.zeros(3, *REID_INPUT_SIZE))
                continue
            crop_rgb = frame_bgr[y1:y2, x1:x2, ::-1].copy()  # BGR→RGB
            crops.append(self.transform(crop_rgb))

        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
            feats = F.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)


class ReIDGallery:
    """
    Rolling appearance gallery with EMA update.
    Stores one embedding per active track ID.
    """

    def __init__(self, max_size: int = 300,
                 sim_threshold: float = 0.85,
                 ema_alpha: float = 0.7,
                 min_inactive_frames: int = REID_MIN_INACTIVE_FRAMES,
                 max_center_dist_ratio: float = REID_MAX_CENTER_DIST_RATIO,
                 min_sim_margin: float = REID_MIN_SIM_MARGIN):
        self.gallery               = {}   # track_id → np.ndarray (D,)
        self.timestamps            = {}   # track_id → last seen frame
        self.last_boxes            = {}   # track_id → np.ndarray (4,)
        self.velocities            = {}   # track_id → np.ndarray (2,)
        self.max_size              = max_size
        self.sim_threshold         = sim_threshold
        self.alpha                 = ema_alpha
        self.min_inactive_frames   = min_inactive_frames
        self.max_center_dist_ratio = max_center_dist_ratio
        self.min_sim_margin        = min_sim_margin

    @staticmethod
    def _center(box: np.ndarray) -> np.ndarray:
        return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)

    @staticmethod
    def _cosine(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float | None:
        if vec_a is None or vec_b is None:
            return None
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a < 1e-6 or norm_b < 1e-6:
            return None
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    # ── update ──────────────────────────────────────────────────────────
    def update(self, track_id: int, feat: np.ndarray,
               frame_idx: int = 0, box: np.ndarray | None = None):
        prev_box = self.last_boxes.get(track_id)
        prev_frame = self.timestamps.get(track_id)

        if track_id in self.gallery:
            self.gallery[track_id] = (
                self.alpha       * self.gallery[track_id] +
                (1 - self.alpha) * feat
            )
        else:
            if len(self.gallery) >= self.max_size:
                oldest = min(self.timestamps, key=self.timestamps.get)
                del self.gallery[oldest]
                del self.timestamps[oldest]
                self.last_boxes.pop(oldest, None)
                self.velocities.pop(oldest, None)
            self.gallery[track_id] = feat.copy()
        self.timestamps[track_id] = frame_idx
        if box is not None:
            box = np.asarray(box, dtype=np.float32).copy()
            self.last_boxes[track_id] = box
            if prev_box is not None and prev_frame is not None and frame_idx - prev_frame == 1:
                inst_velocity = self._center(box) - self._center(prev_box)
                prev_velocity = self.velocities.get(track_id)
                if prev_velocity is None:
                    self.velocities[track_id] = inst_velocity
                else:
                    self.velocities[track_id] = self.alpha * prev_velocity + (1 - self.alpha) * inst_velocity

    # ── query ────────────────────────────────────────────────────────────
    def find_match(self, feat: np.ndarray, query_box: np.ndarray,
                   active_ids: set[int] | None = None, frame_idx: int = 0,
                   frame_shape: tuple[int, int] | None = None,
                   query_velocity: np.ndarray | None = None):
        """Return conservative recovery metadata for a long-lost track, or None."""
        if not self.gallery:
            return None

        active_ids = active_ids or set()
        frame_h, frame_w = frame_shape if frame_shape is not None else (None, None)
        max_center_dist = None
        if frame_h is not None and frame_w is not None:
            max_center_dist = self.max_center_dist_ratio * float(np.hypot(frame_w, frame_h))

        query_center = self._center(np.asarray(query_box, dtype=np.float32))
        candidates = []

        for track_id, gallery_feat in self.gallery.items():
            if track_id in active_ids:
                continue

            last_seen = self.timestamps.get(track_id, frame_idx)
            inactive_for = frame_idx - last_seen
            if inactive_for < self.min_inactive_frames:
                continue

            last_box = self.last_boxes.get(track_id)
            velocity = self.velocities.get(track_id)
            predicted_center = None
            if last_box is not None and max_center_dist is not None:
                last_center = self._center(last_box)
                predicted_center = last_center
                if velocity is not None:
                    predicted_center = last_center + velocity * inactive_for
                center_dist = float(np.linalg.norm(query_center - predicted_center))
                velocity_norm = float(np.linalg.norm(velocity)) if velocity is not None else 0.0
                allowed_dist = max_center_dist + velocity_norm * inactive_for * 1.5
                if center_dist > allowed_dist:
                    continue
            else:
                center_dist = None

            direction_cos = self._cosine(query_velocity, velocity)
            if direction_cos is not None and direction_cos < REID_DIRECTION_MIN_COS:
                continue

            sim = float(gallery_feat @ feat)
            proximity_score = 0.0
            if center_dist is not None and max_center_dist is not None:
                proximity_score = max(0.0, 1.0 - center_dist / max(max_center_dist, 1.0))
            direction_score = 0.0 if direction_cos is None else max(0.0, direction_cos)

            candidates.append({
                "track_id": track_id,
                "sim": sim,
                "inactive_for": inactive_for,
                "center_dist": center_dist,
                "direction_cos": direction_cos,
                "score": sim + REID_MOTION_WEIGHT * (0.6 * proximity_score + 0.4 * direction_score),
            })

        if not candidates:
            return None

        candidates.sort(key=lambda item: item["score"], reverse=True)
        best = candidates[0]
        second_best_score = candidates[1]["score"] if len(candidates) > 1 else -1.0
        best["margin"] = best["score"] - second_best_score

        if best["sim"] < self.sim_threshold or best["margin"] < self.min_sim_margin:
            return None
        return best

    def all_ids(self):
        return list(self.gallery.keys())


# ─────────────────────────────────────────────────────────────────────────────
#  Tracker factory (ByteTrack / BoT-SORT via BoxMOT)
# ─────────────────────────────────────────────────────────────────────────────
def build_tracker(tracker_type: str = "bytetrack"):
    """
    Returns a tracker object with .update(dets, frame) interface.
    dets shape: (N,6)  [x1,y1,x2,y2,conf,cls]
    Returns:   (M,8)  [x1,y1,x2,y2,track_id,conf,cls,det_idx]
    """
    try:
        if tracker_type == "bytetrack":
            try:
                from boxmot.trackers.bytetrack.bytetrack import ByteTrack
            except ImportError:
                from boxmot.trackers.bytetrack import ByteTrack
            return ByteTrack(
                track_thresh  = 0.25,
                track_buffer  = 30,
                match_thresh  = 0.80,
                frame_rate    = 25,
            )
        elif tracker_type == "botsort":
            try:
                from boxmot.trackers.botsort.botsort import BotSort
            except ImportError:
                from boxmot.trackers.botsort import BotSort
            return BotSort(
                reid_weights      = None,       # OSNet handled separately
                device            = DEVICE,
                half              = False,
                track_high_thresh = 0.50,
                track_low_thresh  = 0.10,
                new_track_thresh  = 0.60,
                track_buffer      = 30,
                match_thresh      = 0.80,
                proximity_thresh  = 0.50,
                appearance_thresh = 0.25,
                with_reid         = False,
            )
        else:
            raise ValueError(f"Unknown tracker: {tracker_type}")
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] != "boxmot":
            raise
        print("⚠️   boxmot not installed — using a simple IoU tracker fallback.")
        return SimpleIoUTracker()


class SimpleIoUTracker:
    """
    Minimal single-class IoU tracker used as fallback when boxmot is absent.
    Greedy nearest-IoU assignment.
    """
    def __init__(self, iou_thresh=0.3, max_lost=30):
        self.tracks    = {}   # id → {'box':..., 'lost':0}
        self.next_id   = 1
        self.iou_thresh= iou_thresh
        self.max_lost  = max_lost

    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        if len(dets) == 0:
            for tid in list(self.tracks):
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    del self.tracks[tid]
            return np.empty((0, 8))

        def iou(b1, b2):
            xa = max(b1[0],b2[0]); ya = max(b1[1],b2[1])
            xb = min(b1[2],b2[2]); yb = min(b1[3],b2[3])
            inter = max(0,xb-xa)*max(0,yb-ya)
            a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
            a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
            return inter/(a1+a2-inter+1e-6)

        track_ids  = list(self.tracks.keys())
        matched_tr = set()
        matched_dt = set()
        out_rows   = []

        for di, det in enumerate(dets):
            best_iou, best_tid = 0, None
            for tid in track_ids:
                if tid in matched_tr: continue
                s = iou(det[:4], self.tracks[tid]['box'])
                if s > best_iou:
                    best_iou, best_tid = s, tid
            if best_iou >= self.iou_thresh:
                self.tracks[best_tid]['box']  = det[:4]
                self.tracks[best_tid]['lost'] = 0
                matched_tr.add(best_tid)
                matched_dt.add(di)
                out_rows.append([*det[:4], best_tid, det[4], det[5], di])
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {'box': det[:4], 'lost': 0}
                out_rows.append([*det[:4], tid, det[4], det[5], di])

        for tid in track_ids:
            if tid not in matched_tr:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    del self.tracks[tid]

        return np.array(out_rows) if out_rows else np.empty((0, 8))


# ─────────────────────────────────────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
_PALETTE_CACHE = {}

def _track_colour(track_id: int):
    """Deterministic per-track BGR colour."""
    if track_id not in _PALETTE_CACHE:
        rng = np.random.default_rng(track_id * 137 + 29)
        h   = rng.uniform(0, 1)
        # HSV → BGR
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        _PALETTE_CACHE[track_id] = (int(b*255), int(g*255), int(r*255))
    return _PALETTE_CACHE[track_id]


def remap_label(raw_label: str) -> str:
    """'pedestrian' → 'person', everything else kept as-is."""
    return LABEL_REMAP.get(raw_label.lower(), raw_label)


def draw_track(frame, box, track_id, label, conf,
               reid_sim=None, re_id_tid=None, colour=None):
    x1, y1, x2, y2 = map(int, box)
    c = colour or _track_colour(track_id)

    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

    # Label text
    id_str   = f"#{track_id}"
    cls_str  = f"{label} {conf:.2f}"
    reid_str = ""
    if re_id_tid is not None and re_id_tid != track_id:
        reid_str = f" ReID: #{re_id_tid} ({reid_sim:.2f}) "

    full_label = id_str + "  " + cls_str + reid_str

    (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, y1 - th - 7), (x1 + tw + 4, y1), c, -1)
    cv2.putText(frame, full_label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Keypoint dot at centre
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 3, c, -1)

    return frame


def overlay_hud(frame, frame_idx, fps, n_tracks, tracker_type, reid_on):
    h, w = frame.shape[:2]
    info = (f"Frame:{frame_idx}  FPS:{fps:.1f}  "
            f"Persons:{n_tracks}  Tracker:{tracker_type.upper()}  "
            f"ReID:{'ON' if reid_on else 'OFF'}")
    cv2.putText(frame, info, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def _compute_box_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def _parse_zone_spec(spec: str) -> tuple[float, float, float, float]:
    vals = [float(v.strip()) for v in spec.split(",")]
    if len(vals) != 4:
        raise ValueError("Zone must have four comma-separated values: x1,y1,x2,y2")
    x1, y1, x2, y2 = vals
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("Zone values must be normalized floats in [0,1] with x1<x2 and y1<y2")
    return x1, y1, x2, y2


def _denormalize_zone(zone: tuple[float, float, float, float], frame_shape: tuple[int, int]) -> np.ndarray:
    h, w = frame_shape
    x1, y1, x2, y2 = zone
    return np.array([x1 * w, y1 * h, x2 * w, y2 * h], dtype=np.float32)


def _expand_box(box: np.ndarray, margin_x: float, margin_y: float,
                frame_shape: tuple[int, int] | None = None) -> np.ndarray:
    box = np.asarray(box, dtype=np.float32).copy()
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    box[0] = x1 - w * margin_x
    box[1] = y1 - h * margin_y
    box[2] = x2 + w * margin_x
    box[3] = y2 + h * margin_y
    if frame_shape is not None:
        fh, fw = frame_shape
        box[0] = np.clip(box[0], 0, fw)
        box[1] = np.clip(box[1], 0, fh)
        box[2] = np.clip(box[2], 0, fw)
        box[3] = np.clip(box[3], 0, fh)
    return box


def _boxes_intersect(box_a: np.ndarray, box_b: np.ndarray) -> bool:
    ax1, ay1, ax2, ay2 = np.asarray(box_a, dtype=np.float32)
    bx1, by1, bx2, by2 = np.asarray(box_b, dtype=np.float32)
    return min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)


class DynamicZoneTracker:
    """Track occlusion zones under camera shake using global frame translation."""

    def __init__(self, zones: list[np.ndarray], frame_shape: tuple[int, int]):
        self.frame_shape = frame_shape
        self.base_zones = [np.asarray(z, dtype=np.float32).copy() for z in zones]
        self.current_zones = [z.copy() for z in self.base_zones]
        self.anchor_gray = None
        self.offset = np.zeros(2, dtype=np.float32)

    def update(self, frame: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)
        if self.anchor_gray is None:
            self.anchor_gray = gray_f
            return [z.copy() for z in self.current_zones]

        shift, response = cv2.phaseCorrelate(self.anchor_gray, gray_f)
        dx, dy = shift
        if not np.isfinite(dx) or not np.isfinite(dy):
            dx, dy = 0.0, 0.0
        if response < 0.02:
            dx, dy = float(self.offset[0]), float(self.offset[1])
        else:
            dx = 0.85 * float(self.offset[0]) + 0.15 * float(dx)
            dy = 0.85 * float(self.offset[1]) + 0.15 * float(dy)
            self.offset[:] = (dx, dy)

        fh, fw = self.frame_shape
        updated = []
        for zone in self.base_zones:
            moved = zone.copy()
            moved[[0, 2]] += dx
            moved[[1, 3]] += dy
            moved[0] = np.clip(moved[0], 0, fw)
            moved[1] = np.clip(moved[1], 0, fh)
            moved[2] = np.clip(moved[2], 0, fw)
            moved[3] = np.clip(moved[3], 0, fh)
            updated.append(moved)
        self.current_zones = updated
        return [z.copy() for z in self.current_zones]


# ─────────────────────────────────────────────────────────────────────────────
#  Core tracking + ReID loop
# ─────────────────────────────────────────────────────────────────────────────
def run_tracking(
    source         : str,
    output_path    : str  = None,
    imgsz          : int  = 640,
    conf           : float= 0.5,
    iou            : float= 0.45,
    tracker_type   : str  = "botsort",
    use_reid       : bool = True,
    reid_sim_thresh: float= 0.85,
    reid_ema       : float= 0.70,
    occlusion_zones: list[tuple[float, float, float, float]] | None = None,
    dynamic_occlusion_zones: bool = False,
    show           : bool = False,
    max_frames     : int  = None,
    save_reid_json : bool = True,
) -> list:
    """
    Full pipeline: detect → track → ReID.

    Returns list of per-frame dicts with track info.
    """
    # ── Load components ──────────────────────────────────────────────────
    print(f"\n  Model   : {MODEL_PATH}")
    if not Path(MODEL_PATH).exists():
        sys.exit(f"❌  Model not found: {Path(MODEL_PATH).resolve()}")
    detector = YOLO(MODEL_PATH)

    tracker = build_tracker(tracker_type)
    reid_extractor = OSNetExtractor(DEVICE) if (use_reid and HAS_REID) else None
    gallery        = ReIDGallery(sim_threshold=reid_sim_thresh, ema_alpha=reid_ema)

    # ── Open source ──────────────────────────────────────────────────────
    is_webcam = str(source) in ("0","1","2","3") or source == 0
    cap = cv2.VideoCapture(int(source) if is_webcam else str(source))
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open source: {source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
    if max_frames and total:
        total = min(total, max_frames)

    print(f"  Source  : {source}  ({W}×{H}  {fps_src:.1f}fps)")
    print(f"  Tracker : {tracker_type.upper()}")
    print(f"  ReID    : {'OSNet-x1_0' if (use_reid and HAS_REID) else 'OFF'}")
    if occlusion_zones:
        print(f"  Zones   : {len(occlusion_zones)} occlusion zone(s) enabled")
    if occlusion_zones and dynamic_occlusion_zones:
        print("  Zones   : dynamic zone tracking enabled")
    print(f"  Device  : {DEVICE}\n")

    # ── Writer ───────────────────────────────────────────────────────────
    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_src, (W, H),
        )

    # ── Stats tracking ───────────────────────────────────────────────────
    all_frames_data = []
    fps_history     = []
    unique_ids      = set()
    id_switch_count = 0
    prev_track_ids  = set()
    frame_idx       = 0
    stable_id_map   = {}
    motion_state    = {}
    stable_track_state = {}
    occlusion_bank  = {}
    pixel_zones     = [_denormalize_zone(z, (H, W)) for z in (occlusion_zones or [])]
    zone_tracker    = (DynamicZoneTracker(pixel_zones, (H, W))
                       if pixel_zones and dynamic_occlusion_zones else None)

    def box_hits_occlusion_zone(box: np.ndarray, expand: float = 0.0) -> bool:
        if not pixel_zones:
            return False
        test_box = _expand_box(box, expand, expand, (H, W)) if expand > 0 else np.asarray(box, dtype=np.float32)
        return any(_boxes_intersect(test_box, zone) for zone in pixel_zones)

    def query_occlusion_bank(feat: np.ndarray, box: np.ndarray, active_ids: set[int], now_frame: int):
        if not occlusion_bank:
            return None

        candidates = []
        for stable_id, item in occlusion_bank.items():
            if stable_id in active_ids:
                continue
            if now_frame - item["lost_frame"] > REID_OCCLUSION_RECOVERY_FRAMES:
                continue
            sim = float(item["feat"] @ feat)
            size_ref = max((item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]), 1.0)
            size_now = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
            size_ratio = min(size_now, size_ref) / max(size_now, size_ref)
            score = sim + 0.10 * size_ratio
            candidates.append({
                "track_id": stable_id,
                "sim": sim,
                "score": score,
                "inactive_for": now_frame - item["lost_frame"],
                "center_dist": None,
                "direction_cos": None,
                "zone_recovery": True,
            })

        if not candidates:
            return None

        candidates.sort(key=lambda item: item["score"], reverse=True)
        best = candidates[0]
        second = candidates[1]["score"] if len(candidates) > 1 else -1.0
        best["margin"] = best["score"] - second
        if best["sim"] < REID_OCCLUSION_SIM_THRESH or best["margin"] < REID_OCCLUSION_MIN_MARGIN:
            return None
        return best

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        t_start = time.perf_counter()
        if zone_tracker is not None:
            pixel_zones = zone_tracker.update(frame)

        # ── Detection ────────────────────────────────────────────────────
        det_res = detector(
            frame,
            imgsz   = imgsz,
            conf    = conf,
            iou     = iou,
            verbose = False,
            classes = None,       # all classes from the model
        )[0]

        boxes_xyxy = det_res.boxes.xyxy.cpu().numpy()
        confs_arr  = det_res.boxes.conf.cpu().numpy()
        clses_arr  = det_res.boxes.cls.cpu().numpy()

        # Map class names
        raw_names  = [det_res.names[int(c)] for c in clses_arr]
        disp_names = [remap_label(n) for n in raw_names]

        det_arr = (np.column_stack([boxes_xyxy, confs_arr, clses_arr])
                   if len(boxes_xyxy) else np.empty((0, 6)))

        # ── Tracking ─────────────────────────────────────────────────────
        tracks = tracker.update(det_arr, frame)  # (M, 8)

        # ── ReID ─────────────────────────────────────────────────────────
        reid_results = {}   # track_id → conservative recovery metadata
        active_stable_ids = set()
        if reid_extractor is not None and len(tracks) > 0:
            t_boxes    = tracks[:, :4]
            t_ids      = tracks[:, 4].astype(int)
            reid_feats = reid_extractor.extract(frame, t_boxes)
            current_internal_ids = set(t_ids.tolist())
            active_stable_ids = {
                stable_id_map[tid]
                for tid in current_internal_ids
                if tid in stable_id_map
            }

            for tid, feat, box in zip(t_ids, reid_feats, t_boxes):
                tid = int(tid)
                box = np.asarray(box, dtype=np.float32)
                box_center = _compute_box_center(box)
                state = motion_state.get(tid)
                query_velocity = None
                if state is not None and frame_idx - state["frame_idx"] == 1:
                    query_velocity = box_center - state["center"]

                matched_stable_id = None
                match = None
                if tid in stable_id_map:
                    matched_stable_id = stable_id_map[tid]
                else:
                    use_zone_recovery = box_hits_occlusion_zone(box, expand=REID_OCCLUSION_ZONE_EXPAND)
                    if use_zone_recovery:
                        match = query_occlusion_bank(
                            feat=feat,
                            box=box,
                            active_ids=active_stable_ids,
                            now_frame=frame_idx,
                        )
                    if match is None:
                        match = gallery.find_match(
                            feat=feat,
                            query_box=box,
                            active_ids=active_stable_ids,
                            frame_idx=frame_idx,
                            frame_shape=frame.shape[:2],
                            query_velocity=query_velocity,
                        )
                    matched_stable_id = int(match["track_id"]) if match else tid
                    stable_id_map[tid] = matched_stable_id

                active_stable_ids.add(matched_stable_id)
                gallery.update(matched_stable_id, feat, frame_idx, box=box)
                motion_state[tid] = {"center": box_center, "frame_idx": frame_idx}
                stable_track_state[matched_stable_id] = {
                    "box": box.copy(),
                    "feat": feat.copy(),
                    "frame_idx": frame_idx,
                    "in_zone": box_hits_occlusion_zone(box, expand=0.02),
                }
                occlusion_bank.pop(matched_stable_id, None)

                reid_results[tid] = {
                    "feat"         : feat,
                    "stable_id"    : matched_stable_id,
                    "match_id"     : int(match["track_id"]) if match else None,
                    "sim"          : float(match["sim"]) if match else 0.0,
                    "inactive_for" : int(match["inactive_for"]) if match else 0,
                    "center_dist"  : float(match["center_dist"]) if match and match["center_dist"] is not None else None,
                    "sim_margin"   : float(match["margin"]) if match else 0.0,
                    "direction_cos": float(match["direction_cos"]) if match and match["direction_cos"] is not None else None,
                    "zone_recovery": bool(match.get("zone_recovery", False)) if match else False,
                }

            stale_internal_ids = [tid for tid in motion_state if tid not in current_internal_ids]
            for stale_tid in stale_internal_ids:
                stable_id = stable_id_map.get(stale_tid)
                if stable_id is not None:
                    last_state = stable_track_state.get(stable_id)
                    if last_state and last_state["in_zone"]:
                        occlusion_bank[stable_id] = {
                            "feat": last_state["feat"].copy(),
                            "box": last_state["box"].copy(),
                            "lost_frame": frame_idx,
                        }
                motion_state.pop(stale_tid, None)
                stable_id_map.pop(stale_tid, None)

            expired_ids = [
                stable_id for stable_id, item in occlusion_bank.items()
                if frame_idx - item["lost_frame"] > REID_OCCLUSION_RECOVERY_FRAMES
            ]
            for stable_id in expired_ids:
                occlusion_bank.pop(stable_id, None)

        # ── ID switch detection ───────────────────────────────────────────
        curr_ids = set(
            int(reid_results.get(int(t[4]), {}).get("stable_id", int(t[4])))
            for t in tracks
        ) if len(tracks) else set()
        new_ids  = curr_ids - prev_track_ids
        id_switch_count += len(new_ids)
        unique_ids |= curr_ids
        prev_track_ids = curr_ids

        # ── Annotate frame ────────────────────────────────────────────────
        frame_data = {"frame": frame_idx, "tracks": []}

        for t in tracks:
            x1, y1, x2, y2 = t[:4]
            tid   = int(t[4])
            tconf = float(t[5])
            tcls  = int(t[6])
            tname = remap_label(det_res.names.get(tcls, "person"))

            reid_info   = reid_results.get(tid, {})
            display_id  = int(reid_info.get("stable_id", tid))
            match_id    = reid_info.get("match_id")
            match_sim   = reid_info.get("sim", 0.0)

            draw_track(frame, (x1,y1,x2,y2), display_id, tname, tconf,
                       reid_sim=match_sim if match_id else None,
                       re_id_tid=match_id)

            frame_data["tracks"].append({
                "track_id" : display_id,
                "tracker_id": tid,
                "class"    : tname,
                "box"      : [int(x1),int(y1),int(x2),int(y2)],
                "conf"     : round(tconf, 4),
                "reid_match_id" : match_id,
                "reid_sim"      : round(match_sim, 4),
                "reid_inactive_for": reid_info.get("inactive_for", 0),
                "reid_center_dist": (round(reid_info["center_dist"], 2)
                                      if reid_info.get("center_dist") is not None else None),
                "reid_sim_margin": round(reid_info.get("sim_margin", 0.0), 4),
                "reid_direction_cos": (round(reid_info["direction_cos"], 4)
                                        if reid_info.get("direction_cos") is not None else None),
                "reid_zone_recovery": reid_info.get("zone_recovery", False),
            })

        # ── HUD ──────────────────────────────────────────────────────────
        dt  = time.perf_counter() - t_start
        fps = 1.0 / max(dt, 1e-6)
        fps_history.append(fps)

        overlay_hud(frame, frame_idx, fps, len(tracks),
                    tracker_type, reid_extractor is not None)

        for zone in pixel_zones:
            zx1, zy1, zx2, zy2 = zone.astype(int)
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
            cv2.putText(frame, "occlusion-zone", (zx1, max(18, zy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA)

        # Mini stats panel (bottom-left)
        stats_lines = [
            f"Unique IDs : {len(unique_ids)}",
            f"ID assigns : {id_switch_count}",
            f"Avg FPS    : {np.mean(fps_history[-30:]):.1f}",
        ]
        for li, txt in enumerate(stats_lines):
            cv2.putText(frame, txt, (8, H - 12 - li*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        if writer:
            writer.write(frame)
        if show:
            cv2.imshow("Person ReID Pipeline  [Q=quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  User quit.")
                break

        all_frames_data.append(frame_data)
        frame_idx += 1

        if frame_idx % 200 == 0:
            print(f"  Frame {frame_idx}{'/'+ str(total) if total else ''}  "
                  f"Avg FPS:{np.mean(fps_history[-50:]):.1f}  "
                  f"Unique IDs:{len(unique_ids)}")

    cap.release()
    if writer:
        writer.release()
        print(f"\n  ✅  Video saved → {output_path}")
    cv2.destroyAllWindows()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n  ─── Tracking Summary ───────────────────────────────────")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Unique track IDs : {len(unique_ids)}")
    print(f"  Total ID assigns : {id_switch_count}")
    print(f"  Avg FPS          : {np.mean(fps_history):.1f}")
    print(f"  Min / Max FPS    : {np.min(fps_history):.1f} / {np.max(fps_history):.1f}")
    print(f"  ────────────────────────────────────────────────────────")

    if save_reid_json and output_path:
        json_path = Path(output_path).with_suffix(".json")
        clean = [{"frame": fd["frame"],
                  "tracks": [{k:v for k,v in t.items() if k != "feat"}
                              for t in fd["tracks"]]}
                 for fd in all_frames_data]
        with open(json_path, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"  ✅  Track JSON   → {json_path}")

    return all_frames_data


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation on test images
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    test_img_dir : str  = "test/images",
    imgsz        : int  = 640,
    conf         : float= 0.5,
    iou_thresh   : float= 0.45,
    save_dir     : str  = "reid_results",
    max_images   : int  = None,
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    img_dir = Path(test_img_dir)
    lbl_dir = img_dir.parent / "labels"
    has_gt  = lbl_dir.exists() and any(lbl_dir.glob("*.txt"))

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not images:
        print(f"⚠️   No images found in {img_dir}"); return
    if max_images:
        images = images[:max_images]

    print(f"\n{'='*60}")
    print(f"  Evaluation — {len(images)} images  |  GT labels: {has_gt}")
    print(f"{'='*60}")

    model = YOLO(MODEL_PATH)
    reid  = OSNetExtractor(DEVICE) if HAS_REID else None

    latencies   = []
    det_counts  = []
    cls_counts  = defaultdict(int)
    reid_times  = []

    for img_path in images:
        frame = cv2.imread(str(img_path))

        # Detection
        t0  = time.perf_counter()
        res = model(frame, imgsz=imgsz, conf=conf, iou=iou_thresh, verbose=False)[0]
        latencies.append((time.perf_counter() - t0) * 1000)

        boxes = res.boxes.xyxy.cpu().numpy()
        det_counts.append(len(boxes))

        for box in res.boxes:
            raw = res.names[int(box.cls[0])]
            cls_counts[remap_label(raw)] += 1

        # ReID timing
        if reid is not None and len(boxes):
            t1 = time.perf_counter()
            reid.extract(frame, boxes)
            reid_times.append((time.perf_counter() - t1) * 1000)

    lat  = np.array(latencies)
    print(f"\n  Detection")
    print(f"    Avg latency   : {lat.mean():.1f} ms  ({1000/lat.mean():.1f} FPS)")
    print(f"    Avg dets/img  : {np.mean(det_counts):.1f}")
    print(f"    Total dets    : {sum(det_counts)}")
    print(f"    Class counts  :")
    for cn, cnt in sorted(cls_counts.items(), key=lambda x: -x[1]):
        print(f"      {cn:20s}: {cnt:6d}")

    if reid_times:
        rt = np.array(reid_times)
        print(f"\n  ReID Feature Extraction")
        print(f"    Avg latency  : {rt.mean():.1f} ms / frame")
        print(f"    Total time   : {rt.sum()/1000:.2f} s")

    stats = {
        "n_images"        : len(images),
        "avg_latency_ms"  : float(lat.mean()),
        "avg_fps"         : float(1000/lat.mean()),
        "avg_dets_per_img": float(np.mean(det_counts)),
        "total_dets"      : int(sum(det_counts)),
        "class_counts"    : dict(cls_counts),
    }
    if reid_times:
        stats["reid_avg_ms"] = float(np.mean(reid_times))

    with open(Path(save_dir) / "eval_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── Plots ────────────────────────────────────────────────────────────
    if HAS_MPL and cls_counts:
        _plot_eval(cls_counts, lat, save_dir)

    print(f"\n  ✅  Eval stats saved → {save_dir}/eval_stats.json")
    return stats


def _plot_eval(cls_counts, latencies, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Class bar
    names  = list(cls_counts.keys())
    counts = [cls_counts[n] for n in names]
    ax1.bar(names, counts, color="#2196F3", edgecolor="white", alpha=0.85)
    ax1.set_title("Detections per Class (remapped)", fontweight="bold")
    ax1.set_ylabel("Count"); ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(counts):
        ax1.text(i, v + max(counts)*0.01, str(v), ha="center", fontweight="bold", fontsize=9)

    # Latency histogram
    ax2.hist(latencies, bins=30, color="#E91E63", edgecolor="white", alpha=0.8)
    ax2.axvline(latencies.mean(), color="yellow", linewidth=2, label=f"Mean={latencies.mean():.1f}ms")
    ax2.set_title("Inference Latency Distribution", fontweight="bold")
    ax2.set_xlabel("ms / image"); ax2.set_ylabel("Count")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("YOLOv8n CrowdHuman — Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = Path(save_dir) / "eval_plots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅  Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Speed benchmark
# ─────────────────────────────────────────────────────────────────────────────
def benchmark(
    test_img_dir : str  = "test/images",
    imgsz        : int  = 640,
    n_warmup     : int  = 5,
    n_runs       : int  = 50,
    save_dir     : str  = "reid_results",
):
    images = (sorted(Path(test_img_dir).glob("*.jpg")) +
              sorted(Path(test_img_dir).glob("*.png")))
    if not images:
        print(f"⚠️   No images in {test_img_dir}"); return

    images = (images * ((n_runs + n_warmup) // len(images) + 1))[:n_runs + n_warmup]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    reid  = OSNetExtractor(DEVICE) if HAS_REID else None

    det_lats, reid_lats = [], []

    print("\n  Warming up …")
    for img in images[:n_warmup]:
        frame = cv2.imread(str(img))
        model(frame, imgsz=imgsz, verbose=False)

    print(f"  Running {n_runs} iterations …")
    for img in images[n_warmup:n_warmup + n_runs]:
        frame = cv2.imread(str(img))

        t0  = time.perf_counter()
        res = model(frame, imgsz=imgsz, verbose=False)[0]
        det_lats.append((time.perf_counter() - t0) * 1000)

        boxes = res.boxes.xyxy.cpu().numpy()
        if reid is not None and len(boxes):
            t1 = time.perf_counter()
            reid.extract(frame, boxes)
            reid_lats.append((time.perf_counter() - t1) * 1000)

    def _stats(arr, label):
        arr = np.array(arr)
        print(f"\n  {label}")
        print(f"    Mean   : {arr.mean():.1f} ms  |  FPS: {1000/arr.mean():.1f}")
        print(f"    Median : {np.median(arr):.1f} ms")
        print(f"    P95    : {np.percentile(arr, 95):.1f} ms")
        print(f"    Std    : {arr.std():.1f} ms")
        return arr

    det_arr  = _stats(det_lats,  "Detection (YOLOv8n CrowdHuman)")
    reid_arr = _stats(reid_lats, "ReID Extraction (OSNet)") if reid_lats else None

    if HAS_MPL:
        n_plots = 2 if reid_arr is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        axes[0].hist(det_arr, bins=25, color="#2196F3", edgecolor="white", alpha=0.85)
        axes[0].axvline(det_arr.mean(), color="yellow", lw=2,
                        label=f"Mean={det_arr.mean():.1f}ms")
        axes[0].set_title("Detection Latency", fontweight="bold")
        axes[0].set_xlabel("ms"); axes[0].legend(); axes[0].grid(alpha=0.3)

        if reid_arr is not None:
            axes[1].hist(reid_arr, bins=25, color="#E91E63", edgecolor="white", alpha=0.85)
            axes[1].axvline(reid_arr.mean(), color="yellow", lw=2,
                            label=f"Mean={reid_arr.mean():.1f}ms")
            axes[1].set_title("ReID Extraction Latency", fontweight="bold")
            axes[1].set_xlabel("ms"); axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.suptitle("Benchmark — YOLOv8n + OSNet ReID", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = Path(save_dir) / "benchmark.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"\n  ✅  Benchmark plot → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Side-by-side comparison (with vs without ReID)
# ─────────────────────────────────────────────────────────────────────────────
def compare(
    test_img_dir : str  = "test/images",
    n_images     : int  = 6,
    imgsz        : int  = 640,
    conf         : float= 0.5,
    tracker_type : str  = "bytetrack",
    save_dir     : str  = "reid_results",
):
    """Save a grid: original | detection-only | detection + ReID."""
    if not HAS_MPL:
        print("⚠️   matplotlib not available — skipping compare"); return

    images = (sorted(Path(test_img_dir).glob("*.jpg")) +
              sorted(Path(test_img_dir).glob("*.png")))[:n_images]
    if not images:
        print(f"⚠️   No images in {test_img_dir}"); return

    model   = YOLO(MODEL_PATH)
    tracker = build_tracker(tracker_type)
    reid    = OSNetExtractor(DEVICE) if HAS_REID else None
    gallery = ReIDGallery()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    n_rows  = len(images)
    n_cols  = 3   # original | det only | det + ReID
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4))
    if n_rows == 1:
        axes = [axes]

    for row_ax, img_path in zip(axes, images):
        frame_orig = cv2.imread(str(img_path))

        # ── Col 0: original ──────────────────────────────────────────────
        row_ax[0].imshow(cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB))
        row_ax[0].set_title("Original", fontsize=10, fontweight="bold")
        row_ax[0].axis("off")

        # ── Col 1: detection only ────────────────────────────────────────
        frame_det = frame_orig.copy()
        res = model(frame_det, imgsz=imgsz, conf=conf, verbose=False)[0]
        for box in res.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            c = float(box.conf[0])
            n = remap_label(res.names[int(box.cls[0])])
            col = COLOURS.get(n, COLOURS["default"])
            cv2.rectangle(frame_det,(x1,y1),(x2,y2), col, 2)
            cv2.putText(frame_det, f"{n} {c:.2f}", (x1,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255,255,255), 1)
        row_ax[1].imshow(cv2.cvtColor(frame_det, cv2.COLOR_BGR2RGB))
        row_ax[1].set_title(f"Detection  ({len(res.boxes)} persons)", fontsize=10, fontweight="bold")
        row_ax[1].axis("off")

        # ── Col 2: tracking + ReID ───────────────────────────────────────
        frame_reid = frame_orig.copy()
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confs_arr  = res.boxes.conf.cpu().numpy()
        clses_arr  = res.boxes.cls.cpu().numpy()
        det_arr    = (np.column_stack([boxes_xyxy, confs_arr, clses_arr])
                      if len(boxes_xyxy) else np.empty((0,6)))
        tracks = tracker.update(det_arr, frame_reid)

        if reid is not None and len(tracks):
            t_boxes    = tracks[:, :4]
            t_ids      = tracks[:, 4].astype(int)
            reid_feats = reid.extract(frame_reid, t_boxes)
            for tid, feat in zip(t_ids, reid_feats):
                gallery.update(int(tid), feat)

        for t in tracks:
            x1,y1,x2,y2 = t[:4]
            tid  = int(t[4])
            tc   = float(t[5])
            tn   = remap_label(res.names.get(int(t[6]), "person"))
            draw_track(frame_reid, (x1,y1,x2,y2), tid, tn, tc)

        reid_label = f"Tracking+ReID  ({len(tracks)} active  {len(gallery.all_ids())} gallery)"
        row_ax[2].imshow(cv2.cvtColor(frame_reid, cv2.COLOR_BGR2RGB))
        row_ax[2].set_title(reid_label, fontsize=10, fontweight="bold")
        row_ax[2].axis("off")

        row_ax[0].set_ylabel(img_path.name, fontsize=7, rotation=0,
                             labelpad=60, va="center")

    fig.suptitle("YOLOv8n CrowdHuman — Original | Detection | Tracking+ReID",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = Path(save_dir) / "comparison_grid.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅  Comparison grid → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  ReID Gallery builder  — build gallery from test images, query one image
# ─────────────────────────────────────────────────────────────────────────────
def reid_gallery_mode(
    test_img_dir : str  = "test/images",
    query_image  : str  = None,
    imgsz        : int  = 640,
    conf         : float= 0.5,
    top_k        : int  = 5,
    save_dir     : str  = "reid_results",
):
    """
    1. Run detector on all gallery images, extract ReID features per crop.
    2. If query_image is given, detect persons in it and find top-K
       most similar gallery crops.
    """
    if not HAS_REID:
        print("⚠️   ReID not available"); return

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)
    reid  = OSNetExtractor(DEVICE)

    images = (sorted(Path(test_img_dir).glob("*.jpg")) +
              sorted(Path(test_img_dir).glob("*.png")))
    if not images:
        print(f"⚠️   No gallery images in {test_img_dir}"); return

    print(f"\n  Building gallery from {len(images)} images …")

    # gallery_entries: list of {img_path, box, feat, crop}
    gallery_entries = []

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        res   = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        if not len(boxes):
            continue
        feats = reid.extract(frame, boxes)
        H, W  = frame.shape[:2]
        for box, feat in zip(boxes, feats):
            x1,y1,x2,y2 = [int(v) for v in box]
            crop = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
            gallery_entries.append({
                "img_path": str(img_path),
                "box"     : [x1,y1,x2,y2],
                "feat"    : feat,
                "crop"    : crop,
            })

    print(f"  Gallery: {len(gallery_entries)} person crops indexed")

    if not query_image:
        print("  No --query-image given — gallery built but not queried.")
        print("  Re-run with:  --mode reid_gallery --query-image path/to/person.jpg")
        return

    # ── Query ─────────────────────────────────────────────────────────────
    q_frame = cv2.imread(query_image)
    if q_frame is None:
        print(f"❌  Cannot read query image: {query_image}"); return

    q_res   = model(q_frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    q_boxes = q_res.boxes.xyxy.cpu().numpy()
    if not len(q_boxes):
        print("  No persons detected in query image."); return

    q_feats = reid.extract(q_frame, q_boxes)

    if not HAS_MPL:
        print("⚠️  matplotlib missing — skipping ReID gallery visualisation")
        return

    gallery_stack = np.stack([e["feat"] for e in gallery_entries])  # (G, D)

    for qi, q_feat in enumerate(q_feats):
        sims    = gallery_stack @ q_feat
        top_idx = np.argsort(sims)[::-1][:top_k]

        # ── Plot ──────────────────────────────────────────────────────────
        n_cols = top_k + 1
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols*3, 4))

        # Query crop
        qb = q_boxes[qi].astype(int)
        H, W = q_frame.shape[:2]
        q_crop = q_frame[max(0,qb[1]):min(H,qb[3]), max(0,qb[0]):min(W,qb[2])]
        axes[0].imshow(cv2.cvtColor(q_crop, cv2.COLOR_BGR2RGB))
        axes[0].set_title("QUERY", fontsize=11, fontweight="bold", color="red")
        axes[0].axis("off")

        for rank, gi in enumerate(top_idx):
            e    = gallery_entries[gi]
            crop = e["crop"]
            sim  = sims[gi]
            axes[rank+1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axes[rank+1].set_title(f"Rank {rank+1}\n{sim:.3f}", fontsize=9,
                                   color="green" if rank == 0 else "black")
            axes[rank+1].axis("off")
            # green border for top match
            for spine in axes[rank+1].spines.values():
                spine.set_edgecolor("lime" if rank == 0 else "gray")
                spine.set_linewidth(3 if rank == 0 else 1)

        fig.suptitle(f"ReID Query Result — Person #{qi+1} from {Path(query_image).name}",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        out = Path(save_dir) / f"reid_query_person{qi+1}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅  Query result → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Trajectory / FPS visualisation after tracking
# ─────────────────────────────────────────────────────────────────────────────
def plot_trajectories(frames_data: list, save_dir: str, title: str = "Track Trajectories"):
    if not HAS_MPL or not frames_data:
        return
    traj = defaultdict(list)
    for fd in frames_data:
        for t in fd["tracks"]:
            b = t["box"]
            traj[t["track_id"]].append(((b[0]+b[2])//2, (b[1]+b[3])//2))

    top_ids = sorted(traj, key=lambda x: -len(traj[x]))[:25]
    cmap    = plt.cm.tab20

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, tid in enumerate(top_ids):
        pts = np.array(traj[tid])
        ax.plot(pts[:,0], pts[:,1], "-o", markersize=2,
                linewidth=1.5, color=cmap(i % 20), label=f"#{tid}({len(pts)}f)")
        ax.text(pts[-1,0], pts[-1,1], str(tid), fontsize=7, color=cmap(i % 20))

    ax.invert_yaxis()
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
    ax.legend(fontsize=6, ncol=5, loc="upper right")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    out = Path(save_dir) / "trajectories.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ✅  Trajectories → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Person ReID Pipeline — YOLOv8n CrowdHuman + OSNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", default="video",
                   choices=["eval","compare","video","webcam",
                            "benchmark","reid_gallery","all"],
                   help="Pipeline mode")

    # I/O
    p.add_argument("--source",      default=None,  help="Video file (for video/all mode)")
    p.add_argument("--cam",         default=0, type=int, help="Webcam device ID")
    p.add_argument("--test-img-dir",default="test/images")
    p.add_argument("--save-dir",    default="reid_results")
    p.add_argument("--query-image", default=None,
                   help="Query image for reid_gallery mode")

    # Detection
    p.add_argument("--imgsz",  default=640,  type=int)
    p.add_argument("--conf",   default=0.5, type=float)
    p.add_argument("--iou",    default=0.45, type=float)

    # Tracking
    p.add_argument("--tracker", default="botsort",
                   choices=["bytetrack","botsort"],
                   help="Tracker algorithm")

    # ReID
    p.add_argument("--no-reid",       action="store_true", help="Disable ReID")
    p.add_argument("--reid-sim",      default=0.85, type=float,
                   help="Cosine similarity threshold for ReID match")
    p.add_argument("--reid-ema",      default=0.70, type=float,
                   help="EMA weight for gallery feature update")
    p.add_argument("--occlusion-zone", action="append", default=[],
                   help="Normalized occlusion zone x1,y1,x2,y2. Repeatable.")
    p.add_argument("--ship-barrier-preset", action="store_true",
                   help="Enable the barrier preset from the provided ship-deck screenshot.")
    p.add_argument("--dynamic-occlusion-zone", action="store_true",
                   help="Track occlusion zones frame-to-frame to compensate for camera oscillation.")

    # Display
    p.add_argument("--show",          action="store_true",
                   help="Show live OpenCV window (requires display)")
    p.add_argument("--max-images",    default=None, type=int)
    p.add_argument("--max-frames",    default=None, type=int)
    p.add_argument("--no-save-video", action="store_true",
                   help="Do not write output video to disk")
    p.add_argument("--top-k",         default=5, type=int,
                   help="Top-K matches in reid_gallery mode")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'━'*62}")
    print(f"  Person ReID Pipeline")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Mode    : {args.mode}")
    print(f"  Tracker : {args.tracker}")
    print(f"  Device  : {DEVICE}")
    print(f"  ReID    : {'OFF' if args.no_reid else 'OSNet-x1_0'}")
    print(f"{'━'*62}\n")

    mode     = args.mode
    save_dir = args.save_dir
    use_reid = not args.no_reid
    occlusion_zones = [_parse_zone_spec(spec) for spec in args.occlusion_zone]
    if args.ship_barrier_preset:
        occlusion_zones.append(DEFAULT_SHIP_BARRIER_ZONE)

    # ── helper to resolve video source ────────────────────────────────────
    def resolve_source():
        if args.source:
            return args.source
        vid_dir = Path(args.test_img_dir).parent / "videos"
        if vid_dir.exists():
            vids = list(vid_dir.glob("*.mp4")) + list(vid_dir.glob("*.avi"))
            if vids:
                print(f"  Auto-selected: {vids[0]}")
                return str(vids[0])
        return None

    if mode in ("eval", "all"):
        evaluate(
            test_img_dir = args.test_img_dir,
            imgsz        = args.imgsz,
            conf         = args.conf,
            iou_thresh   = args.iou,
            save_dir     = save_dir,
            max_images   = args.max_images,
        )

    if mode in ("compare", "all"):
        compare(
            test_img_dir = args.test_img_dir,
            imgsz        = args.imgsz,
            conf         = args.conf,
            tracker_type = args.tracker,
            save_dir     = save_dir,
        )

    if mode in ("benchmark", "all"):
        benchmark(
            test_img_dir = args.test_img_dir,
            imgsz        = args.imgsz,
            save_dir     = save_dir,
        )

    if mode in ("video", "all"):
        src = resolve_source()
        if src:
            src_stem    = Path(src).stem
            output_path = (None if args.no_save_video
                           else Path(save_dir) / f"{src_stem}_reid.mp4")
            frames_data = run_tracking(
                source          = src,
                output_path     = str(output_path) if output_path else None,
                imgsz           = args.imgsz,
                conf            = args.conf,
                iou             = args.iou,
                tracker_type    = args.tracker,
                use_reid        = use_reid,
                reid_sim_thresh = args.reid_sim,
                reid_ema        = args.reid_ema,
                occlusion_zones = occlusion_zones,
                dynamic_occlusion_zones = args.dynamic_occlusion_zone,
                show            = args.show,
                max_frames      = args.max_frames,
                save_reid_json  = True,
            )
            plot_trajectories(frames_data, save_dir,
                              title=f"Trajectories — {src_stem}")
        else:
            print("  ⚠️   No video source found. Use --source path/to/video.mp4")

    if mode == "webcam":
        src = str(args.cam)
        run_tracking(
            source          = src,
            output_path     = None,
            imgsz           = args.imgsz,
            conf            = args.conf,
            iou             = args.iou,
            tracker_type    = args.tracker,
            use_reid        = use_reid,
            reid_sim_thresh = args.reid_sim,
            reid_ema        = args.reid_ema,
            occlusion_zones = occlusion_zones,
            dynamic_occlusion_zones = args.dynamic_occlusion_zone,
            show            = True,
            save_reid_json  = False,
        )

    if mode == "reid_gallery":
        reid_gallery_mode(
            test_img_dir = args.test_img_dir,
            query_image  = args.query_image,
            imgsz        = args.imgsz,
            conf         = args.conf,
            top_k        = args.top_k,
            save_dir     = save_dir,
        )

    print(f"\n{'━'*62}")
    print(f"  ✅  All done!  Outputs → {Path(save_dir).resolve()}")
    print(f"{'━'*62}\n")


if __name__ == "__main__":
    main()
