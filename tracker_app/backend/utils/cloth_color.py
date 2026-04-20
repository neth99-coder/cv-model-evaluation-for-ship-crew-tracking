"""
Clothing color recognition utilities for tracker_app.

Pluggable segmenters:
- grabcut
- yolov8n-seg
- yolov8s-seg

Designed to augment person tracking results without changing tracking IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

BASE_COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("Black", (10, 10, 10)),
    ("White", (245, 245, 245)),
    ("Gray", (128, 128, 128)),
    ("Red", (220, 30, 30)),
    ("Orange", (245, 140, 30)),
    ("Yellow", (240, 220, 30)),
    ("Green", (40, 170, 40)),
    ("Blue", (40, 90, 210)),
    ("Purple", (140, 70, 180)),
    ("Pink", (235, 130, 180)),
    ("Brown", (130, 80, 45)),
]


@dataclass
class GarmentColor:
    name: str
    rgb: tuple[int, int, int]
    proportion: float
    confidence: float


@dataclass
class SegDetection:
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    score: float


class _GrabCutSegmenter:
    name = "grabcut"

    def segment_for_bbox(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None

        rect = (x1, y1, x2 - x1, y2 - y1)
        gc_mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(frame, gc_mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
            mask = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype(np.uint8)
            if int(mask.sum() / 255) < 40:
                return None
            return mask
        except Exception:
            return None


class _YOLOSegSegmenter:
    def __init__(self, model_name: str):
        from ultralytics import YOLO

        self.name = model_name
        self.model = YOLO(f"{model_name}.pt")

    def segment(self, frame: np.ndarray, conf: float = 0.35) -> list[SegDetection]:
        h, w = frame.shape[:2]
        detections: list[SegDetection] = []
        results = self.model.predict(frame, classes=[0], conf=conf, verbose=False)
        for r in results:
            if r.masks is None:
                continue
            for b, m in zip(r.boxes, r.masks):
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                mask_data = m.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
                detections.append(SegDetection((x1, y1, x2, y2), mask_bin, float(b.conf)))
        return detections


class ClothColorEngine:
    def __init__(self, segmenter: str = "grabcut", n_colors: int = 3, conf: float = 0.35):
        self.segmenter_name = segmenter
        self.n_colors = n_colors
        self.seg_conf = conf

        if segmenter == "grabcut":
            self.segmenter = _GrabCutSegmenter()
        elif segmenter in {"yolov8n-seg", "yolov8s-seg"}:
            self.segmenter = _YOLOSegSegmenter(segmenter)
        else:
            raise ValueError(f"Unsupported color segmenter: {segmenter}")

    def analyze_tracks(self, frame: np.ndarray, tracks: np.ndarray) -> dict[int, dict]:
        if tracks is None or len(tracks) == 0:
            return {}

        track_boxes: dict[int, tuple[int, int, int, int]] = {}
        for t in tracks:
            tid = _safe_tid(t[4])
            x1, y1, x2, y2 = [int(float(v)) for v in t[:4]]
            track_boxes[tid] = (x1, y1, x2, y2)

        masks_by_tid: dict[int, np.ndarray] = {}

        if isinstance(self.segmenter, _YOLOSegSegmenter):
            seg_dets = self.segmenter.segment(frame, conf=self.seg_conf)
            available = set(range(len(seg_dets)))
            for tid, tb in track_boxes.items():
                best_idx = -1
                best_iou = 0.0
                for i in available:
                    iou = _iou(tb, seg_dets[i].bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                if best_idx >= 0 and best_iou >= 0.15:
                    masks_by_tid[tid] = seg_dets[best_idx].mask
                    available.remove(best_idx)
        else:
            for tid, tb in track_boxes.items():
                m = self.segmenter.segment_for_bbox(frame, tb)
                if m is not None:
                    masks_by_tid[tid] = m

        out: dict[int, dict] = {}
        for tid, bbox in track_boxes.items():
            mask = masks_by_tid.get(tid)
            if mask is None:
                continue

            upper, lower = _split_body(mask, bbox)
            upper_colors = extract_cloth_colors(frame, upper, n_colors=self.n_colors)
            lower_colors = extract_cloth_colors(frame, lower, n_colors=self.n_colors)
            top = upper_colors[0] if upper_colors else None
            bottom = lower_colors[0] if lower_colors else None

            out[tid] = {
                "top": top.name if top else None,
                "bottom": bottom.name if bottom else None,
                "top_confidence": round(top.confidence, 3) if top else None,
                "bottom_confidence": round(bottom.confidence, 3) if bottom else None,
            }
        return out


def extract_cloth_colors(
    frame: np.ndarray,
    mask: np.ndarray,
    n_colors: int = 3,
    min_pixels: int = 200,
) -> list[GarmentColor]:
    roi_pixels = _get_roi_pixels(frame, mask)
    if roi_pixels is None or len(roi_pixels) < min_pixels:
        return []

    roi_pixels = _filter_noise(roi_pixels)
    if len(roi_pixels) < min_pixels:
        return []

    n_colors = min(n_colors, max(1, len(roi_pixels) // 60))
    features = _to_hs_features(roi_pixels)
    labels = _kmeans_labels(features, n_colors)
    centers_rgb = _cluster_centers_rgb(roi_pixels, labels, n_colors)

    total = len(labels)
    colors: list[GarmentColor] = []
    for k in range(n_colors):
        p = float(np.sum(labels == k)) / total
        if p < 0.05:
            continue
        rgb = tuple(int(v) for v in centers_rgb[k])
        name, conf = _name_color(rgb)
        colors.append(GarmentColor(name=name, rgb=rgb, proportion=round(p, 3), confidence=round(conf, 3)))

    colors = _merge_same_color_clusters(colors)
    colors.sort(key=lambda c: c.proportion, reverse=True)
    return colors


def _get_roi_pixels(frame: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    if mask is None or mask.sum() == 0:
        return None
    pixels = frame[mask > 127]
    if pixels.size == 0:
        return None
    return pixels[:, ::-1].astype(np.float32)


def _filter_noise(pixels: np.ndarray, brightness_floor: int = 15) -> np.ndarray:
    return pixels[np.max(pixels, axis=1) > brightness_floor]


def _to_hs_features(rgb: np.ndarray) -> np.ndarray:
    bgr = rgb[:, ::-1].astype(np.uint8)
    hsv = cv2.cvtColor(bgr[np.newaxis], cv2.COLOR_BGR2HSV)[0].astype(np.float32)
    h = hsv[:, 0]
    s = hsv[:, 1]
    h_rad = h * (np.pi / 90.0)
    return np.stack([np.sin(h_rad) * s, np.cos(h_rad) * s, s], axis=1).astype(np.float32)


def _kmeans_labels(features: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(features) < 2:
        return np.zeros((len(features),), dtype=np.int32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.2)
    _compactness, labels, _centers = cv2.kmeans(features, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(-1)


def _cluster_centers_rgb(rgb_pixels: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    centers = np.zeros((k, 3), dtype=np.float32)
    for i in range(k):
        m = labels == i
        if np.any(m):
            centers[i] = rgb_pixels[m].mean(axis=0)
    return centers.astype(int)


def _name_color(rgb: tuple[int, int, int]) -> tuple[str, float]:
    hsv = cv2.cvtColor(np.uint8([[list(rgb[::-1])]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if v < 35 and s < 90:
        name = "Black"
    elif s < 22 and v > 210:
        name = "White"
    elif s < 35:
        name = "Gray"
    elif 8 <= h <= 25 and 40 <= v <= 190 and s >= 35:
        name = "Brown"
    elif h < 10 or h >= 170:
        name = "Red"
    elif h < 22:
        name = "Orange"
    elif h < 36:
        name = "Yellow"
    elif h < 86:
        name = "Green"
    elif h < 132:
        name = "Blue"
    elif h < 160:
        name = "Purple"
    else:
        name = "Pink"

    ref_rgb = np.array(dict(BASE_COLORS)[name], dtype=np.float32)
    dist = np.linalg.norm(np.array(rgb, dtype=np.float32) - ref_rgb)
    conf = 1.0 - float(dist) / (np.sqrt(3) * 255)
    if name not in {"Black", "White", "Gray"} and s < 45:
        conf *= 0.85
    return name, max(0.0, min(1.0, conf))


def _merge_same_color_clusters(colors: list[GarmentColor]) -> list[GarmentColor]:
    merged: dict[str, dict] = {}
    for c in colors:
        if c.name not in merged:
            merged[c.name] = {"p": 0.0, "rgb": np.zeros(3, dtype=np.float64), "conf": 0.0}
        w = float(c.proportion)
        merged[c.name]["p"] += w
        merged[c.name]["rgb"] += np.array(c.rgb, dtype=np.float64) * w
        merged[c.name]["conf"] += float(c.confidence) * w

    out: list[GarmentColor] = []
    for name, m in merged.items():
        if m["p"] <= 0:
            continue
        rgb = np.clip(m["rgb"] / m["p"], 0, 255).astype(int)
        out.append(
            GarmentColor(
                name=name,
                rgb=(int(rgb[0]), int(rgb[1]), int(rgb[2])),
                proportion=round(float(m["p"]), 3),
                confidence=round(float(m["conf"] / m["p"]), 3),
            )
        )
    return out


def _split_body(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox
    h, w = mask.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    mid = y1 + int((y2 - y1) * 0.5)

    upper = np.zeros_like(mask)
    lower = np.zeros_like(mask)
    upper[y1:mid, x1:x2] = mask[y1:mid, x1:x2]
    lower[mid:y2, x1:x2] = mask[mid:y2, x1:x2]
    return upper, lower


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0


def _safe_tid(value) -> int:
    try:
        return int(float(value))
    except Exception:
        s = "".join(ch for ch in str(value) if ch.isdigit())
        return int(s) if s else abs(hash(str(value))) % 1000000
