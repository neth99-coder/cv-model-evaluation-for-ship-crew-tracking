"""
segmenters.py
─────────────
Pluggable segmentation back-ends.

Each class implements:
    segment(frame: np.ndarray) -> List[PersonDetection]

where PersonDetection holds:
    bbox   : (x1, y1, x2, y2)  absolute pixel coords
    mask   : np.ndarray uint8 same H×W as frame (0 or 255)
    score  : float confidence
    label  : str  e.g. "person"

Adding a new back-end = subclass BaseSegmenter + register in REGISTRY.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Type
import time
import logging

log = logging.getLogger(__name__)


# ─── Shared result type ───────────────────────────────────────────────────────

@dataclass
class PersonDetection:
    bbox:   Tuple[int, int, int, int]   # x1 y1 x2 y2
    mask:   np.ndarray                  # uint8 H×W  (0 / 255)
    score:  float
    label:  str = "person"
    source: str = ""                    # which segmenter produced this


# ─── Base class ───────────────────────────────────────────────────────────────

class BaseSegmenter:
    name: str = "base"

    def load(self) -> None:
        """One-time model loading. Called before first segment()."""
        pass

    def segment(self, frame: np.ndarray) -> List[PersonDetection]:
        raise NotImplementedError

    # SAHI-style sliced inference helper
    def segment_sahi(
        self,
        frame:       np.ndarray,
        slice_h:     int  = 320,
        slice_w:     int  = 320,
        overlap:     float = 0.2,
        min_area:    int   = 300,
    ) -> List[PersonDetection]:
        """
        Slice the frame into overlapping tiles, run segment() on each,
        re-project detections back to full-frame coordinates, then NMS.
        """
        H, W = frame.shape[:2]
        step_h = max(1, int(slice_h * (1 - overlap)))
        step_w = max(1, int(slice_w * (1 - overlap)))

        all_dets: List[PersonDetection] = []

        for y0 in range(0, H, step_h):
            for x0 in range(0, W, step_w):
                y1 = min(y0 + slice_h, H)
                x1 = min(x0 + slice_w, W)
                tile = frame[y0:y1, x0:x1]
                th, tw = tile.shape[:2]

                tile_dets = self.segment(tile)
                for det in tile_dets:
                    # Re-project bbox
                    bx1, by1, bx2, by2 = det.bbox
                    bx1 += x0;  by1 += y0
                    bx2 += x0;  by2 += y0

                    # Re-project mask to full frame
                    full_mask = np.zeros((H, W), dtype=np.uint8)
                    full_mask[y0:y0+th, x0:x0+tw] = det.mask

                    area = int(det.mask.sum() / 255)
                    if area < min_area:
                        continue

                    all_dets.append(PersonDetection(
                        bbox=(bx1, by1, bx2, by2),
                        mask=full_mask,
                        score=det.score,
                        label=det.label,
                        source=det.source + "+SAHI",
                    ))

        return _nms_detections(all_dets, iou_thresh=0.4)


# ─── 1. GrabCut (zero dependencies) ──────────────────────────────────────────

class GrabCutSegmenter(BaseSegmenter):
    """
    Uses YOLOv8n for detection + GrabCut for pixel-level mask.
    Falls back to bbox-rectangle mask if YOLO unavailable.
    Pure OpenCV — no extra installs needed.
    """
    name = "GrabCut"

    def __init__(self, conf: float = 0.4):
        self.conf = conf
        self._detector = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
            self._detector = YOLO("yolov8n.pt")
            log.info("GrabCutSegmenter: using YOLOv8n for detection")
        except Exception:
            log.warning("GrabCutSegmenter: YOLO unavailable, using HOG person detector")
            self._detector = None

    def segment(self, frame: np.ndarray) -> List[PersonDetection]:
        bboxes = self._detect_persons(frame)
        results = []
        for bbox in bboxes:
            mask = self._grabcut_mask(frame, bbox)
            results.append(PersonDetection(
                bbox=bbox, mask=mask, score=0.8, source=self.name
            ))
        return results

    def _detect_persons(self, frame):
        if self._detector is not None:
            res = self._detector(frame, classes=[0], conf=self.conf, verbose=False)
            boxes = []
            for r in res:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    boxes.append((x1,y1,x2,y2))
            return boxes
        # HOG fallback
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        found, _ = hog.detectMultiScale(frame, winStride=(8,8), padding=(4,4), scale=1.05)
        boxes = []
        if len(found):
            for (x,y,w,h) in found:
                boxes.append((x, y, x+w, y+h))
        return boxes

    def _grabcut_mask(self, frame, bbox):
        H, W = frame.shape[:2]
        x1,y1,x2,y2 = bbox
        x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
        if x2-x1 < 2 or y2-y1 < 2:
            return np.zeros((H,W), dtype=np.uint8)
        rect = (x1, y1, x2-x1, y2-y1)
        gc_mask = np.zeros((H, W), np.uint8)
        bgd = np.zeros((1,65), np.float64)
        fgd = np.zeros((1,65), np.float64)
        try:
            cv2.grabCut(frame, gc_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        except Exception:
            mask = np.zeros((H,W), np.uint8)
            mask[y1:y2, x1:x2] = 255
            return mask
        fg = np.where((gc_mask==1)|(gc_mask==3), 255, 0).astype(np.uint8)
        return fg


# ─── 2. YOLOv8-Seg ────────────────────────────────────────────────────────────

class YOLOv8SegSegmenter(BaseSegmenter):
    name = "YOLOv8-Seg"

    def __init__(self, model_size: str = "n", conf: float = 0.35):
        self.model_size = model_size
        self.conf = conf
        self._model = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(f"yolov8{self.model_size}-seg.pt")
            log.info(f"YOLOv8SegSegmenter loaded: yolov8{self.model_size}-seg")
        except Exception as e:
            raise RuntimeError(f"Cannot load YOLOv8-Seg: {e}")

    def segment(self, frame: np.ndarray) -> List[PersonDetection]:
        if self._model is None:
            return []
        H, W = frame.shape[:2]
        res = self._model(frame, classes=[0], conf=self.conf, verbose=False)
        out = []
        for r in res:
            if r.masks is None:
                continue
            for i, (b, m) in enumerate(zip(r.boxes, r.masks)):
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                mask_data = m.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_data, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
                out.append(PersonDetection(
                    bbox=(x1,y1,x2,y2),
                    mask=mask_bin,
                    score=float(b.conf),
                    source=self.name,
                ))
        return out


# ─── 3. YOLOv9-Seg ────────────────────────────────────────────────────────────

class YOLOv9SegSegmenter(BaseSegmenter):
    """YOLOv9c-seg via Ultralytics."""
    name = "YOLOv9-Seg"

    def __init__(self, conf: float = 0.35):
        self.conf = conf
        self._model = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO("yolov9c-seg.pt")
            log.info("YOLOv9SegSegmenter loaded")
        except Exception as e:
            raise RuntimeError(f"Cannot load YOLOv9-Seg: {e}")

    def segment(self, frame: np.ndarray) -> List[PersonDetection]:
        if self._model is None:
            return []
        H, W = frame.shape[:2]
        res = self._model(frame, classes=[0], conf=self.conf, verbose=False)
        out = []
        for r in res:
            if r.masks is None:
                continue
            for b, m in zip(r.boxes, r.masks):
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                mask_data = m.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_data, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
                out.append(PersonDetection(
                    bbox=(x1,y1,x2,y2),
                    mask=mask_bin,
                    score=float(b.conf),
                    source=self.name,
                ))
        return out


# ─── 4. MediaPipe Selfie Segmentation ─────────────────────────────────────────

class MediaPipeSegmenter(BaseSegmenter):
    name = "MediaPipe"

    def __init__(self, model_selection: int = 1, person_detector=None):
        """
        model_selection: 0=general(fast), 1=landscape(accurate)
        person_detector: optional callable(frame) → List[(x1,y1,x2,y2)]
                         used to isolate ROI before running MP segmentation
        """
        self.model_selection = model_selection
        self._mp_seg = None
        self._person_detector = person_detector

    def load(self) -> None:
        try:
            import mediapipe as mp
            self._mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=self.model_selection
            )
            log.info(f"MediaPipeSegmenter loaded (model={self.model_selection})")
        except Exception as e:
            raise RuntimeError(f"Cannot load MediaPipe: {e}")

    def segment(self, frame: np.ndarray) -> List[PersonDetection]:
        if self._mp_seg is None:
            return []
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bboxes = self._person_detector(frame) if self._person_detector else [(0, 0, W, H)]
        out = []
        for (x1, y1, x2, y2) in bboxes:
            x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
            if x2-x1 < 10 or y2-y1 < 10:
                continue
            roi_rgb = rgb[y1:y2, x1:x2]
            res = self._mp_seg.process(roi_rgb)
            if res.segmentation_mask is None:
                continue
            roi_mask = (res.segmentation_mask > 0.5).astype(np.uint8) * 255
            full_mask = np.zeros((H, W), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = roi_mask
            out.append(PersonDetection(
                bbox=(x1,y1,x2,y2),
                mask=full_mask,
                score=0.85,
                source=self.name,
            ))
        return out


# ─── 5. Background subtraction (MOG2) — good for static-cam footage ──────────

class BackgroundSubtractorSegmenter(BaseSegmenter):
    """
    Uses MOG2 background subtraction + contour filtering.
    Fast, zero deep-learning, best for static cameras.
    """
    name = "BGSub-MOG2"

    def __init__(self, history: int = 200, var_threshold: int = 40,
                 min_area: int = 2000):
        self.history       = history
        self.var_threshold = var_threshold
        self.min_area      = min_area
        self._bg = None

    def load(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=True,
        )
        log.info("BackgroundSubtractorSegmenter loaded (MOG2)")

    def segment(self, frame: np.ndarray) -> List[PersonDetection]:
        if self._bg is None:
            return []
        H, W = frame.shape[:2]
        fg = self._bg.apply(frame)
        # 127 = shadow (MOG2); keep only definite foreground
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        # morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel, iterations=1)

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x,y,w,h = cv2.boundingRect(c)
            if h < w * 1.2:          # rough person aspect-ratio filter
                continue
            mask = np.zeros((H,W), dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            out.append(PersonDetection(
                bbox=(x, y, x+w, y+h),
                mask=mask,
                score=min(1.0, area / 20000),
                source=self.name,
            ))
        return out


# ─── Registry ─────────────────────────────────────────────────────────────────

REGISTRY: Dict[str, Type[BaseSegmenter]] = {
    "grabcut":      GrabCutSegmenter,
    "yolov8n-seg":  lambda: YOLOv8SegSegmenter(model_size="n"),
    "yolov8s-seg":  lambda: YOLOv8SegSegmenter(model_size="s"),
    "yolov8m-seg":  lambda: YOLOv8SegSegmenter(model_size="m"),
    "yolov9-seg":   YOLOv9SegSegmenter,
    "mediapipe":    MediaPipeSegmenter,
    "bgsub":        BackgroundSubtractorSegmenter,
}


def get_segmenter(name: str) -> BaseSegmenter:
    name = name.lower()
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown segmenter '{name}'. Available: {list(REGISTRY.keys())}"
        )
    factory = REGISTRY[name]
    seg = factory() if callable(factory) else factory
    return seg


# ─── NMS helper ───────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def _nms_detections(dets: List[PersonDetection], iou_thresh=0.4) -> List[PersonDetection]:
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept = []
    for d in dets:
        if all(_iou(d.bbox, k.bbox) < iou_thresh for k in kept):
            kept.append(d)
    return kept
