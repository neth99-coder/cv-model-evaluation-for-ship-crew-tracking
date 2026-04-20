"""
pipeline.py
───────────
Main cloth-colour detection pipeline.

Usage
-----
    from pipeline import ClothColorPipeline

    pipe = ClothColorPipeline(segmenter_name="grabcut", use_sahi=True)
    pipe.process_video("test/video.mp4", output_path="output/result.mp4")
"""

from __future__ import annotations
import cv2
import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from segmenters import BaseSegmenter, PersonDetection, get_segmenter
from color_utils  import extract_cloth_colors, GarmentColor
from body_regions import split_body, morphological_refine

log = logging.getLogger(__name__)


# ─── Per-frame result ─────────────────────────────────────────────────────────

@dataclass
class PersonResult:
    detection:    PersonDetection
    upper_colors: List[GarmentColor] = field(default_factory=list)
    lower_colors: List[GarmentColor] = field(default_factory=list)

@dataclass
class FrameResult:
    frame_idx:     int
    persons:       List[PersonResult]
    inference_ms:  float = 0.0
    color_ms:      float = 0.0


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class ClothColorPipeline:
    def __init__(
        self,
        segmenter_name:   str   = "grabcut",
        segmenter_kwargs: dict  = None,
        use_sahi:         bool  = False,
        sahi_slice_h:     int   = 320,
        sahi_slice_w:     int   = 320,
        sahi_overlap:     float = 0.2,
        n_colors:         int   = 3,
        use_hs_only:      bool  = True,    # shadow-invariant clustering
        skip_frames:      int   = 2,       # process every N-th frame
        display_scale:    float = 1.0,
        segmenter:        Optional[BaseSegmenter] = None,
    ):
        if segmenter is not None:
            self.seg = segmenter
        else:
            self.seg = get_segmenter(segmenter_name)

        if segmenter_kwargs:
            for k, v in segmenter_kwargs.items():
                setattr(self.seg, k, v)

        self.use_sahi      = use_sahi
        self.sahi_slice_h  = sahi_slice_h
        self.sahi_slice_w  = sahi_slice_w
        self.sahi_overlap  = sahi_overlap
        self.n_colors      = n_colors
        self.use_hs_only   = use_hs_only
        self.skip_frames   = max(1, skip_frames)
        self.display_scale = display_scale

        self._loaded = False

    # ── Loading ──────────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if not self._loaded:
            log.info(f"Loading segmenter: {self.seg.name}")
            self.seg.load()
            self._loaded = True

    # ── Inference on a single frame ──────────────────────────────────────────

    def infer_frame(self, frame: np.ndarray) -> FrameResult:
        self._ensure_loaded()

        t0 = time.perf_counter()
        if self.use_sahi:
            dets = self.seg.segment_sahi(
                frame,
                slice_h  = self.sahi_slice_h,
                slice_w  = self.sahi_slice_w,
                overlap  = self.sahi_overlap,
            )
        else:
            dets = self.seg.segment(frame)
        infer_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        persons: List[PersonResult] = []
        for det in dets:
            mask = morphological_refine(det.mask, ksize=5)
            regions = split_body(mask, det.bbox)
            upper_colors = extract_cloth_colors(
                frame, regions.upper,
                n_colors   = self.n_colors,
                use_hs_only = self.use_hs_only,
            )
            lower_colors = extract_cloth_colors(
                frame, regions.lower,
                n_colors   = self.n_colors,
                use_hs_only = self.use_hs_only,
            )
            # Keep only the dominant color for each garment region.
            upper_colors = upper_colors[:1]
            lower_colors = lower_colors[:1]
            persons.append(PersonResult(
                detection    = det,
                upper_colors = upper_colors,
                lower_colors = lower_colors,
            ))
        color_ms = (time.perf_counter() - t1) * 1000

        return FrameResult(frame_idx=0, persons=persons,
                           inference_ms=infer_ms, color_ms=color_ms)

    # ── Video processing ─────────────────────────────────────────────────────

    def process_video(
        self,
        video_path:   str,
        output_path:  Optional[str] = None,
        max_frames:   Optional[int] = None,
        show_preview: bool = False,
    ) -> List[FrameResult]:
        self._ensure_loaded()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {video_path}")

        W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        writer = None
        if output_path:
            ow = int(W * self.display_scale)
            oh = int(H * self.display_scale)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (ow, oh))

        results: List[FrameResult] = []
        frame_idx = 0
        last_result: Optional[FrameResult] = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and frame_idx >= max_frames:
                break

            if frame_idx % self.skip_frames == 0:
                result = self.infer_frame(frame)
                result.frame_idx = frame_idx
                results.append(result)
                last_result = result

                log.info(
                    f"Frame {frame_idx:05d} | "
                    f"persons={len(result.persons)} | "
                    f"infer={result.inference_ms:.1f}ms | "
                    f"color={result.color_ms:.1f}ms"
                )

            # Always draw (use cached result on skipped frames)
            draw_result = last_result or FrameResult(frame_idx=frame_idx, persons=[])
            vis = draw_frame(frame, draw_result, scale=self.display_scale)

            if writer:
                writer.write(vis)
            if show_preview:
                cv2.imshow("Cloth Color Detection", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        return results

    # ── Single image ─────────────────────────────────────────────────────────

    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
    ) -> FrameResult:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
        result = self.infer_frame(frame)
        vis = draw_frame(frame, result, scale=self.display_scale)
        if output_path:
            cv2.imwrite(output_path, vis)
            log.info(f"Saved to {output_path}")
        return result


# ─── Visualisation ────────────────────────────────────────────────────────────

COLORS_BGR = [
    (0,   220, 255),
    (0,   180, 0),
    (255, 120, 0),
    (200, 0,   200),
    (0,   120, 255),
]


def draw_frame(
    frame:  np.ndarray,
    result: FrameResult,
    scale:  float = 1.0,
) -> np.ndarray:
    vis = frame.copy()

    for i, pr in enumerate(result.persons):
        det   = pr.detection
        color = COLORS_BGR[i % len(COLORS_BGR)]
        x1,y1,x2,y2 = det.bbox

        # Mask overlay
        if det.mask is not None and det.mask.sum() > 0:
            overlay = vis.copy()
            overlay[det.mask > 0] = [int(c * 0.5) + int(p * 0.5)
                                     for c, p in zip(color,
                                                     overlay[det.mask > 0].mean(axis=0))]
            vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        # Bounding box
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)

        # Color labels
        text_y = y1 - 8
        label_lines = [f"[{i+1}] {det.source} {det.score:.2f}"]
        for gc in pr.upper_colors[:2]:
            label_lines.append(f"  TOP: {gc.name} ({gc.proportion*100:.0f}%)")
        for gc in pr.lower_colors[:2]:
            label_lines.append(f"  BTM: {gc.name} ({gc.proportion*100:.0f}%)")

        for line in reversed(label_lines):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, text_y - th - 2), (x1 + tw + 4, text_y + 2),
                          (0,0,0), -1)
            cv2.putText(vis, line, (x1+2, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            text_y -= (th + 6)

        # Small colour swatches in top-right of bbox
        _draw_swatches(vis, pr.upper_colors, x2, y1, label="▲")
        _draw_swatches(vis, pr.lower_colors, x2, y1 + 30, label="▼")

    # HUD
    hud = (f"Seg: {result.persons[0].detection.source if result.persons else 'none'} | "
           f"infer={result.inference_ms:.0f}ms | "
           f"color={result.color_ms:.0f}ms | "
           f"persons={len(result.persons)}")
    cv2.rectangle(vis, (0, 0), (len(hud)*8 + 10, 20), (0,0,0), -1)
    cv2.putText(vis, hud, (5,14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200,255,200), 1, cv2.LINE_AA)

    if scale != 1.0:
        H, W = vis.shape[:2]
        vis = cv2.resize(vis, (int(W*scale), int(H*scale)))
    return vis


def _draw_swatches(
    vis: np.ndarray,
    colors: List[GarmentColor],
    x: int, y: int,
    label: str = "",
    sw: int = 20, sh: int = 20, gap: int = 2,
):
    H, W = vis.shape[:2]
    for j, gc in enumerate(colors[:3]):
        sx = x - (j+1)*(sw+gap)
        sy = y
        if sx < 0 or sy < 0 or sx+sw >= W or sy+sh >= H:
            continue
        r, g, b = gc.rgb
        cv2.rectangle(vis, (sx, sy), (sx+sw, sy+sh), (b,g,r), -1)
        cv2.rectangle(vis, (sx, sy), (sx+sw, sy+sh), (255,255,255), 1)
