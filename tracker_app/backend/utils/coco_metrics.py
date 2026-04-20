"""
COCO detection metric helpers for tracker outputs.

Computes:
- mAP @ IoU=0.50 (map_iou_50)
- mAP Small (map_small)

These metrics require a COCO-format ground-truth annotation file.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np


class CocoMetricEvaluator:
    """Collect detections by frame and evaluate with pycocotools COCOeval."""

    def __init__(self, annotation_path: Path):
        self.annotation_path = Path(annotation_path)
        self._enabled = False
        self._reason = None
        self._detections: list[dict] = []
        self._frame_to_image: dict[int, int] = {}
        self._coco_gt = None

        try:
            from pycocotools.coco import COCO  # type: ignore

            self._coco_gt = COCO(str(self.annotation_path))
            self._frame_to_image = self._build_frame_to_image_map(self._coco_gt)
            self._enabled = True
        except Exception as exc:
            self._reason = f"COCO metric setup failed: {exc}"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def reason(self) -> str | None:
        return self._reason

    def add_tracks(self, frame_idx: int, tracks) -> None:
        if not self._enabled:
            return

        image_id = self._frame_to_image.get(frame_idx)
        if image_id is None:
            return

        for t in tracks:
            if len(t) < 4:
                continue
            x1 = float(t[0])
            y1 = float(t[1])
            x2 = float(t[2])
            y2 = float(t[3])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue

            score = float(t[5]) if len(t) > 5 else 1.0
            self._detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": 1,  # COCO person
                    "bbox": [x1, y1, w, h],
                    "score": score,
                }
            )

    def evaluate(self) -> dict:
        """Return map_iou_50/map_small (or null metrics with reason if unavailable)."""
        if not self._enabled:
            return {
                "map_iou_50": None,
                "map_small": None,
                "map_note": self._reason
                or "COCO metrics unavailable (missing annotation file or pycocotools).",
            }

        if not self._detections:
            return {
                "map_iou_50": None,
                "map_small": None,
                "map_note": "No detections matched annotated frames for COCO evaluation.",
            }

        try:
            from pycocotools.cocoeval import COCOeval  # type: ignore

            coco_dt = self._coco_gt.loadRes(self._detections)
            coco_eval = COCOeval(self._coco_gt, coco_dt, iouType="bbox")
            coco_eval.params.iouThrs = np.array([0.50])

            # Keep API logs clean by silencing COCOeval summarize() prints.
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

            # stats[0] = AP @[IoU=0.50:0.95]; since iouThrs=[0.50], this becomes AP@0.50
            ap50 = float(coco_eval.stats[0]) if coco_eval.stats is not None else -1.0
            ap_small = float(coco_eval.stats[3]) if coco_eval.stats is not None else -1.0

            return {
                "map_iou_50": round(ap50, 4) if ap50 >= 0 else None,
                "map_small": round(ap_small, 4) if ap_small >= 0 else None,
                "map_note": None,
            }
        except Exception as exc:
            return {
                "map_iou_50": None,
                "map_small": None,
                "map_note": f"COCO metric evaluation failed: {exc}",
            }

    @staticmethod
    def _build_frame_to_image_map(coco_gt) -> dict[int, int]:
        imgs = coco_gt.dataset.get("images", [])
        if not imgs:
            return {}

        # Prefer explicit frame index when present.
        with_frame_id = [img for img in imgs if "frame_id" in img]
        if with_frame_id:
            return {int(img["frame_id"]): int(img["id"]) for img in with_frame_id}

        # Otherwise assume images are in video frame order by image id.
        ordered = sorted(imgs, key=lambda i: int(i.get("id", 0)))
        return {idx: int(img["id"]) for idx, img in enumerate(ordered)}


def resolve_coco_annotation_path(video_path: Path, explicit: str | None = None) -> Path | None:
    """
    Resolve COCO annotation file path.

    Resolution order:
    1) explicit path (absolute or relative)
    2) video_path with .json suffix
    3) video_path with .coco.json suffix
    4) test/test_coco.json
    5) test/annotations.json
    """
    candidates: list[Path] = []

    if explicit:
        p = Path(explicit)
        candidates.append(p)

    candidates.append(video_path.with_suffix(".json"))
    candidates.append(video_path.with_suffix(".coco.json"))

    root = Path(__file__).resolve().parent.parent
    candidates.append(root / "test" / "test_coco.json")
    candidates.append(root / "test" / "annotations.json")

    for candidate in candidates:
        # Treat relative inputs as relative to backend root.
        p = candidate if candidate.is_absolute() else (root / candidate)
        if p.exists() and p.is_file():
            return p
    return None
