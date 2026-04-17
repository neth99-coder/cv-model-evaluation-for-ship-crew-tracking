"""
Evaluation logic: run person detection on test videos,
compute metrics, and save annotated output videos + result files.

Ground-truth support (optional)
--------------------------------
If ``test/ground_truth.json`` exists, per-frame mAP@IoU=0.5 is computed.
Without it, only detection-count and FPS metrics are reported.

Ground-truth JSON format
-------------------------
{
  "video1.mp4": {
    "0":  [[x1,y1,x2,y2], [x1,y1,x2,y2]],
    "30": [[x1,y1,x2,y2]]
  },
  "video2.mp4": { ... }
}

Keys are frame indices (as strings). Missing frames are treated as
containing zero persons.
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from models.base import BasePersonDetector

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# ── Annotation colours (BGR) ──────────────────────────────────────────────────
BOX_COLOUR  = (0, 200, 0)     # green box
TEXT_BG     = (0, 200, 0)     # green label background
TEXT_FG     = (0, 0, 0)       # black label text
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.55
FONT_THICK  = 1
BOX_THICK   = 2


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def load_ground_truth(test_dir: Path) -> dict[str, dict[int, list[list[int]]]]:
    """
    Load optional ground-truth JSON.

    Returns ``{video_name: {frame_idx: [[x1,y1,x2,y2], ...]}}``
    or an empty dict if the file does not exist.
    """
    gt_file = test_dir / "ground_truth.json"
    if not gt_file.exists():
        return {}

    with open(gt_file, encoding="utf-8") as fh:
        raw: dict[str, dict[str, list]] = json.load(fh)

    return {
        video: {int(fidx): boxes for fidx, boxes in frames.items()}
        for video, frames in raw.items()
    }


# ---------------------------------------------------------------------------
# mAP computation
# ---------------------------------------------------------------------------

def _iou(boxA: list[int], boxB: list[int]) -> float:
    """Compute Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-9)


def compute_ap50(
    all_detections: list[dict],
    gt_frames: dict[int, list[list[int]]],
) -> float:
    """
    Compute Average Precision at IoU=0.5 for a single video.

    Parameters
    ----------
    all_detections:
        List of ``{"frame_idx": int, "bbox": (x1,y1,x2,y2), "confidence": float}``
        across the entire video.
    gt_frames:
        ``{frame_idx: [[x1,y1,x2,y2], ...]}`` ground-truth boxes.

    Returns
    -------
    AP@0.5 as float in [0, 1].
    """
    if not gt_frames:
        return float("nan")

    # Total ground-truth instances
    total_gt = sum(len(boxes) for boxes in gt_frames.values())
    if total_gt == 0:
        return float("nan")

    # Sort detections by confidence descending
    sorted_dets = sorted(all_detections, key=lambda d: d["confidence"], reverse=True)

    # Track which GT boxes have been matched
    matched: dict[int, list[bool]] = {
        fidx: [False] * len(boxes)
        for fidx, boxes in gt_frames.items()
    }

    tp_list: list[int] = []
    fp_list: list[int] = []

    for det in sorted_dets:
        fidx = det["frame_idx"]
        pred_box = list(det["bbox"])
        gt_boxes = gt_frames.get(fidx, [])

        best_iou = 0.0
        best_j   = -1
        for j, gt_box in enumerate(gt_boxes):
            iou = _iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j   = j

        if best_iou >= 0.5 and best_j >= 0 and not matched[fidx][best_j]:
            matched[fidx][best_j] = True
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    # Precision-recall curve (cumulative)
    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall    = tp_cum / (total_gt + 1e-9)

    # Append sentinel points
    recall    = np.concatenate([[0.0], recall,    [recall[-1]]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    # Area under the PR curve (trapezoidal)
    ap = float(np.trapz(precision, recall))
    return round(ap, 4)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _annotate_frame(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    gt_boxes: list[list[int]] | None = None,
) -> np.ndarray:
    """
    Draw detection boxes (green) and optional GT boxes (blue) on a copy of *frame*.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Ground-truth boxes (blue, dashed appearance via double rect)
    if gt_boxes:
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 100, 0), 1)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]

        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOUR, BOX_THICK)

        label = f"person {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)

        # Clamp label so it stays within the frame
        lx = max(0, min(x1, w - tw - 2))
        ly = max(th + baseline + 2, y1)          # above box if room
        if ly - th - baseline - 2 < 0:
            ly = y2 + th + baseline + 2           # below box

        # Background rectangle
        cv2.rectangle(
            out,
            (lx, ly - th - baseline - 2),
            (lx + tw + 2, ly),
            TEXT_BG, -1,
        )
        cv2.putText(out, label, (lx + 1, ly - baseline - 1),
                    FONT, FONT_SCALE, TEXT_FG, FONT_THICK, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def _process_video(
    model: "BasePersonDetector",
    video_path: Path,
    annotated_dir: Path,
    gt_frames: dict[int, list[list[int]]] | None,
    frame_step: int,
    max_frames: int | None,
    time_limit: float | None,
) -> dict | None:
    """
    Run detection on a single video file.

    Annotation behavior:
    - Inference runs every ``frame_step`` frames.
    - Intermediate frames are still rendered with the last detections so
      boxes remain visible throughout the output clip.
    - If ``max_frames`` or ``time_limit`` is set, output video stops once
      the budget is exhausted; collected frames are still saved.

    Returns a per-video metrics dict, or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open video: {video_path.name}")
        return None

    fps_native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer (same codec as source when possible, fallback to mp4v)
    out_path = annotated_dir / f"{video_path.stem}_annotated.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps_native, (frame_w, frame_h))

    inference_times: list[float] = []
    detection_counts: list[int]  = []
    all_detections_for_map: list[dict] = []   # used for mAP if GT available

    frame_idx  = 0
    frames_processed = 0
    last_detections: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    print(f"  → {video_path.name}  ({total_frames_in_file} frames, "
          f"{frame_w}×{frame_h}, {fps_native:.1f} fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames is not None and frames_processed >= max_frames:
            # Stop once the configured frame budget is reached.
            break

        if time_limit is not None and (time.perf_counter() - t_start) >= time_limit:
            print(f"    [TIME LIMIT] Stopping after {time_limit:.0f}s "
                  f"({frames_processed} frames evaluated).")
            break

        if frame_idx % frame_step == 0:
            detections, ms = model.detect_timed(frame)
            inference_times.append(ms)
            detection_counts.append(len(detections))
            frames_processed += 1
            last_detections = detections

            for det in detections:
                all_detections_for_map.append({
                    "frame_idx":  frame_idx,
                    "bbox":       det["bbox"],
                    "confidence": det["confidence"],
                })

            gt_boxes_this_frame = (gt_frames or {}).get(frame_idx)
            annotated = _annotate_frame(frame, detections, gt_boxes_this_frame)
            writer.write(annotated)
        else:
            # Keep overlays visible between sampled inference frames.
            gt_boxes_this_frame = (gt_frames or {}).get(frame_idx)
            annotated = _annotate_frame(frame, last_detections, gt_boxes_this_frame)
            writer.write(annotated)

        frame_idx += 1

    cap.release()
    writer.release()

    if not inference_times:
        print(f"    [WARN] No frames processed for {video_path.name}")
        return None

    avg_ms   = sum(inference_times) / len(inference_times)
    avg_fps  = 1_000.0 / avg_ms if avg_ms > 0 else 0.0
    avg_dets = sum(detection_counts) / len(detection_counts)

    ap50 = (
        compute_ap50(all_detections_for_map, gt_frames)
        if gt_frames
        else None
    )

    result = {
        "video":                   video_path.name,
        "annotated_video":         f"annotated_videos/{out_path.name}",
        "native_fps":              round(fps_native, 2),
        "total_frames_in_file":    total_frames_in_file,
        "frames_processed":        frames_processed,
        "frame_step":              frame_step,
        "total_detections":        sum(detection_counts),
        "avg_detections_per_frame":round(avg_dets, 3),
        "max_detections_in_frame": max(detection_counts),
        "min_detections_in_frame": min(detection_counts),
        "avg_inference_ms":        round(avg_ms, 2),
        "avg_fps":                 round(avg_fps, 2),
        "min_inference_ms":        round(min(inference_times), 2),
        "max_inference_ms":        round(max(inference_times), 2),
    }
    if ap50 is not None:
        result["ap50"] = ap50

    print(f"    frames={frames_processed}  "
          f"avg_fps={avg_fps:.1f}  "
          f"avg_dets/frame={avg_dets:.2f}"
          + (f"  AP50={ap50:.4f}" if ap50 is not None else ""))

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_metrics(per_video: list[dict]) -> dict:
    """Compute overall averages across all processed videos."""
    n = len(per_video)

    def _mean(key: str) -> float:
        vals = [v[key] for v in per_video if key in v]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    agg: dict[str, Any] = {
        "total_videos":            n,
        "total_detections":        sum(v["total_detections"] for v in per_video),
        "avg_fps":                 _mean("avg_fps"),
        "avg_inference_ms":        _mean("avg_inference_ms"),
        "avg_detections_per_frame":_mean("avg_detections_per_frame"),
    }

    ap50_vals = [v["ap50"] for v in per_video if "ap50" in v and v["ap50"] == v["ap50"]]  # exclude NaN
    if ap50_vals:
        agg["mAP50"] = round(sum(ap50_vals) / len(ap50_vals), 4)

    return agg


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    model: "BasePersonDetector",
    test_dir: Path,
    results_base: Path,
    frame_step: int = 1,
    max_frames: int | None = None,
    time_limit: float | None = 60.0,
) -> dict:
    """
    Main evaluation entry point for one model.

    1. Discovers all videos under ``test_dir/videos/``.
    2. Loads optional ground-truth from ``test_dir/ground_truth.json``.
    3. Processes each video frame-by-frame (honouring *frame_step*).
    4. Writes annotated output videos.
    5. Saves ``detailed_results.json``, ``summary.json``, and
       ``per_video_results.csv`` to ``results_base/<model_name>/``.

    Returns the aggregated metrics dict (empty dict on failure).
    """
    out_dir = results_base / model.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = out_dir / "annotated_videos"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = test_dir / "videos"
    if not videos_dir.exists():
        print(f"[ERROR] {videos_dir} does not exist.")
        return {}

    video_files = sorted(
        p for p in videos_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_files:
        print(f"[ERROR] No video files found in {videos_dir}")
        return {}

    ground_truth_all = load_ground_truth(test_dir)
    has_gt = bool(ground_truth_all)

    print(f"\n{'='*60}")
    print(f"  Model : {model.model_name}")
    if has_gt:
        print("  Ground-truth loaded – AP50 will be computed.")
    else:
        print("  No ground_truth.json found – AP50 skipped.")
    print(f"{'='*60}")
    print(f"Processing {len(video_files)} video(s)…")

    per_video_results: list[dict] = []
    for video_path in video_files:
        gt_frames = ground_truth_all.get(video_path.name)
        result = _process_video(
            model=model,
            video_path=video_path,
            annotated_dir=annotated_dir,
            gt_frames=gt_frames,
            frame_step=frame_step,
            max_frames=max_frames,
            time_limit=time_limit,
        )
        if result:
            per_video_results.append(result)

    if not per_video_results:
        return {}

    metrics = _aggregate_metrics(per_video_results)
    metrics["model"]     = model.model_name
    metrics["timestamp"] = datetime.now().isoformat()

    print(f"\n  Avg FPS               : {metrics.get('avg_fps', 0):.1f}")
    print(f"  Avg inference (ms)    : {metrics.get('avg_inference_ms', 0):.1f}")
    print(f"  Avg detections/frame  : {metrics.get('avg_detections_per_frame', 0):.2f}")
    print(f"  Total detections      : {metrics.get('total_detections', 0)}")
    if "mAP50" in metrics:
        print(f"  mAP@50               : {metrics['mAP50']:.4f}")

    detailed = {
        "model":     model.model_name,
        "metrics":   metrics,
        "per_video": per_video_results,
    }
    _write_json(out_dir / "detailed_results.json", detailed)
    _write_json(out_dir / "summary.json", metrics)

    if per_video_results:
        _write_csv(out_dir / "per_video_results.csv", per_video_results)

    print(f"  Results saved → {out_dir}")
    return metrics


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: object) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Comparison report (called from main.py after all models finish)
# ---------------------------------------------------------------------------

def generate_comparison_report(all_metrics: list[dict], results_dir: Path) -> None:
    """Write a Markdown comparison table and a CSV summary."""
    has_map = any("mAP50" in m for m in all_metrics)

    # ── Markdown ──────────────────────────────────────────────────────────────
    header_cols = ["Model", "Avg FPS", "Avg ms/frame", "Avg dets/frame",
                   "Total dets", "Videos"]
    if has_map:
        header_cols.append("mAP@50")

    sep = "|" + "|".join("-" * (len(c) + 2) for c in header_cols) + "|"
    header = "| " + " | ".join(header_cols) + " |"

    rows_md = []
    for m in all_metrics:
        row = (
            f"| {m.get('model', 'N/A')} "
            f"| {m.get('avg_fps', 0):.1f} "
            f"| {m.get('avg_inference_ms', 0):.1f} "
            f"| {m.get('avg_detections_per_frame', 0):.2f} "
            f"| {m.get('total_detections', 0)} "
            f"| {m.get('total_videos', 0)} "
        )
        if has_map:
            row += f"| {m.get('mAP50', float('nan')):.4f} "
        row += "|"
        rows_md.append(row)

    lines = [
        "# Person Detection Model Evaluation Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Summary\n",
        header,
        sep,
        *rows_md,
        "\n## Notes\n",
        "- **Avg FPS** is the model's inference throughput (excludes video I/O).",
        "- **mAP@50** requires `ground_truth.json` in the test directory. "
          "Without it, this column is absent.",
        "- Annotated videos are saved under each model's `annotated_videos/` subfolder.",
    ]

    md_path = results_dir / "comparison_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_fields = [
        "model", "avg_fps", "avg_inference_ms", "avg_detections_per_frame",
        "total_detections", "total_videos", "timestamp",
    ]
    if has_map:
        csv_fields.append("mAP50")

    csv_path = results_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"\nComparison report → {md_path}")
    print(f"Comparison CSV    → {csv_path}")
