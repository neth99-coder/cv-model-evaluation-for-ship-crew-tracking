#!/usr/bin/env python3
"""
evaluate.py — Standalone evaluation script for Multi-Tracker Person Tracking System

Usage:
    python evaluate.py                            # Use default test/test.mp4
    python evaluate.py --video path/to/video.mp4  # Custom video
    python evaluate.py --output results.json       # Custom output file
    python evaluate.py --frameworks boxmot deepsort # Filter frameworks
    python evaluate.py --max-frames 500            # Limit frames per combo

Output:
    JSON file with per-combination metrics + summary table printed to stdout.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Add backend to path ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from trackers.boxmot_tracker import BoxMOTTracker
from trackers.fairmot_tracker import FairMOTTracker
from trackers.deepsort_tracker import DeepSORTTracker
from utils.metrics import TrackingMetrics


# ── Combinations to evaluate ─────────────────────────────────────────────────
ALL_COMBINATIONS = [
    # (framework, tracker, detector, reid)
    ("boxmot",   "bytetrack",   "yolov8n",  None),
    ("boxmot",   "bytetrack",   "yolov8s",  None),
    ("boxmot",   "botsort",     "yolov8n",  "osnet_x0_25"),
    ("boxmot",   "botsort",     "yolov8s",  "osnet_x1_0"),
    ("boxmot",   "ocsort",      "yolov8n",  None),
    ("boxmot",   "deepocsort",  "yolov8n",  "osnet_x0_25"),
    ("boxmot",   "strongsort",  "yolov8s",  "osnet_x1_0"),
    ("fairmot",  "fairmot",     "dla34",    None),
    ("fairmot",  "fairmot",     "resnet50", None),
    ("deepsort", "deepsort",    "yolov8n",  "osnet_x0_25"),
    ("deepsort", "deepsort",    "yolov8s",  "osnet_x1_0"),
    ("deepsort", "deepsort",    "yolov8n",  "resnet50"),
]


def build_tracker(framework, tracker_name, detector, reid, conf=0.4):
    if framework == "boxmot":
        return BoxMOTTracker(tracker_name, detector, reid, conf)
    elif framework == "fairmot":
        return FairMOTTracker(detector, conf)
    elif framework == "deepsort":
        return DeepSORTTracker(detector, reid, conf)
    raise ValueError(f"Unknown framework: {framework}")


def draw_tracks(frame, tracks):
    for t in tracks:
        x1, y1, x2, y2, tid = map(int, t[:5])
        np.random.seed(tid * 13)
        color = tuple(np.random.randint(80, 255, 3).tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{tid}", (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


def evaluate_combination(video_path, framework, tracker_name, detector, reid,
                          max_frames, save_video, output_dir, conf=0.4):
    label = f"{framework}/{tracker_name}/{detector}" + (f"/{reid}" if reid else "")
    print(f"\n{'─'*60}")
    print(f"  Evaluating: {label}")
    print(f"{'─'*60}")

    tracker = build_tracker(framework, tracker_name, detector, reid, conf)
    cap = cv2.VideoCapture(str(video_path))

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_video:
        out_name = f"{framework}_{tracker_name}_{detector}_{reid or 'noid'}.mp4"
        out_path = Path(output_dir) / out_name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (w, h))

    metrics = TrackingMetrics()
    frame_idx = 0
    limit = min(max_frames, total_frames) if max_frames else total_frames

    while frame_idx < limit:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        tracks = tracker.update(frame)
        elapsed = time.time() - t0
        metrics.update(tracks, elapsed, frame_idx)

        if writer is not None:
            annotated = draw_tracks(frame.copy(), tracks)
            writer.write(annotated)

        frame_idx += 1
        if frame_idx % 100 == 0:
            pct = frame_idx / limit * 100
            summary = metrics.summary()
            print(f"  Frame {frame_idx}/{limit} ({pct:.0f}%) | "
                  f"FPS: {summary.get('avg_fps', 0):.1f} | "
                  f"Tracks: {summary.get('avg_tracks_per_frame', 0):.1f}")

    cap.release()
    if writer:
        writer.release()

    result = {
        "framework": framework,
        "tracker": tracker_name,
        "detector": detector,
        "reid": reid,
        "frames_evaluated": frame_idx,
        "status": "ok",
        **metrics.summary(),
    }

    # Print per-combo summary
    s = metrics.summary()
    print(f"\n  ✓ Done — {frame_idx} frames")
    print(f"    Avg FPS:          {s.get('avg_fps', 0):.2f}")
    print(f"    Unique tracks:    {s.get('unique_tracks', 0)}")
    print(f"    Avg tracks/frame: {s.get('avg_tracks_per_frame', 0):.2f}")
    print(f"    Avg lifetime:     {s.get('avg_track_lifetime_frames', 0):.1f} frames")

    return result


def print_summary_table(results):
    print("\n" + "═" * 90)
    print("  EVALUATION SUMMARY")
    print("═" * 90)
    header = f"{'Framework':<10} {'Tracker':<12} {'Detector':<10} {'RE-ID':<15} " \
             f"{'Avg FPS':>8} {'Tracks':>7} {'Avg/fr':>7}"
    print(header)
    print("─" * 90)
    for r in results:
        if r.get("status") == "error":
            print(f"  ERROR: {r.get('framework')}/{r.get('tracker')} — {r.get('error', 'unknown')}")
            continue
        print(
            f"  {r['framework']:<10} {r['tracker']:<12} {r['detector']:<10} "
            f"{(r['reid'] or 'None'):<15} "
            f"{r.get('avg_fps', 0):>8.2f} "
            f"{r.get('unique_tracks', 0):>7} "
            f"{r.get('avg_tracks_per_frame', 0):>7.2f}"
        )
    print("═" * 90)

    # Best combo by FPS
    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        best_fps = max(ok, key=lambda r: r.get("avg_fps", 0))
        print(f"\n  🏆 Best FPS:    {best_fps['framework']}/{best_fps['tracker']}/{best_fps['detector']}"
              f" — {best_fps.get('avg_fps', 0):.2f} FPS")
        best_tracks = max(ok, key=lambda r: r.get("unique_tracks", 0))
        print(f"  🔍 Most tracks: {best_tracks['framework']}/{best_tracks['tracker']}/{best_tracks['detector']}"
              f" — {best_tracks.get('unique_tracks', 0)} unique IDs")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-tracker person tracking combinations."
    )
    parser.add_argument("--video", type=str, default="test/test.mp4",
                        help="Path to test video (default: test/test.mp4)")
    parser.add_argument("--output", type=str, default="outputs/evaluation_results.json",
                        help="Output JSON file path")
    parser.add_argument("--frameworks", nargs="+", default=None,
                        choices=["boxmot", "fairmot", "deepsort"],
                        help="Filter to specific frameworks")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Max frames to evaluate per combination (0=all)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Detection confidence threshold")
    parser.add_argument("--save-videos", action="store_true",
                        help="Save annotated output videos for each combination")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save output videos")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    combos = ALL_COMBINATIONS
    if args.frameworks:
        combos = [c for c in combos if c[0] in args.frameworks]

    max_frames = args.max_frames if args.max_frames > 0 else None

    print(f"\n{'═'*60}")
    print("  Multi-Tracker Person Tracking Evaluator")
    print(f"{'═'*60}")
    print(f"  Video:       {video_path}")
    print(f"  Combos:      {len(combos)}")
    print(f"  Max frames:  {max_frames or 'all'}")
    print(f"  Save videos: {args.save_videos}")
    print(f"  Output:      {args.output}")

    results = []
    for framework, tracker, detector, reid in combos:
        try:
            r = evaluate_combination(
                video_path, framework, tracker, detector, reid,
                max_frames, args.save_videos, args.output_dir, args.conf
            )
            results.append(r)
        except Exception as e:
            print(f"[ERROR] {framework}/{tracker}: {e}")
            results.append({
                "framework": framework, "tracker": tracker,
                "detector": detector, "reid": reid,
                "status": "error", "error": str(e),
            })

    # Save JSON
    with open(args.output, "w") as f:
        json.dump({"combinations": results, "total": len(results)}, f, indent=2)
    print(f"\n[✓] Results saved to: {args.output}")

    print_summary_table(results)


if __name__ == "__main__":
    main()
