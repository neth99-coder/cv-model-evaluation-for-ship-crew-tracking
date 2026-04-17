#!/usr/bin/env python3
"""
Live object detection on a video source.

Supported models are aligned with the main evaluation script:
    - yolov8
    - yolov5
    - faster_rcnn
    - ssd_mobilenet

Examples:
    python live_object_detection.py --model yolov8
    python live_object_detection.py --model yolov5 --yolov5-size m
    python live_object_detection.py --model faster_rcnn --max-frames 120
    python live_object_detection.py --model ssd_mobilenet --save-out results/live_ssd.mp4

Controls (when display window is enabled):
    q  -> quit
"""

import argparse
import time
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).parent
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _auto_pick_source() -> Path:
    videos_dir = ROOT / "test" / "videos"
    if not videos_dir.exists():
        raise FileNotFoundError(f"Missing test video folder: {videos_dir}")

    videos = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No video files found in {videos_dir}")

    return videos[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live object detection on a video file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to input video. If omitted, first file under test/videos is used.",
    )
    parser.add_argument(
        "--model",
        choices=["yolov8", "yolov5", "faster_rcnn", "ssd_mobilenet"],
        default="yolov8",
        help="Model backend to run for live detection.",
    )
    parser.add_argument(
        "--yolov8-size",
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 size variant (used when --model yolov8).",
    )
    parser.add_argument(
        "--yolov5-size",
        default="s",
        choices=["s", "m", "l", "x"],
        help="YOLOv5 size variant (used when --model yolov5).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold in [0, 1].",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing this many frames.",
    )
    parser.add_argument(
        "--save-out",
        type=Path,
        default=None,
        help="Optional output path for annotated video.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable window display (useful for headless environments).",
    )
    return parser.parse_args()


def build_detector(args: argparse.Namespace):
    """Build detector instance from CLI selection."""
    if args.model == "yolov8":
        from models.yolov8 import YOLOv8Detector
        return YOLOv8Detector(
            model_size=args.yolov8_size,
            confidence_threshold=args.conf,
        )

    if args.model == "yolov5":
        from models.yolov5 import YOLOv5Detector
        return YOLOv5Detector(
            model_size=args.yolov5_size,
            confidence_threshold=args.conf,
        )

    if args.model == "faster_rcnn":
        from models.faster_rcnn import FasterRCNNDetector
        return FasterRCNNDetector(confidence_threshold=args.conf)

    if args.model == "ssd_mobilenet":
        from models.ssd_mobilenet import SSDMobileNetDetector
        return SSDMobileNetDetector(confidence_threshold=args.conf)

    raise ValueError(f"Unsupported model: {args.model}")


def annotate_frame(frame: Any, detections: list[dict[str, Any]]) -> Any:
    """Draw detector outputs on a frame."""
    h, w = frame.shape[:2]
    out = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = float(det["confidence"])
        label = f"person {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx = max(0, min(x1, w - tw - 4))
        ly = max(th + base + 4, y1)
        cv2.rectangle(out, (lx, ly - th - base - 4), (lx + tw + 4, ly), (0, 220, 0), -1)
        cv2.putText(
            out,
            label,
            (lx + 2, ly - base - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return out


def main() -> None:
    args = parse_args()

    source = args.source if args.source is not None else _auto_pick_source()
    if not source.exists():
        raise FileNotFoundError(f"Input video does not exist: {source}")

    print("=" * 60)
    print("Live Object Detection")
    print(f"Source   : {source}")
    print(f"Model    : {args.model}")
    if args.model == "yolov8":
        print(f"Variant  : {args.yolov8_size}")
    if args.model == "yolov5":
        print(f"Variant  : {args.yolov5_size}")
    print(f"Conf     : {args.conf}")
    print(f"Img size : {args.imgsz}")
    if args.max_frames is not None:
        print(f"Max frames: {args.max_frames}")
    if args.save_out is not None:
        print(f"Save out : {args.save_out}")
    print("=" * 60)

    detector = build_detector(args)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps_native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_out is not None:
        args.save_out.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save_out), fourcc, fps_native, (frame_w, frame_h))

    total_frames = 0
    total_detections = 0
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.max_frames is not None and total_frames >= args.max_frames:
            break

        infer_start = time.perf_counter()
        detections = detector.detect(frame)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        annotated = annotate_frame(frame, detections)
        frame_detections = len(detections)

        total_frames += 1
        total_detections += frame_detections

        elapsed = max(time.perf_counter() - t0, 1e-9)
        live_fps = total_frames / elapsed
        hud = f"FPS {live_fps:.1f} | Infer {infer_ms:.1f}ms | Objects {frame_detections}"
        cv2.rectangle(annotated, (8, 8), (430, 36), (30, 30, 30), -1)
        cv2.putText(annotated, hud, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

        if writer is not None:
            writer.write(annotated)

        if not args.no_show:
            cv2.imshow("Live Object Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    total_elapsed = max(time.perf_counter() - t0, 1e-9)
    avg_fps = total_frames / total_elapsed
    avg_objects = (total_detections / total_frames) if total_frames else 0.0

    print("\nRun complete")
    print(f"Frames processed : {total_frames}")
    print(f"Avg FPS          : {avg_fps:.2f}")
    print(f"Avg objects/frame: {avg_objects:.2f}")
    if writer is not None:
        print(f"Saved annotated video to: {args.save_out}")


if __name__ == "__main__":
    main()
