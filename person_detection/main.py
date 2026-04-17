#!/usr/bin/env python3
"""
Person Detection Model Evaluation
===================================
Evaluates four COCO-pretrained models on test videos:

  • YOLOv8   (Ultralytics)            – yolov8n/s/m/l/x
  • YOLOv5   (Ultralytics / PyTorch Hub) – yolov5s/m/l/x
  • Faster R-CNN (torchvision)        – ResNet-50 FPN v2
  • SSD MobileNetV3 (torchvision)     – SSDLite320

Test-data layout expected under ``person_detection/test/``:

    test/
    ├── videos/                    ← video files to evaluate
    │   ├── clip1.mp4
    │   └── clip2.avi
    └── ground_truth.json          ← optional; enables mAP@50 computation
                                     format: {"clip1.mp4": {"0": [[x1,y1,x2,y2]], ...}}

Usage examples:
    # evaluate all models (default frame step = 1, i.e. every frame)
    python main.py

    # evaluate specific models only
    python main.py --models yolov8 faster_rcnn

    # process every 5th frame (faster, lower accuracy)
    python main.py --frame-step 5

    # limit to the first 500 frames per video
    python main.py --max-frames 500

    # custom confidence threshold
    python main.py --confidence 0.4

    # custom model variant sizes
    python main.py --yolov8-size s --yolov5-size m

    # custom paths
    python main.py --test-dir /data/videos --results-dir /data/results
"""

import argparse
import os
import ssl
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def configure_ssl_certificates() -> None:
    """
    Configure CA bundle paths for urllib/requests/torch downloads.

    On some macOS Python setups, default certificate discovery fails for
    ``urllib`` even when internet access is available. Pointing all relevant
    env vars to certifi's bundle makes model downloads much more reliable.
    """
    try:
        import certifi
    except Exception:
        return

    ca_file = certifi.where()
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ.setdefault(key, ca_file)

    # Ensure urllib uses the same CA bundle.
    def _https_context(*args, **kwargs):
        return ssl.create_default_context(cafile=ca_file)

    ssl._create_default_https_context = _https_context

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Person detection model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["yolov8", "yolov5", "faster_rcnn", "ssd_mobilenet", "all"],
        default=["all"],
        metavar="MODEL",
        help=(
            "Models to evaluate: yolov8 yolov5 faster_rcnn ssd_mobilenet all "
            "(default: all)"
        ),
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=ROOT / "test",
        help="Root of the test dataset (default: ./test)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Output folder for results (default: ./results)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        metavar="N",
        help="Process every Nth frame (default: 1 = every frame). "
             "Higher values run faster at lower temporal resolution.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Maximum frames to evaluate per video (default: all frames).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Wall-clock time budget per model per video in seconds "
             "(default: 60). Evaluation stops cleanly when the limit is "
             "reached; use 0 to disable.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        metavar="THRESH",
        help="Detection confidence threshold, 0–1 (default: 0.5).",
    )
    parser.add_argument(
        "--yolov8-size",
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size (default: n = nano).",
    )
    parser.add_argument(
        "--yolov5-size",
        default="s",
        choices=["s", "m", "l", "x"],
        help="YOLOv5 model size (default: s = small).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_models(args: argparse.Namespace) -> list:
    selected = set(args.models)
    use_all  = "all" in selected
    models   = []
    debug_load_errors = os.getenv("PERSON_DET_DEBUG_LOAD_ERRORS", "0") == "1"

    def _try_load(label: str, factory):
        try:
            m = factory()
            print(f"  [LOADED] {label}")
            return m
        except Exception as exc:
            print(f"  [SKIP]   {label}: {exc}")
            if debug_load_errors:
                traceback.print_exc()
            return None

    if use_all or "yolov8" in selected:
        from models.yolov8 import YOLOv8Detector
        m = _try_load(
            f"YOLOv8{args.yolov8_size.upper()} (Ultralytics)",
            lambda: YOLOv8Detector(
                model_size=args.yolov8_size,
                confidence_threshold=args.confidence,
            ),
        )
        if m:
            models.append(m)

    if use_all or "yolov5" in selected:
        from models.yolov5 import YOLOv5Detector
        m = _try_load(
            f"YOLOv5{args.yolov5_size.upper()} (Ultralytics)",
            lambda: YOLOv5Detector(
                model_size=args.yolov5_size,
                confidence_threshold=args.confidence,
            ),
        )
        if m:
            models.append(m)

    if use_all or "faster_rcnn" in selected:
        from models.faster_rcnn import FasterRCNNDetector
        m = _try_load(
            "Faster R-CNN ResNet-50 FPN v2 (torchvision)",
            lambda: FasterRCNNDetector(confidence_threshold=args.confidence),
        )
        if m:
            models.append(m)

    if use_all or "ssd_mobilenet" in selected:
        from models.ssd_mobilenet import SSDMobileNetDetector
        m = _try_load(
            "SSDLite320 MobileNetV3 (torchvision)",
            lambda: SSDMobileNetDetector(confidence_threshold=args.confidence),
        )
        if m:
            models.append(m)

    return models


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    configure_ssl_certificates()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Person Detection Model Evaluation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"  Test dir    : {args.test_dir}")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Frame step  : {args.frame_step}")
    if args.max_frames:
        print(f"  Max frames  : {args.max_frames}")
    if args.time_limit > 0:
        print(f"  Time limit  : {args.time_limit:.0f}s per model/video")
    print(f"  Confidence  : {args.confidence}")
    print()

    print("Loading models…")
    models = build_models(args)
    if not models:
        print("\n[ERROR] No models were loaded. Exiting.")
        sys.exit(1)
    print(f"\n{len(models)} model(s) loaded.\n")

    from evaluate import run_evaluation, generate_comparison_report

    all_metrics: list[dict] = []
    for model in models:
        try:
            metrics = run_evaluation(
                model=model,
                test_dir=args.test_dir,
                results_base=args.results_dir,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                time_limit=args.time_limit if args.time_limit > 0 else None,
            )
            if metrics:
                all_metrics.append(metrics)
        except Exception as exc:
            print(f"\n[ERROR] {model.model_name} failed: {exc}")
            traceback.print_exc()

    if all_metrics:
        generate_comparison_report(all_metrics, args.results_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
