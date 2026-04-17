#!/usr/bin/env python3
"""Hybrid person tracking evaluation: BoxMOT when available, custom fallback."""

import argparse
import os
import ssl
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).parent


@dataclass(frozen=True)
class TrackerSpec:
    key: str
    label: str
    boxmot_tracker: str | None
    boxmot_reid: str | None
    custom_factory: Callable[..., object] | None


def _build_deepsort_custom(detector: str, conf: float, imgsz: int):
    from trackers.deepsort import DeepSORTTracker
    return DeepSORTTracker(detector_weights=detector, conf=conf, imgsz=imgsz)


def _build_bytetrack_custom(detector: str, conf: float, imgsz: int):
    from trackers.bytetrack import ByteTrackTracker
    return ByteTrackTracker(detector_weights=detector, conf=conf, imgsz=imgsz)


def _build_botsort_custom(detector: str, conf: float, imgsz: int):
    from trackers.botsort import BoTSORTTracker
    return BoTSORTTracker(detector_weights=detector, conf=conf, imgsz=imgsz)


TRACKER_SPECS: dict[str, TrackerSpec] = {
    # DeepSORT-equivalent in BoxMOT is StrongSORT.
    "deepsort": TrackerSpec("deepsort", "DeepSORT", "strongsort", "osnet_x0_25_msmt17", _build_deepsort_custom),
    "bytetrack": TrackerSpec("bytetrack", "ByteTrack", "bytetrack", None, _build_bytetrack_custom),
    "botsort": TrackerSpec("botsort", "BoT-SORT", "botsort", "osnet_x0_25_msmt17", _build_botsort_custom),
    # BoxMOT-only SOTA options
    "strongsort": TrackerSpec("strongsort", "StrongSORT", "strongsort", "osnet_x0_25_msmt17", None),
    "deepocsort": TrackerSpec("deepocsort", "DeepOCSORT", "deepocsort", "osnet_x0_25_msmt17", None),
    "ocsort": TrackerSpec("ocsort", "OCSORT", "ocsort", None, None),
    "hybridsort": TrackerSpec("hybridsort", "HybridSORT", "hybridsort", "osnet_x0_25_msmt17", None),
    "boosttrack": TrackerSpec("boosttrack", "BoostTrack", "boosttrack", "osnet_x0_25_msmt17", None),
    "sfsort": TrackerSpec("sfsort", "SFSORT", "sfsort", None, None),
}


def configure_ssl_certificates() -> None:
    try:
        import certifi
    except Exception:
        return

    ca_file = certifi.where()
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ.setdefault(key, ca_file)

    def _https_context(*args, **kwargs):
        return ssl.create_default_context(cafile=ca_file)

    ssl._create_default_https_context = _https_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid person tracking evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*TRACKER_SPECS.keys(), "all"],
        default=["all"],
        help="Trackers to evaluate (default: all).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "boxmot", "custom"],
        default="auto",
        help="Execution backend: auto prefers BoxMOT and falls back to custom.",
    )
    parser.add_argument("--test-dir", type=Path, default=ROOT / "test", help="Dataset root (default: ./test)")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results", help="Output directory (default: ./results)")

    parser.add_argument(
        "--detector",
        default="yolov8n.pt",
        help="Detector spec or weights. For BoxMOT this is detector name/spec; for custom this is YOLO weights path.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Detector inference size")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames per video")
    parser.add_argument("--time-limit", type=float, default=60.0, help="Seconds per model/video (0 disables)")

    return parser.parse_args()


def _boxmot_available() -> bool:
    try:
        import boxmot  # noqa: F401
        return True
    except Exception:
        return False


def _is_likely_yolo_detector(detector: str) -> bool:
    val = Path(detector).stem.lower()
    return val.startswith("yolov")


def main() -> None:
    args = parse_args()
    configure_ssl_certificates()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    from evaluate import run_boxmot_evaluation, run_custom_evaluation, generate_comparison_report

    selected_keys = list(TRACKER_SPECS.keys()) if "all" in set(args.models) else list(args.models)

    has_boxmot = _boxmot_available()

    print("=" * 60)
    print("  Person Tracking Evaluation (Hybrid)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"  Backend mode     : {args.backend}")
    print(f"  BoxMOT available : {has_boxmot}")
    print(f"  Detector         : {args.detector}")
    print(f"  Confidence       : {args.conf}")
    print(f"  Img size         : {args.imgsz}")
    if args.max_frames is not None:
        print(f"  Max frames/video : {args.max_frames}")
    if args.time_limit > 0:
        print(f"  Time limit/video : {args.time_limit:.0f}s")
    print()

    all_metrics: list[dict] = []

    for key in selected_keys:
        spec = TRACKER_SPECS[key]

        # Decide preferred backend.
        preferred = None
        if args.backend == "boxmot":
            preferred = "boxmot"
        elif args.backend == "custom":
            preferred = "custom"
        else:
            # auto: if BoxMOT has this tracker combination, use it first.
            if has_boxmot and spec.boxmot_tracker is not None:
                preferred = "boxmot"
            elif spec.custom_factory is not None:
                preferred = "custom"
            else:
                preferred = "boxmot"

            # In auto mode, non-YOLO detector often means BoxMOT is preferable.
            if preferred == "custom" and not _is_likely_yolo_detector(args.detector) and has_boxmot and spec.boxmot_tracker is not None:
                preferred = "boxmot"

        tried = []

        def _run_boxmot() -> dict:
            if not has_boxmot:
                raise RuntimeError("BoxMOT is not installed")
            if spec.boxmot_tracker is None:
                raise RuntimeError(f"{spec.label} has no BoxMOT mapping")
            return run_boxmot_evaluation(
                tracker_name=spec.boxmot_tracker,
                tracker_label=spec.label,
                detector=args.detector,
                reid_model=spec.boxmot_reid,
                test_dir=args.test_dir,
                results_base=args.results_dir,
                conf=args.conf,
                imgsz=args.imgsz,
                max_frames=args.max_frames,
                time_limit=args.time_limit if args.time_limit > 0 else None,
            )

        def _run_custom() -> dict:
            if spec.custom_factory is None:
                raise RuntimeError(f"{spec.label} has no custom adapter")
            tracker = spec.custom_factory(args.detector, args.conf, args.imgsz)
            return run_custom_evaluation(
                tracker=tracker,
                tracker_label=spec.label,
                test_dir=args.test_dir,
                results_base=args.results_dir,
                max_frames=args.max_frames,
                time_limit=args.time_limit if args.time_limit > 0 else None,
            )

        backends = [preferred]
        if args.backend == "auto":
            backends = [preferred, "custom" if preferred == "boxmot" else "boxmot"]

        metrics = {}
        for backend in backends:
            if backend in tried:
                continue
            tried.append(backend)
            try:
                print(f"\n[RUN] {spec.label} via {backend}")
                metrics = _run_boxmot() if backend == "boxmot" else _run_custom()
                if metrics:
                    all_metrics.append(metrics)
                break
            except Exception as exc:
                print(f"[WARN] {spec.label} via {backend} failed: {exc}")
                if os.getenv("PERSON_TRACK_DEBUG_LOAD_ERRORS", "0") == "1":
                    traceback.print_exc()

        if not metrics:
            print(f"[SKIP] {spec.label}: no working backend from {tried}")

    if all_metrics:
        generate_comparison_report(all_metrics, args.results_dir)
    else:
        print("\n[ERROR] No tracker evaluations succeeded.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
