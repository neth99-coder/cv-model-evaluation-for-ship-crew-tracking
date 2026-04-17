#!/usr/bin/env python3
"""Live person tracking with hybrid backend support (BoxMOT + custom)."""

import argparse
import os
import shutil
import ssl
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2

ROOT = Path(__file__).parent
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


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
    "deepsort": TrackerSpec("deepsort", "DeepSORT", "strongsort", "osnet_x0_25_msmt17", _build_deepsort_custom),
    "bytetrack": TrackerSpec("bytetrack", "ByteTrack", "bytetrack", None, _build_bytetrack_custom),
    "botsort": TrackerSpec("botsort", "BoT-SORT", "botsort", "osnet_x0_25_msmt17", _build_botsort_custom),
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


def _boxmot_available() -> bool:
    try:
        import boxmot  # noqa: F401
        return True
    except Exception:
        return False


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


def _parse_source(raw: str | None) -> str | int:
    if raw is None:
        return str(_auto_pick_source())
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    return raw


def _normalize_boxmot_detector_name(detector: str) -> str:
    stem = Path(detector).stem.lower()
    if stem.startswith("yolov"):
        return stem
    return detector


def _track_color(track_id: int) -> tuple[int, int, int]:
    return (
        (37 * track_id) % 255,
        (17 * track_id + 80) % 255,
        (29 * track_id + 160) % 255,
    )


def _annotate_custom_frame(frame: Any, tracks: list[dict[str, Any]]) -> Any:
    out = frame.copy()
    h, w = out.shape[:2]

    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        tid = int(t["track_id"])
        conf = float(t.get("confidence", 0.0))

        color = _track_color(tid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"ID {tid} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        lx = max(0, min(x1, w - tw - 4))
        ly = max(th + base + 4, y1)
        cv2.rectangle(out, (lx, ly - th - base - 4), (lx + tw + 4, ly), color, -1)
        cv2.putText(out, label, (lx + 2, ly - base - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

    return out


def _trim_video(video_path: Path, out_path: Path, max_frames_to_keep: int) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    kept = 0
    while kept < max_frames_to_keep:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        kept += 1

    cap.release()
    writer.release()
    return out_path


def run_custom_live(
    spec: TrackerSpec,
    source: str | int,
    detector: str,
    conf: float,
    imgsz: int,
    max_frames: int | None,
    time_limit: float | None,
    show: bool,
    save_out: Path | None,
) -> None:
    if spec.custom_factory is None:
        raise RuntimeError(f"{spec.label} has no custom backend")

    tracker = spec.custom_factory(detector, conf, imgsz)
    tracker.reset()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_out is not None:
        save_out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(save_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    frames = 0
    total_tracks = 0
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames is not None and max_frames > 0 and frames >= max_frames:
            break
        if time_limit is not None and time_limit > 0 and (time.perf_counter() - t0) >= time_limit:
            break

        infer_start = time.perf_counter()
        tracks = tracker.update(frame)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        annotated = _annotate_custom_frame(frame, tracks)
        total_tracks += len(tracks)
        frames += 1

        elapsed = max(time.perf_counter() - t0, 1e-9)
        live_fps = frames / elapsed
        hud = f"{spec.label} | FPS {live_fps:.1f} | Infer {infer_ms:.1f}ms | Tracks {len(tracks)}"
        cv2.rectangle(annotated, (8, 8), (540, 36), (30, 30, 30), -1)
        cv2.putText(annotated, hud, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

        if writer is not None:
            writer.write(annotated)

        if show:
            cv2.imshow("Live Person Tracking", annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - t0, 1e-9)
    avg_fps = frames / elapsed if elapsed > 0 else 0.0
    avg_tracks = (total_tracks / frames) if frames > 0 else 0.0
    print(f"\nRun complete ({spec.label}/custom)")
    print(f"Frames processed : {frames}")
    print(f"Avg FPS          : {avg_fps:.2f}")
    print(f"Avg tracks/frame : {avg_tracks:.2f}")
    if save_out is not None:
        print(f"Saved annotated video to: {save_out}")


def run_boxmot_live(
    spec: TrackerSpec,
    source: str | int,
    detector: str,
    conf: float,
    imgsz: int,
    max_frames: int | None,
    time_limit: float | None,
    show: bool,
    save_out: Path | None,
) -> None:
    if spec.boxmot_tracker is None:
        raise RuntimeError(f"{spec.label} has no BoxMOT backend")
    if not _boxmot_available():
        raise RuntimeError("BoxMOT is not installed")

    from boxmot import Boxmot

    project_dir = ROOT / "runs_live"
    project_dir.mkdir(parents=True, exist_ok=True)

    source_for_run = source
    tmp_input = None
    if isinstance(source, str) and Path(source).exists() and (max_frames is not None or time_limit is not None):
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        frame_cap = None
        if max_frames is not None and max_frames > 0:
            frame_cap = max_frames
        if time_limit is not None and time_limit > 0:
            by_time = max(1, int(round(fps * time_limit)))
            frame_cap = by_time if frame_cap is None else min(frame_cap, by_time)

        if frame_cap is not None and frame_cap < total:
            td = tempfile.mkdtemp(prefix="person_tracking_live_boxmot_")
            tmp_input = Path(td) / f"{Path(source).stem}_short.mp4"
            _trim_video(Path(source), tmp_input, frame_cap)
            source_for_run = str(tmp_input)

    runner = Boxmot(
        detector=_normalize_boxmot_detector_name(detector),
        reid=spec.boxmot_reid,
        tracker=spec.boxmot_tracker,
        classes=[0],
        project=project_dir,
    )

    run = runner.track(
        source=source_for_run,
        imgsz=imgsz,
        conf=conf,
        save=save_out is not None,
        save_txt=False,
        show=show,
        verbose=False,
        device="cpu",
    )

    if save_out is not None and run.video_path:
        saved_path = Path(run.video_path)
        if saved_path.exists():
            save_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(saved_path, save_out)
            print(f"Saved annotated video to: {save_out}")

    summary = run.summary if isinstance(run.summary, dict) else {}
    timings = run.timings if isinstance(run.timings, dict) else {}

    print(f"\nRun complete ({spec.label}/boxmot)")
    print(f"Frames processed : {summary.get('frames', 'n/a')}")
    print(f"Avg FPS          : {timings.get('fps', 0):.2f}")
    print(f"Unique tracks    : {summary.get('unique_tracks', 'n/a')}")

    if tmp_input is not None:
        shutil.rmtree(tmp_input.parent, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live person tracking with auto BoxMOT/custom backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=list(TRACKER_SPECS.keys()), default="bytetrack", help="Tracker model key")
    parser.add_argument("--backend", choices=["auto", "boxmot", "custom"], default="auto", help="Execution backend")
    parser.add_argument("--source", default=None, help="Video path or webcam index (e.g. 0). If omitted, first test video is used")
    parser.add_argument("--detector", default="yolov8n.pt", help="Detector spec/weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--time-limit", type=float, default=None, help="Wall-clock seconds limit (for file inputs this trims BoxMOT input clip)")
    parser.add_argument("--save-out", type=Path, default=None, help="Optional path for annotated output video")
    parser.add_argument("--no-show", action="store_true", help="Disable on-screen window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_ssl_certificates()

    spec = TRACKER_SPECS[args.model]
    source = _parse_source(args.source)
    show = not args.no_show
    has_boxmot = _boxmot_available()

    preferred = args.backend
    if preferred == "auto":
        if has_boxmot and spec.boxmot_tracker is not None:
            preferred = "boxmot"
        else:
            preferred = "custom"

    backends = [preferred]
    if args.backend == "auto":
        backends = [preferred, "custom" if preferred == "boxmot" else "boxmot"]

    print("=" * 60)
    print("Live Person Tracking")
    print(f"Source           : {source}")
    print(f"Model            : {spec.label} ({spec.key})")
    print(f"Backend mode     : {args.backend}")
    print(f"BoxMOT available : {has_boxmot}")
    print(f"Detector         : {args.detector}")
    print(f"Conf             : {args.conf}")
    print(f"Img size         : {args.imgsz}")
    if args.max_frames is not None:
        print(f"Max frames       : {args.max_frames}")
    if args.time_limit is not None:
        print(f"Time limit (s)   : {args.time_limit}")
    if args.save_out is not None:
        print(f"Save out         : {args.save_out}")
    print(f"Show window      : {show}")
    print("=" * 60)

    last_error = None
    for backend in backends:
        try:
            if backend == "boxmot":
                run_boxmot_live(
                    spec=spec,
                    source=source,
                    detector=args.detector,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    max_frames=args.max_frames,
                    time_limit=args.time_limit,
                    show=show,
                    save_out=args.save_out,
                )
            else:
                run_custom_live(
                    spec=spec,
                    source=source,
                    detector=args.detector,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    max_frames=args.max_frames,
                    time_limit=args.time_limit,
                    show=show,
                    save_out=args.save_out,
                )
            return
        except Exception as exc:
            last_error = exc
            print(f"[WARN] {spec.label} via {backend} failed: {exc}")

    raise RuntimeError(f"No working backend for {spec.label}. Last error: {last_error}")


if __name__ == "__main__":
    main()
