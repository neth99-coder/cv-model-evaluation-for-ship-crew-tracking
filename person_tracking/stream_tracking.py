#!/usr/bin/env python3
"""MJPEG realtime tracking stream for GUI integration.

This stream endpoint is intentionally custom-tracker based because it emits
frame-by-frame annotated output for browser display.
"""

import argparse
import os
import ssl
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2

ROOT = Path(__file__).parent


@dataclass(frozen=True)
class TrackerSpec:
    key: str
    label: str
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
    "deepsort": TrackerSpec("deepsort", "DeepSORT", _build_deepsort_custom),
    "bytetrack": TrackerSpec("bytetrack", "ByteTrack", _build_bytetrack_custom),
    "botsort": TrackerSpec("botsort", "BoT-SORT", _build_botsort_custom),
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


def _parse_source(raw: str) -> str | int:
    s = str(raw).strip()
    if s.isdigit():
        return int(s)
    return s


def _track_color(track_id: int) -> tuple[int, int, int]:
    return (
        (37 * track_id) % 255,
        (17 * track_id + 80) % 255,
        (29 * track_id + 160) % 255,
    )


def _annotate_custom_frame(frame: Any, tracks: list[dict[str, Any]], label: str, fps: float) -> Any:
    out = frame.copy()
    h, w = out.shape[:2]

    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        tid = int(t["track_id"])
        conf = float(t.get("confidence", 0.0))

        color = _track_color(tid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        text = f"ID {tid} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        lx = max(0, min(x1, w - tw - 4))
        ly = max(th + base + 4, y1)
        cv2.rectangle(out, (lx, ly - th - base - 4), (lx + tw + 4, ly), color, -1)
        cv2.putText(out, text, (lx + 2, ly - base - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

    hud = f"{label} | FPS {fps:.1f}"
    cv2.rectangle(out, (8, 8), (320, 36), (30, 30, 30), -1)
    cv2.putText(out, hud, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime MJPEG tracking stream")
    parser.add_argument("--model", choices=list(TRACKER_SPECS.keys()), required=True)
    parser.add_argument("--source", required=True, help="Camera index or video/stream URL")
    parser.add_argument("--detector", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_ssl_certificates()

    spec = TRACKER_SPECS[args.model]
    if spec.custom_factory is None:
        raise RuntimeError(f"No custom tracker for {spec.key}")

    tracker = spec.custom_factory(args.detector, args.conf, args.imgsz)
    tracker.reset()

    cap = cv2.VideoCapture(_parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    frames = 0
    t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.max_frames and args.max_frames > 0 and frames >= args.max_frames:
                break
            if args.time_limit and args.time_limit > 0 and (time.perf_counter() - t0) >= args.time_limit:
                break

            tracks = tracker.update(frame)
            elapsed = max(time.perf_counter() - t0, 1e-9)
            fps = (frames + 1) / elapsed
            annotated = _annotate_custom_frame(frame, tracks, spec.label, fps)

            ok_jpg, encoded = cv2.imencode(".jpg", annotated)
            if not ok_jpg:
                continue

            jpg = encoded.tobytes()
            sys.stdout.buffer.write(b"--frame\r\n")
            sys.stdout.buffer.write(b"Content-Type: image/jpeg\r\n")
            sys.stdout.buffer.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
            sys.stdout.buffer.write(jpg)
            sys.stdout.buffer.write(b"\r\n")
            sys.stdout.flush()

            frames += 1

    finally:
        cap.release()


if __name__ == "__main__":
    main()
