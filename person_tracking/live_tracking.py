#!/usr/bin/env python3
"""Run live or file-based person tracking with pluggable detector/tracker backends."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from runtime import annotate_tracks, build_runtime
from trackers import expand_tracker_specs

ROOT = Path(__file__).parent
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _default_source() -> str:
	videos_dir = ROOT / "test" / "videos"
	if videos_dir.exists():
		candidates = sorted(path for path in videos_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
		if candidates:
			return str(candidates[0])
	return "0"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Live person tracking with pluggable detectors and trackers")
	parser.add_argument("--tracker", default="botsort", help="Tracker selection, e.g. botsort, deepsort, or boxmot:strongsort")
	parser.add_argument("--detector", default="yolov8n", help="Detector selection, e.g. yolov8n, yolov5s, faster_rcnn")
	parser.add_argument("--source", default=_default_source(), help="Video path, URL, or camera index")
	parser.add_argument("--save-out", type=Path, default=None, help="Optional output path for annotated video")
	parser.add_argument("--no-show", action="store_true", help="Disable the preview window.")
	parser.add_argument("--max-frames", type=int, default=None)
	parser.add_argument("--confidence", type=float, default=0.25)
	parser.add_argument("--reid", default=None)
	parser.add_argument("--device", default="cpu")
	parser.add_argument("--half", action="store_true")
	parser.add_argument(
		"--deepsort-embedder",
		default="mobilenet",
		choices=[
			"mobilenet",
			"torchreid",
			"clip_RN50",
			"clip_RN101",
			"clip_RN50x4",
			"clip_RN50x16",
			"clip_ViT-B/32",
			"clip_ViT-B/16",
		],
	)
	parser.add_argument("--deepsort-embedder-model", default=None)
	parser.add_argument("--deepsort-embedder-weights", default=None)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	tracker_spec = expand_tracker_specs([args.tracker])[0]
	runtime = build_runtime(
		detector_spec=args.detector,
		tracker_spec=tracker_spec,
		confidence_threshold=args.confidence,
		reid_model=args.reid,
		device=args.device,
		half=args.half,
		deepsort_embedder=args.deepsort_embedder,
		deepsort_embedder_model=args.deepsort_embedder_model,
		deepsort_embedder_weights=args.deepsort_embedder_weights,
	)
	runtime.reset()

	source: str | int = int(args.source) if str(args.source).isdigit() else args.source
	cap = cv2.VideoCapture(source)
	if not cap.isOpened():
		raise SystemExit(f"Failed to open source: {args.source}")

	writer = None
	if args.save_out is not None:
		args.save_out.parent.mkdir(parents=True, exist_ok=True)
		fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
		frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		writer = cv2.VideoWriter(
			str(args.save_out),
			cv2.VideoWriter_fourcc(*"mp4v"),
			fps,
			(frame_w, frame_h),
		)

	frame_count = 0
	total_ms = 0.0
	started_at = time.perf_counter()

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			if args.max_frames is not None and frame_count >= args.max_frames:
				break

			result = runtime.process_frame(frame)
			frame_count += 1
			total_ms += result.total_ms

			avg_fps = 1_000.0 / (total_ms / frame_count) if frame_count else 0.0
			annotated = annotate_tracks(
				frame,
				result.tracks,
				header_lines=(
					f"Detector: {runtime.detector_name}",
					f"Tracker: {runtime.tracker_name} ({runtime.backend})",
					f"FPS: {avg_fps:.2f}",
				),
			)

			if writer is not None:
				writer.write(annotated)

			if not args.no_show:
				cv2.imshow("Live Tracking", annotated)
				key = cv2.waitKey(1) & 0xFF
				if key in (ord("q"), 27):
					break
	finally:
		cap.release()
		if writer is not None:
			writer.release()
		cv2.destroyAllWindows()

	elapsed = time.perf_counter() - started_at
	avg_fps = 1_000.0 / (total_ms / frame_count) if frame_count else 0.0
	print(f"Processed {frame_count} frame(s) in {elapsed:.2f}s. Average tracking FPS: {avg_fps:.2f}")


if __name__ == "__main__":
	main()
