#!/usr/bin/env python3
"""Evaluate pluggable detector and tracker combinations on test videos."""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from detectors import DEFAULT_DETECTORS, SUPPORTED_DETECTORS, expand_detector_specs
from runtime import annotate_tracks, append_mot_rows, build_runtime
from trackers import BOXMOT_TRACKERS, CUSTOM_TRACKERS, expand_tracker_specs

ROOT = Path(__file__).parent
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _process_video(
	runtime: Any,
	video_path: Path,
	out_dir: Path,
	frame_step: int,
	max_frames: int | None,
	time_limit: float | None,
) -> dict[str, Any] | None:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		print(f"  [WARN] Cannot open video: {video_path.name}")
		return None

	runtime.reset()

	fps_native = cap.get(cv2.CAP_PROP_FPS) or 25.0
	total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	annotated_dir = out_dir / "annotated_videos"
	annotated_dir.mkdir(parents=True, exist_ok=True)
	tracks_dir = out_dir / "tracks_txt"
	tracks_dir.mkdir(parents=True, exist_ok=True)

	out_video_path = annotated_dir / f"{video_path.stem}_tracked.mp4"
	out_tracks_path = tracks_dir / f"{video_path.stem}.txt"
	writer = cv2.VideoWriter(
		str(out_video_path),
		cv2.VideoWriter_fourcc(*"mp4v"),
		fps_native,
		(frame_w, frame_h),
	)

	detector_times: list[float] = []
	tracker_times: list[float] = []
	total_times: list[float] = []
	detection_counts: list[int] = []
	track_counts: list[int] = []
	unique_track_ids: set[int] = set()

	frame_idx = 0
	frames_processed = 0
	last_tracks: list[dict[str, Any]] = []
	started_at = time.perf_counter()

	print(
		f"  → {video_path.name} ({total_frames_in_file} frames, {frame_w}x{frame_h}, {fps_native:.1f} fps)"
	)

	with open(out_tracks_path, "w", encoding="utf-8") as handle:
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			if max_frames is not None and frames_processed >= max_frames:
				break

			if time_limit is not None and (time.perf_counter() - started_at) >= time_limit:
				print(f"    [TIME LIMIT] Stopping after {time_limit:.0f}s")
				break

			if frame_idx % frame_step == 0:
				result = runtime.process_frame(frame)
				frames_processed += 1
				detector_times.append(result.detector_ms)
				tracker_times.append(result.tracker_ms)
				total_times.append(result.total_ms)
				detection_counts.append(len(result.detections))
				track_counts.append(len(result.tracks))
				unique_track_ids.update(int(track["track_id"]) for track in result.tracks)
				last_tracks = result.tracks
				append_mot_rows(handle, frame_idx + 1, result.tracks)

			annotated = annotate_tracks(
				frame,
				last_tracks,
				header_lines=(
					f"Detector: {runtime.detector_name}",
					f"Tracker: {runtime.tracker_name} ({runtime.backend})",
					f"Tracks: {len(last_tracks)}",
				),
			)
			writer.write(annotated)
			frame_idx += 1

	cap.release()
	writer.release()

	if not total_times:
		print(f"    [WARN] No frames processed for {video_path.name}")
		return None

	avg_total_ms = sum(total_times) / len(total_times)
	avg_fps = 1_000.0 / avg_total_ms if avg_total_ms > 0 else 0.0
	result = {
		"video": video_path.name,
		"input_clip": video_path.name,
		"native_fps": round(fps_native, 2),
		"total_frames_in_file": total_frames_in_file,
		"frames_processed": frames_processed,
		"frame_step": frame_step,
		"avg_detection_ms": round(sum(detector_times) / len(detector_times), 3),
		"avg_tracking_ms": round(sum(tracker_times) / len(tracker_times), 3),
		"avg_inference_ms": round(avg_total_ms, 3),
		"avg_fps": round(avg_fps, 3),
		"detections": sum(detection_counts),
		"tracks": sum(track_counts),
		"avg_detections": round(sum(detection_counts) / len(detection_counts), 3),
		"avg_tracks": round(sum(track_counts) / len(track_counts), 3),
		"unique_tracks": len(unique_track_ids),
		"annotated_video": f"annotated_videos/{out_video_path.name}",
		"tracks_txt": f"tracks_txt/{out_tracks_path.name}",
		"backend": runtime.backend,
	}
	print(
		f"    frames={frames_processed} avg_fps={avg_fps:.2f} unique_tracks={len(unique_track_ids)} avg_tracks={result['avg_tracks']:.2f}"
	)
	return result


def _aggregate_metrics(runtime: Any, per_video: list[dict[str, Any]]) -> dict[str, Any]:
	def _mean(key: str) -> float:
		values = [row[key] for row in per_video if key in row]
		return round(sum(values) / len(values), 3) if values else 0.0

	return {
		"model": runtime.tracker_name,
		"backend": runtime.backend,
		"detector": runtime.detector_name,
		"tracker_backend": runtime.tracker_backend,
		"reid": runtime.reid_name,
		"timestamp": datetime.now().isoformat(),
		"total_videos": len(per_video),
		"total_frames_processed": sum(row["frames_processed"] for row in per_video),
		"avg_detection_ms": _mean("avg_detection_ms"),
		"avg_tracking_ms": _mean("avg_tracking_ms"),
		"avg_inference_ms": _mean("avg_inference_ms"),
		"avg_fps": _mean("avg_fps"),
		"avg_unique_tracks": _mean("unique_tracks"),
		"avg_detections": _mean("avg_detections"),
		"avg_tracks": _mean("avg_tracks"),
		"total_detections": sum(row["detections"] for row in per_video),
		"total_tracks": sum(row["tracks"] for row in per_video),
	}


def run_evaluation(
	runtime: Any,
	test_dir: Path,
	results_base: Path,
	frame_step: int = 1,
	max_frames: int | None = None,
	time_limit: float | None = 60.0,
) -> dict[str, Any]:
	out_dir = results_base / runtime.run_name
	out_dir.mkdir(parents=True, exist_ok=True)

	videos_dir = test_dir / "videos"
	if not videos_dir.exists():
		print(f"[ERROR] {videos_dir} does not exist.")
		return {}

	video_files = sorted(path for path in videos_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
	if not video_files:
		print(f"[ERROR] No video files found in {videos_dir}")
		return {}

	print(f"\n{'=' * 72}")
	print(f"  Tracker : {runtime.tracker_name}")
	print(f"  Detector: {runtime.detector_name}")
	print(f"  Backend : {runtime.backend}")
	if runtime.reid_name:
		print(f"  ReID    : {runtime.reid_name}")
	print(f"{'=' * 72}")

	per_video_results: list[dict[str, Any]] = []
	for video_path in video_files:
		result = _process_video(
			runtime=runtime,
			video_path=video_path,
			out_dir=out_dir,
			frame_step=frame_step,
			max_frames=max_frames,
			time_limit=time_limit,
		)
		if result:
			per_video_results.append(result)

	if not per_video_results:
		return {}

	metrics = _aggregate_metrics(runtime, per_video_results)
	detailed = {
		"model": runtime.tracker_name,
		"tracker_backend": runtime.tracker_backend,
		"detector": runtime.detector_name,
		"reid": runtime.reid_name,
		"metrics": metrics,
		"per_video": per_video_results,
	}

	_write_json(out_dir / "detailed_results.json", detailed)
	_write_json(out_dir / "summary.json", metrics)
	_write_csv(out_dir / "per_video_results.csv", per_video_results)

	print(f"  Results saved -> {out_dir}")
	return metrics


def generate_comparison_report(all_metrics: list[dict[str, Any]], results_dir: Path) -> None:
	if not all_metrics:
		return

	header_cols = [
		"Tracker",
		"Detector",
		"ReID",
		"Backend",
		"Avg FPS",
		"Avg ms/frame",
		"Avg unique tracks",
	]
	header = "| " + " | ".join(header_cols) + " |"
	sep = "|" + "|".join("-" * (len(col) + 2) for col in header_cols) + "|"

	rows_md = []
	for metrics in all_metrics:
		rows_md.append(
			"| "
			+ " | ".join(
				[
					str(metrics.get("model", "N/A")),
					str(metrics.get("detector", "N/A")),
					str(metrics.get("reid") or "-"),
					str(metrics.get("backend", "N/A")),
					f"{metrics.get('avg_fps', 0):.2f}",
					f"{metrics.get('avg_inference_ms', 0):.2f}",
					f"{metrics.get('avg_unique_tracks', 0):.2f}",
				]
			)
			+ " |"
		)

	report = "\n".join(
		[
			"# Person Tracking Model Evaluation Report",
			"",
			f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
			"",
			"## Summary",
			"",
			header,
			sep,
			*rows_md,
			"",
			"## Notes",
			"",
			"- Custom trackers accept any detector from this repo through the shared detector registry.",
			"- BoxMOT client runs the requested BoxMOT tracker backend while reusing the same detector registry.",
			"- Runtime metrics here are throughput and track-count proxies; MOT metrics require annotated identities.",
		]
	)

	(results_dir / "comparison_report.md").write_text(report, encoding="utf-8")
	_write_csv(results_dir / "comparison_summary.csv", all_metrics)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate pluggable person-tracking pipelines")
	parser.add_argument(
		"--trackers",
		nargs="+",
		default=["all"],
		metavar="TRACKER",
		help=(
			"Tracker selections. Use custom trackers botsort bytetrack deepsort, "
			"or boxmot:<tracker> for BoxMOT backends. Use 'all' for the default suite."
		),
	)
	parser.add_argument(
		"--detectors",
		nargs="+",
		default=["yolov8n"],
		metavar="DETECTOR",
		help=f"Detector selections. Supported: {', '.join(SUPPORTED_DETECTORS)} or 'all' ({', '.join(DEFAULT_DETECTORS)}).",
	)
	parser.add_argument("--test-dir", type=Path, default=ROOT / "test")
	parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
	parser.add_argument("--frame-step", type=int, default=1)
	parser.add_argument("--max-frames", type=int, default=None)
	parser.add_argument("--time-limit", type=float, default=60.0)
	parser.add_argument("--confidence", type=float, default=0.25)
	parser.add_argument("--reid", default=None, help="BoxMOT ReID model name or weights path.")
	parser.add_argument("--device", default="cpu", help="Tracking device for BoxMOT trackers, e.g. cpu or 0.")
	parser.add_argument("--half", action="store_true", help="Enable half precision where supported.")
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
	args.results_dir.mkdir(parents=True, exist_ok=True)

	print("=" * 72)
	print("  Person Tracking Model Evaluation")
	print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print("=" * 72)
	print(f"  Test dir    : {args.test_dir}")
	print(f"  Results dir : {args.results_dir}")
	print(f"  Trackers    : {' '.join(args.trackers)}")
	print(f"  Detectors   : {' '.join(args.detectors)}")
	print(f"  Confidence  : {args.confidence}")
	print(f"  Frame step  : {args.frame_step}")
	if args.max_frames:
		print(f"  Max frames  : {args.max_frames}")
	if args.time_limit > 0:
		print(f"  Time limit  : {args.time_limit:.0f}s per combination/video")
	print()

	tracker_specs = expand_tracker_specs(args.trackers)
	detector_specs = expand_detector_specs(args.detectors)
	print(
		f"Resolved {len(tracker_specs)} tracker selection(s) and {len(detector_specs)} detector selection(s)."
	)

	all_metrics: list[dict[str, Any]] = []
	for detector_spec in detector_specs:
		for tracker_spec in tracker_specs:
			try:
				runtime = build_runtime(
					detector_spec=detector_spec,
					tracker_spec=tracker_spec,
					confidence_threshold=args.confidence,
					reid_model=args.reid,
					device=args.device,
					half=args.half,
					deepsort_embedder=args.deepsort_embedder,
					deepsort_embedder_model=args.deepsort_embedder_model,
					deepsort_embedder_weights=args.deepsort_embedder_weights,
				)
				metrics = run_evaluation(
					runtime=runtime,
					test_dir=args.test_dir,
					results_base=args.results_dir,
					frame_step=args.frame_step,
					max_frames=args.max_frames,
					time_limit=args.time_limit if args.time_limit > 0 else None,
				)
				if metrics:
					all_metrics.append(metrics)
			except Exception as exc:
				print(f"\n[ERROR] {tracker_spec.label} + {detector_spec} failed: {exc}")
				traceback.print_exc()

	if all_metrics:
		generate_comparison_report(all_metrics, args.results_dir)

	print("\nDone.")


def _write_json(path: Path, data: object) -> None:
	with open(path, "w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
	if not rows:
		return
	with open(path, "w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
		writer.writeheader()
		writer.writerows(rows)


if __name__ == "__main__":
	main()
