"""Evaluation logic for hybrid person tracking backends (BoxMOT + custom)."""

import csv
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _write_json(path: Path, data: object) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _normalize_boxmot_detector_name(detector: str) -> str:
    p = Path(detector)
    stem = p.stem.lower()
    if stem.startswith("yolov"):
        return stem
    return detector


def _trim_video(video_path: Path, out_path: Path, max_frames_to_keep: int) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
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


def _prepare_bounded_input_video(
    video_path: Path,
    tmp_dir: Path,
    max_frames: int | None,
    time_limit: float | None,
) -> tuple[Path, int | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_cap = None
    if max_frames is not None and max_frames > 0:
        frame_cap = max_frames

    if time_limit is not None and time_limit > 0:
        by_time = max(1, int(round(fps * time_limit)))
        frame_cap = by_time if frame_cap is None else min(frame_cap, by_time)

    if frame_cap is None or frame_cap >= total_frames:
        return video_path, None

    trimmed = tmp_dir / f"{video_path.stem}_first_{frame_cap}.mp4"
    _trim_video(video_path, trimmed, frame_cap)
    return trimmed, frame_cap


# ---------------------------------------------------------------------------
# Custom-backend evaluation
# ---------------------------------------------------------------------------

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


def _compute_track_stability(track_frames: dict[int, list[int]]) -> dict[str, float]:
    if not track_frames:
        return {
            "unique_track_ids": 0.0,
            "avg_track_length_frames": 0.0,
            "track_fragmentation": 0.0,
        }

    lengths = [len(frames) for frames in track_frames.values()]
    frags: list[int] = []
    for frames in track_frames.values():
        if not frames:
            frags.append(0)
            continue
        gaps = 0
        prev = frames[0]
        for f in frames[1:]:
            if f != prev + 1:
                gaps += 1
            prev = f
        frags.append(gaps)

    return {
        "unique_track_ids": float(len(track_frames)),
        "avg_track_length_frames": round(sum(lengths) / len(lengths), 3),
        "track_fragmentation": round(sum(frags) / len(frags), 3),
    }


def run_custom_evaluation(
    tracker: Any,
    tracker_label: str,
    test_dir: Path,
    results_base: Path,
    max_frames: int | None = None,
    time_limit: float | None = 60.0,
) -> dict[str, Any]:
    out_dir = results_base / tracker_label
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = out_dir / "annotated_videos"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = test_dir / "videos"
    if not videos_dir.exists():
        print(f"[ERROR] {videos_dir} does not exist")
        return {}

    videos = sorted(p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        print(f"[ERROR] No videos found in {videos_dir}")
        return {}

    print(f"\n{'=' * 60}")
    print(f"  Tracker : {tracker_label} (custom backend)")
    print(f"{'=' * 60}")

    per_video: list[dict[str, Any]] = []

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [WARN] Cannot open video: {video_path.name}")
            continue

        fps_native = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = annotated_dir / f"{video_path.stem}_tracked.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_native, (frame_w, frame_h))

        tracker.reset()

        inference_times: list[float] = []
        active_counts: list[int] = []
        track_frames: dict[int, list[int]] = defaultdict(list)

        frame_idx = 0
        frames_processed = 0
        t_start = time.perf_counter()

        print(f"  -> {video_path.name} ({total_frames_in_file} frames)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames is not None and max_frames > 0 and frames_processed >= max_frames:
                break
            if time_limit is not None and time_limit > 0 and (time.perf_counter() - t_start) >= time_limit:
                print(f"     [TIME LIMIT] {time_limit:.0f}s reached")
                break

            infer_start = time.perf_counter()
            tracks = tracker.update(frame)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0

            for t in tracks:
                track_frames[int(t["track_id"])].append(frame_idx)

            annotated = _annotate_custom_frame(frame, tracks)
            hud = f"Tracks {len(tracks)} | Infer {infer_ms:.1f}ms"
            cv2.rectangle(annotated, (8, 8), (360, 36), (30, 30, 30), -1)
            cv2.putText(annotated, hud, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(annotated)

            inference_times.append(infer_ms)
            active_counts.append(len(tracks))
            frame_idx += 1
            frames_processed += 1

        cap.release()
        writer.release()

        if not inference_times:
            continue

        avg_ms = sum(inference_times) / len(inference_times)
        avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        avg_active = sum(active_counts) / len(active_counts)
        stability = _compute_track_stability(track_frames)

        row = {
            "video": video_path.name,
            "frames_processed": frames_processed,
            "avg_inference_ms": round(avg_ms, 3),
            "avg_fps": round(avg_fps, 3),
            "avg_active_tracks": round(avg_active, 3),
            "max_active_tracks": max(active_counts),
            "min_active_tracks": min(active_counts),
            "unique_tracks": int(stability["unique_track_ids"]),
            "avg_track_length_frames": stability["avg_track_length_frames"],
            "track_fragmentation": stability["track_fragmentation"],
            "annotated_video": f"annotated_videos/{out_path.name}",
            "backend": "custom",
        }
        per_video.append(row)

    if not per_video:
        return {}

    metrics = {
        "model": tracker_label,
        "backend": "custom",
        "timestamp": datetime.now().isoformat(),
        "total_videos": len(per_video),
        "total_frames_processed": int(sum(int(v["frames_processed"]) for v in per_video)),
        "avg_fps": _safe_mean(per_video, "avg_fps"),
        "avg_inference_ms": _safe_mean(per_video, "avg_inference_ms"),
        "avg_unique_tracks": _safe_mean(per_video, "unique_tracks"),
        "avg_active_tracks": _safe_mean(per_video, "avg_active_tracks"),
        "avg_track_length_frames": _safe_mean(per_video, "avg_track_length_frames"),
        "avg_track_fragmentation": _safe_mean(per_video, "track_fragmentation"),
    }

    detailed = {"model": tracker_label, "metrics": metrics, "per_video": per_video}
    _write_json(out_dir / "summary.json", metrics)
    _write_json(out_dir / "detailed_results.json", detailed)
    _write_csv(out_dir / "per_video_results.csv", per_video)

    return metrics


# ---------------------------------------------------------------------------
# BoxMOT-backend evaluation
# ---------------------------------------------------------------------------

def run_boxmot_evaluation(
    tracker_name: str,
    tracker_label: str,
    detector: str,
    reid_model: str | None,
    test_dir: Path,
    results_base: Path,
    conf: float = 0.25,
    imgsz: int = 640,
    max_frames: int | None = None,
    time_limit: float | None = 60.0,
) -> dict[str, Any]:
    from boxmot import Boxmot

    out_dir = results_base / tracker_label
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = out_dir / "annotated_videos"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = out_dir / "tracks_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp_inputs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = test_dir / "videos"
    if not videos_dir.exists():
        print(f"[ERROR] {videos_dir} does not exist")
        return {}

    videos = sorted(p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        print(f"[ERROR] No videos found in {videos_dir}")
        return {}

    detector_name = _normalize_boxmot_detector_name(detector)

    print(f"\n{'=' * 60}")
    print(f"  Tracker : {tracker_label} (BoxMOT: {tracker_name})")
    print(f"{'=' * 60}")

    per_video: list[dict[str, Any]] = []

    for video_path in videos:
        try:
            input_video, applied_cap = _prepare_bounded_input_video(
                video_path=video_path,
                tmp_dir=tmp_dir,
                max_frames=max_frames,
                time_limit=time_limit,
            )

            print(f"  -> {video_path.name}")
            if applied_cap is not None:
                print(f"     using shortened clip: first {applied_cap} frames")

            runner = Boxmot(
                detector=detector_name,
                reid=reid_model,
                tracker=tracker_name,
                classes=[0],
                project=out_dir,
            )

            run = runner.track(
                source=str(input_video),
                imgsz=imgsz,
                conf=conf,
                save=True,
                save_txt=True,
                show=False,
                verbose=False,
                device="cpu",
            )

            summary = run.summary if isinstance(run.summary, dict) else {}
            timings = run.timings if isinstance(run.timings, dict) else {}

            saved_video = Path(run.video_path) if run.video_path else None
            saved_txt = Path(run.text_path) if run.text_path else None

            annotated_rel = ""
            if saved_video and saved_video.exists():
                target = annotated_dir / f"{video_path.stem}_tracked.mp4"
                shutil.copy2(saved_video, target)
                annotated_rel = f"annotated_videos/{target.name}"

            txt_rel = ""
            if saved_txt and saved_txt.exists():
                target = txt_dir / f"{video_path.stem}.txt"
                shutil.copy2(saved_txt, target)
                txt_rel = f"tracks_txt/{target.name}"

            row = {
                "video": video_path.name,
                "input_clip": input_video.name,
                "frames_processed": int(summary.get("frames", 0)),
                "avg_inference_ms": round(float(timings.get("avg_total", 0.0)), 3),
                "avg_fps": round(float(timings.get("fps", 0.0)), 3),
                "detections": int(summary.get("detections", 0)),
                "tracks": int(summary.get("tracks", 0)),
                "unique_tracks": int(summary.get("unique_tracks", 0)),
                "det_ms_total": round(float(timings.get("det", 0.0)), 3),
                "reid_ms_total": round(float(timings.get("reid", 0.0)), 3),
                "assoc_ms_total": round(float(timings.get("track", 0.0)), 3),
                "annotated_video": annotated_rel,
                "tracks_txt": txt_rel,
                "backend": "boxmot",
            }
            per_video.append(row)
        except Exception as exc:
            print(f"  [WARN] Failed on {video_path.name}: {exc}")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if not per_video:
        return {}

    metrics = {
        "model": tracker_label,
        "backend": "boxmot",
        "timestamp": datetime.now().isoformat(),
        "total_videos": len(per_video),
        "total_frames_processed": int(sum(int(v["frames_processed"]) for v in per_video)),
        "avg_fps": _safe_mean(per_video, "avg_fps"),
        "avg_inference_ms": _safe_mean(per_video, "avg_inference_ms"),
        "avg_unique_tracks": _safe_mean(per_video, "unique_tracks"),
        "avg_detections": _safe_mean(per_video, "detections"),
        "avg_tracks": _safe_mean(per_video, "tracks"),
    }

    detailed = {
        "model": tracker_label,
        "tracker_backend": tracker_name,
        "detector": detector_name,
        "reid": reid_model,
        "metrics": metrics,
        "per_video": per_video,
    }

    _write_json(out_dir / "summary.json", metrics)
    _write_json(out_dir / "detailed_results.json", detailed)
    _write_csv(out_dir / "per_video_results.csv", per_video)

    return metrics


def generate_comparison_report(all_metrics: list[dict[str, Any]], results_dir: Path) -> None:
    lines = [
        "# Person Tracking Model Evaluation Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Summary\n",
        "| Tracker | Backend | Avg FPS | Avg ms/frame | Avg unique tracks |",
        "|---------|---------|---------|--------------|-------------------|",
    ]

    for m in all_metrics:
        lines.append(
            f"| {m.get('model', 'N/A')} "
            f"| {m.get('backend', 'n/a')} "
            f"| {m.get('avg_fps', 0):.2f} "
            f"| {m.get('avg_inference_ms', 0):.2f} "
            f"| {m.get('avg_unique_tracks', 0):.2f} |"
        )

    lines.extend(
        [
            "\n## Notes\n",
            "- Backend can be `boxmot` or `custom` depending on availability and selection mode.",
            "- Without MOT ground truth, metrics are runtime and track-count stability proxies.",
        ]
    )

    md_path = results_dir / "comparison_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    csv_fields = [
        "model",
        "backend",
        "avg_fps",
        "avg_inference_ms",
        "avg_unique_tracks",
        "total_videos",
        "total_frames_processed",
        "timestamp",
    ]

    csv_path = results_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"\nComparison report -> {md_path}")
    print(f"Comparison CSV    -> {csv_path}")
