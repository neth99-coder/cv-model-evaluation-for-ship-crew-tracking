"""
Multi-Tracker Person Tracking Backend API
Supports: BOXMOT, FairMOT, DeepSORT with pluggable detectors, trackers, RE-ID
"""

import os
import uuid
import asyncio
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import json

from trackers.boxmot_tracker import BoxMOTTracker
from trackers.fairmot_tracker import FairMOTTracker
from trackers.deepsort_tracker import DeepSORTTracker
from utils.session_manager import SessionManager
from utils.metrics import TrackingMetrics
from utils.coco_metrics import CocoMetricEvaluator, resolve_coco_annotation_path

app = FastAPI(title="Multi-Tracker Person Tracking API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TEST_DIR = Path("test")

for d in [UPLOAD_DIR, OUTPUT_DIR, TEST_DIR]:
    d.mkdir(exist_ok=True)

session_manager = SessionManager()

# ─── Config ──────────────────────────────────────────────────────────────────

TRACKER_CAPABILITIES = {
    "boxmot": {
        "trackers": ["bytetrack", "botsort", "ocsort", "deepocsort", "strongsort"],
        "detectors": ["yolov8n", "yolov8s", "yolov8m", "yolov5n", "yolov5s", "fasterrcnn", "ssd_mobilenet"],
        "reid": {
            "bytetrack": False,
            "botsort": True,
            "ocsort": False,
            "deepocsort": True,
            "strongsort": True,
        },
        "reid_models": ["osnet_x0_25", "osnet_x1_0", "resnet50", "mlfn"],
    },
    "fairmot": {
        "trackers": ["fairmot"],
        "detectors": ["dla34", "hrnet", "resnet50"],
        "reid": {"fairmot": True},
        "reid_models": ["built-in"],
    },
    "deepsort": {
        "trackers": ["deepsort"],
        "detectors": ["yolov8n", "yolov8s", "yolov8m", "yolov5n", "yolov5s", "fasterrcnn", "ssd_mobilenet"],
        "reid": {"deepsort": True},
        "reid_models": ["osnet_x0_25", "osnet_x1_0", "resnet50"],
    },
}

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "Multi-Tracker Person Tracking API", "version": "1.0.0"}


@app.get("/api/capabilities")
async def get_capabilities():
    """Return tracker/detector/reid capabilities for each framework."""
    return TRACKER_CAPABILITIES


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and return its ID."""
    video_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".mp4"
    dest = UPLOAD_DIR / f"{video_id}{ext}"
    content = await file.read()
    dest.write_bytes(content)
    return {"video_id": video_id, "filename": file.filename, "path": str(dest)}


@app.get("/api/test-video")
async def get_test_video(name: Optional[str] = None):
    """Return test video info; defaults to the primary test clip when name is not provided."""
    videos = _list_test_videos()
    if not videos:
        return JSONResponse({"error": "No test videos found in test/ folder"}, status_code=404)

    if name:
        requested = next((v for v in videos if v["filename"] == name), None)
        if not requested:
            return JSONResponse({"error": f"Test video '{name}' not found"}, status_code=404)
        return requested

    # Backward compatibility: existing clients using /api/test-video get the primary clip.
    return videos[0]


@app.get("/api/test-videos")
async def list_test_videos():
    """List all available test videos for selection in UI."""
    videos = _list_test_videos()
    return {"videos": videos, "count": len(videos)}


@app.post("/api/track/file")
async def track_to_file(
    background_tasks: BackgroundTasks,
    video_id: str,
    framework: str = "boxmot",
    tracker: str = "bytetrack",
    detector: str = "yolov8n",
    reid_model: Optional[str] = None,
    coco_annotations: Optional[str] = None,
    conf_threshold: float = 0.4,
):
    """Start tracking job that saves output video + metrics."""
    video_path = _resolve_video(video_id)
    if not video_path:
        return JSONResponse({"error": "Video not found"}, status_code=404)

    job_id = str(uuid.uuid4())
    output_path = OUTPUT_DIR / f"{job_id}_tracked.mp4"
    metrics_path = OUTPUT_DIR / f"{job_id}_metrics.json"

    session_manager.create_job(job_id, {
        "status": "queued",
        "framework": framework,
        "tracker": tracker,
        "detector": detector,
        "reid_model": reid_model,
        "coco_annotations": coco_annotations,
        "video_id": video_id,
        "output_path": str(output_path),
        "metrics_path": str(metrics_path),
    })

    background_tasks.add_task(
        _run_tracking_job,
        job_id, video_path, output_path, metrics_path,
        framework, tracker, detector, reid_model, coco_annotations, conf_threshold
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/track/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a tracking job."""
    job = session_manager.get_job(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return job


@app.get("/api/track/download/{job_id}")
async def download_output(job_id: str):
    """Download tracked output video."""
    job = session_manager.get_job(job_id)
    if not job or job["status"] != "done":
        return JSONResponse({"error": "Output not ready"}, status_code=404)
    return FileResponse(job["output_path"], media_type="video/mp4", filename=f"tracked_{job_id}.mp4")


@app.get("/api/track/metrics/{job_id}")
async def get_metrics(job_id: str):
    """Get tracking metrics for a completed job."""
    job = session_manager.get_job(job_id)
    if not job or job["status"] != "done":
        return JSONResponse({"error": "Metrics not ready"}, status_code=404)
    with open(job["metrics_path"]) as f:
        return json.load(f)


@app.websocket("/ws/track/{session_id}")
async def realtime_tracking(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time tracking stream."""
    await websocket.accept()
    session_manager.set_ws_active(session_id, True)

    try:
        # Receive config
        config_raw = await websocket.receive_text()
        config = json.loads(config_raw)

        video_path = _resolve_video(config.get("video_id", ""))
        if not video_path:
            await websocket.send_text(json.dumps({"error": "Video not found"}))
            return

        framework = config.get("framework", "boxmot")
        tracker_name = config.get("tracker", "bytetrack")
        detector = config.get("detector", "yolov8n")
        reid_model = config.get("reid_model")
        conf = float(config.get("conf_threshold", 0.4))

        tracker = _build_tracker(framework, tracker_name, detector, reid_model, conf)
        runtime_pipeline = _runtime_pipeline(tracker, framework, tracker_name, detector, reid_model)
        cap = cv2.VideoCapture(str(video_path))

        await websocket.send_text(json.dumps({
            "type": "pipeline",
            "pipeline": runtime_pipeline,
        }))

        fps_real = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        metrics = TrackingMetrics()
        frame_idx = 0

        while session_manager.is_ws_active(session_id):
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            tracks = tracker.update(frame)
            elapsed = time.time() - t0

            metrics.update(tracks, elapsed, frame_idx)
            annotated = _draw_tracks(frame.copy(), tracks)

            # Encode frame as JPEG
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = __import__("base64").b64encode(buf).decode()

            payload = {
                "type": "frame",
                "frame": b64,
                "frame_idx": frame_idx,
                "total_frames": total_frames,
                "fps": round(1.0 / elapsed if elapsed > 0 else 0, 1),
                "track_count": len(tracks),
                "metrics": metrics.summary(),
            }
            await websocket.send_text(json.dumps(payload))
            frame_idx += 1

            # throttle to ~real-time
            await asyncio.sleep(max(0, (1.0 / fps_real) - elapsed))

        cap.release()
        await websocket.send_text(json.dumps({
            "type": "done",
            "metrics": metrics.summary(),
            "pipeline": runtime_pipeline,
        }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": str(e),
            }))
        except Exception:
            pass
    finally:
        session_manager.set_ws_active(session_id, False)


@app.post("/api/track/stop/{session_id}")
async def stop_realtime(session_id: str):
    session_manager.set_ws_active(session_id, False)
    return {"stopped": True}


# ─── Evaluate ─────────────────────────────────────────────────────────────────

@app.post("/api/evaluate")
async def evaluate_combinations(background_tasks: BackgroundTasks, video_id: Optional[str] = None):
    """Run evaluate.py-style batch evaluation on all combinations."""
    video_path = _resolve_video(video_id or "test")
    if not video_path:
        return JSONResponse({"error": "Video not found"}, status_code=404)

    eval_id = str(uuid.uuid4())
    session_manager.create_job(eval_id, {"status": "running", "results": []})
    background_tasks.add_task(_run_evaluation, eval_id, video_path, None)
    return {"eval_id": eval_id}


@app.post("/api/evaluate/with-annotations")
async def evaluate_combinations_with_annotations(
    background_tasks: BackgroundTasks,
    video_id: Optional[str] = None,
    coco_annotations: Optional[str] = None,
):
    """Run batch evaluation and include COCO mAP metrics when annotations are provided."""
    video_path = _resolve_video(video_id or "test")
    if not video_path:
        return JSONResponse({"error": "Video not found"}, status_code=404)

    eval_id = str(uuid.uuid4())
    session_manager.create_job(eval_id, {
        "status": "running",
        "results": [],
        "coco_annotations": coco_annotations,
    })
    background_tasks.add_task(_run_evaluation, eval_id, video_path, coco_annotations)
    return {"eval_id": eval_id}


@app.get("/api/evaluate/{eval_id}")
async def get_evaluation(eval_id: str):
    return session_manager.get_job(eval_id) or JSONResponse({"error": "Not found"}, status_code=404)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_video(video_id: str) -> Optional[Path]:
    if video_id == "test":
        videos = _list_test_videos()
        if not videos:
            return None
        return Path(videos[0]["path"])

    # Explicit test video id format: test::<filename>
    if video_id.startswith("test::"):
        filename = video_id.split("::", 1)[1]
        p = TEST_DIR / Path(filename).name
        return p if p.exists() else None

    # Support raw filename for compatibility if caller sends the test file name directly.
    test_candidate = TEST_DIR / Path(video_id).name
    if test_candidate.exists() and test_candidate.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        return test_candidate

    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        p = UPLOAD_DIR / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def _list_test_videos() -> list[dict]:
    allowed = {".mp4", ".avi", ".mov", ".mkv"}
    videos = []
    for p in sorted(TEST_DIR.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file() or p.suffix.lower() not in allowed:
            continue
        videos.append(
            {
                "video_id": f"test::{p.name}",
                "filename": p.name,
                "path": str(p),
            }
        )

    # Keep historic default first if present.
    videos.sort(key=lambda v: (0 if v["filename"] == "test.mp4" else 1, v["filename"].lower()))
    return videos


def _build_tracker(framework, tracker_name, detector, reid_model, conf):
    if framework == "boxmot":
        return BoxMOTTracker(tracker_name, detector, reid_model, conf)
    elif framework == "fairmot":
        return FairMOTTracker(detector, conf)
    elif framework == "deepsort":
        return DeepSORTTracker(detector, reid_model, conf)
    raise ValueError(f"Unknown framework: {framework}")


def _runtime_pipeline(tracker, framework, tracker_name, detector, reid_model):
    if hasattr(tracker, "runtime_config"):
        return tracker.runtime_config()
    return {
        "framework": framework,
        "tracker": tracker_name,
        "detector": detector,
        "reid_model": reid_model,
    }


def _draw_tracks(frame, tracks):
    colors = {}
    for track in tracks:
        tid = _as_int_track_id(track[4])
        x1, y1, x2, y2 = [_as_int_coord(v) for v in track[:4]]
        if tid not in colors:
            np.random.seed(tid * 7)
            colors[tid] = tuple(np.random.randint(80, 255, 3).tolist())
        color = colors[tid]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"person_{tid}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def _as_int_coord(value) -> int:
    return int(float(value))


def _as_int_track_id(value) -> int:
    """Convert track ids that may come as int/float/string into a stable int."""
    try:
        return int(float(value))
    except Exception:
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else abs(hash(str(value))) % 1000000


async def _run_tracking_job(job_id, video_path, output_path, metrics_path,
                             framework, tracker_name, detector, reid_model, coco_annotations, conf):
    session_manager.update_job(job_id, {"status": "running", "progress": 0})
    try:
        tracker = _build_tracker(framework, tracker_name, detector, reid_model, conf)
        runtime_pipeline = _runtime_pipeline(tracker, framework, tracker_name, detector, reid_model)
        coco_path = resolve_coco_annotation_path(video_path, coco_annotations)
        coco_eval = CocoMetricEvaluator(coco_path) if coco_path else None
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        metrics = TrackingMetrics()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.time()
            tracks = tracker.update(frame)
            elapsed = time.time() - t0
            metrics.update(tracks, elapsed, frame_idx)
            if coco_eval:
                coco_eval.add_tracks(frame_idx, tracks)
            annotated = _draw_tracks(frame.copy(), tracks)
            writer.write(annotated)
            frame_idx += 1
            progress = int((frame_idx / max(total, 1)) * 100)
            session_manager.update_job(job_id, {"progress": progress})

        cap.release()
        writer.release()
        summary = metrics.summary()
        if coco_eval:
            summary.update(coco_eval.evaluate())
        else:
            summary.update({
                "map_iou_50": None,
                "map_small": None,
                "map_note": "COCO metrics unavailable (no annotation file found).",
            })
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)

        session_manager.update_job(job_id, {
            "status": "done",
            "progress": 100,
            "metrics": summary,
            "pipeline": runtime_pipeline,
        })
    except Exception as e:
        session_manager.update_job(job_id, {"status": "error", "error": str(e)})


async def _run_evaluation(eval_id, video_path, coco_annotations):
    combinations = [
        ("boxmot", "bytetrack", "yolov8n", None),
        ("boxmot", "botsort", "yolov8n", "osnet_x0_25"),
        ("boxmot", "strongsort", "yolov8s", "osnet_x1_0"),
        ("boxmot", "deepocsort", "yolov8n", "osnet_x0_25"),
        ("fairmot", "fairmot", "dla34", None),
        ("deepsort", "deepsort", "yolov8n", "osnet_x0_25"),
    ]
    results = []
    for framework, tracker_name, detector, reid in combinations:
        try:
            tracker = _build_tracker(framework, tracker_name, detector, reid, 0.4)
            coco_path = resolve_coco_annotation_path(video_path, coco_annotations)
            coco_eval = CocoMetricEvaluator(coco_path) if coco_path else None
            cap = cv2.VideoCapture(str(video_path))
            metrics = TrackingMetrics()
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= 300:
                    break
                t0 = time.time()
                tracks = tracker.update(frame)
                elapsed = time.time() - t0
                metrics.update(tracks, elapsed, frame_idx)
                if coco_eval:
                    coco_eval.add_tracks(frame_idx, tracks)
                frame_idx += 1
            cap.release()
            summary = metrics.summary()
            if coco_eval:
                summary.update(coco_eval.evaluate())
            else:
                summary.update({
                    "map_iou_50": None,
                    "map_small": None,
                    "map_note": "COCO metrics unavailable (no annotation file found).",
                })
            results.append({
                "framework": framework,
                "tracker": tracker_name,
                "detector": detector,
                "reid": reid,
                **summary,
                "status": "ok",
            })
        except Exception as e:
            results.append({
                "framework": framework, "tracker": tracker_name,
                "detector": detector, "reid": reid,
                "status": "error", "error": str(e),
            })

    out_path = OUTPUT_DIR / f"eval_{eval_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    session_manager.update_job(eval_id, {
        "status": "done",
        "results": results,
        "output_file": str(out_path),
    })
