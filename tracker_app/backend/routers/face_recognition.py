import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

import cv2
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from face_models.auraface import AuraFaceRecognizer
from face_models.compreface import CompreFaceRecognizer
from face_models.facenet_model import FaceNetRecognizer
from face_models.mobilefacenet import MobileFaceNetRecognizer
from utils.session_manager import SessionManager

router = APIRouter()
api_router = APIRouter()

ROOT_DIR = Path(__file__).resolve().parents[3]
FACE_PROJECT_DIR = ROOT_DIR / "face_recognition"
FACE_TEST_DIR = FACE_PROJECT_DIR / "test"
FACE_TEST_IMAGES_DIR = FACE_TEST_DIR / "test_images"
FACE_UPLOAD_DIR = Path("uploads") / "face_recognition"
FACE_OUTPUT_DIR = Path("outputs") / "face_recognition"

for directory in [FACE_UPLOAD_DIR, FACE_OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

face_session_manager = SessionManager()

MODEL_FACTORIES: dict[str, tuple[str, Callable[[], object]]] = {
    "auraface": ("AuraFace (ResNet100)", AuraFaceRecognizer),
    "mobilefacenet": ("MobileFaceNet", MobileFaceNetRecognizer),
    "facenet": ("FaceNet (Sandberg)", FaceNetRecognizer),
    "compreface": ("CompreFace", CompreFaceRecognizer),
}


@router.get("/models")
@api_router.get("/api/face/models")
async def list_models():
    models = []
    for key, (label, _) in MODEL_FACTORIES.items():
        models.append({"id": key, "label": label})
    return {"models": models, "count": len(models)}


@router.get("/test-assets")
@api_router.get("/api/face/test-assets")
async def list_test_assets():
    media = []
    if FACE_TEST_IMAGES_DIR.exists():
        for path in sorted(FACE_TEST_IMAGES_DIR.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.name == "ground_truth.json":
                continue
            kind = "video" if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"} else "image"
            media.append({
                "asset_id": f"media::{path.name}",
                "filename": path.name,
                "kind": kind,
            })

    references = []
    if FACE_TEST_DIR.exists():
        for person_dir in sorted(FACE_TEST_DIR.iterdir(), key=lambda p: p.name.lower()):
            if not person_dir.is_dir() or person_dir.name == "test_images":
                continue
            for image_path in sorted(person_dir.iterdir(), key=lambda p: p.name.lower()):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                references.append({
                    "asset_id": f"reference::{person_dir.name}::{image_path.name}",
                    "name": person_dir.name,
                    "filename": image_path.name,
                })
                break

    return {"media": media, "references": references}


@api_router.post("/api/face/upload")
async def upload_face_media(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    dest = FACE_UPLOAD_DIR / f"{uuid.uuid4()}_{Path(file.filename).name}"
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    kind = "video" if dest.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"} else "image"
    return {
        "media_id": f"upload::{dest.name}",
        "filename": file.filename,
        "path": str(dest),
        "kind": kind,
    }


@api_router.get("/api/face/runs")
async def list_face_runs():
    runs = []
    for job in face_session_manager.list_jobs():
        runs.append({
            "job_id": job["job_id"],
            "status": job.get("status"),
            "model_name": job.get("model_name"),
            "target_filename": job.get("target_filename"),
            "target_kind": job.get("target_kind"),
            "created_at": job.get("created_at"),
            "completed_at": job.get("completed_at"),
            "stats": job.get("stats", {}),
            "output_url": job.get("output_url"),
            "error": job.get("error"),
        })
    runs.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return {"runs": runs, "count": len(runs)}


@router.post("/jobs")
async def create_face_job(
    background_tasks: BackgroundTasks,
    media_file: UploadFile | None = File(default=None),
    model_name: str = Form(...),
    test_asset_id: str | None = Form(default=None),
    reference_asset_ids: str = Form(default="[]"),
    sample_names: list[str] = Form(default=[]),
    sample_files: list[UploadFile] = File(default=[]),
):
    if model_name not in MODEL_FACTORIES:
        raise HTTPException(status_code=400, detail="Invalid model name.")

    target_path, target_filename, target_kind = await _resolve_target_media(media_file, test_asset_id)
    if target_path is None:
        raise HTTPException(status_code=400, detail="Upload a media file or choose a default test asset.")

    try:
        selected_reference_asset_ids = json.loads(reference_asset_ids or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="reference_asset_ids must be valid JSON.") from exc

    if len(sample_names) != len(sample_files):
        raise HTTPException(status_code=400, detail="Each uploaded sample face must have a matching name.")

    uploaded_samples = []
    for name, sample_file in zip(sample_names, sample_files):
        cleaned_name = (name or "").strip()
        if not cleaned_name:
            continue
        sample_dest = FACE_UPLOAD_DIR / f"{uuid.uuid4()}_{Path(sample_file.filename or 'sample.jpg').name}"
        with sample_dest.open("wb") as buffer:
            shutil.copyfileobj(sample_file.file, buffer)
        uploaded_samples.append({"name": cleaned_name, "path": str(sample_dest)})

    reference_assets = []
    for asset_id in selected_reference_asset_ids:
        ref = _resolve_reference_asset(asset_id)
        if ref:
            reference_assets.append(ref)

    job_id = str(uuid.uuid4())
    output_ext = ".mp4" if target_kind == "video" else Path(target_filename).suffix or ".jpg"
    output_path = FACE_OUTPUT_DIR / f"{job_id}_processed{output_ext}"
    face_session_manager.create_job(job_id, {
        "status": "queued",
        "created_at": time.time(),
        "model_name": model_name,
        "target_filename": target_filename,
        "target_kind": target_kind,
        "output_path": str(output_path),
        "detections": [],
        "stats": {},
    })

    background_tasks.add_task(
        _run_face_job,
        job_id,
        model_name,
        str(target_path),
        target_kind,
        str(output_path),
        reference_assets,
        uploaded_samples,
    )
    return {"job_id": job_id, "status": "queued"}


@api_router.post("/api/face/run/file")
async def create_face_file_run(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
    media_id: str | None = Form(default=None),
    reference_asset_ids: str = Form(default="[]"),
    sample_names: list[str] = Form(default=[]),
    sample_files: list[UploadFile] = File(default=[]),
):
    if model_name not in MODEL_FACTORIES:
        raise HTTPException(status_code=400, detail="Invalid model name.")

    target_path, target_filename, target_kind = _resolve_media_id(media_id)
    if target_path is None:
        raise HTTPException(status_code=400, detail="Select or upload media before starting a face run.")

    try:
        selected_reference_asset_ids = json.loads(reference_asset_ids or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="reference_asset_ids must be valid JSON.") from exc

    if len(sample_names) != len(sample_files):
        raise HTTPException(status_code=400, detail="Each uploaded sample face must have a matching name.")

    uploaded_samples = []
    for name, sample_file in zip(sample_names, sample_files):
        cleaned_name = (name or "").strip()
        if not cleaned_name:
            continue
        sample_dest = FACE_UPLOAD_DIR / f"{uuid.uuid4()}_{Path(sample_file.filename or 'sample.jpg').name}"
        with sample_dest.open("wb") as buffer:
            shutil.copyfileobj(sample_file.file, buffer)
        uploaded_samples.append({"name": cleaned_name, "path": str(sample_dest)})

    reference_assets = []
    for asset_id in selected_reference_asset_ids:
        ref = _resolve_reference_asset(asset_id)
        if ref:
            reference_assets.append(ref)

    job_id = str(uuid.uuid4())
    output_ext = ".mp4" if target_kind == "video" else Path(target_filename).suffix or ".jpg"
    output_path = FACE_OUTPUT_DIR / f"{job_id}_processed{output_ext}"
    face_session_manager.create_job(job_id, {
        "status": "queued",
        "created_at": time.time(),
        "model_name": model_name,
        "target_filename": target_filename,
        "target_kind": target_kind,
        "output_path": str(output_path),
        "detections": [],
        "stats": {},
    })

    background_tasks.add_task(
        _run_face_job,
        job_id,
        model_name,
        str(target_path),
        target_kind,
        str(output_path),
        reference_assets,
        uploaded_samples,
    )
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
async def get_face_job(job_id: str):
    job = face_session_manager.get_job(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return job


@api_router.get("/api/face/run/status/{job_id}")
async def get_face_run_status(job_id: str):
    return await get_face_job(job_id)


@router.get("/download/{job_id}")
async def download_face_output(job_id: str):
    job = face_session_manager.get_job(job_id)
    if not job or job.get("status") != "done":
        return JSONResponse({"error": "Output not ready"}, status_code=404)
    output_path = Path(job["output_path"])
    if not output_path.exists():
        return JSONResponse({"error": "Processed output missing"}, status_code=404)
    media_type = "video/mp4" if job.get("target_kind") == "video" else "image/jpeg"
    return FileResponse(str(output_path), media_type=media_type, filename=output_path.name)


@api_router.get("/api/face/run/download/{job_id}")
async def download_face_run_output(job_id: str):
    return await download_face_output(job_id)


def _build_recognizer(model_name: str):
    _, factory = MODEL_FACTORIES[model_name]
    return factory()


async def _resolve_target_media(media_file: UploadFile | None, test_asset_id: str | None):
    if media_file is not None and media_file.filename:
        dest = FACE_UPLOAD_DIR / f"{uuid.uuid4()}_{Path(media_file.filename).name}"
        with dest.open("wb") as buffer:
            shutil.copyfileobj(media_file.file, buffer)
        kind = "video" if dest.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"} else "image"
        return dest, media_file.filename, kind

    if test_asset_id:
        asset = _resolve_media_asset(test_asset_id)
        if asset is not None:
            return asset["path"], asset["filename"], asset["kind"]

    return None, None, None


def _resolve_media_asset(asset_id: str):
    if not asset_id.startswith("media::"):
        return None
    filename = asset_id.split("::", 1)[1]
    path = FACE_TEST_IMAGES_DIR / Path(filename).name
    if not path.exists():
        return None
    kind = "video" if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"} else "image"
    return {"filename": path.name, "path": path, "kind": kind}


def _resolve_media_id(media_id: str | None):
    if not media_id:
        return None, None, None

    if media_id.startswith("upload::"):
        filename = media_id.split("::", 1)[1]
        path = FACE_UPLOAD_DIR / Path(filename).name
        if not path.exists():
            return None, None, None
        kind = "video" if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"} else "image"
        return path, path.name, kind

    asset = _resolve_media_asset(media_id)
    if asset is not None:
        return asset["path"], asset["filename"], asset["kind"]

    return None, None, None


def _resolve_reference_asset(asset_id: str):
    if not asset_id.startswith("reference::"):
        return None
    _, person_name, filename = asset_id.split("::", 2)
    path = FACE_TEST_DIR / person_name / Path(filename).name
    if not path.exists():
        return None
    return {"name": person_name, "path": str(path)}


def _run_face_job(
    job_id: str,
    model_name: str,
    target_path: str,
    target_kind: str,
    output_path: str,
    reference_assets: list[dict],
    uploaded_samples: list[dict],
):
    face_session_manager.update_job(job_id, {"status": "running", "progress": 0})
    temp_dir = Path(tempfile.mkdtemp(prefix=f"face_job_{job_id}_"))
    created_uploads = [Path(sample["path"]) for sample in uploaded_samples]
    try:
        recognizer = _build_recognizer(model_name)

        enrolled = []
        for ref in [*reference_assets, *uploaded_samples]:
            if recognizer.enroll(ref["name"], ref["path"]):
                enrolled.append(ref["name"])

        if target_kind == "video":
            result = _process_video(recognizer, Path(target_path), Path(output_path), temp_dir, job_id)
        else:
            result = _process_image(recognizer, Path(target_path), Path(output_path))

        face_session_manager.update_job(job_id, {
            "status": "done",
            "progress": 100,
            "completed_at": time.time(),
            "model_name": model_name,
            "target_kind": target_kind,
            "output_path": output_path,
            "output_url": f"/api/face/run/download/{job_id}",
            "detections": result["detections"],
            "stats": {
                **result["stats"],
                "enrolled_names": enrolled,
                "enrolled_count": len(enrolled),
            },
        })
    except Exception as exc:
        face_session_manager.update_job(job_id, {
            "status": "error",
            "completed_at": time.time(),
            "error": str(exc),
        })
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        for path in created_uploads:
            path.unlink(missing_ok=True)


def _process_image(recognizer, input_path: Path, output_path: Path):
    detections, elapsed_ms = recognizer.recognize_faces(str(input_path))
    if not detections:
        fallback_boxes = recognizer.detect_bboxes(str(input_path))
        if fallback_boxes:
            detections = [{"name": "unknown", "confidence": 0.0, "bbox": box} for box in fallback_boxes]

    image = cv2.imread(str(input_path))
    if image is None:
        raise RuntimeError(f"Unable to read image: {input_path.name}")
    annotated = _annotate_frame(image, detections)
    cv2.imwrite(str(output_path), annotated)
    return {
        "detections": [_serialize_detection(d) for d in detections],
        "stats": {
            "faces_detected": len(detections),
            "inference_time_ms": round(elapsed_ms, 2),
            **_summarize_detections(detections),
        },
    }


def _process_video(recognizer, input_path: Path, output_path: Path, temp_dir: Path, job_id: str):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path.name}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    total_faces = 0
    identified_faces = 0
    unknown_faces = 0
    inference_times = []
    confidence_sum = 0.0
    confidence_count = 0
    preview_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = temp_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        detections, elapsed_ms = recognizer.recognize_faces(str(frame_path))
        if not detections:
            fallback_boxes = recognizer.detect_bboxes(str(frame_path))
            if fallback_boxes:
                detections = [{"name": "unknown", "confidence": 0.0, "bbox": box} for box in fallback_boxes]

        annotated = _annotate_frame(frame, detections)
        writer.write(annotated)

        total_faces += len(detections)
        stats = _summarize_detections(detections)
        identified_faces += stats["identified_faces"]
        unknown_faces += stats["unknown_faces"]
        confidence_sum += stats["_confidence_sum"]
        confidence_count += stats["_confidence_count"]
        inference_times.append(elapsed_ms)
        if len(preview_detections) < 20 and detections:
            preview_detections.append({
                "frame_idx": frame_idx,
                "detections": [_serialize_detection(d) for d in detections],
            })

        frame_idx += 1
        progress = int((frame_idx / max(total_frames, 1)) * 100)
        face_session_manager.update_job(job_id, {"progress": progress})

    cap.release()
    writer.release()

    avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0.0
    return {
        "detections": preview_detections,
        "stats": {
            "frames_processed": frame_idx,
            "total_faces_detected": total_faces,
            "avg_inference_time_ms": round(avg_inference, 2),
            "identified_faces": identified_faces,
            "unknown_faces": unknown_faces,
            "avg_match_confidence": round(confidence_sum / confidence_count, 3) if confidence_count else 0.0,
        },
    }


def _summarize_detections(detections: list[dict] | None = None, all_detections: list[dict] | None = None):
    items = detections if detections is not None else all_detections or []
    identified = [det for det in items if str(det.get("name", "unknown")) != "unknown"]
    confidences = [
        float(det.get("confidence", 0.0))
        for det in identified
        if det.get("confidence") is not None
    ]
    return {
        "identified_faces": len(identified),
        "unknown_faces": max(0, len(items) - len(identified)),
        "avg_match_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        "_confidence_sum": sum(confidences),
        "_confidence_count": len(confidences),
    }


def _annotate_frame(frame, detections: list[dict]):
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for det in detections:
        bbox = det.get("bbox")
        if bbox is None:
            continue

        name = str(det.get("name", "unknown"))
        score = float(det.get("confidence", 0.0))
        color = (0, 180, 0) if name != "unknown" else (0, 0, 255)
        label = name if name == "unknown" else f"{name} {score:.2f}"

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_x = x1
        text_y = y1 - 8
        if text_y - text_h < 0:
            text_y = y1 + text_h + 8
        if text_x + text_w >= width:
            text_x = max(0, width - text_w - 4)

        bg_top = max(0, text_y - text_h - 4)
        bg_bottom = min(height - 1, text_y + 4)
        bg_right = min(width - 1, text_x + text_w + 4)
        cv2.rectangle(annotated, (text_x, bg_top), (bg_right, bg_bottom), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            label,
            (text_x + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _serialize_detection(det: dict) -> dict:
    bbox = det.get("bbox")
    return {
        "name": det.get("name", "unknown"),
        "confidence": round(float(det.get("confidence", 0.0)), 4),
        "bbox": list(bbox) if bbox is not None else None,
    }
