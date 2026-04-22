# TrackVision — Tracking And Face Recognition Evaluation App

A full-stack application for running and evaluating person tracking pipelines and face-recognition runs. The tracking side supports BOXMOT, FairMOT, and DeepSORT with pluggable detectors and Re-ID models, while the face-recognition side supports multiple recognizers, uploaded or built-in reference galleries, saved outputs, and run comparison tables.

---

## Project Structure

```
tracker_app/
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── evaluate.py              # Standalone CLI evaluation script
│   ├── requirements.txt
│   ├── routers/
│   │   └── face_recognition.py  # Face-recognition API + background jobs
│   ├── trackers/
│   │   ├── __init__.py
│   │   ├── boxmot_tracker.py    # BOXMOT wrapper (ByteTrack/BoT-SORT/OC-SORT/DeepOCSORT/StrongSORT)
│   │   ├── fairmot_tracker.py   # FairMOT wrapper (DLA34/HRNet/ResNet50 backbone)
│   │   └── deepsort_tracker.py  # DeepSORT wrapper with pluggable Re-ID
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── session_manager.py   # In-memory job/session state
│   │   └── metrics.py           # Tracking metrics collector
│   ├── test/
│   │   └── test.mp4             # ← Place your tracking test video here
│   ├── uploads/                 # Uploaded videos/images (auto-created)
│   └── outputs/                 # Tracking outputs, face outputs, evaluation results
│
└── frontend/
    ├── index.html
    ├── vite.config.js
    ├── package.json
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── index.css
        ├── components/
        │   ├── Navbar.jsx
        │   ├── ModelSelector.jsx         # Framework/tracker/detector/reid selection
        │   ├── VideoUploader.jsx         # Drag-drop video upload
        │   ├── RealtimePlayer.jsx        # WebSocket live stream viewer
        │   ├── FileTrackingJob.jsx       # Background tracking job submission + polling
        │   ├── ComparisonPanel.jsx       # Tracking evaluation table
        │   ├── FaceRunComparisonPanel.jsx # Face-recognition run table
        │   └── MetricsPanel.jsx          # FPS/track count metrics display
        └── pages/
            ├── ObjectTracking.jsx        # Main tracking page
            ├── FaceDetection.jsx         # Face-recognition page
            └── FaceRecognition.jsx       # Alias to FaceDetection for compatibility
```

---

## Quick Start

### 1. Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Place test video
cp /path/to/your/video.mp4 test/test.mp4

# Start API server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev       # http://localhost:5173
```

---

## Main Workflows

### Object Tracking

- Upload or choose a server test video.
- Configure framework, tracker, detector, Re-ID, and color options.
- Run either real-time WebSocket tracking or file-based processing.
- Compare completed tracking runs in the built-in evaluation table.

### Face Recognition

- Choose a face model from the face-recognition page.
- Upload an image/video or select a built-in test asset.
- Add built-in reference faces and optional uploaded named samples.
- Start a background face run, poll its status, preview/download the annotated output, and compare completed runs in the face-run evaluation table.

---

## Tracker + Detector + Re-ID Matrix

| Framework | Tracker    | Detectors              | Re-ID Support | Re-ID Models                            |
| --------- | ---------- | ---------------------- | ------------- | --------------------------------------- |
| BOXMOT    | bytetrack  | yolov8/11/26 n/s/m/l/x, yolov5 n/s/m/l/x | ✗             | —                                       |
| BOXMOT    | botsort    | yolov8/11/26 n/s/m/l/x, yolov5 n/s/m/l/x | ✓             | osnet_x0_25, osnet_x1_0, resnet50, mlfn |
| BOXMOT    | ocsort     | yolov8/11/26 n/s/m/l/x, yolov5 n/s/m/l/x | ✗             | —                                       |
| BOXMOT    | deepocsort | yolov8/11/26 n/s/m/l/x, yolov5 n/s/m/l/x | ✓             | osnet_x0_25, osnet_x1_0, resnet50, mlfn |
| BOXMOT    | strongsort | yolov8/11/26 n/s/m/l/x, yolov5 n/s/m/l/x | ✓             | osnet_x0_25, osnet_x1_0, resnet50, mlfn |
| FairMOT   | fairmot    | dla34, hrnet, resnet50 | ✓ (built-in)  | Joint detection-embedding               |
| DeepSORT  | deepsort   | yolov8/11/26 n/s/m/l/x, yolov5 n/s/m/l/x | ✓             | osnet_x0_25, osnet_x1_0, resnet50       |

---

## Running Evaluations

### CLI (evaluate.py)

```bash
cd backend

# Evaluate all combinations on default test video (300 frames each)
python evaluate.py

# Full video, save annotated output videos
python evaluate.py --max-frames 0 --save-videos

# Filter to specific frameworks
python evaluate.py --frameworks boxmot deepsort

# Custom video and output
python evaluate.py --video /path/to/video.mp4 --output results/eval.json

# Custom confidence threshold
python evaluate.py --conf 0.5
```

Output:

- JSON results file with per-combination metrics
- Console table comparing all combinations
- Optional annotated MP4 files per combination

### Via API

```bash
# Start evaluation (uses test/test.mp4 by default)
curl -X POST "http://localhost:8000/api/evaluate"

# Poll for results
curl "http://localhost:8000/api/evaluate/{eval_id}"
```

---

## API Reference

### Tracking API

| Method | Endpoint                       | Description                             |
| ------ | ------------------------------ | --------------------------------------- |
| GET    | `/api/capabilities`            | Tracker/detector/reid capability matrix |
| POST   | `/api/upload`                  | Upload a video file                     |
| GET    | `/api/test-video`              | Get default test video info             |
| GET    | `/api/test-videos`             | List available test videos              |
| POST   | `/api/track/file`              | Start background tracking job           |
| GET    | `/api/track/status/{job_id}`   | Poll job status + progress              |
| GET    | `/api/track/download/{job_id}` | Download output video                   |
| GET    | `/api/track/metrics/{job_id}`  | Get tracking metrics JSON               |
| WS     | `/ws/track/{session_id}`       | Real-time WebSocket tracking stream     |
| POST   | `/api/track/stop/{session_id}` | Stop a live session                     |
| POST   | `/api/evaluate`                | Run batch evaluation                    |
| GET    | `/api/evaluate/{eval_id}`      | Get evaluation results                  |

### Face Recognition API

| Method | Endpoint                            | Description                                         |
| ------ | ----------------------------------- | --------------------------------------------------- |
| GET    | `/api/face/models`                  | List available face-recognition models              |
| GET    | `/api/face/test-assets`             | List built-in media assets and reference faces      |
| POST   | `/api/face/upload`                  | Upload face-recognition target media                |
| POST   | `/api/face/run/file`                | Start a background face-recognition run             |
| GET    | `/api/face/run/status/{job_id}`     | Poll face run status, progress, stats, detections   |
| GET    | `/api/face/run/download/{job_id}`   | Download the annotated face-recognition output      |
| GET    | `/api/face/runs`                    | List previously created face-recognition runs       |

### Legacy Compatibility

- Legacy face routes are still mounted under `/face-recognition/...` for compatibility with older clients.
- New frontend code uses the normalized `/api/face/...` API shape so face jobs behave more like tracking jobs.

---

## Key Design Notes

- **Auto-install on first run**: If required ML libraries are missing, trackers attempt to install them automatically using the active Python environment, then retry initialization.
- **Re-ID gating**: The frontend disables Re-ID selection automatically when the chosen tracker doesn't support it (e.g. ByteTrack, OC-SORT).
- **FairMOT Re-ID**: FairMOT uses a built-in joint detection+embedding head — no external Re-ID model is selected.
- **WebSocket streaming**: Real-time mode sends JPEG frames over WebSocket at the source video's native FPS.
- **Job polling**: File mode uses 1-second polling against the `/api/track/status/{job_id}` endpoint.
- **Normalized face jobs**: Face recognition now follows the same general pattern as tracking: select/upload input, start a background file run, poll status, then open/download the result.
- **Face evaluation table**: The face-recognition page keeps a run table with metrics such as detected faces, identified faces, unknown faces, average match confidence, inference time, and gallery size.
- **Dev proxy**: Vite proxies both `/api` and `/face-recognition` to the backend, defaulting to `127.0.0.1:8000`. Optional overrides are available through `VITE_BACKEND_URL` and `VITE_BACKEND_WS_URL`.

---

## FairMOT Installation

FairMOT requires installing from source:

```bash
git clone https://github.com/ifzhang/FairMOT.git
cd FairMOT
pip install -r requirements.txt
pip install -e .
```

Or use the community PyPI fork:

```bash
pip install fairmot
```
