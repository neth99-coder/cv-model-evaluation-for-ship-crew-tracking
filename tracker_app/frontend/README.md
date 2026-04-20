# TrackVision — Multi-Model Person Tracking System

A full-stack application for running and evaluating person tracking pipelines using BOXMOT, FairMOT, and DeepSORT with pluggable detectors and Re-ID models.

---

## Project Structure

```
tracker_app/
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── evaluate.py              # Standalone CLI evaluation script
│   ├── requirements.txt
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
│   │   └── test.mp4             # ← Place your test video here
│   ├── uploads/                 # Uploaded videos (auto-created)
│   └── outputs/                 # Job outputs + evaluation results (auto-created)
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
        │   ├── ModelSelector.jsx     # Framework/tracker/detector/reid selection
        │   ├── VideoUploader.jsx     # Drag-drop video upload
        │   ├── RealtimePlayer.jsx    # WebSocket live stream viewer
        │   ├── FileTrackingJob.jsx   # Background job submission + polling
        │   └── MetricsPanel.jsx      # FPS/track count metrics display
        └── pages/
            ├── ObjectTracking.jsx    # Main tracking page
            └── FaceDetection.jsx     # Face detection page
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
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev       # http://localhost:3000
```

---

## Tracker + Detector + Re-ID Matrix

| Framework | Tracker    | Detectors              | Re-ID Support | Re-ID Models                            |
| --------- | ---------- | ---------------------- | ------------- | --------------------------------------- |
| BOXMOT    | bytetrack  | yolov8n/s/m, yolov5n/s | ✗             | —                                       |
| BOXMOT    | botsort    | yolov8n/s/m, yolov5n/s | ✓             | osnet_x0_25, osnet_x1_0, resnet50, mlfn |
| BOXMOT    | ocsort     | yolov8n/s/m, yolov5n/s | ✗             | —                                       |
| BOXMOT    | deepocsort | yolov8n/s/m, yolov5n/s | ✓             | osnet_x0_25, osnet_x1_0, resnet50, mlfn |
| BOXMOT    | strongsort | yolov8n/s/m, yolov5n/s | ✓             | osnet_x0_25, osnet_x1_0, resnet50, mlfn |
| FairMOT   | fairmot    | dla34, hrnet, resnet50 | ✓ (built-in)  | Joint detection-embedding               |
| DeepSORT  | deepsort   | yolov8n/s/m, yolov5n/s | ✓             | osnet_x0_25, osnet_x1_0, resnet50       |

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

| Method | Endpoint                       | Description                             |
| ------ | ------------------------------ | --------------------------------------- |
| GET    | `/api/capabilities`            | Tracker/detector/reid capability matrix |
| POST   | `/api/upload`                  | Upload a video file                     |
| GET    | `/api/test-video`              | Get default test video info             |
| POST   | `/api/track/file`              | Start background tracking job           |
| GET    | `/api/track/status/{job_id}`   | Poll job status + progress              |
| GET    | `/api/track/download/{job_id}` | Download output video                   |
| GET    | `/api/track/metrics/{job_id}`  | Get tracking metrics JSON               |
| WS     | `/ws/track/{session_id}`       | Real-time WebSocket tracking stream     |
| POST   | `/api/track/stop/{session_id}` | Stop a live session                     |
| POST   | `/api/evaluate`                | Run batch evaluation                    |
| GET    | `/api/evaluate/{eval_id}`      | Get evaluation results                  |

---

## Key Design Notes

- **Auto-install on first run**: If required ML libraries are missing, trackers attempt to install them automatically using the active Python environment, then retry initialization.
- **Re-ID gating**: The frontend disables Re-ID selection automatically when the chosen tracker doesn't support it (e.g. ByteTrack, OC-SORT).
- **FairMOT Re-ID**: FairMOT uses a built-in joint detection+embedding head — no external Re-ID model is selected.
- **WebSocket streaming**: Real-time mode sends JPEG frames over WebSocket at the source video's native FPS.
- **Job polling**: File mode uses 1-second polling against the `/api/track/status/{job_id}` endpoint.

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
