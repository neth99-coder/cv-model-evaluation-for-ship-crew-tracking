# Person Tracking Model Evaluation

Evaluates free industrial person-tracking models on test videos and saves annotated outputs plus comparison reports.

## Trackers

| Tracker Key  | Label      | Availability    | Notes                                                      |
| ------------ | ---------- | --------------- | ---------------------------------------------------------- |
| `deepsort`   | DeepSORT   | BoxMOT + Custom | BoxMOT maps this to StrongSORT; custom adapter also exists |
| `bytetrack`  | ByteTrack  | BoxMOT + Custom | Fast and practical baseline                                |
| `botsort`    | BoT-SORT   | BoxMOT + Custom | Strong association quality                                 |
| `strongsort` | StrongSORT | BoxMOT only     | DeepSORT-family tracker with re-ID                         |
| `deepocsort` | DeepOCSORT | BoxMOT only     | SOTA-style ID robustness                                   |
| `ocsort`     | OCSORT     | BoxMOT only     | ReID-free motion/association tracker                       |
| `hybridsort` | HybridSORT | BoxMOT only     | Modern hybrid association                                  |
| `boosttrack` | BoostTrack | BoxMOT only     | Strong MOT17 performance                                   |
| `sfsort`     | SFSORT     | BoxMOT only     | Lightweight BoxMOT option                                  |

All trackers in this repo track the `person` class only.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Test-data layout

```text
person_tracking/test/
└── videos/
    ├── clip1.mp4
    ├── clip2.mp4
    └── ...
```

---

## Usage

```bash
# Evaluate all trackers
python main.py

# Evaluate selected trackers
python main.py --models deepsort bytetrack deepocsort

# Backend policy
python main.py --backend auto     # default: BoxMOT first, fallback to custom
python main.py --backend boxmot   # force BoxMOT only
python main.py --backend custom   # force custom only (deepsort/bytetrack/botsort)

# Detector spec / weights
python main.py --detector yolov8s.pt

# Bound runtime on heavy models
python main.py --time-limit 60

# Evaluate only first N frames/video
python main.py --max-frames 300
```

### CLI options

- `--models`: any tracker key listed above, or `all`
- `--backend`: `auto`, `boxmot`, `custom`
- `--detector`: detector spec/weights shared across backends (default: `yolov8n.pt`)
- `--conf`: detection confidence threshold
- `--imgsz`: detector image size
- `--max-frames`: frame cap per video
- `--time-limit`: seconds per model/video (`0` disables cap)

---

## Live tracking tool

Use the separate live runner for real-time playback on a video source:

```bash
# Auto source (first file in test/videos), auto backend
python live_tracking.py --model bytetrack

# Force custom backend
python live_tracking.py --model deepsort --backend custom

# Force BoxMOT backend
python live_tracking.py --model ocsort --backend boxmot

# Specific source, headless, save output
python live_tracking.py \
    --model deepocsort \
    --backend auto \
    --source /path/to/video.mp4 \
    --no-show \
    --max-frames 300 \
    --save-out results/live_deepocsort.mp4
```

Live CLI options:

- `--model`: one tracker key (same set as `main.py`)
- `--backend`: `auto`, `boxmot`, `custom`
- `--source`: video path or webcam index (example: `0`)
- `--detector`: detector spec/weights
- `--conf`, `--imgsz`: detector parameters
- `--max-frames`, `--time-limit`: runtime bounds
- `--save-out`: save annotated output video
- `--no-show`: disable UI window

When the UI window is enabled, press `q` to quit.

---

## Output layout

```text
person_tracking/results/
├── comparison_report.md
├── comparison_summary.csv
├── DeepSORT/
│   ├── summary.json
│   ├── detailed_results.json
│   ├── per_video_results.csv
│   └── annotated_videos/
│       └── clip1_tracked.mp4
├── ByteTrack/
│   └── ...
└── BoT-SORT/
    └── ...
```

---

## Reported metrics

Without identity ground truth, this project reports practical proxy metrics for production benchmarking:

- `avg_fps`
- `avg_inference_ms`
- `avg_unique_tracks`

Interpretation:

- Higher `avg_fps` is faster.
- Higher `avg_unique_tracks` often indicates more tracked identities detected.

---

## Notes

- First run may download YOLO weights from Ultralytics.
- `auto` mode uses BoxMOT when a model combination is available and falls back to custom otherwise.
- For non-YOLO detector experimentation, prefer `--backend boxmot` first.
- Custom backend currently supports `deepsort`, `bytetrack`, and `botsort`.
- On macOS, PyTorch detection/tracking typically runs CPU or mixed backend depending on available ops.
