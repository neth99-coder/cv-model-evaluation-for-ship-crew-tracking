# Person Detection Model Evaluation

Evaluates four COCO-pretrained person detection models on test videos and produces annotated output videos, per-model JSON/CSV results, and a cross-model comparison report.

## Models

| Model                         | Backend     | Notes                            |
| ----------------------------- | ----------- | -------------------------------- |
| YOLOv8 (nano–xlarge)          | Ultralytics | Fastest inference; good accuracy |
| YOLOv5 (small–xlarge)         | Ultralytics | Lightweight alternative to v8    |
| Faster R-CNN ResNet-50 FPN v2 | torchvision | Highest accuracy; slower         |
| SSDLite320 MobileNetV3        | torchvision | Very fast; lower accuracy        |

All models are COCO-pretrained and detect the **person** class only.

---

## Setup

```bash
pip install -r requirements.txt
```

> **macOS / CPU-only:** `torch` and `torchvision` will run on CPU automatically. GPU is used automatically when available.

---

## Test-data layout

```
person_detection/test/
├── videos/                  ← video files (.mp4, .avi, .mov, .mkv, …)
│   ├── clip1.mp4
│   └── clip2.avi
└── ground_truth.json        ← OPTIONAL – enables mAP@50 computation
```

### Ground-truth format (optional)

```json
{
  "clip1.mp4": {
    "0": [[120, 80, 300, 450]],
    "30": [
      [120, 80, 300, 450],
      [400, 100, 560, 420]
    ],
    "60": []
  },
  "clip2.avi": {
    "0": [[50, 30, 200, 380]]
  }
}
```

Keys are **frame indices** (integers, as strings). Each value is a list of `[x1, y1, x2, y2]` bounding boxes. Frames omitted from the JSON are treated as containing zero persons.

Without `ground_truth.json`, only FPS and detection-count metrics are reported.

---

## Usage

```bash
# Evaluate all four models (every frame, default confidence = 0.5)
python main.py

# Keep heavy models bounded (default is already 60s per model/video)
python main.py --time-limit 60

# Evaluate specific models
python main.py --models yolov8 faster_rcnn

# Process every 5th frame (faster)
python main.py --frame-step 5

# Limit to first 500 frames per video
python main.py --max-frames 500

# Custom confidence threshold
python main.py --confidence 0.4

# Choose model size variants
python main.py --yolov8-size s --yolov5-size m

# Custom paths
python main.py --test-dir /path/to/videos --results-dir /path/to/output

# Disable time limit (run until all frames are done)
python main.py --time-limit 0
```

### Key runtime options

- `--frame-step N`: run inference every Nth frame; intermediate frames are still written with overlays.
- `--max-frames N`: stop after N evaluated frames (output video is intentionally shortened).
- `--time-limit SECONDS`: wall-clock budget per model per video (default: `60`). Use `0` to disable.

---

## Live detection script

A separate script is available for live object detection playback on a test video:

```bash
# Auto-picks first video from test/videos
python live_object_detection.py --model yolov8

# Choose a different backend (same model family names as main.py)
python live_object_detection.py --model yolov5 --yolov5-size m
python live_object_detection.py --model faster_rcnn
python live_object_detection.py --model ssd_mobilenet

# Use a specific video and save annotated output
python live_object_detection.py --source test/videos/test_1.mp4 --save-out results/live_test_1.mp4

# Headless run (no display window)
python live_object_detection.py --no-show --max-frames 300 --save-out results/live_preview.mp4
```

Controls:

- Press `q` to quit when display is enabled.
- For YOLO models, you can set variants via `--yolov8-size` or `--yolov5-size`.

---

## Results layout

```
person_detection/results/
├── comparison_report.md         ← cross-model Markdown table
├── comparison_summary.csv       ← cross-model CSV
├── live_test_1.mp4              ← optional output from live_object_detection.py
├── YOLOv8N/
│   ├── summary.json
│   ├── detailed_results.json
│   ├── per_video_results.csv
│   └── annotated_videos/
│       └── clip1_annotated.mp4
├── YOLOv5S/
│   └── …
├── FasterRCNN_ResNet50_FPN_v2/
│   └── …
└── SSDLite320_MobileNetV3/
    └── …
```

### Metrics reported

| Metric                     | Description                             | Requires GT? |
| -------------------------- | --------------------------------------- | ------------ |
| `avg_fps`                  | Model inference throughput (frames/sec) | No           |
| `avg_inference_ms`         | Mean inference time per frame (ms)      | No           |
| `avg_detections_per_frame` | Mean person count per sampled frame     | No           |
| `total_detections`         | Sum across all frames and videos        | No           |
| `mAP50`                    | Mean Average Precision @ IoU=0.5        | **Yes**      |

---

## Notes

- Annotated videos draw **green boxes** for model predictions and **blue boxes** for ground-truth (when available).
- The `--frame-step N` flag samples every Nth frame for inference while keeping overlays visible between sampled frames.
- The default `--time-limit` is `60` seconds per model/video to avoid very long runs on heavy models like Faster R-CNN.
- `--max-frames` and `--time-limit` both produce shorter output videos by design.
- YOLOv5/YOLOv8 weights are downloaded from Ultralytics on first run and cached locally.
- Faster R-CNN and SSD weights are downloaded by torchvision and cached at `~/.cache/torch/`.
