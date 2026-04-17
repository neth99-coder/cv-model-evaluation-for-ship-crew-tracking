# Person Tracking Evaluation

This module now supports pluggable detector and ReID backends across the tracking pipeline.

## What is included

- Custom tracker adapters in `trackers/` for `botsort`, `bytetrack`, and `deepsort`.
- A generic BoxMOT client in `trackers/boxmot_client.py` that can run any supported BoxMOT tracker backend against the same detector registry.
- A detector registry that reuses the existing models from `person_detection`:
  - `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
  - `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`
  - `faster_rcnn`
  - `ssd_mobilenet`
- An evaluation script in `evaluate.py` and a live preview script in `live_tracking.py`.

## Setup

```bash
pip install -r requirements.txt
```

## Tracker modes

- `botsort`: custom wrapper over the BoxMOT BoT-SORT tracker, fed by any detector from this repo.
- `bytetrack`: custom wrapper over the BoxMOT ByteTrack tracker, fed by any detector from this repo.
- `deepsort`: custom `deep-sort-realtime` wrapper with configurable embedders.
- `boxmot:<tracker>`: generic BoxMOT client for any installed BoxMOT tracker backend such as `strongsort`, `deepocsort`, or `ocsort`.

## ReID support

- `botsort` and ReID-capable `boxmot:<tracker>` modes accept `--reid` with either a BoxMOT ReID model name such as `osnet_x0_25_msmt17` or a weights path.
- `deepsort` accepts `--deepsort-embedder` with one of:
  - `mobilenet`
  - `torchreid`
  - `clip_RN50`
  - `clip_RN101`
  - `clip_RN50x4`
  - `clip_RN50x16`
  - `clip_ViT-B/32`
  - `clip_ViT-B/16`

## Evaluate combinations

```bash
# Default single combination
python evaluate.py

# Compare multiple custom trackers against one detector
python evaluate.py --trackers botsort bytetrack deepsort --detectors yolov8n

# Compare the default tracker suite against the default detector suite
python evaluate.py --trackers all --detectors all

# Run a generic BoxMOT backend with a repo detector
python evaluate.py --trackers boxmot:strongsort --detectors faster_rcnn --reid osnet_x0_25_msmt17

# Use DeepSORT with a different appearance encoder
python evaluate.py --trackers deepsort --detectors yolov5s --deepsort-embedder torchreid --deepsort-embedder-model osnet_x0_25
```

### Evaluation outputs

Each detector/tracker combination writes a folder under `results/` containing:

- `summary.json`
- `detailed_results.json`
- `per_video_results.csv`
- `annotated_videos/*.mp4`
- `tracks_txt/*.txt`

The root `results/` directory also gets:

- `comparison_report.md`
- `comparison_summary.csv`

The reported metrics are runtime-oriented:

- `avg_detection_ms`
- `avg_tracking_ms`
- `avg_inference_ms`
- `avg_fps`
- `avg_unique_tracks`
- `avg_detections`
- `avg_tracks`

## Live tracking

```bash
# Auto-pick the first test video
python live_tracking.py --tracker botsort --detector yolov8n

# Camera source
python live_tracking.py --source 0 --tracker deepsort --detector yolov8n

# Save annotated output and disable preview window
python live_tracking.py --tracker boxmot:strongsort --detector faster_rcnn --reid osnet_x0_25_msmt17 --save-out results/live_strongsort.mp4 --no-show
```

Controls:

- Press `q` or `Esc` to stop the live preview.

## Notes

- BoxMOT's own detector registry does not cover every detector used in this repo, so the BoxMOT client here feeds detections from the local detector registry into BoxMOT trackers directly.
- `bytetrack` does not use ReID.
- `frame_step`, `max_frames`, and `time_limit` are available in `evaluate.py` to bound long-running evaluations.
