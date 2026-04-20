# Cloth Colour Detection

Real-time clothing colour identification from video or images.
Handles the three hard challenges out of the box:

| Challenge | Solution |
|-----------|----------|
| **Shadows** (blue jumpsuit looks black) | Cluster in **HS space** — brightness (V) is ignored |
| **Small / distant people** | **SAHI** (Slicing Aided Hyper Inference) tiles the frame |
| **"K-Means gives numbers"** | **KD-Tree** on a curated 60-colour clothing palette → human names |

---

## Project layout

```
cloth_color_detection/
├── run_detection.py   ← run on a single video / image / webcam
├── evaluate.py        ← benchmark & compare multiple segmenters
├── pipeline.py        ← core pipeline (infer + draw)
├── segmenters.py      ← pluggable segmentation back-ends
├── color_utils.py     ← HSV clustering + colour naming
├── body_regions.py    ← upper / lower body mask split
├── requirements.txt
├── test/              ← put your test videos here
└── output/            ← results written here
```

---

## Quick start

```bash
pip install -r requirements.txt

# Put your test video in test/
python run_detection.py --input test/video.mp4
```

---

## Segmentation back-ends

| Name | Description | Requires |
|------|-------------|----------|
| `grabcut` | YOLO detect → GrabCut pixel mask | OpenCV only (YOLO optional) |
| `yolov8n-seg` | YOLOv8-nano segmentation | `ultralytics` + torch |
| `yolov8s-seg` | YOLOv8-small segmentation | `ultralytics` + torch |
| `yolov8m-seg` | YOLOv8-medium segmentation | `ultralytics` + torch |
| `yolov9-seg` | YOLOv9c segmentation | `ultralytics` + torch |
| `mediapipe` | MediaPipe selfie segmentation | `mediapipe` |
| `bgsub` | MOG2 background subtraction | OpenCV only (static cameras) |

Add a new back-end in **segmenters.py** by subclassing `BaseSegmenter` and
adding it to `REGISTRY`.

---

## run_detection.py — all options

```bash
python run_detection.py --input test/video.mp4 --model grabcut
python run_detection.py --input test/photo.jpg --model yolov8n-seg --sahi
python run_detection.py --webcam --model mediapipe --show

Options:
  --model        {grabcut,yolov8n-seg,yolov8s-seg,yolov8m-seg,yolov9-seg,mediapipe,bgsub}
  --output       Output file path (auto if omitted)
  --show         Open live preview window
  --sahi         Enable SAHI for small/distant people
  --sahi-size    H W slice dimensions  [default: 320 320]
  --sahi-overlap Overlap fraction       [default: 0.2]
  --n-colors     Dominant colours per body region [default: 3]
  --rgb-cluster  Cluster in RGB (disables shadow fix)
  --skip         Process every N frames [default: 2]
  --max-frames   Cap total frames processed
  --scale        Output display scale   [default: 1.0]
```

---

## evaluate.py — compare all models

```bash
# Compare all available models
python evaluate.py --input test/video.mp4

# Compare specific models
python evaluate.py --input test/video.mp4 --models grabcut mediapipe bgsub

# With SAHI enabled
python evaluate.py --input test/video.mp4 --sahi --max-frames 60
```

**Outputs:**
- `output/eval_report.csv` — per-model metrics table
- `output/eval_comparison_grid.jpg` — side-by-side frame grid

**Metrics reported:**
- `avg_infer_ms` — mean inference time per frame
- `avg_total_ms` — inference + colour extraction
- `detection_rate` — fraction of frames with ≥1 person found
- `avg_mask_coverage` — how much of the bounding box is masked (quality proxy)
- `avg_conf` — mean detection confidence
- `top_colors` — five most frequently detected colour names

---

## How the colour extraction works

```
Frame
  └─► Segmenter → binary mask (0/255)
        └─► Body split (upper 50% / lower 50% of bbox)
              └─► Convert ROI pixels to HSV, extract only H and S channels
                    (V/brightness is DROPPED → shadows don't change the cluster)
              └─► K-Means on (H_sin, H_cos, S) encoding (circular hue)
              └─► Mean RGB per cluster → KD-Tree lookup → colour name
```

### Why HS-only clustering?
A navy jumpsuit in full sun reads ~(0, 0, 128) RGB.
In shadow it reads ~(0, 0, 40) RGB.
In **HS space** both points have the same hue and saturation; only V differs.
By dropping V we keep them in the same cluster.

### Why circular hue encoding?
Hue 0° (red) and 179° (also red) are numerically far apart.
We encode hue as `(sin(h), cos(h))` before feeding K-Means so the geometry
is correct and red shades cluster together.

---

## Adding a new segmenter

```python
# segmenters.py

class MyCustomSegmenter(BaseSegmenter):
    name = "my-model"

    def load(self):
        self._model = load_my_model()

    def segment(self, frame: np.ndarray) -> List[PersonDetection]:
        dets = self._model(frame)
        return [
            PersonDetection(
                bbox   = (x1, y1, x2, y2),
                mask   = binary_mask_uint8,   # same H×W as frame
                score  = confidence_float,
                source = self.name,
            )
            for ... in dets
        ]

# Register it
REGISTRY["my-model"] = MyCustomSegmenter
```

That's it — it will automatically appear in `run_detection.py --model my-model`
and be benchmarked by `evaluate.py`.
