# Face Recognition Model Evaluation

Evaluates four face recognition models against a local dataset of ship crew members and reports accuracy, F1, precision, recall, and inference speed per model.

Supports both **single-person** and **group (multi-face)** test images — a photo of 10 people on deck can be passed as a single test image and the system will identify each person in it.

## Models

| Model                    | Backend                                | Notes                                      |
| ------------------------ | -------------------------------------- | ------------------------------------------ |
| **AuraFace (ResNet100)** | InsightFace `antelopev2` / `glintr100` | Best accuracy; GPU recommended             |
| **MobileFaceNet**        | InsightFace `buffalo_s` / `w600k_mbf`  | Lightweight; fast on CPU                   |
| **FaceNet (Sandberg)**   | DeepFace `Facenet512`                  | 512-d embeddings; auto-downloads weights   |
| **CompreFace**           | REST API (Docker)                      | No local model; requires running container |

---

## Project Structure

```
face_recognition/
├── main.py              ← Entry point
├── evaluate.py          ← Enrollment, recognition, and metrics logic
├── requirements.txt     ← Python dependencies
├── README.md
├── models/
│   ├── base.py          ← Abstract base class (cosine similarity matching)
│   ├── auraface.py      ← AuraFace / ResNet100
│   ├── mobilefacenet.py ← MobileFaceNet
│   ├── facenet_model.py ← FaceNet (Sandberg / Facenet512)
│   └── compreface.py    ← CompreFace REST client
├── test/                ← Your test dataset (create this)
│   ├── <person_name>/   ← One folder per person
│   │   ├── photo_1.jpg  ← One or more reference images per person (different angles/lighting)
│   │   └── photo_2.jpg
│   └── test_images/     ← Images to identify (single or group photos)
│       ├── alice_1.jpg
│       ├── group_deck.jpg
│       └── ground_truth.json  ← required for group photos
└── results/             ← Auto-created; one sub-folder per model
    ├── AuraFace_ResNet100/
    │   ├── detailed_results.json
    │   ├── summary.json
    │   └── per_image_results.csv
    ├── comparison_report.md
    └── comparison_summary.csv
```

---

## Test Data Layout

```
test/
├── alice/
│   ├── alice_front.jpg      ← multiple reference images per person are supported
│   ├── alice_side.jpg
│   └── alice_lowlight.jpg
├── bob/
│   ├── bob_front.jpg
│   └── bob_hat.jpg
├── carol/
│   ├── carol_front.jpg
│   └── carol_profile.jpg
└── test_images/
    ├── alice_outdoor.jpg    ← single-person: prefix before _ = ground truth
    ├── bob_hat.jpg
    ├── group_deck.jpg       ← group photo: list all people in ground_truth.json
    └── ground_truth.json    ← recommended; required for group photos
```

### Ground Truth

Ground truth is inferred automatically from the image filename — the text **before the first `_`, `-`, or space** is used as the person name. This works for single-person images only.

| Filename            | Inferred name |
| ------------------- | ------------- |
| `alice_outdoor.jpg` | `alice`       |
| `bob-side.jpg`      | `bob`         |
| `carol.jpg`         | `carol`       |

For group photos or non-standard filenames, place a `ground_truth.json` in `test_images/`. Values can be a single name (string) or a list of names:

```json
{
  "alice_outdoor.jpg": "alice",
  "crew01.jpg": "bob",
  "group_deck.jpg": ["alice", "bob", "carol"],
  "crew_meeting.jpg": ["alice", "dave"]
}
```

---

## Setup

### 1. Install Python dependencies

```bash
cd face_recognition
pip install -r requirements.txt
```

For GPU acceleration, replace `onnxruntime` with `onnxruntime-gpu`:

```bash
pip install onnxruntime-gpu
```

### 2. CompreFace (optional)

CompreFace runs as a Docker container. Start it before running the evaluation:

```bash
docker run -p 8000:8000 exadel/compreface
```

Then create a **Recognition** service in the CompreFace web UI (`http://localhost:8000`) and copy the API key.
The key must be a UUID for that Recognition service.

---

## Usage

Run from inside the `face_recognition/` directory:

```bash
# Evaluate all models
python main.py

# Evaluate specific models only
python main.py --models auraface mobilefacenet

# Custom test and results directories
python main.py --test-dir /data/test --results-dir /data/results

# CompreFace with API key
python main.py --models compreface --compreface-key YOUR_API_KEY

# Or use an environment variable
export COMPREFACE_API_KEY=YOUR_API_KEY
python main.py --models compreface

# Or set it in face_recognition/.env (loaded automatically by main.py)
# COMPREFACE_API_KEY=YOUR_API_KEY
python main.py --models compreface

# CompreFace on a different host/port
python main.py --models compreface \
    --compreface-host http://192.168.1.10 \
    --compreface-port 8080 \
    --compreface-key YOUR_API_KEY
```

### Available model names

| Flag value      | Model                |
| --------------- | -------------------- |
| `auraface`      | AuraFace (ResNet100) |
| `mobilefacenet` | MobileFaceNet        |
| `facenet`       | FaceNet (Sandberg)   |
| `compreface`    | CompreFace (REST)    |
| `all`           | All models (default) |

---

## Output

Results are saved to `results/<ModelName>/` for each model:

| File                    | Description                                                                      |
| ----------------------- | -------------------------------------------------------------------------------- |
| `summary.json`          | Accuracy, macro F1/precision/recall, timing stats                                |
| `detailed_results.json` | Full per-image breakdown + summary                                               |
| `per_image_results.csv` | Spreadsheet-friendly per-image predictions                                       |
| `annotated_images/`     | One output image per test image with face boxes and labels (`name` or `unknown`) |

Comparison files are written to `results/`:

| File                     | Description                              |
| ------------------------ | ---------------------------------------- |
| `comparison_report.md`   | Markdown table comparing all models      |
| `comparison_summary.csv` | One row per model; easy to open in Excel |

### Example summary output

```
============================================================
  Model : AuraFace_ResNet100
============================================================
Enrolling reference faces…
  [OK ] alice ← alice.jpg
  [OK ] bob   ← bob.jpg
  [OK ] carol ← carol.jpg

Running recognition on 4 image(s)…
  ✓  alice_outdoor.jpg    expected=['alice']         got=['alice']         TP=1 FP=0 FN=0  43.2ms
  ✓  bob_hat.jpg          expected=['bob']           got=['bob']           TP=1 FP=0 FN=0  41.8ms
  ✓  group_deck.jpg       expected=['alice','bob']   got=['alice','bob']   TP=2 FP=0 FN=0  89.4ms
  ✗  crew_meeting.jpg     expected=['alice','carol'] got=['alice']         TP=1 FP=0 FN=1  86.1ms

  Accuracy : 75.00%
  Macro F1 : 0.8750
  Avg time : 65.1 ms/image
```

**Image-level accuracy** counts an image as correct only when the predicted set of people exactly matches the ground-truth set. Per-person TP/FP/FN are accumulated across all images to compute precision, recall, and F1.

---

## Metrics Explained

- **Accuracy** — percentage of test images where the predicted set of people exactly matches the ground-truth set
- **TP / FP / FN** — counted per person across all images:
  - **TP**: person was present and correctly identified
  - **FP**: person was predicted but not actually in the image
  - **FN**: person was present but not detected / recognised
- **Macro Precision / Recall / F1** — per-person metrics averaged across all enrolled persons
- **Inference time** — time to detect all faces + match each against the database (ms per image)
- **Threshold** — cosine similarity cut-off below which a face is labelled `unknown`

| Model         | Default threshold              |
| ------------- | ------------------------------ |
| AuraFace      | 0.35                           |
| MobileFaceNet | 0.35                           |
| FaceNet       | 0.40                           |
| CompreFace    | 0.80 (native similarity scale) |

---

## Troubleshooting

**`No face detected` warnings during enrollment**

- Ensure the reference image is front-facing and well-lit.
- Try a higher-resolution image.
- Add more reference images per person with varied angles/lighting so matching is more robust.

**Group photo — some faces not detected**

- Use a high-resolution image so small faces are visible.
- InsightFace's `det_size=(640, 640)` handles groups well; very large images may need tiling.
- For FaceNet, make sure faces are at least ~80 px tall for the OpenCV detector.

**All people in a group photo return `unknown`**

- Confirm enrollment images are correct and high quality.
- Try lowering the cosine threshold (e.g. `threshold = 0.28`) in the relevant model file.

**InsightFace model download fails**

- Models are downloaded automatically to `~/.insightface/models/`.
- Ensure you have an internet connection on first run.

**CompreFace returns 401 / 403**

- Double-check the API key matches the Recognition service in the CompreFace UI.

**CompreFace says `Service API key should be UUID`**

- The key being sent is empty or not a valid UUID.
- Pass `--compreface-key <UUID>` or set `COMPREFACE_API_KEY`.
- Make sure you copied the key from a **Recognition** service (not another service type).

**DeepFace / FaceNet slow on first run**

- Weights are downloaded once to `~/.deepface/weights/`. Subsequent runs are faster.
