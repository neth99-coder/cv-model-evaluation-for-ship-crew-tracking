"""
YOLOv8 Multi-Tracker Detection with Webcam (Front Camera)
==========================================================
Trackers (all supported by ultralytics built-in):
  1. ByteTrack  — fast, IoU-based,  No Re-ID
  2. BoT-SORT   — Kalman + GMC,     No Re-ID
  3. BoT-SORT   — Kalman + GMC,     With Re-ID  (OSNet)
  4. ByteTrack  — aggressive config (low thresh, long buffer)

NOTE: StrongSORT was removed because older ultralytics versions
      only ship 'bytetrack' and 'botsort' tracker backends.
      To get StrongSORT, run:  pip install -U ultralytics

Requirements:
    pip install -U ultralytics opencv-python

Controls:
    1-4  → switch tracker   q → quit
"""

import os
import cv2
from ultralytics import YOLO
import ultralytics

# ──────────────────────────────────────────────────────────────────────────────
# Detect whether StrongSORT is available in this ultralytics install
# ──────────────────────────────────────────────────────────────────────────────

try:
    from ultralytics.trackers.track import TRACKER_MAP
    HAS_STRONGSORT = "strongsort" in TRACKER_MAP
except Exception:
    HAS_STRONGSORT = False

print(f"ultralytics {ultralytics.__version__}  |  StrongSORT available: {HAS_STRONGSORT}")

# ──────────────────────────────────────────────────────────────────────────────
# Write YAML configs
# ──────────────────────────────────────────────────────────────────────────────

YAML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracker_cfgs")
os.makedirs(YAML_DIR, exist_ok=True)

# Tracker 1 — ByteTrack standard
BT_STD = """\
tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 30
match_thresh: 0.8
fuse_score: True
"""

# Tracker 2 — BoT-SORT, no Re-ID
BOT_NOID = """\
tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 30
match_thresh: 0.7
fuse_score: True
gmc_method: sparseOptFlow
with_reid: False
proximity_thresh: 0.5
appearance_thresh: 0.25
"""

# Tracker 3 — BoT-SORT with Re-ID
BOT_REID = """\
tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 30
match_thresh: 0.7
fuse_score: True
gmc_method: sparseOptFlow
with_reid: True
model_weights: osnet_x0_25_msmt17.pt
proximity_thresh: 0.5
appearance_thresh: 0.25
"""

# Tracker 4a — ByteTrack aggressive (recovers lost tracks faster)
BT_AGG = """\
tracker_type: bytetrack
track_high_thresh: 0.35
track_low_thresh: 0.05
new_track_thresh: 0.4
track_buffer: 60
match_thresh: 0.9
fuse_score: True
"""

# Tracker 4b — StrongSORT (only written if available)
STRONG = """\
tracker_type: strongsort
model_weights: osnet_x0_25_msmt17.pt
device: ""
fp16: False
per_class: False
max_dist: 0.2
max_iou_dist: 0.7
max_age: 30
n_init: 3
nn_budget: 100
mc_lambda: 0.995
ema_alpha: 0.9
"""

yaml_map = {
    "bytetrack_std.yaml":  BT_STD,
    "botsort_noid.yaml":   BOT_NOID,
    "botsort_reid.yaml":   BOT_REID,
    "bytetrack_agg.yaml":  BT_AGG,
    "strongsort.yaml":     STRONG,
}
for fname, content in yaml_map.items():
    with open(os.path.join(YAML_DIR, fname), "w") as f:
        f.write(content)

# ──────────────────────────────────────────────────────────────────────────────
# Tracker registry — swap slot 4 based on availability
# ──────────────────────────────────────────────────────────────────────────────

if HAS_STRONGSORT:
    slot4 = {
        "name":  "StrongSORT  (With Re-ID)",
        "yaml":  os.path.join(YAML_DIR, "strongsort.yaml"),
        "color": (180, 0, 255),
    }
else:
    slot4 = {
        "name":  "ByteTrack-Aggressive (No Re-ID)",
        "yaml":  os.path.join(YAML_DIR, "bytetrack_agg.yaml"),
        "color": (180, 0, 255),
    }

TRACKERS = {
    "1": {"name": "ByteTrack       (No Re-ID)",   "yaml": os.path.join(YAML_DIR, "bytetrack_std.yaml"), "color": (0, 200, 255)},
    "2": {"name": "BoT-SORT        (No Re-ID)",   "yaml": os.path.join(YAML_DIR, "botsort_noid.yaml"),  "color": (0, 255, 100)},
    "3": {"name": "BoT-SORT        (With Re-ID)", "yaml": os.path.join(YAML_DIR, "botsort_reid.yaml"),  "color": (255, 140,  0)},
    "4": slot4,
}

# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_box(frame, box, track_id, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"ID:{track_id}  {label}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def draw_hud(frame, tracker_info, fps, n_tracks):
    w = frame.shape[1]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 54), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"Tracker: {tracker_info['name']}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, tracker_info["color"], 2)
    cv2.putText(frame,
                f"FPS:{fps:.1f}  Tracks:{n_tracks}  "
                "[1]ByteTrack [2]BoT-SORT [3]BoT-SORT+ReID [4]"
                + ("StrongSORT" if HAS_STRONGSORT else "ByteTrack-Agg")
                + "  [q]Quit",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading YOLOv8n model ...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    active_key = "1"
    prev_key   = None
    timer      = cv2.TickMeter()

    print("Controls:  [1] ByteTrack  [2] BoT-SORT  [3] BoT-SORT+ReID  "
          f"[4] {'StrongSORT' if HAS_STRONGSORT else 'ByteTrack-Aggressive'}  [q] Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed - retrying ...")
            continue

        cfg = TRACKERS[active_key]

        if active_key != prev_key:
            model.predictor = None
            prev_key = active_key

        timer.reset()
        timer.start()

        try:
            results = model.track(
                source=frame,
                tracker=cfg["yaml"],
                persist=True,
                verbose=False,
                conf=0.35,
                iou=0.45,
                classes=[0],   # person only — remove to track all classes
            )
        except Exception as e:
            print(f"Tracker error: {e}")
            model.predictor = None
            continue

        timer.stop()
        fps = 1.0 / max(timer.getTimeSec(), 1e-6)

        n_tracks = 0
        r = results[0]
        if r.boxes is not None and r.boxes.id is not None:
            for box, tid, cls, conf in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.id.cpu().numpy().astype(int),
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy(),
            ):
                draw_box(frame, box, tid, f"{model.names[cls]} {conf:.2f}", cfg["color"])
                n_tracks += 1

        draw_hud(frame, cfg, fps, n_tracks)
        cv2.imshow("YOLOv8 Multi-Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        k = chr(key)
        if k in TRACKERS and k != active_key:
            active_key = k
            print(f"  -> Switched to {TRACKERS[k]['name']}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()