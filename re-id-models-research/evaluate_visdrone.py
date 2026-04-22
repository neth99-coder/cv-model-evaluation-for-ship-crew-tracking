"""
╔══════════════════════════════════════════════════════════════════════════╗
║         VisDrone — Model Evaluation & Real-Time Inference               ║
║         Models : YOLOv11n  |  YOLOv8n  (pretrained on VisDrone)        ║
║                                                                          ║
║  Folder layout expected:                                                 ║
║    pretrained_models/                                                    ║
║        yolo11n_visdrone.pt                                               ║
║        yolov8n_visdrone.pt                                               ║
║    test/                                                                 ║
║        images/   ← .jpg / .png test images                              ║
║        labels/   ← .txt YOLO-format ground-truth (optional)             ║
║        videos/   ← .mp4 / .avi test videos (optional)                  ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage
-----
# Evaluate on test images
python evaluate_visdrone.py --mode eval

# Run on a video file
python evaluate_visdrone.py --mode video --source test/videos/drone1.mp4

# Real-time webcam
python evaluate_visdrone.py --mode webcam --cam 0

# All modes in sequence
python evaluate_visdrone.py --mode all --source test/videos/drone1.mp4

# Compare both models side-by-side on test images
python evaluate_visdrone.py --mode compare
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")          # headless-safe; change to "TkAgg" if you have a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

# ── Try importing ultralytics ──────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("❌  ultralytics not installed.  Run:  pip install ultralytics")

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_COLOURS = {
    "pedestrian"     : (255, 87,  34),
    "people"         : (255, 193,   7),
    "bicycle"        : (76,  175,  80),
    "car"            : (33,  150, 243),
    "van"            : (156,  39, 176),
    "truck"          : (0,   188, 212),
    "tricycle"       : (255,  64,  64),
    "awning-tricycle": (121,  85,  72),
    "bus"            : (63,   81, 181),
    "motor"          : (0,   150, 136),
}

MODELS_CFG = {
    "YOLOv26n": "pretrained_models/yolov26n_visdrone.pt",
    "YOLOv8n" : "pretrained_models/yolov8n_visdrone.pt",
    "YOLOv8n-CROWDHUMAN" : "pretrained_models/yolov8n_crowdhuman.pt",
}

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_model(path: str, name: str = "") -> YOLO:
    p = Path(path)
    if not p.exists():
        sys.exit(f"❌  Model not found: {p}\n"
                 f"    Place your .pt file at  {p.resolve()}")
    print(f"  Loading {name or p.name} …")
    model = YOLO(str(p))
    return model


def colour_for(class_name: str):
    """Return BGR colour tuple for a class name."""
    rgb = CLASS_COLOURS.get(class_name, (200, 200, 200))
    return (rgb[2], rgb[1], rgb[0])          # RGB → BGR for OpenCV


def draw_detections(frame: np.ndarray, results, conf_thresh: float = 0.5) -> np.ndarray:
    """Draw bounding boxes + labels on a frame in-place. Returns frame."""
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        cls_idx = int(box.cls[0])
        label   = CLASS_NAMES[cls_idx] if cls_idx < NUM_CLASSES else str(cls_idx)
        colour  = colour_for(label)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label background
        text        = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def overlay_info(frame: np.ndarray, model_name: str, fps: float,
                 n_dets: int, h_off: int = 0) -> np.ndarray:
    """Overlay model name, FPS, detection count."""
    y = 28 + h_off
    cv2.putText(frame, f"{model_name}   FPS:{fps:.1f}   Dets:{n_dets}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  1.  EVALUATION on test images
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_models(
    test_img_dir : str  = "test/images",
    imgsz        : int  = 640,
    conf         : float= 0.5,
    iou          : float= 0.45,
    save_dir     : str  = "eval_results",
    max_images   : int  = None,
):
    """
    Run both models on all test images, collect metrics, save plots.
    If test/labels/ exists → compute mAP with val().
    Otherwise → inference-only statistics.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    img_dir  = Path(test_img_dir)
    lbl_dir  = img_dir.parent / "labels"
    has_gt   = lbl_dir.exists() and any(lbl_dir.glob("*.txt"))

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not images:
        print(f"⚠️   No images found in {img_dir}")
        return
    if max_images:
        images = images[:max_images]

    print(f"\n{'='*60}")
    print(f"  Evaluation — {len(images)} images  |  GT labels: {has_gt}")
    print(f"{'='*60}")

    all_stats = {}

    for model_name, model_path in MODELS_CFG.items():
        print(f"\n── {model_name} ──────────────────────────────────────────")
        model = load_model(model_path, model_name)

        # ── mAP via val() if ground-truth labels exist ──────────────────
        if has_gt:
            # Build a minimal YAML for val()
            yaml_path = Path(save_dir) / "test_data.yaml"
            yaml_content = (
                f"path: {img_dir.parent.resolve()}\n"
                f"test: images\n"
                f"nc: {NUM_CLASSES}\n"
                f"names: {CLASS_NAMES}\n"
            )
            yaml_path.write_text(yaml_content)

            metrics = model.val(
                data     = str(yaml_path),
                split    = "test",
                imgsz    = imgsz,
                conf     = conf,
                iou      = iou,
                save_json= True,
                project  = save_dir,
                name     = model_name,
                exist_ok = True,
                verbose  = False,
            )
            map50    = float(metrics.box.map50)
            map5095  = float(metrics.box.map)
            prec     = float(metrics.box.mp)
            rec      = float(metrics.box.mr)
            ap_cls   = metrics.box.ap50          # per-class AP@0.5
            ap_idxs  = metrics.box.ap_class_index

            print(f"  mAP@0.5      : {map50:.4f}")
            print(f"  mAP@0.5-0.95 : {map5095:.4f}")
            print(f"  Precision    : {prec:.4f}")
            print(f"  Recall       : {rec:.4f}")

            all_stats[model_name] = {
                "mAP@0.5"    : map50,
                "mAP@0.5-95" : map5095,
                "Precision"  : prec,
                "Recall"     : rec,
                "AP_per_class": {CLASS_NAMES[i]: float(ap_cls[j])
                                 for j, i in enumerate(ap_idxs)
                                 if i < NUM_CLASSES},
            }
        else:
            # ── Inference-only stats ─────────────────────────────────────
            latencies, det_counts, cls_counts = [], [], {}

            for img_path in images:
                t0 = time.perf_counter()
                res = model(str(img_path), imgsz=imgsz, conf=conf,
                            iou=iou, verbose=False)[0]
                latencies.append((time.perf_counter() - t0) * 1000)
                det_counts.append(len(res.boxes))
                for box in res.boxes:
                    cn = CLASS_NAMES[int(box.cls[0])]
                    cls_counts[cn] = cls_counts.get(cn, 0) + 1

            lat  = np.array(latencies)
            mean_ms = lat.mean()
            fps_avg = 1000 / mean_ms
            print(f"  Avg latency  : {mean_ms:.1f} ms  ({fps_avg:.1f} FPS)")
            print(f"  Avg dets/img : {np.mean(det_counts):.1f}")
            print(f"  Total dets   : {sum(det_counts)}")
            print(f"  Class breakdown:")
            for cn, cnt in sorted(cls_counts.items(), key=lambda x: -x[1]):
                print(f"    {cn:20s}: {cnt:6d}")

            all_stats[model_name] = {
                "avg_latency_ms" : float(mean_ms),
                "avg_fps"        : float(fps_avg),
                "avg_dets_per_img": float(np.mean(det_counts)),
                "total_dets"     : int(sum(det_counts)),
                "class_counts"   : cls_counts,
            }

    # ── Save stats JSON ──────────────────────────────────────────────────
    out_json = Path(save_dir) / "evaluation_stats.json"
    with open(out_json, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Saved stats → {out_json}")

    # ── Plots ────────────────────────────────────────────────────────────
    _plot_eval_results(all_stats, save_dir, has_gt)

    return all_stats


def _plot_eval_results(stats: dict, save_dir: str, has_gt: bool):
    save_dir = Path(save_dir)

    if has_gt:
        # ── mAP bar chart ────────────────────────────────────────────────
        models  = list(stats.keys())
        map50   = [stats[m]["mAP@0.5"]    for m in models]
        map5095 = [stats[m]["mAP@0.5-95"] for m in models]
        prec    = [stats[m]["Precision"]   for m in models]
        rec     = [stats[m]["Recall"]      for m in models]

        x  = np.arange(len(models))
        w  = 0.2
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - 1.5*w, map50,   w, label="mAP@0.5",     color="#2196F3")
        ax.bar(x - 0.5*w, map5095, w, label="mAP@0.5-0.95",color="#4CAF50")
        ax.bar(x + 0.5*w, prec,    w, label="Precision",    color="#FF9800")
        ax.bar(x + 1.5*w, rec,     w, label="Recall",       color="#E91E63")
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=12)
        ax.set_ylim(0, 1.1); ax.set_ylabel("Score"); ax.legend()
        ax.set_title("VisDrone — Detection Metrics Comparison", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for i, (a, b, c, d) in enumerate(zip(map50, map5095, prec, rec)):
            for j, v in zip([-1.5, -0.5, 0.5, 1.5], [a, b, c, d]):
                ax.text(i + j*w, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(save_dir / "metrics_comparison.png", dpi=150)
        plt.close()
        print(f"  Saved → {save_dir/'metrics_comparison.png'}")

        # ── Per-class AP heatmap ─────────────────────────────────────────
        ap_data = {}
        for m in models:
            for cls, ap in stats[m].get("AP_per_class", {}).items():
                ap_data.setdefault(cls, {})[m] = ap

        if ap_data:
            df = pd.DataFrame(ap_data).T.reindex(CLASS_NAMES).dropna(how="all")
            fig, ax = plt.subplots(figsize=(max(6, len(models)*2 + 2), 7))
            im = ax.imshow(df.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns, fontsize=11)
            ax.set_yticks(range(len(df.index)));   ax.set_yticklabels(df.index,   fontsize=10)
            for i in range(len(df.index)):
                for j in range(len(df.columns)):
                    val = df.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)
            plt.colorbar(im, ax=ax, label="AP@0.5")
            ax.set_title("Per-Class AP@0.5 — VisDrone", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(save_dir / "per_class_ap.png", dpi=150)
            plt.close()
            print(f"  Saved → {save_dir/'per_class_ap.png'}")

    else:
        # ── Class distribution bar chart ─────────────────────────────────
        all_classes = sorted(
            set(c for m in stats.values() for c in m.get("class_counts", {}))
        )
        fig, ax = plt.subplots(figsize=(14, 6))
        x  = np.arange(len(all_classes))
        w  = 0.35
        colours = ["#2196F3", "#E91E63"]
        for i, (model_name, colour) in enumerate(zip(stats.keys(), colours)):
            counts = [stats[model_name]["class_counts"].get(c, 0) for c in all_classes]
            bars   = ax.bar(x + (i - 0.5) * w, counts, w, label=model_name, color=colour, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(all_classes, rotation=35, ha="right")
        ax.set_ylabel("Detection Count"); ax.legend()
        ax.set_title("Detected Objects per Class — Test Set", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "class_detection_counts.png", dpi=150)
        plt.close()
        print(f"  Saved → {save_dir/'class_detection_counts.png'}")

        # ── Latency comparison ───────────────────────────────────────────
        names   = list(stats.keys())
        lat_vals = [stats[m]["avg_latency_ms"] for m in names]
        fps_vals = [stats[m]["avg_fps"]        for m in names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.bar(names, lat_vals, color=["#2196F3","#E91E63"])
        ax1.set_ylabel("ms / image"); ax1.set_title("Avg Inference Latency", fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)
        for i, v in enumerate(lat_vals):
            ax1.text(i, v + 0.5, f"{v:.1f} ms", ha="center", fontweight="bold")

        ax2.bar(names, fps_vals, color=["#4CAF50","#FF9800"])
        ax2.set_ylabel("FPS"); ax2.set_title("Avg Throughput (FPS)", fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)
        for i, v in enumerate(fps_vals):
            ax2.text(i, v + 0.5, f"{v:.1f}", ha="center", fontweight="bold")

        plt.suptitle("Model Speed Comparison — VisDrone Test Set", fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_dir / "latency_comparison.png", dpi=150)
        plt.close()
        print(f"  Saved → {save_dir/'latency_comparison.png'}")


# ─────────────────────────────────────────────────────────────────────────────
#  2.  SIDE-BY-SIDE COMPARE on sample images (saves grid PNG)
# ─────────────────────────────────────────────────────────────────────────────
def compare_models_images(
    test_img_dir : str  = "test/images",
    n_images     : int  = 6,
    imgsz        : int  = 640,
    conf         : float= 0.5,
    save_dir     : str  = "eval_results",
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    images = (sorted(Path(test_img_dir).glob("*.jpg")) +
              sorted(Path(test_img_dir).glob("*.png")))[:n_images]
    if not images:
        print(f"⚠️   No images in {test_img_dir}"); return

    models = {name: load_model(path, name) for name, path in MODELS_CFG.items()}
    model_names = list(models.keys())

    n_cols  = len(model_names) + 1        # original + one col per model
    n_rows  = len(images)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    if n_rows == 1:
        axes = [axes]

    col_titles = ["Original"] + model_names
    for j, title in enumerate(col_titles):
        axes[0][j].set_title(title, fontsize=11, fontweight="bold", pad=6)

    for i, img_path in enumerate(images):
        frame_orig = cv2.imread(str(img_path))
        axes[i][0].imshow(cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB))
        axes[i][0].axis("off")
        axes[i][0].set_ylabel(img_path.name, fontsize=7, rotation=0, labelpad=60, va="center")

        for j, (mname, model) in enumerate(models.items(), start=1):
            frame = frame_orig.copy()
            t0    = time.perf_counter()
            res   = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
            ms    = (time.perf_counter() - t0) * 1000
            frame = draw_detections(frame, res, conf)
            dets  = len(res.boxes)
            cv2.putText(frame, f"Dets:{dets}  {ms:.0f}ms",
                        (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            axes[i][j].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[i][j].axis("off")

    # Legend
    patches = [mpatches.Patch(color=tuple(v/255 for v in rgb), label=name)
               for name, rgb in CLASS_COLOURS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("VisDrone — YOLOv11n vs YOLOv8n Detection Comparison",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = Path(save_dir) / "model_comparison_grid.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅  Saved comparison grid → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  3.  VIDEO INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def run_video(
    source       : str,
    model_name   : str  = "YOLOv26n",
    imgsz        : int  = 640,
    conf         : float= 0.5,
    iou          : float= 0.45,
    save_dir     : str  = "eval_results",
    show         : bool = False,
    save_video   : bool = True,
    max_frames   : int  = None,
):
    model_path = MODELS_CFG.get(model_name)
    if not model_path:
        print(f"❌  Unknown model '{model_name}'. Choose from: {list(MODELS_CFG)}")
        return

    model = load_model(model_path, model_name)
    cap   = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌  Cannot open source: {source}"); return

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    writer = None
    if save_video:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        src_name = Path(source).stem if source != "0" else "webcam"
        out_path = str(Path(save_dir) / f"{src_name}_{model_name}.mp4")
        writer   = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (W, H)
        )

    fps_history = []
    frame_idx   = 0
    print(f"\n  Running {model_name} on: {source}")
    print(f"  Resolution: {W}×{H}  |  Source FPS: {fps_src:.1f}  |  Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        t0  = time.perf_counter()
        res = model(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
        dt  = time.perf_counter() - t0
        fps = 1.0 / dt
        fps_history.append(fps)

        frame = draw_detections(frame, res, conf)
        overlay_info(frame, model_name, fps, len(res.boxes))

        # Mini class count bar overlay (top-right corner)
        cls_in_frame = {}
        for box in res.boxes:
            cn = CLASS_NAMES[int(box.cls[0])]
            cls_in_frame[cn] = cls_in_frame.get(cn, 0) + 1
        y_off = 28
        for cn, cnt in sorted(cls_in_frame.items(), key=lambda x: -x[1])[:5]:
            colour_bgr = colour_for(cn)
            label_str  = f"{cn}: {cnt}"
            (tw, _), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(frame, label_str, (W - tw - 8, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour_bgr, 1, cv2.LINE_AA)
            y_off += 18

        if writer:
            writer.write(frame)
        if show:
            cv2.imshow(f"VisDrone — {model_name}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  User quit.")
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total}  |  Avg FPS: {np.mean(fps_history[-50:]):.1f}")

    cap.release()
    if writer:
        writer.release()
        print(f"\n  ✅  Saved video → {out_path}")
    cv2.destroyAllWindows()

    print(f"  Processed {frame_idx} frames")
    print(f"  Avg FPS   : {np.mean(fps_history):.1f}")
    print(f"  Min FPS   : {np.min(fps_history):.1f}")
    print(f"  Max FPS   : {np.max(fps_history):.1f}")

    return fps_history


# ─────────────────────────────────────────────────────────────────────────────
#  4.  REAL-TIME WEBCAM  (side-by-side both models)
# ─────────────────────────────────────────────────────────────────────────────
def run_webcam_dual(
    cam_id  : int   = 0,
    imgsz   : int   = 640,
    conf    : float = 0.5,
):
    """
    Run BOTH models on the webcam feed and display side-by-side.
    Press Q to quit, S to save a snapshot.
    """
    models = {name: load_model(path, name) for name, path in MODELS_CFG.items()}
    cap    = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"❌  Cannot open camera {cam_id}"); return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Webcam opened  {W}×{H}  |  Press Q to quit  |  S to save snapshot")

    snapshot_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        panels = []
        for mname, model in models.items():
            panel = frame.copy()
            t0    = time.perf_counter()
            res   = model(panel, imgsz=imgsz, conf=conf, verbose=False)[0]
            fps   = 1.0 / (time.perf_counter() - t0)
            panel = draw_detections(panel, res, conf)
            overlay_info(panel, mname, fps, len(res.boxes))
            # divider line on right edge
            cv2.line(panel, (W-2, 0), (W-2, H), (255,255,255), 3)
            panels.append(panel)

        combined = np.hstack(panels)
        cv2.imshow("VisDrone — YOLOv11n | YOLOv8n   [Q=quit  S=snapshot]", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"snapshot_{snapshot_idx:04d}.jpg"
            cv2.imwrite(fname, combined)
            print(f"  📸  Saved {fname}")
            snapshot_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  5.  CONFUSION MATRIX (requires GT labels)
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    model_name   : str  = "YOLOv26n",
    test_img_dir : str  = "test/images",
    imgsz        : int  = 640,
    conf         : float= 0.5,
    iou_match    : float= 0.5,
    save_dir     : str  = "eval_results",
):
    """Compute and plot confusion matrix from GT labels vs predictions."""
    import itertools

    img_dir = Path(test_img_dir)
    lbl_dir = img_dir.parent / "labels"
    if not lbl_dir.exists():
        print("⚠️   No labels/ folder found — skipping confusion matrix"); return

    model    = load_model(MODELS_CFG[model_name], model_name)
    cm       = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)
    # rows = GT class (last = background), cols = Pred class (last = background)

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    def xywhn_to_xyxy(box, w, h):
        cx, cy, bw, bh = box
        return [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]

    def iou(b1, b2):
        xa = max(b1[0],b2[0]); ya = max(b1[1],b2[1])
        xb = min(b1[2],b2[2]); yb = min(b1[3],b2[3])
        inter = max(0,xb-xa)*max(0,yb-ya)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter/(a1+a2-inter+1e-6)

    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img  = Image.open(img_path)
        W, H = img.size
        gts  = []
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            cls   = int(parts[0])
            box   = list(map(float, parts[1:5]))
            gts.append((cls, xywhn_to_xyxy(box, W, H)))

        res   = model(str(img_path), imgsz=imgsz, conf=conf, verbose=False)[0]
        preds = [(int(b.cls[0]), b.xyxy[0].tolist()) for b in res.boxes]

        matched_gt  = set()
        matched_pred= set()
        for pi, (pcls, pbox) in enumerate(preds):
            best_iou, best_gi = 0, -1
            for gi, (gcls, gbox) in enumerate(gts):
                if gi in matched_gt: continue
                s = iou(pbox, gbox)
                if s > best_iou:
                    best_iou, best_gi = s, gi
            if best_iou >= iou_match:
                gcls = gts[best_gi][0]
                if gcls < NUM_CLASSES and pcls < NUM_CLASSES:
                    cm[gcls][pcls] += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
            else:
                # False positive → predicted but no GT
                if pcls < NUM_CLASSES:
                    cm[NUM_CLASSES][pcls] += 1

        for gi in range(len(gts)):
            if gi not in matched_gt:
                gcls = gts[gi][0]
                if gcls < NUM_CLASSES:
                    cm[gcls][NUM_CLASSES] += 1

    # Normalise
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_norm, row_sums, where=row_sums!=0)

    labels   = CLASS_NAMES + ["Background"]
    fig, ax  = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name} (VisDrone)", fontsize=13, fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.6 else "black", fontsize=7)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out = Path(save_dir) / f"confusion_matrix_{model_name}.png"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ✅  Confusion matrix saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  6.  SPEED BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_speed(
    test_img_dir : str = "test/images",
    imgsz        : int = 640,
    n_warmup     : int = 5,
    n_runs       : int = 50,
    save_dir     : str = "eval_results",
):
    """Measure inference latency distribution for each model."""
    images = (sorted(Path(test_img_dir).glob("*.jpg")) +
              sorted(Path(test_img_dir).glob("*.png")))
    if not images:
        print(f"⚠️   No images in {test_img_dir}"); return
    images = (images * ((n_runs // len(images)) + 1))[:n_runs + n_warmup]

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    all_lats = {}

    for model_name, model_path in MODELS_CFG.items():
        model = load_model(model_path, model_name)
        lats  = []

        # Warmup
        for img in images[:n_warmup]:
            model(str(img), imgsz=imgsz, verbose=False)

        # Benchmark
        for img in images[n_warmup:n_warmup + n_runs]:
            t0 = time.perf_counter()
            model(str(img), imgsz=imgsz, verbose=False)
            lats.append((time.perf_counter() - t0) * 1000)

        all_lats[model_name] = lats
        arr = np.array(lats)
        print(f"\n  {model_name}")
        print(f"    Mean   : {arr.mean():.1f} ms  ({1000/arr.mean():.1f} FPS)")
        print(f"    Median : {np.median(arr):.1f} ms")
        print(f"    P95    : {np.percentile(arr, 95):.1f} ms")
        print(f"    Std    : {arr.std():.1f} ms")

    # Box plot
    fig, ax = plt.subplots(figsize=(8, 5))
    data    = [all_lats[m] for m in all_lats]
    bp = ax.boxplot(data, labels=list(all_lats.keys()), patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    colours_bp = ["#2196F3", "#E91E63"]
    for patch, c in zip(bp["boxes"], colours_bp):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Inference Time (ms)"); ax.set_title("Speed Benchmark — VisDrone Test Images", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Secondary FPS axis
    ax2 = ax.twinx()
    ax2.set_ylim(1000 / ax.get_ylim()[1], 1000 / max(ax.get_ylim()[0], 0.1))
    ax2.set_ylabel("FPS")

    plt.tight_layout()
    out = Path(save_dir) / "speed_benchmark.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  ✅  Benchmark plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="VisDrone evaluation — YOLOv11n & YOLOv8n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["eval","compare","video","webcam","benchmark","confusion","all"],
        default="eval",
        help=(
            "eval       → evaluate on test/images with metrics + plots\n"
            "compare    → side-by-side detection grid image\n"
            "video      → run on a video file (--source required)\n"
            "webcam     → live dual-model webcam feed\n"
            "benchmark  → latency / FPS benchmark\n"
            "confusion  → confusion matrix (requires labels)\n"
            "all        → run eval + compare + benchmark + video (if --source given)"
        ),
    )
    p.add_argument("--source",  default=None,       help="Video file path (for --mode video)")
    p.add_argument("--cam",     default=0, type=int, help="Webcam device ID (default 0)")
    p.add_argument("--model",   default="YOLOv26n", choices=list(MODELS_CFG.keys()),
                   help="Which model to use for video/webcam single-model modes")
    p.add_argument("--imgsz",   default=640,  type=int,   help="Inference image size")
    p.add_argument("--conf",    default=0.5, type=float, help="Confidence threshold")
    p.add_argument("--iou",     default=0.45, type=float, help="IoU NMS threshold")
    p.add_argument("--test-img-dir", default="test/images", help="Test images folder")
    p.add_argument("--save-dir",     default="eval_results", help="Output folder")
    p.add_argument("--show",    action="store_true", help="Show video window (requires display)")
    p.add_argument("--max-images",  default=None, type=int, help="Limit eval to N images")
    p.add_argument("--max-frames",  default=None, type=int, help="Limit video to N frames")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n{'━'*60}")
    print(f"  VisDrone Evaluation Script")
    print(f"  Mode     : {args.mode}")
    print(f"  ImgSz    : {args.imgsz}")
    print(f"  Conf     : {args.conf}")
    print(f"  Save dir : {args.save_dir}")
    print(f"{'━'*60}\n")

    mode = args.mode

    if mode in ("eval", "all"):
        evaluate_models(
            test_img_dir = args.test_img_dir,
            imgsz        = args.imgsz,
            conf         = args.conf,
            iou          = args.iou,
            save_dir     = args.save_dir,
            max_images   = args.max_images,
        )

    if mode in ("compare", "all"):
        compare_models_images(
            test_img_dir = args.test_img_dir,
            imgsz        = args.imgsz,
            conf         = args.conf,
            save_dir     = args.save_dir,
        )

    if mode in ("benchmark", "all"):
        benchmark_speed(
            test_img_dir = args.test_img_dir,
            imgsz        = args.imgsz,
            save_dir     = args.save_dir,
        )

    if mode == "confusion":
        plot_confusion_matrix(
            model_name   = args.model,
            test_img_dir = args.test_img_dir,
            imgsz        = args.imgsz,
            conf         = args.conf,
            save_dir     = args.save_dir,
        )

    if mode in ("video", "all"):
        src = args.source
        if not src:
            # Try to find a video in test/videos/
            vid_dir = Path(args.test_img_dir).parent / "videos"
            videos  = list(vid_dir.glob("*.mp4")) + list(vid_dir.glob("*.avi")) if vid_dir.exists() else []
            if videos:
                src = str(videos[0])
                print(f"  Auto-selected video: {src}")
            else:
                print("  ⚠️   No --source given and no videos found in test/videos/ — skipping video mode")
        if src:
            run_video(
                source     = src,
                model_name = args.model,
                imgsz      = args.imgsz,
                conf       = args.conf,
                iou        = args.iou,
                save_dir   = args.save_dir,
                show       = args.show,
                max_frames = args.max_frames,
            )

    if mode == "webcam":
        run_webcam_dual(
            cam_id = args.cam,
            imgsz  = args.imgsz,
            conf   = args.conf,
        )

    print(f"\n{'━'*60}")
    print(f"  ✅  Done!  Results saved in: {Path(args.save_dir).resolve()}")
    print(f"{'━'*60}\n")


if __name__ == "__main__":
    main()
