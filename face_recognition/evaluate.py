"""
Evaluation logic: enroll reference images, run recognition on test images,
compute metrics, and save results.

Supports both single-person and multi-person (group) test images.
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from models.base import BaseFaceRecognizer

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(test_images_dir: Path) -> dict[str, list[str]]:
    """
    Return a mapping of ``filename -> [person_name, ...]`` for every image.

    Values are always lists so the rest of the pipeline is uniform.

    Priority:
    1. ``ground_truth.json`` – supports both single names and lists:
         ``{"alice.jpg": "alice", "group.jpg": ["alice", "bob"]}``
    2. Filename inference: text before the first ``_``, ``-``, or space.
         ``alice_outdoor.jpg`` → ``["alice"]``
    """
    gt_file = test_images_dir / "ground_truth.json"
    if gt_file.exists():
        with open(gt_file) as fh:
            raw = json.load(fh)
        # Normalise every value to a list
        return {
            fname: ([v] if isinstance(v, str) else list(v))
            for fname, v in raw.items()
        }

    ground_truth: dict[str, list[str]] = {}
    for img_path in sorted(test_images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        person_name = re.split(r"[_\-\s]", img_path.stem)[0]
        ground_truth[img_path.name] = [person_name]

    return ground_truth


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    """
    Compute image-level accuracy and per-person precision/recall/F1.

    Each result entry has:
      - ground_truth_people : list[str]
      - predicted_people    : list[str]
      - correct             : bool  (predicted set == ground-truth set)
    """
    if not results:
        return {}

    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total

    # Collect all person names mentioned anywhere
    all_names: set[str] = set()
    for r in results:
        all_names.update(r["ground_truth_people"])
        all_names.update(p for p in r["predicted_people"] if p != "unknown")

    per_person: dict[str, dict] = {}
    for name in sorted(all_names):
        tp = sum(
            1 for r in results
            if name in r["ground_truth_people"] and name in r["predicted_people"]
        )
        fp = sum(
            1 for r in results
            if name not in r["ground_truth_people"] and name in r["predicted_people"]
        )
        fn = sum(
            1 for r in results
            if name in r["ground_truth_people"] and name not in r["predicted_people"]
        )

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0.0)

        per_person[name] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    macro_p  = sum(v["precision"] for v in per_person.values()) / len(per_person) if per_person else 0.0
    macro_r  = sum(v["recall"]    for v in per_person.values()) / len(per_person) if per_person else 0.0
    macro_f1 = sum(v["f1"]        for v in per_person.values()) / len(per_person) if per_person else 0.0

    times = [r["inference_time_ms"] for r in results]

    return {
        "total_images":           total,
        "correct":                correct,
        "accuracy":               round(accuracy, 4),
        "macro_precision":        round(macro_p,  4),
        "macro_recall":           round(macro_r,  4),
        "macro_f1":               round(macro_f1, 4),
        "avg_inference_time_ms":  round(sum(times) / len(times), 2),
        "min_inference_time_ms":  round(min(times), 2),
        "max_inference_time_ms":  round(max(times), 2),
        "per_person":             per_person,
    }


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    model: "BaseFaceRecognizer",
    test_dir: Path,
    results_base: Path,
) -> dict:
    """
    1. Enroll one reference image per person (``test/<name>/<image>``).
    2. Run recognition on every image in ``test/test_images/``.
       Each image may contain one *or many* faces (group photos supported).
    3. Save per-image CSV + JSON and a summary JSON.

    Returns the metrics dict (empty dict on failure).
    """
    out_dir = results_base / model.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = out_dir / "annotated_images"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ enroll
    print(f"\n{'='*60}")
    print(f"  Model : {model.model_name}")
    print(f"{'='*60}")
    print("Enrolling reference faces…")

    enrolled = 0
    for person_dir in sorted(test_dir.iterdir()):
        if person_dir.name == "test_images" or not person_dir.is_dir():
            continue
        images = [p for p in person_dir.iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            print(f"  [WARN] No images found in {person_dir}")
            continue
        ref_image = images[0]          # exactly one reference image expected
        ok = model.enroll(person_dir.name, str(ref_image))
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {person_dir.name} ← {ref_image.name}")
        if ok:
            enrolled += 1

    print(f"  Enrolled {enrolled} person(s).\n")

    # ----------------------------------------------------------- recognition
    test_images_dir = test_dir / "test_images"
    if not test_images_dir.exists():
        print(f"[ERROR] test_images/ not found inside {test_dir}")
        return {}

    ground_truth = load_ground_truth(test_images_dir)
    if not ground_truth:
        print("[ERROR] No test images found or ground_truth is empty.")
        return {}

    print(f"Running recognition on {len(ground_truth)} image(s)…")

    per_image_results: list[dict] = []
    for filename, true_people in sorted(ground_truth.items()):
        img_path = test_images_dir / filename
        if not img_path.exists():
            print(f"  [WARN] Image missing: {filename}")
            continue

        detections, time_ms = model.recognize_faces(str(img_path))

        # Annotation fallback: if recognition returns no faces, draw detector-only boxes as unknown.
        if not detections:
            fallback_boxes = model.detect_bboxes(str(img_path))
            if fallback_boxes:
                detections = [
                    {"name": "unknown", "confidence": 0.0, "bbox": box}
                    for box in fallback_boxes
                ]

        predicted_people = [d["name"] for d in detections if d["name"] != "unknown"]
        confidences      = {
            d["name"]: round(float(d["confidence"]), 4)
            for d in detections
        }

        annotated_rel = f"annotated_images/{filename}"
        annotated_path = out_dir / annotated_rel
        _save_annotated_image(img_path, detections, annotated_path)

        true_set      = set(true_people)
        predicted_set = set(predicted_people)
        correct       = true_set == predicted_set

        mark = "✓" if correct else "✗"
        tp   = len(true_set & predicted_set)
        fp   = len(predicted_set - true_set)
        fn   = len(true_set - predicted_set)

        print(
            f"  {mark}  {filename:<35} "
            f"expected={sorted(true_people)}  "
            f"got={sorted(predicted_people)}  "
            f"TP={tp} FP={fp} FN={fn}  {time_ms:.1f}ms"
        )

        per_image_results.append({
            "image":              filename,
            "annotated_image":    annotated_rel,
            "ground_truth_people": sorted(true_people),
            "predicted_people":   sorted(predicted_people),
            "confidences":        confidences,
            "inference_time_ms":  round(time_ms, 2),
            "tp":                 tp,
            "fp":                 fp,
            "fn":                 fn,
            "correct":            correct,
        })

    # ---------------------------------------------------------------- metrics
    metrics = compute_metrics(per_image_results)
    metrics["model"]          = model.model_name
    metrics["timestamp"]      = datetime.now().isoformat()
    metrics["enrolled_count"] = enrolled

    print(f"\n  Accuracy : {metrics.get('accuracy', 0):.2%}")
    print(f"  Macro F1 : {metrics.get('macro_f1', 0):.4f}")
    print(f"  Avg time : {metrics.get('avg_inference_time_ms', 0):.1f} ms/image")

    # ------------------------------------------------------------------ save
    detailed = {
        "model":     model.model_name,
        "metrics":   metrics,
        "per_image": per_image_results,
    }
    _write_json(out_dir / "detailed_results.json", detailed)
    _write_json(out_dir / "summary.json", metrics)

    if per_image_results:
        _write_csv_multiface(out_dir / "per_image_results.csv", per_image_results)

    print(f"  Results saved → {out_dir}")
    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: object) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _write_csv_multiface(path: Path, rows: list[dict]) -> None:
    """Write per-image results as CSV, serialising list columns to strings."""
    fieldnames = [
        "image", "annotated_image", "ground_truth_people", "predicted_people",
        "confidences", "inference_time_ms", "tp", "fp", "fn", "correct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "image":               row["image"],
                "annotated_image":     row.get("annotated_image", ""),
                "ground_truth_people": "|".join(row["ground_truth_people"]),
                "predicted_people":    "|".join(row["predicted_people"]),
                "confidences":         json.dumps(row["confidences"]),
                "inference_time_ms":   row["inference_time_ms"],
                "tp":                  row["tp"],
                "fp":                  row["fp"],
                "fn":                  row["fn"],
                "correct":             row["correct"],
            })


def _save_annotated_image(src_path: Path, detections: list[dict], out_path: Path) -> None:
    """Draw face boxes + labels and save one image per test input."""
    img = cv2.imread(str(src_path))
    if img is None:
        return

    height, width = img.shape[:2]

    for det in detections:
        bbox = det.get("bbox")
        name = str(det.get("name", "unknown"))
        score = float(det.get("confidence", 0.0))

        if name == "unknown":
            color = (0, 0, 255)  # red
            label = "unknown"
        else:
            color = (0, 180, 0)  # green
            label = f"{name} {score:.2f}"

        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )
            text_x = x1
            text_y = y1 - 8
            if text_y - text_h < 0:
                text_y = y1 + text_h + 8
            if text_x + text_w >= width:
                text_x = max(0, width - text_w - 4)

            bg_top = max(0, text_y - text_h - 4)
            bg_bottom = min(height - 1, text_y + 4)
            bg_right = min(width - 1, text_x + text_w + 4)
            cv2.rectangle(img, (text_x, bg_top), (bg_right, bg_bottom), (0, 0, 0), -1)
            cv2.putText(
                img,
                label,
                (text_x + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(out_path), img)

