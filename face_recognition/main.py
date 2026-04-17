#!/usr/bin/env python3
"""
Face Recognition Model Evaluation
===================================
Evaluates four models against a local test dataset:

  • AuraFace (ResNet100)  – InsightFace antelopev2 / glintr100
  • MobileFaceNet         – InsightFace buffalo_s  / w600k_mbf
  • FaceNet (Sandberg)    – DeepFace Facenet512
  • CompreFace            – REST service (Docker)

Test-data layout expected under ``face_recognition/test/``:

    test/
    ├── alice/                 ← one front-facing reference image
    │   └── alice.jpg
    ├── bob/
    │   └── bob.jpg
    └── test_images/           ← images to identify
        ├── alice_outdoor.jpg  ← filename prefix = ground-truth person name
        ├── bob_side.jpg
        └── ground_truth.json  ← optional, overrides filename inference
                                  format: {"alice_outdoor.jpg": "alice", ...}

Usage examples:
    # evaluate all models
    python main.py

    # evaluate specific models only
    python main.py --models auraface mobilefacenet

    # custom paths
    python main.py --test-dir /data/test --results-dir /data/results

    # CompreFace with API key
    python main.py --models compreface --compreface-key YOUR_KEY
"""

import argparse
import csv
import json
import os
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent


def load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE entries from a .env file without overriding existing env vars."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if ((value.startswith('"') and value.endswith('"'))
                or (value.startswith("'") and value.endswith("'"))):
            value = value[1:-1]

        os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face recognition model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["auraface", "mobilefacenet", "facenet", "compreface", "all"],
        default=["all"],
        metavar="MODEL",
        help="Models to evaluate: auraface mobilefacenet facenet compreface all (default: all)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=ROOT / "test",
        help="Root of the test dataset (default: ./test)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Output folder for results (default: ./results)",
    )
    parser.add_argument("--compreface-host", default="http://localhost",
                        help="CompreFace host (default: http://localhost)")
    parser.add_argument("--compreface-port", type=int, default=8000,
                        help="CompreFace port (default: 8000)")
    parser.add_argument("--compreface-key", default="",
                        help="CompreFace Recognition service API key (UUID). If omitted, COMPREFACE_API_KEY is used.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_models(args: argparse.Namespace) -> list:
    selected = set(args.models)
    use_all  = "all" in selected
    models   = []

    def _try_load(label: str, factory):
        try:
            m = factory()
            print(f"  [LOADED] {label}")
            return m
        except Exception as exc:
            print(f"  [SKIP]   {label}: {exc}")
            return None

    if use_all or "auraface" in selected:
        from models.auraface import AuraFaceRecognizer
        m = _try_load("AuraFace (ResNet100)", AuraFaceRecognizer)
        if m:
            models.append(m)

    if use_all or "mobilefacenet" in selected:
        from models.mobilefacenet import MobileFaceNetRecognizer
        m = _try_load("MobileFaceNet", MobileFaceNetRecognizer)
        if m:
            models.append(m)

    if use_all or "facenet" in selected:
        from models.facenet_model import FaceNetRecognizer
        m = _try_load("FaceNet (Sandberg / Facenet512)", FaceNetRecognizer)
        if m:
            models.append(m)

    if use_all or "compreface" in selected:
        from models.compreface import CompreFaceRecognizer
        key = (args.compreface_key or os.getenv("COMPREFACE_API_KEY", "")).strip()
        if not key:
            print(
                "  [SKIP]   CompreFace (REST): API key missing. "
                "Use --compreface-key or set COMPREFACE_API_KEY."
            )
            return models
        try:
            uuid.UUID(key)
        except ValueError:
            print(
                "  [SKIP]   CompreFace (REST): API key must be a valid UUID "
                "from a CompreFace Recognition service."
            )
            return models

        m = _try_load(
            "CompreFace (REST)",
            lambda: CompreFaceRecognizer(
                host=args.compreface_host,
                port=args.compreface_port,
                api_key=key,
            ),
        )
        if m:
            models.append(m)

    return models


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def generate_comparison_report(all_metrics: list[dict], results_dir: Path) -> None:
    """Write a Markdown comparison report and a CSV summary."""

    # ---- Markdown ----
    lines = [
        "# Face Recognition Model Evaluation Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Summary\n",
        "| Model | Accuracy | Macro F1 | Precision | Recall | Avg Inference (ms) |",
        "|-------|----------|----------|-----------|--------|-------------------|",
    ]
    for m in all_metrics:
        lines.append(
            f"| {m.get('model', 'N/A')} "
            f"| {m.get('accuracy', 0):.2%} "
            f"| {m.get('macro_f1', 0):.4f} "
            f"| {m.get('macro_precision', 0):.4f} "
            f"| {m.get('macro_recall', 0):.4f} "
            f"| {m.get('avg_inference_time_ms', 0):.1f} |"
        )

    lines.append("\n## Per-Model Details\n")
    for m in all_metrics:
        lines += [
            f"### {m.get('model', 'N/A')}",
            f"- **Total images**: {m.get('total_images', 0)}",
            f"- **Correct**:      {m.get('correct', 0)}",
            f"- **Accuracy**:     {m.get('accuracy', 0):.2%}",
            f"- **Macro F1**:     {m.get('macro_f1', 0):.4f}",
            f"- **Avg time**:     {m.get('avg_inference_time_ms', 0):.1f} ms",
            "",
        ]
        per_person = m.get("per_person", {})
        if per_person:
            lines += [
                "**Per-person metrics:**\n",
                "| Person | Precision | Recall | F1 |",
                "|--------|-----------|--------|-----|",
            ]
            for person, pm in sorted(per_person.items()):
                lines.append(
                    f"| {person} | {pm['precision']:.4f} | {pm['recall']:.4f} | {pm['f1']:.4f} |"
                )
            lines.append("")

    report_path = results_dir / "comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Markdown report → {report_path}")

    # ---- CSV ----
    fields = [
        "model", "accuracy", "macro_f1", "macro_precision", "macro_recall",
        "avg_inference_time_ms", "min_inference_time_ms", "max_inference_time_ms",
        "total_images", "correct", "enrolled_count",
    ]
    csv_path = results_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"  CSV summary      → {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Allow setting COMPREFACE_API_KEY and other options via face_recognition/.env
    load_env_file(ROOT / ".env")

    args = parse_args()

    # Validate test directory
    if not args.test_dir.exists():
        print(f"[ERROR] Test directory not found: {args.test_dir}\n")
        print("Expected layout:")
        print("  test/")
        print("    <person_name>/        ← one reference image per person")
        print("      <image>.jpg")
        print("    test_images/          ← images to identify")
        print("      <person>_variant.jpg  (or add ground_truth.json)")
        sys.exit(1)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models…")
    models = build_models(args)

    if not models:
        print("[ERROR] No models loaded. Check your dependencies (see requirements.txt).")
        sys.exit(1)

    # Run evaluations
    from evaluate import run_evaluation

    all_metrics: list[dict] = []
    for model in models:
        try:
            metrics = run_evaluation(model, args.test_dir, args.results_dir)
            if metrics:
                all_metrics.append(metrics)
        except Exception:
            print(f"[ERROR] Evaluation failed for {model.model_name}:")
            traceback.print_exc()

    # Comparison report
    if all_metrics:
        print(f"\n{'='*60}")
        print("COMPARISON REPORT")
        print(f"{'='*60}")
        generate_comparison_report(all_metrics, args.results_dir)

    print(f"\nDone. All results in: {args.results_dir}")


if __name__ == "__main__":
    main()
