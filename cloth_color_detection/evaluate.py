"""
evaluate.py
───────────
Benchmark multiple segmentation back-ends on the same video/image
and produce a side-by-side comparison report (CSV + visual grid).

Usage
-----
  # Evaluate all available segmenters on a video
  python evaluate.py --input test/video.mp4 --max-frames 60

  # Evaluate specific segmenters
  python evaluate.py --input test/video.mp4 --models grabcut mediapipe bgsub

  # On an image
  python evaluate.py --input test/sample.jpg

Output
------
  output/eval_report.csv          — per-segmenter metrics
  output/eval_grid_<frame>.jpg    — side-by-side frame comparison
"""

import argparse
import csv
import os
import time
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from segmenters import REGISTRY, get_segmenter
from pipeline   import ClothColorPipeline, FrameResult, draw_frame

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("evaluate")


# ─── Metrics dataclass ────────────────────────────────────────────────────────

@dataclass
class ModelMetrics:
    model:            str
    avg_infer_ms:     float = 0.0
    avg_color_ms:     float = 0.0
    avg_total_ms:     float = 0.0
    avg_persons:      float = 0.0
    avg_conf:         float = 0.0
    frames_processed: int   = 0
    detection_rate:   float = 0.0   # fraction of frames with ≥1 person
    avg_mask_coverage:float = 0.0   # mean fraction of person bbox covered by mask
    top_colors:       List[str] = field(default_factory=list)
    error:            Optional[str] = None


# ─── Evaluator ────────────────────────────────────────────────────────────────

class SegmenterEvaluator:
    def __init__(
        self,
        input_path:  str,
        models:      List[str],
        max_frames:  int   = 60,
        skip_frames: int   = 1,
        use_sahi:    bool  = False,
        use_hs_only: bool  = True,
        output_dir:  str   = "output",
        grid_frames: int   = 3,       # how many frames to render in the grid
    ):
        self.input_path  = input_path
        self.models      = models
        self.max_frames  = max_frames
        self.skip_frames = skip_frames
        self.use_sahi    = use_sahi
        self.use_hs_only = use_hs_only
        self.output_dir  = Path(output_dir)
        self.grid_frames = grid_frames
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.is_video = Path(input_path).suffix.lower() in (
            ".mp4", ".avi", ".mov", ".mkv", ".webm"
        )

    # ── Sample frames from source ────────────────────────────────────────────

    def _load_frames(self) -> List[np.ndarray]:
        if not self.is_video:
            img = cv2.imread(self.input_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read {self.input_path}")
            return [img] * max(1, self.grid_frames)

        cap = cv2.VideoCapture(self.input_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target = min(self.max_frames, total)
        step   = max(1, total // target)

        frames = []
        idx = 0
        while len(frames) < target:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
            idx += step
        cap.release()
        log.info(f"Loaded {len(frames)} frames from {self.input_path}")
        return frames

    # ── Evaluate one model ────────────────────────────────────────────────────

    def _eval_model(
        self,
        model_name:  str,
        frames:      List[np.ndarray],
    ) -> tuple[ModelMetrics, List[FrameResult]]:
        metrics = ModelMetrics(model=model_name)
        frame_results: List[FrameResult] = []

        try:
            seg  = get_segmenter(model_name)
            pipe = ClothColorPipeline(
                segmenter    = seg,
                use_sahi     = self.use_sahi,
                use_hs_only  = self.use_hs_only,
                skip_frames  = self.skip_frames,
                n_colors     = 3,
            )

            infer_times = []
            color_times  = []
            person_counts = []
            confs         = []
            coverages     = []
            all_color_names: Dict[str, int] = {}

            for fi, frame in enumerate(frames):
                try:
                    t0 = time.perf_counter()
                    res = pipe.infer_frame(frame)
                    total_ms = (time.perf_counter() - t0) * 1000

                    infer_times.append(res.inference_ms)
                    color_times.append(res.color_ms)
                    person_counts.append(len(res.persons))

                    for pr in res.persons:
                        confs.append(pr.detection.score)
                        # mask coverage
                        bx1,by1,bx2,by2 = pr.detection.bbox
                        bbox_area = max(1, (bx2-bx1)*(by2-by1))
                        mask_area = int(pr.detection.mask.sum() / 255)
                        coverages.append(mask_area / bbox_area)
                        # colour histogram
                        for gc in pr.upper_colors + pr.lower_colors:
                            all_color_names[gc.name] = \
                                all_color_names.get(gc.name, 0) + 1

                    frame_results.append(res)
                    log.debug(
                        f"  {model_name} frame {fi}: "
                        f"persons={len(res.persons)} "
                        f"infer={res.inference_ms:.1f}ms"
                    )

                except Exception as e:
                    log.warning(f"  {model_name} frame {fi} error: {e}")

            n = len(infer_times) or 1
            metrics.avg_infer_ms      = round(sum(infer_times) / n, 2)
            metrics.avg_color_ms      = round(sum(color_times)  / n, 2)
            metrics.avg_total_ms      = round(metrics.avg_infer_ms + metrics.avg_color_ms, 2)
            metrics.avg_persons       = round(sum(person_counts) / n, 2)
            metrics.avg_conf          = round(sum(confs) / max(1, len(confs)), 3)
            metrics.frames_processed  = len(frames)
            metrics.detection_rate    = round(
                sum(1 for c in person_counts if c > 0) / n, 3
            )
            metrics.avg_mask_coverage = round(
                sum(coverages) / max(1, len(coverages)), 3
            )
            metrics.top_colors = [
                k for k, _ in sorted(
                    all_color_names.items(), key=lambda x: x[1], reverse=True
                )[:5]
            ]

        except Exception as e:
            metrics.error = str(e)
            log.error(f"  {model_name} FAILED: {e}")
            traceback.print_exc()

        return metrics, frame_results

    # ── Grid visualisation ────────────────────────────────────────────────────

    def _build_comparison_grid(
        self,
        model_results: Dict[str, tuple],
        frames:        List[np.ndarray],
        grid_n:        int = 3,
    ) -> np.ndarray:
        """
        Build a grid:  rows = models,  cols = sampled frames.
        Each cell = annotated frame at reduced size.
        """
        CELL_W, CELL_H = 480, 270
        sample_indices = np.linspace(0, len(frames)-1, grid_n, dtype=int)

        rows = []
        for model_name, (metrics, frame_results) in model_results.items():
            if metrics.error:
                # placeholder row
                row_imgs = []
                for fi in sample_indices:
                    ph = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
                    cv2.putText(ph, model_name, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,200), 2)
                    cv2.putText(ph, f"ERROR: {metrics.error[:40]}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,80,200), 1)
                    row_imgs.append(ph)
                rows.append(np.hstack(row_imgs))
                continue

            row_imgs = []
            for fi in sample_indices:
                frame   = frames[fi]
                fr_list = [r for r in frame_results if r is not None]
                if fi < len(fr_list):
                    vis = draw_frame(frame, fr_list[fi])
                else:
                    vis = frame.copy()
                    cv2.putText(vis, "(no result)", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)

                # Resize to cell size
                vis = cv2.resize(vis, (CELL_W, CELL_H))

                # Model label banner
                banner = np.zeros((40, CELL_W, 3), dtype=np.uint8)
                label  = (f"{model_name} | {metrics.avg_infer_ms:.0f}ms | "
                          f"{metrics.avg_persons:.1f}p | "
                          f"det={metrics.detection_rate:.0%}")
                cv2.putText(banner, label, (4, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230,230,230), 1)
                cell = np.vstack([banner, vis])
                row_imgs.append(cell)

            rows.append(np.hstack(row_imgs))

        # Equalise row widths
        max_w = max(r.shape[1] for r in rows)
        padded = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
                r = np.hstack([r, pad])
            padded.append(r)

        return np.vstack(padded)

    # ── CSV report ────────────────────────────────────────────────────────────

    def _write_csv(self, all_metrics: List[ModelMetrics]) -> str:
        csv_path = self.output_dir / "eval_report.csv"
        fields = [
            "model", "avg_infer_ms", "avg_color_ms", "avg_total_ms",
            "avg_persons", "avg_conf", "detection_rate", "avg_mask_coverage",
            "frames_processed", "top_colors", "error",
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for m in all_metrics:
                row = asdict(m)
                row["top_colors"] = " | ".join(m.top_colors)
                w.writerow({k: row[k] for k in fields})
        return str(csv_path)

    # ── Print leaderboard ─────────────────────────────────────────────────────

    @staticmethod
    def _print_leaderboard(all_metrics: List[ModelMetrics]):
        valid = [m for m in all_metrics if not m.error]
        if not valid:
            log.warning("No valid model results to rank.")
            return

        # Composite score: detection_rate * mask_coverage / (total_ms / 1000)
        def score(m):
            speed_bonus = 1.0 / max(1.0, m.avg_total_ms / 1000)
            return m.detection_rate * m.avg_mask_coverage * speed_bonus

        ranked = sorted(valid, key=score, reverse=True)

        print("\n" + "═"*70)
        print(f"{'SEGMENTER LEADERBOARD':^70}")
        print("═"*70)
        hdr = f"{'Rank':<5} {'Model':<20} {'DetRate':>7} {'Cover':>6} "
        hdr += f"{'Infer(ms)':>10} {'Total(ms)':>10} {'Score':>8}"
        print(hdr)
        print("─"*70)
        for rank, m in enumerate(ranked, 1):
            s = score(m)
            print(
                f"{rank:<5} {m.model:<20} {m.detection_rate:>7.1%} "
                f"{m.avg_mask_coverage:>6.2f} "
                f"{m.avg_infer_ms:>10.1f} {m.avg_total_ms:>10.1f} "
                f"{s:>8.4f}"
            )
        print("═"*70)
        print(f"\n🏆 Best model: {ranked[0].model}")
        print(f"   Detection rate : {ranked[0].detection_rate:.1%}")
        print(f"   Mask coverage  : {ranked[0].avg_mask_coverage:.2f}")
        print(f"   Avg infer time : {ranked[0].avg_infer_ms:.1f} ms")
        if ranked[0].top_colors:
            print(f"   Top colours    : {' | '.join(ranked[0].top_colors)}")
        print()

    # ── Main entry ───────────────────────────────────────────────────────────

    def run(self) -> List[ModelMetrics]:
        log.info(f"Loading frames from: {self.input_path}")
        frames = self._load_frames()

        log.info(f"Evaluating {len(self.models)} model(s) on {len(frames)} frames …")
        model_results: Dict[str, tuple] = {}
        all_metrics:   List[ModelMetrics] = []

        for model_name in self.models:
            log.info(f"\n── Evaluating: {model_name} ──")
            metrics, frame_results = self._eval_model(model_name, frames)
            model_results[model_name] = (metrics, frame_results)
            all_metrics.append(metrics)

        # Leaderboard
        self._print_leaderboard(all_metrics)

        # CSV
        csv_path = self._write_csv(all_metrics)
        log.info(f"Report saved → {csv_path}")

        # Visual grid
        grid = self._build_comparison_grid(
            model_results, frames, grid_n=min(self.grid_frames, len(frames))
        )
        grid_path = self.output_dir / "eval_comparison_grid.jpg"
        cv2.imwrite(str(grid_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        log.info(f"Visual grid saved → {grid_path}")

        return all_metrics


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Evaluate cloth-colour segmenters on a video or image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",       required=True,
                   help="Path to test video or image (e.g. test/video.mp4)")
    p.add_argument("--models",      nargs="+",
                   default=list(REGISTRY.keys()),
                   help=f"Segmenters to compare. Available: {list(REGISTRY.keys())}")
    p.add_argument("--max-frames",  type=int, default=60,
                   help="Max frames to sample from the video")
    p.add_argument("--grid-frames", type=int, default=3,
                   help="Columns in comparison grid image")
    p.add_argument("--sahi",        action="store_true",
                   help="Enable SAHI (sliced inference) for tiny persons")
    p.add_argument("--rgb-cluster", action="store_true",
                   help="Cluster in RGB instead of HS (disables shadow fix)")
    p.add_argument("--output-dir",  default="output",
                   help="Directory for reports and grids")
    p.add_argument("--verbose",     action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ev = SegmenterEvaluator(
        input_path  = args.input,
        models      = args.models,
        max_frames  = args.max_frames,
        use_sahi    = args.sahi,
        use_hs_only = not args.rgb_cluster,
        output_dir  = args.output_dir,
        grid_frames = args.grid_frames,
    )
    ev.run()


if __name__ == "__main__":
    main()
