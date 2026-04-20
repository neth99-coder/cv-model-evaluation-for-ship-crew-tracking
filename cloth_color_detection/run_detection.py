"""
run_detection.py
────────────────
Simple CLI to run cloth-colour detection on a video or image.

Examples
--------
  # Basic — GrabCut on a video (default)
  python run_detection.py --input test/video.mp4

  # YOLOv8-seg on image, with SAHI for small people
  python run_detection.py --input test/photo.jpg --model yolov8n-seg --sahi

  # Background subtraction on CCTV footage (static camera)
  python run_detection.py --input test/cctv.mp4 --model bgsub

  # Live webcam
  python run_detection.py --webcam --model mediapipe

  # MediaPipe — fast, no GPU needed
  python run_detection.py --input test/video.mp4 --model mediapipe --show
"""

import argparse
import logging
import sys
from pathlib import Path
from pprint import pprint

from segmenters import REGISTRY
from pipeline   import ClothColorPipeline

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("run_detection")


def build_parser():
    p = argparse.ArgumentParser(
        description="Cloth colour detection on video / image / webcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input",   help="Path to video or image file")
    src.add_argument("--webcam",  action="store_true", help="Use webcam (index 0)")

    p.add_argument(
        "--model", "-m",
        default="grabcut",
        choices=list(REGISTRY.keys()),
        help=f"Segmentation back-end. Choices: {list(REGISTRY.keys())}",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Output video/image path (auto-generated if omitted)",
    )
    p.add_argument("--show",        action="store_true",
                   help="Show live preview window")
    p.add_argument("--sahi",        action="store_true",
                   help="Enable SAHI sliced inference for tiny people")
    p.add_argument("--sahi-size",   type=int, nargs=2, default=[320, 320],
                   metavar=("H", "W"), help="SAHI slice dimensions")
    p.add_argument("--sahi-overlap",type=float, default=0.2,
                   help="SAHI tile overlap fraction")
    p.add_argument("--n-colors",    type=int, default=3,
                   help="Max dominant colours to extract per region")
    p.add_argument("--rgb-cluster", action="store_true",
                   help="Cluster in RGB space (disables shadow-fix HSV mode)")
    p.add_argument("--skip",        type=int, default=2,
                   help="Process every N-th frame (1 = every frame)")
    p.add_argument("--max-frames",  type=int, default=None,
                   help="Stop after this many frames (video only)")
    p.add_argument("--scale",       type=float, default=1.0,
                   help="Display scale for output video")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def auto_output_path(input_path: str, model: str) -> str:
    src = Path(input_path)
    return str(Path("results") / f"{src.stem}_{model}{src.suffix}")


def run_webcam(args):
    """Live webcam demo."""
    import cv2
    from pipeline import ClothColorPipeline, draw_frame, FrameResult

    pipe = ClothColorPipeline(
        segmenter_name = args.model,
        use_sahi       = args.sahi,
        sahi_slice_h   = args.sahi_size[0],
        sahi_slice_w   = args.sahi_size[1],
        sahi_overlap   = args.sahi_overlap,
        n_colors       = args.n_colors,
        use_hs_only    = not args.rgb_cluster,
        skip_frames    = args.skip,
        display_scale  = args.scale,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open webcam")
        sys.exit(1)

    log.info("Webcam started. Press Q to quit.")
    last_result = FrameResult(frame_idx=0, persons=[])
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % args.skip == 0:
            last_result = pipe.infer_frame(frame)
            last_result.frame_idx = fi
        vis = draw_frame(frame, last_result, scale=args.scale)
        cv2.imshow("Cloth Colour Detection — webcam", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        fi += 1
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = build_parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    Path("results").mkdir(exist_ok=True)

    if args.webcam:
        run_webcam(args)
        return

    input_path  = args.input
    output_path = args.output or auto_output_path(input_path, args.model)

    log.info(f"Model    : {args.model}")
    log.info(f"Input    : {input_path}")
    log.info(f"Output   : {output_path}")
    log.info(f"SAHI     : {args.sahi}")
    log.info(f"HS-only  : {not args.rgb_cluster}")
    log.info(f"Skip     : every {args.skip} frame(s)")

    pipe = ClothColorPipeline(
        segmenter_name = args.model,
        use_sahi       = args.sahi,
        sahi_slice_h   = args.sahi_size[0],
        sahi_slice_w   = args.sahi_size[1],
        sahi_overlap   = args.sahi_overlap,
        n_colors       = args.n_colors,
        use_hs_only    = not args.rgb_cluster,
        skip_frames    = args.skip,
        display_scale  = args.scale,
    )

    import cv2
    suffix = Path(input_path).suffix.lower()
    is_image = suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

    if is_image:
        result = pipe.process_image(input_path, output_path=output_path)
        print(f"\n── Results for {input_path} ({args.model}) ──")
        for i, pr in enumerate(result.persons):
            print(f"\n  Person {i+1} | conf={pr.detection.score:.2f}")
            print(f"    Upper colours:")
            for gc in pr.upper_colors:
                print(f"      {gc.name:20s}  {gc.proportion*100:.0f}%  "
                      f"{gc.hex_code}  conf={gc.confidence:.2f}")
            print(f"    Lower colours:")
            for gc in pr.lower_colors:
                print(f"      {gc.name:20s}  {gc.proportion*100:.0f}%  "
                      f"{gc.hex_code}  conf={gc.confidence:.2f}")
        print(f"\nSaved → {output_path}")
    else:
        results = pipe.process_video(
            input_path,
            output_path  = output_path,
            max_frames   = args.max_frames,
            show_preview = args.show,
        )
        # Summary stats
        total_persons = sum(len(r.persons) for r in results)
        avg_infer = (sum(r.inference_ms for r in results) / max(1, len(results)))
        print(f"\n── Summary ({args.model}) ──")
        print(f"  Frames processed : {len(results)}")
        print(f"  Total detections : {total_persons}")
        print(f"  Avg infer time   : {avg_infer:.1f} ms")
        print(f"  Output video     : {output_path}")

        # Colour frequency summary
        color_freq: dict = {}
        for r in results:
            for pr in r.persons:
                for gc in pr.upper_colors + pr.lower_colors:
                    color_freq[gc.name] = color_freq.get(gc.name, 0) + 1
        if color_freq:
            top = sorted(color_freq.items(), key=lambda x: x[1], reverse=True)[:8]
            print(f"\n  Top detected colours:")
            for name, cnt in top:
                print(f"    {name:22s}  {cnt:4d} occurrences")


if __name__ == "__main__":
    main()
