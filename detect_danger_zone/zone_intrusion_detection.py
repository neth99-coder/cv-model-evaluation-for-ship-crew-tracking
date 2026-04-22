"""
Zone Intrusion Detection — Danger Zone with Foot-Based Triggering
==================================================================
Detects when a PERSON's FEET (bottom of bounding box) enter a
user-defined danger zone polygon. Head/body being in the zone does
NOT count — only when the person is standing inside it.

Requirements:
    pip install ultralytics opencv-python numpy

Model (yolo11n.pt ~6 MB) downloads automatically on first run.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Draw your danger zone
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python zone_intrusion_detection.py --setup

    • Left-click  : add a polygon vertex
    • Right-click : undo last vertex
    • ENTER       : confirm zone and save to zone.json
    • ESC         : quit without saving

STEP 2 — Run detection
━━━━━━━━━━━━━━━━━━━━━━
    python zone_intrusion_detection.py
    python zone_intrusion_detection.py --video path/to/video.mp4
    python zone_intrusion_detection.py --conf 0.4

Live controls:
    Q / ESC  — Quit
    P        — Pause / Resume
    Z        — Toggle zone overlay
    +  /  -  — Raise / lower confidence threshold
    R        — Re-draw zone (restarts setup)
"""

import cv2
import numpy as np
import argparse
import json
import os
import sys

# ── Constants ─────────────────────────────────────────────────────────────────

ZONE_FILE       = "zone.json"
DEFAULT_VIDEO   = "test/test.mp4"
DEFAULT_MODEL   = "yolo11n.pt"
DEFAULT_CONF    = 0.35

PERSON_CLASS_ID = 0          # COCO class 0 = person

COLOR_SAFE      = (0,   220,  80)   # green   — person outside zone
COLOR_DANGER    = (0,    40, 255)   # red     — person inside zone
COLOR_ZONE_FILL = (0,    30, 200)   # zone polygon fill (translucent)
COLOR_ZONE_LINE = (0,    80, 255)   # zone polygon border
COLOR_FOOT_DOT  = (255, 255,   0)   # yellow dot at foot midpoint

ALERT_TEXT      = "IN DANGER ZONE"
SAFE_TEXT       = "SAFE"

# ── Geometry helpers ──────────────────────────────────────────────────────────

def point_in_polygon(point: tuple, polygon: np.ndarray) -> bool:
    """
    Uses OpenCV's pointPolygonTest.
    Returns True if point is inside or on the boundary.
    """
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def foot_point(x1: int, y1: int, x2: int, y2: int) -> tuple:
    """
    Returns the midpoint of the BOTTOM edge of the bounding box —
    i.e. where the person's feet touch the ground.
    """
    return ((x1 + x2) // 2, y2)


# ── Zone Drawing (Setup Mode) ─────────────────────────────────────────────────

class ZoneDrawer:
    def __init__(self, frame: np.ndarray):
        self.frame   = frame.copy()
        self.points  = []
        self.done    = False
        self.saved   = False

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()

    def render(self) -> np.ndarray:
        display = self.frame.copy()
        pts = self.points

        # Draw filled polygon preview
        if len(pts) >= 3:
            overlay = display.copy()
            cv2.fillPoly(overlay, [np.array(pts, np.int32)], COLOR_ZONE_FILL)
            cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)
            cv2.polylines(display, [np.array(pts, np.int32)],
                          isClosed=True, color=COLOR_ZONE_LINE, thickness=2)

        # Draw vertices
        for i, p in enumerate(pts):
            cv2.circle(display, p, 5, COLOR_ZONE_LINE, -1)
            if i > 0:
                cv2.line(display, pts[i - 1], p, COLOR_ZONE_LINE, 2)

        # Closing line preview
        if len(pts) >= 2:
            cv2.line(display, pts[-1], pts[0], (100, 100, 255), 1)

        # Instructions
        lines = [
            "ZONE SETUP MODE",
            f"Vertices: {len(pts)}",
            "Left-click  : add point",
            "Right-click : undo point",
            "ENTER       : save & continue",
            "ESC         : cancel",
        ]
        for i, txt in enumerate(lines):
            color = (0, 255, 200) if i == 0 else (220, 220, 220)
            cv2.putText(display, txt, (12, 28 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, txt, (12, 28 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        return display

    def run(self) -> list | None:
        win = "Draw Danger Zone  —  ENTER to confirm, ESC to cancel"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.setMouseCallback(win, self.mouse_cb)

        while True:
            cv2.imshow(win, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key == 13:  # ENTER
                if len(self.points) >= 3:
                    self.saved = True
                    break
                else:
                    print("[WARN] Need at least 3 points to define a zone.")
            elif key == 27:  # ESC
                break

        cv2.destroyAllWindows()
        return self.points if self.saved else None


def save_zone(points: list, path: str = ZONE_FILE):
    with open(path, "w") as f:
        json.dump({"zone": points}, f)
    print(f"[INFO] Zone saved to {path}  ({len(points)} vertices)")


def load_zone(path: str = ZONE_FILE) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    pts = data.get("zone", [])
    if len(pts) < 3:
        return None
    return np.array(pts, np.int32)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_bounding_box(frame, x1, y1, x2, y2, in_danger: bool, conf: float, track_id=None):
    color = COLOR_DANGER if in_danger else COLOR_SAFE
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label_parts = []
    if track_id is not None:
        label_parts.append(f"#{track_id}")
    label_parts.append(f"{conf:.0%}")
    label_parts.append(ALERT_TEXT if in_danger else SAFE_TEXT)
    label = "  ".join(label_parts)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)
    bar_x2 = min(x1 + tw + 8, frame.shape[1])
    cv2.rectangle(frame, (x1, y1 - th - 10), (bar_x2, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)


def draw_foot_dot(frame, fx, fy, in_danger: bool):
    color = COLOR_DANGER if in_danger else COLOR_FOOT_DOT
    cv2.circle(frame, (fx, fy), 6,  (0, 0, 0), -1)   # outline
    cv2.circle(frame, (fx, fy), 4,  color,     -1)   # fill


def draw_zone_overlay(frame, polygon: np.ndarray, any_danger: bool):
    """Draw the zone polygon with a translucent fill."""
    overlay = frame.copy()
    fill_color = (0, 20, 180) if not any_danger else (0, 0, 160)
    cv2.fillPoly(overlay, [polygon], fill_color)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
    border_color = COLOR_DANGER if any_danger else COLOR_ZONE_LINE
    cv2.polylines(frame, [polygon], isClosed=True, color=border_color, thickness=2)

    # Zone label at centroid
    M = cv2.moments(polygon)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        label = "! DANGER ZONE !" if any_danger else "DANGER ZONE"
        color = COLOR_DANGER if any_danger else (180, 180, 255)
        cv2.putText(frame, label, (cx - 60, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (cx - 60, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)


def draw_hud(frame, n_people, n_danger, fps, conf_thresh, show_zone, paused):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (310, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    lines = [
        ("Foot-based zone detection",  (180, 180, 180)),
        (f"FPS      : {fps:.1f}",       (200, 200, 200)),
        (f"People   : {n_people}",      (0, 220, 80)),
        (f"In Zone  : {n_danger}",      COLOR_DANGER if n_danger else (0, 220, 80)),
        (f"Conf     : {conf_thresh:.0%}   Zone: {'ON' if show_zone else 'OFF'}",
                                        (200, 200, 200)),
    ]
    if paused:
        lines.append(("  ▐▐  PAUSED", (255, 220, 0)))

    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (8, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)


def flash_alert(frame, n_danger: int):
    """Full-screen red flash border when someone is in the danger zone."""
    if n_danger == 0:
        return
    h, w = frame.shape[:2]
    thickness = 8
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_DANGER, thickness)
    txt = f"⚠  INTRUSION DETECTED — {n_danger} person{'s' if n_danger > 1 else ''} in danger zone"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cx = (w - tw) // 2
    cv2.rectangle(frame, (cx - 8, h - 45), (cx + tw + 8, h - 8), (0, 0, 0), -1)
    cv2.putText(frame, txt, (cx, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_DANGER, 2, cv2.LINE_AA)


# ── Main Detection Loop ───────────────────────────────────────────────────────

def run_detection(video_path: str, model_name: str,
                  conf_thresh: float, zone_polygon: np.ndarray):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.  Run:  pip install ultralytics")
        sys.exit(1)

    print(f"[INFO] Loading model: {model_name}")
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] {os.path.basename(video_path)}  {w}x{h}  {fps:.1f} fps")

    show_zone  = True
    paused     = False
    fps_disp   = fps
    t_prev     = cv2.getTickCount()
    frame_idx  = 0

    win = "Zone Intrusion Detection  |  Q=quit  P=pause  Z=zone  +/-=conf  R=redraw"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1280), min(h, 720))

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video.")
                break
            frame_idx += 1

            # ── YOLO inference (person class only) ───────────────────────
            results = model.track(
                frame,
                persist=True,
                classes=[PERSON_CLASS_ID],
                conf=conf_thresh,
                verbose=False,
            )

            n_people = 0
            n_danger = 0

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf  = float(box.conf[0])
                    tid   = int(box.id[0]) if box.id is not None else None
                    n_people += 1

                    # ── Foot point = midpoint of bottom edge ──────────────
                    fx, fy = foot_point(x1, y1, x2, y2)
                    in_danger = point_in_polygon((fx, fy), zone_polygon)
                    if in_danger:
                        n_danger += 1

                    # ── Draw ──────────────────────────────────────────────
                    draw_bounding_box(frame, x1, y1, x2, y2, in_danger, conf, tid)
                    draw_foot_dot(frame, fx, fy, in_danger)

            # ── Zone + HUD overlays ───────────────────────────────────────
            if show_zone:
                draw_zone_overlay(frame, zone_polygon, n_danger > 0)

            flash_alert(frame, n_danger)

            # ── FPS rolling average ───────────────────────────────────────
            t_now   = cv2.getTickCount()
            elapsed = (t_now - t_prev) / cv2.getTickFrequency()
            t_prev  = t_now
            fps_disp = 0.85 * fps_disp + 0.15 * (1.0 / max(elapsed, 1e-6))

            draw_hud(frame, n_people, n_danger, fps_disp,
                     conf_thresh, show_zone, paused)

            cv2.imshow(win, frame)

        # ── Key handling ─────────────────────────────────────────────────
        key = cv2.waitKey(1 if not paused else 50) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('z'):
            show_zone = not show_zone
        elif key in (ord('+'), ord('=')):
            conf_thresh = min(0.95, round(conf_thresh + 0.05, 2))
            print(f"[INFO] conf → {conf_thresh:.0%}")
        elif key == ord('-'):
            conf_thresh = max(0.10, round(conf_thresh - 0.05, 2))
            print(f"[INFO] conf → {conf_thresh:.0%}")
        elif key == ord('r'):
            cap.release()
            cv2.destroyAllWindows()
            return "redraw"   # signal caller to re-enter setup

    cap.release()
    cv2.destroyAllWindows()
    return "done"


# ── Setup (zone drawing) ──────────────────────────────────────────────────────

def setup_zone(video_path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    # Grab first non-black frame
    first_frame = None
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            first_frame = frame
    cap.release()

    if first_frame is None:
        print("[ERROR] Could not read a frame from video.")
        sys.exit(1)

    drawer  = ZoneDrawer(first_frame)
    points  = drawer.run()
    if points is None or len(points) < 3:
        print("[INFO] Zone setup cancelled.")
        return None

    save_zone(points)
    return np.array(points, np.int32)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Zone Intrusion Detection")
    parser.add_argument("--video",  default=DEFAULT_VIDEO,
                        help=f"Path to video file (default: {DEFAULT_VIDEO})")
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help=f"YOLO model weights (default: {DEFAULT_MODEL})")
    parser.add_argument("--conf",   type=float, default=DEFAULT_CONF,
                        help=f"Detection confidence threshold (default: {DEFAULT_CONF})")
    parser.add_argument("--setup",  action="store_true",
                        help="Run zone-drawing setup then exit")
    parser.add_argument("--zone",   default=ZONE_FILE,
                        help=f"Path to zone JSON file (default: {ZONE_FILE})")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    # ── Setup mode ────────────────────────────────────────────────────────────
    if args.setup:
        setup_zone(args.video)
        return

    # ── Load or create zone ───────────────────────────────────────────────────
    zone = load_zone(args.zone)
    if zone is None:
        print(f"[INFO] No zone file found at '{args.zone}'. Starting zone setup...")
        zone = setup_zone(args.video)
        if zone is None:
            print("[ERROR] No zone defined. Exiting.")
            sys.exit(1)
    else:
        print(f"[INFO] Loaded zone from '{args.zone}'  ({len(zone)} vertices)")

    # ── Detection loop (with optional redraw) ─────────────────────────────────
    while True:
        result = run_detection(args.video, args.model, args.conf, zone)
        if result == "redraw":
            zone = setup_zone(args.video)
            if zone is None:
                print("[ERROR] Zone redraw cancelled. Exiting.")
                sys.exit(1)
        else:
            break


if __name__ == "__main__":
    main()
