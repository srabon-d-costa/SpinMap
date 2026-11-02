"""SpinMap Phase 2.4 — interactive floor-plane ROI calibration.

Select four coplanar FLOOR points in clockwise image order. Do NOT click the
four tabletop corners: the table top and players' feet are different planes.

Without measured dimensions, the saved coordinates are NORMALIZED map units,
not metres. Supply --width-m and --depth-m only when the selected four points
are the corners of a real, measured rectangular floor reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def read_frame(source, frame_index: int):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source!r}")
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {source!r}")
    return frame, (fps if fps > 0 else None), max(frame_no, 0)


def order_is_valid(points: list[tuple[float, float]]) -> bool:
    if len(points) != 4:
        return False
    poly = np.asarray(points, dtype=np.float32)
    area = abs(cv2.contourArea(poly))
    if area < 1000:
        return False
    # The selected polygon must be convex and non-self-intersecting.
    hull = cv2.convexHull(poly)
    return len(hull) == 4


def main() -> int:
    parser = argparse.ArgumentParser(description="SpinMap floor-plane calibration")
    parser.add_argument("--source", required=True, help="Camera index (0) or video path")
    parser.add_argument("--frame", type=int, default=0, help="Video frame used for clicking")
    parser.add_argument("--output", default="configs/floor_calibration.json")
    parser.add_argument("--width-m", type=float, default=None,
                        help="Measured width in metres of the selected floor rectangle")
    parser.add_argument("--depth-m", type=float, default=None,
                        help="Measured depth in metres of the selected floor rectangle")
    parser.add_argument("--display-width", type=int, default=1280)
    args = parser.parse_args()

    if args.frame < 0:
        raise SystemExit("--frame must be >= 0")
    if (args.width_m is None) != (args.depth_m is None):
        raise SystemExit("Provide both --width-m and --depth-m, or neither.")
    metric = args.width_m is not None
    if metric and (args.width_m <= 0 or args.depth_m <= 0):
        raise SystemExit("Measured floor width/depth must be positive.")

    source = parse_source(args.source)
    frame, source_fps, frame_no = read_frame(source, args.frame)
    image_h, image_w = frame.shape[:2]
    scale = min(1.0, args.display_width / image_w)
    display_w = max(1, int(round(image_w * scale)))
    display_h = max(1, int(round(image_h * scale)))
    base = cv2.resize(frame, (display_w, display_h)) if scale != 1 else frame.copy()

    # Map output has a fixed visual resolution in normalized mode. In metric
    # mode it uses a common pixels-per-metre scale while retaining metres in JSON.
    if metric:
        px_per_m = 300
        map_w = max(300, int(round(args.width_m * px_per_m)))
        map_h = max(200, int(round(args.depth_m * px_per_m)))
        target_w = float(args.width_m)
        target_h = float(args.depth_m)
        units = "metres"
    else:
        map_w, map_h = 1000, 600
        target_w, target_h = 1.0, 1.0
        units = "normalized"

    points: list[tuple[float, float]] = []
    names = ["FLOOR TOP-LEFT", "FLOOR TOP-RIGHT", "FLOOR BOTTOM-RIGHT", "FLOOR BOTTOM-LEFT"]
    win = "SpinMap - Floor Plane Calibration"

    def draw():
        canvas = base.copy()
        for i, (x, y) in enumerate(points):
            p = (int(round(x * scale)), int(round(y * scale)))
            cv2.circle(canvas, p, 8, (0, 255, 255), -1)
            cv2.putText(canvas, str(i + 1), (p[0] + 10, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, str(i + 1), (p[0] + 10, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        if len(points) >= 2:
            pts = np.array([[int(round(x * scale)), int(round(y * scale))] for x, y in points], np.int32)
            cv2.polylines(canvas, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)
        if len(points) == 4:
            pts = np.array([[int(round(x * scale)), int(round(y * scale))] for x, y in points], np.int32)
            cv2.polylines(canvas, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
        mode = "METRIC RECTANGLE" if metric else "NORMALIZED FLOOR ROI (NOT METRES)"
        cv2.rectangle(canvas, (8, 8), (min(display_w - 8, 1120), 94), (20, 20, 20), -1)
        cv2.putText(canvas, f"SpinMap | {mode} | Points: {len(points)}/4",
                    (20, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Click floor rectangle TL, TR, BR, BL | C clear | S save | Q quit",
                    (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    print("FLOOR CALIBRATION ONLY — do not select tabletop corners.")
    print("Select four corners of a coplanar floor rectangle: TL, TR, BR, BL.")
    if not metric:
        print("No dimensions supplied: output will be normalized, NOT metres.")
    else:
        print(f"Measured rectangle: {args.width_m:.3f} m x {args.depth_m:.3f} m")
    print("If the video has no marked/measured floor rectangle, use normalized mode only.")

    saved = False
    try:
        while True:
            cv2.imshow(win, draw())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('c'):
                points.clear()
                print("Cleared floor points.")
            if key == ord('s'):
                if not order_is_valid(points):
                    print("Need four non-degenerate floor points in TL, TR, BR, BL order.")
                    continue
                src = np.asarray(points, dtype=np.float32)
                dst = np.asarray([
                    [0.0, 0.0], [target_w, 0.0],
                    [target_w, target_h], [0.0, target_h]
                ], dtype=np.float32)
                H = cv2.getPerspectiveTransform(src, dst)
                if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
                    print("Invalid transform; clear and select points again.")
                    continue
                payload = {
                    "schema_version": 1,
                    "calibration_type": "floor_plane_homography",
                    "source": str(args.source),
                    "frame_index": frame_no,
                    "source_fps": source_fps,
                    "image_width_px": image_w,
                    "image_height_px": image_h,
                    "source_points_px": [list(map(float, p)) for p in points],
                    "point_order": names,
                    "homography_image_to_floor": H.tolist(),
                    "coordinate_units": units,
                    "metric_calibrated": bool(metric),
                    "floor_rectangle_width_m": args.width_m if metric else None,
                    "floor_rectangle_depth_m": args.depth_m if metric else None,
                    "map_width": map_w,
                    "map_height": map_h,
                    "target_width": target_w,
                    "target_depth": target_h,
                    "warning": (
                        "Metric coordinates are valid only if selected points are the corners "
                        "of a measured rectangular floor reference."
                        if metric else
                        "Normalized spatial map only. Do not report metres, speed, or physical distance."
                    ),
                }
                destination = Path(args.output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"Saved floor calibration: {destination}")
                print(f"Coordinate units: {units}; metric_calibrated={metric}")
                saved = True
                break
    finally:
        cv2.destroyAllWindows()

    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
