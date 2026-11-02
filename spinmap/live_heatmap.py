"""SpinMap: render player movement heatmaps and tracking overlays onto a video.

Uses an existing SpinMap player_tracks.csv; it does NOT rerun YOLO.
Heatmaps are cumulative camera-image-space occupancy maps, not floor coordinates.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render SpinMap player heatmaps, trails and boxes onto a match video."
    )
    p.add_argument("--source", default="data/raw/match.mp4", help="Input video path or camera index")
    p.add_argument("--csv", default="outputs/player_tracks.csv", help="SpinMap tracking CSV")
    p.add_argument("--output", default="outputs/spinmap_live_heatmap.mp4", help="Output annotated MP4")
    p.add_argument("--trail-length", type=int, default=80, help="Recent positions per player shown as trails")
    p.add_argument("--heatmap-scale", type=float, default=0.25, help="Accumulator scale (0.25 = quarter width/height)")
    p.add_argument("--alpha", type=float, default=0.58, help="Maximum heatmap overlay opacity, 0..1")
    p.add_argument("--max-frames", type=int, default=0, help="0 = process the whole video")
    p.add_argument("--preview", action="store_true", help="Also show a preview window while rendering")
    p.add_argument("--include-proxy", action="store_true", help="Include bbox_proxy feet in heatmaps (excluded by default)")
    return p.parse_args()


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value is None or value.strip() == "":
        return None
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def read_tracks(csv_path: Path) -> dict[int, dict[str, dict[str, Any]]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Tracking CSV not found: {csv_path}")
    by_frame: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    required = {"frame", "player", "detected", "foot_x_px", "foot_y_px"}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            try:
                frame_no = int(row["frame"])
            except (TypeError, ValueError):
                continue
            player = row.get("player", "").strip()
            if player not in ("P1", "P2"):
                continue
            detected = row.get("detected", "0").strip() == "1"
            by_frame[frame_no][player] = {
                "detected": detected,
                "x": as_float(row, "foot_x_px"),
                "y": as_float(row, "foot_y_px"),
                "method": row.get("foot_method", "").strip(),
                "confidence": as_float(row, "confidence"),
                "track_id": row.get("tracker_id", "").strip(),
                "bbox": tuple(as_float(row, k) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
                if all(k in fields for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")) else None,
            }
    return by_frame


def make_heat_overlay(accum: np.ndarray, frame_shape: tuple[int, ...], player: str,
                      alpha_max: float) -> np.ndarray:
    """Return a BGR heat tint with per-pixel alpha encoded in a 4-channel image."""
    blurred = cv2.GaussianBlur(accum, (0, 0), sigmaX=5.0, sigmaY=5.0)
    peak = float(blurred.max())
    h, w = frame_shape[:2]
    if peak <= 0:
        return np.zeros((h, w, 4), dtype=np.uint8)
    # Log scaling keeps both frequent and less-frequent positions visible.
    norm = np.log1p(blurred) / max(math.log1p(peak), 1e-8)
    norm = cv2.resize(norm, (w, h), interpolation=cv2.INTER_CUBIC)
    norm = np.clip(norm, 0.0, 1.0)
    # P1 = cyan/blue; P2 = orange/red. Low-density pixels remain transparent.
    if player == "P1":
        color = np.zeros((h, w, 3), dtype=np.uint8)
        color[..., 0] = 255
        color[..., 1] = (70 + 150 * norm).astype(np.uint8)
        color[..., 2] = (15 + 45 * norm).astype(np.uint8)
    else:
        color = np.zeros((h, w, 3), dtype=np.uint8)
        color[..., 0] = (5 + 20 * norm).astype(np.uint8)
        color[..., 1] = (65 + 90 * norm).astype(np.uint8)
        color[..., 2] = 255
    a = np.where(norm > 0.035, np.clip(norm * alpha_max, 0, alpha_max), 0.0)
    out = np.dstack((color, (a * 255).astype(np.uint8)))
    return out


def blend_heat(frame: np.ndarray, overlay: np.ndarray) -> None:
    a = overlay[..., 3:4].astype(np.float32) / 255.0
    if not np.any(a):
        return
    frame[:] = np.clip(frame.astype(np.float32) * (1.0 - a) + overlay[..., :3].astype(np.float32) * a, 0, 255).astype(np.uint8)


def main() -> int:
    args = parse_args()
    if args.trail_length < 1 or not 0 < args.heatmap_scale <= 1 or not 0 <= args.alpha <= 1:
        raise SystemExit("Invalid settings: trail-length >= 1, heatmap-scale in (0,1], alpha in [0,1].")

    source: str | int = args.source
    if args.source.isdigit():
        source = int(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps < 1 or fps > 240:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise SystemExit("Could not read input video dimensions.")

    tracks = read_tracks(Path(args.csv))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not create output video: {out_path}. Try a different output path/codec.")

    small_w = max(1, int(round(width * args.heatmap_scale)))
    small_h = max(1, int(round(height * args.heatmap_scale)))
    accum = {"P1": np.zeros((small_h, small_w), dtype=np.float32),
             "P2": np.zeros((small_h, small_w), dtype=np.float32)}
    trails = {"P1": deque(maxlen=args.trail_length), "P2": deque(maxlen=args.trail_length)}
    colors = {"P1": (255, 190, 35), "P2": (40, 170, 255)}  # BGR
    frame_no = 0
    heat_points = {"P1": 0, "P2": 0}
    total_detections = {"P1": 0, "P2": 0}

    print(f"Input: {width}x{height} @ {fps:.2f} FPS | CSV frames: {len(tracks)}")
    print("Rendering cumulative image-space heatmaps. Press Q in preview to stop early." if args.preview
          else "Rendering video. No preview window; wait for the completion message.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            if args.max_frames and frame_no > args.max_frames:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

            frame_rows = tracks.get(frame_no, {})
            for player in ("P1", "P2"):
                row = frame_rows.get(player)
                if not row or not row["detected"]:
                    continue
                total_detections[player] += 1
                x, y = row["x"], row["y"]
                method = row["method"].lower()
                valid_point = (x is not None and y is not None and
                               0 <= x < width and 0 <= y < height)
                # Primary heatmaps use actual ankle landmarks only.
                use_heat = valid_point and (method == "ankle" or args.include_proxy)
                if use_heat:
                    sx = int(round(x * args.heatmap_scale))
                    sy = int(round(y * args.heatmap_scale))
                    if 0 <= sx < small_w and 0 <= sy < small_h:
                        cv2.circle(accum[player], (sx, sy), 5, 1.0, -1, lineType=cv2.LINE_AA)
                        trails[player].append((int(round(x)), int(round(y))))
                        heat_points[player] += 1
                bbox = row["bbox"]
                if bbox and all(v is not None for v in bbox):
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
                    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colors[player], 2)
                    tid = row["track_id"] or "?"
                    conf = row["confidence"]
                    conf_text = f"{conf:.2f}" if conf is not None else "n/a"
                    cv2.putText(frame, f"{player} ID:{tid} {conf_text}", (x1, max(24, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors[player], 2, cv2.LINE_AA)
                if valid_point:
                    pt = (int(round(x)), int(round(y)))
                    marker_color = colors[player] if method == "ankle" else (0, 255, 255)
                    cv2.circle(frame, pt, 6, marker_color, -1, cv2.LINE_AA)
                    if method != "ankle":
                        cv2.putText(frame, "proxy", (pt[0] + 7, pt[1] - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

            # Blend cumulative P1/P2 occupancy maps, then draw recent trails on top.
            for player in ("P1", "P2"):
                overlay = make_heat_overlay(accum[player], frame.shape, player, args.alpha)
                blend_heat(frame, overlay)
                pts = list(trails[player])
                if len(pts) >= 2:
                    for i in range(1, len(pts)):
                        strength = i / len(pts)
                        thickness = max(1, int(1 + 3 * strength))
                        cv2.line(frame, pts[i - 1], pts[i], colors[player], thickness, cv2.LINE_AA)

            elapsed = frame_no / fps
            banner = (f"SpinMap | frame {frame_no} | {elapsed:.1f}s | "
                      f"P1 heat points {heat_points['P1']} | P2 heat points {heat_points['P2']} | Q quit")
            cv2.rectangle(frame, (0, 0), (width, 42), (20, 20, 20), -1)
            cv2.putText(frame, banner, (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (245, 245, 245), 1, cv2.LINE_AA)
            writer.write(frame)
            if frame_no % 200 == 0:
                print(f"Rendered {frame_no} frames...")
            if args.preview:
                preview, scale = frame, 1.0
                max_w, max_h = 1280, 760
                if width > max_w or height > max_h:
                    scale = min(max_w / width, max_h / height)
                    preview = cv2.resize(frame, (int(width * scale), int(height * scale)))
                cv2.imshow("SpinMap - Heatmap Video", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("Stopped early by user.")
                    break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Finished: {frame_no} frames written to {out_path}")
    print(f"Detected rows: P1={total_detections['P1']}, P2={total_detections['P2']}")
    print(f"Ankle heatmap points: P1={heat_points['P1']}, P2={heat_points['P2']}")
    print("Note: heatmaps and trails are in camera-pixel space, not calibrated floor coordinates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
