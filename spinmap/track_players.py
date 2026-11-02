"""SpinMap Phase 2.2 — two-competitor tracking and pixel-space CSV export.

Designed for the fixed, side-on OpenTTGames clip: player 1 appears on the
left and player 2 on the right. The umpire is near the middle/background.
The filters are *heuristics*, not an evaluated re-identification model.

Ankle locations are still IMAGE PIXELS. Do not treat them as floor coordinates
or compute physical speed/distance until floor-plane calibration is added.
"""

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from .common import fit_frame, open_capture, overlay_text, source_arg


@dataclass
class PersonDetection:
    bbox: tuple[float, float, float, float]
    confidence: float
    track_id: int | None
    foot_x: float
    foot_y: float
    foot_method: str
    pose: object = None

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]


class TwoPlayerAssigner:
    """Keep player labels P1/P2 independent of ByteTrack's numerical IDs."""

    def __init__(self, width, center_margin=0.10, reacquire_distance=0.28):
        self.width = width
        self.margin = center_margin
        self.reacquire_distance = reacquire_distance * width
        self.last = {"P1": None, "P2": None}
        self.tracker_ids = {"P1": None, "P2": None}

    def choose(self, candidates):
        chosen = {}
        used = set()
        for player in ("P1", "P2"):
            previous = self.last[player]
            previous_id = self.tracker_ids[player]
            valid = []
            for idx, det in enumerate(candidates):
                if idx in used:
                    continue
                # Prioritize a known tracker ID; otherwise require an
                # appropriate side to avoid selecting the center umpire.
                same_id = previous_id is not None and det.track_id == previous_id
                on_side = (det.foot_x < self.width * (0.5 - self.margin)
                           if player == "P1" else
                           det.foot_x > self.width * (0.5 + self.margin))
                if not (same_id or on_side):
                    continue
                distance = (math.dist(previous, (det.foot_x, det.foot_y))
                            if previous is not None else 0.0)
                # Reject suspiciously large jumps even if ByteTrack reuses an ID.
                if previous is not None and distance > self.reacquire_distance:
                    continue
                score = (10.0 if same_id else 0.0) + det.confidence
                if previous is not None:
                    score -= distance / self.width * 2.0
                else:
                    # Initial assignment favors players near their own side.
                    score += abs(det.foot_x / self.width - 0.5)
                valid.append((score, idx))
            if valid:
                _, best_idx = max(valid)
                det = candidates[best_idx]
                chosen[player] = det
                used.add(best_idx)
                self.last[player] = (det.foot_x, det.foot_y)
                self.tracker_ids[player] = det.track_id
            else:
                chosen[player] = None
        return chosen


def extract_people(result, width, height, min_box_height_ratio, min_foot_y_ratio):
    """Parse pose result and keep plausible near-table competitors."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    ids = (boxes.id.int().cpu().numpy().tolist()
           if boxes.id is not None else [None] * len(xyxy))
    points = (result.keypoints.xy.cpu().numpy()
              if result.keypoints is not None else None)
    point_conf = (result.keypoints.conf.cpu().numpy()
                  if result.keypoints is not None and result.keypoints.conf is not None
                  else None)

    candidates = []
    for i, (bbox, confidence) in enumerate(zip(xyxy, confidences)):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if y2 - y1 < min_box_height_ratio * height:
            continue
        valid_ankles = []
        pose = points[i] if points is not None and i < len(points) else None
        if pose is not None and pose.shape[0] >= 17:
            for j in (15, 16):
                x, y = float(pose[j][0]), float(pose[j][1])
                kp_ok = (point_conf is None or point_conf[i][j] >= 0.35)
                if x > 0 and y > 0 and kp_ok:
                    valid_ankles.append((x, y))
        if valid_ankles:
            foot_x = sum(v[0] for v in valid_ankles) / len(valid_ankles)
            foot_y = sum(v[1] for v in valid_ankles) / len(valid_ankles)
            method = "ankle"
        else:
            foot_x, foot_y = (x1 + x2) / 2, y2
            method = "bbox_proxy"
        # The seated umpire usually fails both the height and foot-y tests.
        if foot_y < min_foot_y_ratio * height:
            continue
        candidates.append(PersonDetection((x1, y1, x2, y2), float(confidence),
                                          ids[i], foot_x, foot_y, method, pose))
    return candidates


def draw_player(frame, label, det):
    color = (255, 180, 40) if label == "P1" else (50, 200, 255)
    x1, y1, x2, y2 = [round(v) for v in det.bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    caption = f"{label}  ID:{det.track_id if det.track_id is not None else '?'}  {det.confidence:.2f}"
    overlay_text(frame, caption, max(x1, 5), max(y1 - 8, 22), 0.7)
    pt = (round(det.foot_x), round(det.foot_y))
    cv2.circle(frame, pt, 7, color, -1)
    foot_label = "ankle px" if det.foot_method == "ankle" else "bbox foot proxy"
    overlay_text(frame, f"{label} {foot_label}", min(pt[0] + 8, frame.shape[1]-160),
                 min(pt[1] + 10, frame.shape[0]-12), 0.5)


def main():
    parser = argparse.ArgumentParser(description="SpinMap Phase 2.2 player tracker")
    parser.add_argument("--source", type=source_arg, default=0)
    parser.add_argument("--model", default="yolo26n-pose.pt")
    parser.add_argument("--output", default="outputs/player_tracks.csv")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="0 = entire video; e.g., 600 for a short test")
    parser.add_argument("--min-box-height-ratio", type=float, default=0.23)
    parser.add_argument("--min-foot-y-ratio", type=float, default=0.55)
    parser.add_argument("--center-margin-ratio", type=float, default=0.10)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    cap = open_capture(args.source)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not 1 <= source_fps <= 1000:
        source_fps = 30.0
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["frame", "timestamp_sec", "player", "detected", "tracker_id",
               "confidence", "foot_x_px", "foot_y_px", "foot_method",
               "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    frame_idx = 0
    elapsed_start = time.perf_counter()
    prev_time = None
    smoothed_fps = 0.0
    assigner = None
    camera = isinstance(args.source, int)

    print("SpinMap Phase 2.2 | only the left/right players are reported")
    print("All positions are PIXELS; floor mapping comes later.")
    try:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Video ended or camera disconnected.")
                    break
                frame_idx += 1
                height, width = frame.shape[:2]
                if assigner is None:
                    assigner = TwoPlayerAssigner(width, args.center_margin_ratio)
                result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                     classes=[0], verbose=False)[0]
                candidates = extract_people(result, width, height,
                                            args.min_box_height_ratio,
                                            args.min_foot_y_ratio)
                players = assigner.choose(candidates)
                if camera:
                    timestamp_sec = time.perf_counter() - elapsed_start
                else:
                    video_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    timestamp_sec = (video_ms / 1000 if video_ms > 0
                                     else (frame_idx - 1) / source_fps)

                for label, det in players.items():
                    row = {"frame": frame_idx, "timestamp_sec": round(timestamp_sec, 4),
                           "player": label, "detected": int(det is not None)}
                    if det is not None:
                        draw_player(frame, label, det)
                        x1, y1, x2, y2 = det.bbox
                        row.update(tracker_id=det.track_id, confidence=round(det.confidence, 4),
                                   foot_x_px=round(det.foot_x, 2),
                                   foot_y_px=round(det.foot_y, 2),
                                   foot_method=det.foot_method,
                                   bbox_x1=round(x1, 2), bbox_y1=round(y1, 2),
                                   bbox_x2=round(x2, 2), bbox_y2=round(y2, 2))
                    writer.writerow(row)

                now = time.perf_counter()
                if prev_time is not None:
                    instantaneous = 1.0 / max(now - prev_time, 1e-6)
                    smoothed_fps = (instantaneous if not smoothed_fps
                                    else 0.85 * smoothed_fps + 0.15 * instantaneous)
                prev_time = now
                overlay_text(frame, f"SpinMap P1/P2 | processing FPS {smoothed_fps:.1f} | Q quit")
                display, _ = fit_frame(frame)
                cv2.imshow("SpinMap - Player Tracking Phase 2.2", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if args.max_frames and frame_idx >= args.max_frames:
                    print(f"Reached max frames: {args.max_frames}")
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Saved {frame_idx * 2} player-frame records to: {path}")


if __name__ == "__main__":
    main()
