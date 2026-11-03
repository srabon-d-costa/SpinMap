
"""
SpinMap Phase 2.7 — Segment-Aware Player Trajectory Analysis

Features:
1. Camera-segment-aware trajectory analysis.
2. Correct 1-based CSV to 0-based frame conversion.
3. Consecutive-frame displacement calculation.
4. Handles temporary tracker-ID changes without breaking
   trajectories, provided the player label remains consistent.
5. Prevents movement calculations across missing frames.
6. Prevents movement calculations across camera cuts.
7. Detects suspicious player-position jumps.
8. Audits tracker-ID transitions.
9. Generates segment-wise movement statistics.
10. Exports CSV and JSON reports.

IMPORTANT:
All distances are camera-image pixel measurements.
They are NOT physical court distances or metres.

Run:
    python -m spinmap.segment_trajectory
"""

from __future__ import annotations

import argparse
import csv
import json
import math

from collections import defaultdict
from pathlib import Path
from statistics import mean, median


PLAYERS = ("P1", "P2")


# ============================================================
# 1. BASIC HELPERS
# ============================================================

def is_detected(value) -> bool:
    """Interpret detection status from CSV."""

    return str(value).strip().lower() in {
        "1", "true", "yes", "y"
    }


def load_segments(path):

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Camera segment configuration not found: {path}"
        )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    segments = data.get("segments", [])

    if not segments:
        raise ValueError("No camera segments found.")

    segments = sorted(
        segments,
        key=lambda x: int(x["start"])
    )

    total_frames = int(data["total_frames"])

    if total_frames <= 0:
        raise ValueError("Invalid total_frames.")

    if int(segments[0]["start"]) != 0:
        raise ValueError(
            "First camera segment must start at frame 0."
        )

    seen_ids = set()

    for i, seg in enumerate(segments):

        seg["start"] = int(seg["start"])
        seg["end"] = int(seg["end"])

        if not seg.get("id"):
            raise ValueError("Every segment needs an ID.")

        if seg["id"] in seen_ids:
            raise ValueError(
                f"Duplicate segment ID: {seg['id']}"
            )

        seen_ids.add(seg["id"])

        if seg["start"] < 0:
            raise ValueError("Negative segment start.")

        if seg["end"] < seg["start"]:
            raise ValueError(
                f"Invalid segment: {seg['id']}"
            )

        if i > 0:

            previous = segments[i - 1]

            if previous["end"] + 1 != seg["start"]:
                raise ValueError(
                    "Camera segments overlap or contain gaps."
                )

    if segments[-1]["end"] != total_frames - 1:
        raise ValueError(
            "Last segment does not cover the final video frame."
        )

    fps = float(data.get("fps", 30.0))

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid video FPS.")

    return segments, fps, data


def find_segment(frame_index, segments):

    for seg in segments:

        if seg["start"] <= frame_index <= seg["end"]:
            return seg

    return None


# ============================================================
# 2. READ AND VALIDATE TRACKING CSV
# ============================================================

def read_tracking(csv_path, segments):

    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Tracking CSV not found: {csv_path}"
        )

    required = {
        "frame",
        "player",
        "detected",
        "tracker_id",
        "confidence",
        "foot_x_px",
        "foot_y_px",
    }

    observations = {}

    duplicate_count = 0
    outside_count = 0

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        original_fields = reader.fieldnames or []

        missing = required - set(original_fields)

        if missing:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing)}"
            )

        for row in reader:

            if not is_detected(row.get("detected")):
                continue

            player = str(
                row.get("player", "")
            ).strip().upper()

            if player not in PLAYERS:
                continue

            # CSV frame numbering is 1-based.
            # Camera segment indices are 0-based.

            try:
                csv_frame = int(row["frame"])

                if csv_frame < 1:
                    continue

                frame_index = csv_frame - 1

            except (ValueError, TypeError):
                continue

            seg = find_segment(
                frame_index,
                segments
            )

            if seg is None:
                outside_count += 1
                continue

            try:

                x = float(row["foot_x_px"])
                y = float(row["foot_y_px"])
                confidence = float(row["confidence"])

            except (ValueError, TypeError):
                continue

            if not all(
                map(math.isfinite, [x, y, confidence])
            ):
                continue

            tracker_id = str(
                row.get("tracker_id", "")
            ).strip()

            if not tracker_id:
                tracker_id = "UNKNOWN"

            key = (frame_index, player)

            record = {
                **row,

                "frame_index_0based": frame_index,

                "segment_id": seg["id"],

                "segment_frame_index":
                    frame_index - seg["start"],

                "tracker_id": tracker_id,

                "foot_x_px": x,
                "foot_y_px": y,
                "confidence": confidence,
            }

            # Retain the highest-confidence detection
            # if duplicate player detections occur.

            if key in observations:

                duplicate_count += 1

                if confidence <= observations[key]["confidence"]:
                    continue

            observations[key] = record

    observations = sorted(
        observations.values(),
        key=lambda r: (
            r["frame_index_0based"],
            r["player"]
        )
    )

    return (
        observations,
        original_fields,
        duplicate_count,
        outside_count,
    )


# ============================================================
# 3. CALCULATE PLAYER DISPLACEMENTS
# ============================================================

def calculate_displacements(
    observations,
    jump_threshold
):

    groups = defaultdict(list)

    # IMPORTANT:
    # Group by segment and player, NOT tracker ID.
    #
    # Temporary tracker-ID changes should not break
    # a player's trajectory when the player label
    # remains consistent.

    for row in observations:

        key = (
            row["segment_id"],
            row["player"],
        )

        groups[key].append(row)

    results = []

    for (segment_id, player), rows in groups.items():

        rows.sort(
            key=lambda r: r["frame_index_0based"]
        )

        for previous, current in zip(
            rows,
            rows[1:]
        ):

            gap = (
                current["frame_index_0based"]
                - previous["frame_index_0based"]
            )

            # Never bridge missing frames.
            # Never bridge camera cuts.

            if gap != 1:
                continue

            dx = (
                current["foot_x_px"]
                - previous["foot_x_px"]
            )

            dy = (
                current["foot_y_px"]
                - previous["foot_y_px"]
            )

            distance = math.hypot(dx, dy)

            old_id = previous["tracker_id"]
            new_id = current["tracker_id"]

            id_changed = old_id != new_id

            results.append({

                "segment_id": segment_id,

                "player": player,

                "tracker_id": (
                    f"{old_id}->{new_id}"
                    if id_changed
                    else old_id
                ),

                "from_tracker_id": old_id,

                "to_tracker_id": new_id,

                "tracker_id_changed": id_changed,

                "from_frame": previous["frame"],

                "to_frame": current["frame"],

                "dx_px": round(dx, 3),

                "dy_px": round(dy, 3),

                "displacement_px": round(
                    distance, 3
                ),

                "suspicious_jump": (
                    distance > jump_threshold
                ),
            })

    return results


# ============================================================
# 4. TRACKER-ID TRANSITION AUDIT
# ============================================================

def audit_id_transitions(observations):

    groups = defaultdict(list)

    for row in observations:

        groups[
            (
                row["segment_id"],
                row["player"]
            )
        ].append(row)

    transitions = []

    for (segment_id, player), rows in groups.items():

        rows.sort(
            key=lambda r: r["frame_index_0based"]
        )

        for a, b in zip(rows, rows[1:]):

            if a["tracker_id"] == b["tracker_id"]:
                continue

            frame_gap = (
                b["frame_index_0based"]
                - a["frame_index_0based"]
            )

            distance = math.hypot(
                b["foot_x_px"] - a["foot_x_px"],
                b["foot_y_px"] - a["foot_y_px"]
            )

            transitions.append({

                "segment_id": segment_id,

                "player": player,

                "from_frame": a["frame"],

                "to_frame": b["frame"],

                "old_tracker_id": a["tracker_id"],

                "new_tracker_id": b["tracker_id"],

                "frame_gap": frame_gap,

                "displacement_px": round(
                    distance, 3
                ),

                "consecutive_transition": (
                    frame_gap == 1
                ),
            })

    return transitions


# ============================================================
# 5. SEGMENT-WISE MOVEMENT STATISTICS
# ============================================================

def summarize_segments(
    segments,
    observations,
    displacements,
    transitions
):

    summary = []

    for seg in segments:

        for player in PLAYERS:

            rows = [
                r for r in observations

                if r["segment_id"] == seg["id"]

                and r["player"] == player
            ]

            moves = [
                d for d in displacements

                if d["segment_id"] == seg["id"]

                and d["player"] == player
            ]

            distances = [
                d["displacement_px"]
                for d in moves
            ]

            clean_distances = [
                d["displacement_px"]
                for d in moves

                if not d["suspicious_jump"]
            ]

            expected = (
                seg["end"] - seg["start"] + 1
            )

            frame_count = len({

                r["frame_index_0based"]
                for r in rows

            })

            ids = sorted({

                r["tracker_id"]
                for r in rows

            })

            id_transition_count = sum(

                t["segment_id"] == seg["id"]

                and t["player"] == player

                for t in transitions

            )

            summary.append({

                "segment_id": seg["id"],

                "start_frame_index": seg["start"],

                "end_frame_index": seg["end"],

                "player": player,

                "expected_frames": expected,

                "detected_frames": frame_count,

                "missing_frames": expected - frame_count,

                "tracker_ids": ids,

                "consecutive_movements": len(moves),

                "total_path_px": round(
                    sum(distances), 2
                ),

                "path_without_suspicious_jumps_px": round(
                    sum(clean_distances), 2
                ),

                "mean_displacement_px": (
                    round(mean(distances), 2)
                    if distances else None
                ),

                "median_displacement_px": (
                    round(median(distances), 2)
                    if distances else None
                ),

                "maximum_displacement_px": (
                    round(max(distances), 2)
                    if distances else None
                ),

                "suspicious_jumps": sum(

                    d["suspicious_jump"]
                    for d in moves

                ),

                "tracker_id_transitions": (
                    id_transition_count
                ),
            })

    return {

        "total_observations": len(observations),

        "total_movements": len(displacements),

        "total_id_transitions": len(transitions),

        "segments": summary,

        "note": (
            "Movement is measured only between consecutive "
            "frames within the same camera segment and player. "
            "Temporary tracker-ID changes are retained and "
            "flagged separately. Missing frames and camera "
            "cuts are never bridged. Suspicious jumps require "
            "visual verification. All distances are camera "
            "pixel measurements, not physical distances."
        ),
    }


# ============================================================
# 6. OUTPUT HELPERS
# ============================================================

def write_csv(path, rows, fields):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore"
        )

        writer.writeheader()

        writer.writerows(rows)


def write_json(path, data):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ============================================================
# 7. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "SpinMap Phase 2.7 — "
            "Segment-aware trajectory analysis"
        )
    )

    parser.add_argument(
        "--csv",
        default="outputs/player_tracks_final.csv",
        help="Phase 2.2 tracking CSV"
    )

    parser.add_argument(
        "--segments",
        default="configs/camera_segments.json",
        help="Camera segment configuration"
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/phase27",
        help="Output directory"
    )

    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=100.0,
        help="Suspicious displacement threshold in pixels"
    )

    args = parser.parse_args()

    if (
        not math.isfinite(args.jump_threshold)
        or args.jump_threshold <= 0
    ):
        raise SystemExit(
            "Jump threshold must be a positive finite number."
        )

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD CAMERA SEGMENTS
    # --------------------------------------------------------

    segments, fps, _ = load_segments(
        args.segments
    )

    print("\n=== SPINMAP PHASE 2.7 ===")

    print("SEGMENT-AWARE TRAJECTORY ANALYSIS")

    print("--------------------------------")

    print(f"Camera segments: {len(segments)}")

    print(f"Video FPS: {fps}")

    for seg in segments:

        print(
            f"{seg['id']}: "
            f"{seg['start']} - {seg['end']}"
        )

    # --------------------------------------------------------
    # LOAD TRACKING OBSERVATIONS
    # --------------------------------------------------------

    (
        observations,
        original_fields,
        duplicate_count,
        outside_count,
    ) = read_tracking(
        args.csv,
        segments
    )

    if not observations:
        raise SystemExit(
            "No valid tracking observations found."
        )

    print(
        f"\nValid observations: {len(observations)}"
    )

    print(
        f"Duplicate records: {duplicate_count}"
    )

    print(
        f"Outside segments: {outside_count}"
    )

    # --------------------------------------------------------
    # CALCULATE DISPLACEMENTS
    # --------------------------------------------------------

    displacements = calculate_displacements(
        observations,
        args.jump_threshold
    )

    # --------------------------------------------------------
    # AUDIT TRACKER-ID TRANSITIONS
    # --------------------------------------------------------

    transitions = audit_id_transitions(
        observations
    )

    # --------------------------------------------------------
    # GENERATE SUMMARY
    # --------------------------------------------------------

    summary = summarize_segments(
        segments,
        observations,
        displacements,
        transitions
    )

    # --------------------------------------------------------
    # EXPORT SEGMENT TRACKING
    # --------------------------------------------------------

    track_fields = list(dict.fromkeys(
        list(original_fields) + [
            "frame_index_0based",
            "segment_id",
            "segment_frame_index",
        ]
    ))

    write_csv(
        output_dir / "segment_tracking.csv",
        observations,
        track_fields
    )

    # --------------------------------------------------------
    # EXPORT DISPLACEMENTS
    # --------------------------------------------------------

    displacement_fields = [

        "segment_id",
        "player",
        "tracker_id",
        "from_tracker_id",
        "to_tracker_id",
        "tracker_id_changed",
        "from_frame",
        "to_frame",
        "dx_px",
        "dy_px",
        "displacement_px",
        "suspicious_jump",

    ]

    write_csv(
        output_dir / "segment_displacements.csv",
        displacements,
        displacement_fields
    )

    # --------------------------------------------------------
    # EXPORT TRACKER-ID TRANSITIONS
    # --------------------------------------------------------

    transition_fields = [

        "segment_id",
        "player",
        "from_frame",
        "to_frame",
        "old_tracker_id",
        "new_tracker_id",
        "frame_gap",
        "displacement_px",
        "consecutive_transition",

    ]

    write_csv(
        output_dir / "tracker_id_transitions.csv",
        transitions,
        transition_fields
    )

    # --------------------------------------------------------
    # EXPORT SEGMENT SUMMARY
    # --------------------------------------------------------

    summary_fields = [

        "segment_id",
        "start_frame_index",
        "end_frame_index",
        "player",
        "expected_frames",
        "detected_frames",
        "missing_frames",
        "tracker_ids",
        "consecutive_movements",
        "total_path_px",
        "path_without_suspicious_jumps_px",
        "mean_displacement_px",
        "median_displacement_px",
        "maximum_displacement_px",
        "suspicious_jumps",
        "tracker_id_transitions",

    ]

    write_csv(
        output_dir / "segment_summary.csv",
        summary["segments"],
        summary_fields
    )

    write_json(
        output_dir / "segment_summary.json",
        {
            **summary,
            "fps": fps,
            "duplicate_records": duplicate_count,
            "outside_segment_records": outside_count,
            "jump_threshold_px": args.jump_threshold,
        }
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n=== SEGMENT SUMMARY ===")

    for item in summary["segments"]:

        print(

            f"{item['segment_id']} | "

            f"{item['player']} | "

            f"Detected: "
            f"{item['detected_frames']}/"
            f"{item['expected_frames']} | "

            f"IDs: {item['tracker_ids']} | "

            f"Movements: "
            f"{item['consecutive_movements']} | "

            f"Path: "
            f"{item['total_path_px']} px | "

            f"Clean path: "
            f"{item['path_without_suspicious_jumps_px']} px | "

            f"Jumps: "
            f"{item['suspicious_jumps']} | "

            f"ID changes: "
            f"{item['tracker_id_transitions']}"

        )

    print("\n=== TRACKER-ID TRANSITIONS ===")

    for item in transitions:

        print(

            f"{item['segment_id']} | "

            f"{item['player']} | "

            f"Frame {item['from_frame']} -> "
            f"{item['to_frame']} | "

            f"ID {item['old_tracker_id']} -> "
            f"{item['new_tracker_id']} | "

            f"Gap: {item['frame_gap']} | "

            f"Distance: {item['displacement_px']} px"

        )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print("\n=== PHASE 2.7 COMPLETE ===")

    print(f"Output directory: {output_dir}")

    print(
        f"Total observations: "
        f"{len(observations)}"
    )

    print(
        f"Total movements: "
        f"{len(displacements)}"
    )

    print(
        f"Tracker-ID transitions: "
        f"{len(transitions)}"
    )

    print(
        f"Suspicious jumps: "
        f"{sum(d['suspicious_jump'] for d in displacements)}"
    )

    print("\nGenerated files:")

    print("1. segment_tracking.csv")

    print("2. segment_displacements.csv")

    print("3. tracker_id_transitions.csv")

    print("4. segment_summary.csv")

    print("5. segment_summary.json")

    print("\nAll distances are camera-pixel measurements.")

    print("No movement is calculated across camera cuts.")

    print(
        "Tracker-ID changes are retained and flagged."
    )

    print(
        "Suspicious jumps require visual verification."
    )


if __name__ == "__main__":
    main()
