
"""
SpinMap Phase 2.6
Automatic Broadcast Camera-Cut Detection

Detects abrupt changes in camera perspective using:
1. HSV colour histogram distance
2. Consecutive-frame pixel difference
3. Combined transition score

Outputs:
- camera_cut_scores.csv
- camera_cuts.csv
- camera_cut_contact_sheet.jpg

Frame numbering is 1-based, matching player_tracks_final.csv.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="SpinMap automatic camera-cut detection"
    )

    parser.add_argument(
        "--source",
        default="data/raw/match.mp4"
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/diagnostics"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30
    )

    parser.add_argument(
        "--hist-min",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--pixel-min",
        type=float,
        default=0.12
    )

    parser.add_argument(
        "--min-gap",
        type=int,
        default=5
    )

    parser.add_argument(
        "--max-contact-events",
        type=int,
        default=20
    )

    return parser.parse_args()


def extract_features(frame):

    small = cv2.resize(frame, (320, 180))

    small = cv2.GaussianBlur(
        small, (5, 5), 0
    )

    hsv = cv2.cvtColor(
        small, cv2.COLOR_BGR2HSV
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    cv2.normalize(
        hist, hist, 1.0, 0.0, cv2.NORM_L1
    )

    gray = cv2.cvtColor(
        small, cv2.COLOR_BGR2GRAY
    )

    return hist, gray


def compare_frames(previous, current):

    hist_previous, gray_previous = previous
    hist_current, gray_current = current

    hist_distance = cv2.compareHist(
        hist_previous,
        hist_current,
        cv2.HISTCMP_BHATTACHARYYA
    )

    pixel_difference = (
        cv2.absdiff(
            gray_previous,
            gray_current
        ).mean() / 255.0
    )

    # Combined transition score.
    normalized_pixel = min(
        pixel_difference / 0.25,
        1.0
    )

    score = (
        0.70 * hist_distance
        + 0.30 * normalized_pixel
    )

    return (
        float(hist_distance),
        float(pixel_difference),
        float(score)
    )


def cluster_candidates(candidates, min_gap):

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda x: x["frame_after"]
    )

    clusters = []

    for candidate in candidates:

        if not clusters:
            clusters.append([candidate])
            continue

        previous = clusters[-1][-1]

        if (
            candidate["frame_after"]
            - previous["frame_after"]
            <= min_gap
        ):
            clusters[-1].append(candidate)

        else:
            clusters.append([candidate])

    # Keep the strongest transition from each cluster.
    events = [
        max(cluster, key=lambda x: x["cut_score"])
        for cluster in clusters
    ]

    return events


def save_csv(path, rows, fields):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)


def create_contact_sheet(
    source,
    events,
    output_path,
    fps
):

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot reopen video for contact sheet."
        )

    panel_w = 480
    panel_h = 300

    columns = 4
    rows_count = len(events)

    sheet = np.zeros(
        (
            rows_count * panel_h,
            columns * panel_w,
            3
        ),
        dtype=np.uint8
    )

    def get_frame(frame_number):

        frame_number = max(1, frame_number)

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number - 1
        )

        ok, frame = cap.read()

        if not ok:
            return None

        return cv2.resize(
            frame,
            (panel_w, 270)
        )

    for row_index, event in enumerate(events):

        before = event["frame_before"]
        after = event["frame_after"]

        selected_frames = [
            max(1, before - 1),
            before,
            after,
            after + 1
        ]

        for col_index, frame_number in enumerate(
            selected_frames
        ):

            frame = get_frame(frame_number)

            if frame is None:
                continue

            x = col_index * panel_w
            y = row_index * panel_h

            panel = sheet[
                y:y + panel_h,
                x:x + panel_w
            ]

            panel[:270] = frame

            title = (
                f"Frame {frame_number} | "
                f"Transition {before}->{after}"
            )

            cv2.putText(
                panel,
                title,
                (8, 291),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            if frame_number == after:

                cv2.rectangle(
                    panel,
                    (2, 2),
                    (panel_w - 3, 267),
                    (0, 255, 255),
                    3
                )

    cap.release()

    if not cv2.imwrite(
        str(output_path),
        sheet
    ):
        raise OSError(
            f"Could not save contact sheet: {output_path}"
        )


def main():

    args = parse_args()

    source = args.source

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {source}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        raise RuntimeError("Invalid video FPS.")

    print("\n=== SPINMAP PHASE 2.6 ===")
    print("AUTOMATIC CAMERA-CUT DETECTION")
    print("--------------------------------")
    print(f"Source: {source}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames / fps:.2f} seconds")

    ok, previous_frame = cap.read()

    if not ok:
        raise RuntimeError(
            "Cannot read first video frame."
        )

    previous = extract_features(
        previous_frame
    )

    frame_no = 1

    scores = []
    candidates = []

    while True:

        ok, current_frame = cap.read()

        if not ok:
            break

        frame_no += 1

        current = extract_features(
            current_frame
        )

        hist_dist, pixel_diff, score = compare_frames(
            previous,
            current
        )

        is_candidate = (
            score >= args.threshold
            and (
                hist_dist >= args.hist_min
                or pixel_diff >= args.pixel_min
            )
        )

        record = {
            "frame_before": frame_no - 1,
            "frame_after": frame_no,
            "timestamp_sec": round(
                (frame_no - 1) / fps, 4
            ),
            "hist_distance": round(
                hist_dist, 5
            ),
            "pixel_difference": round(
                pixel_diff, 5
            ),
            "cut_score": round(
                score, 5
            ),
            "candidate": int(is_candidate)
        }

        scores.append(record)

        if is_candidate:
            candidates.append(record)

        previous = current

        if frame_no % 100 == 0:
            print(
                f"Processed {frame_no}/{total_frames} frames"
            )

    cap.release()

    events = cluster_candidates(
        candidates,
        args.min_gap
    )

    score_file = (
        output_dir / "camera_cut_scores.csv"
    )

    event_file = (
        output_dir / "camera_cuts.csv"
    )

    save_csv(
        score_file,
        scores,
        list(scores[0].keys()) if scores else []
    )

    event_fields = [
        "frame_before",
        "frame_after",
        "timestamp_sec",
        "hist_distance",
        "pixel_difference",
        "cut_score",
        "candidate"
    ]

    save_csv(
        event_file,
        events,
        event_fields
    )

    print("\n=== DETECTION RESULTS ===")
    print("Raw candidates:", len(candidates))
    print("Clustered transitions:", len(events))

    for event in events:

        print(
            f"Frames {event['frame_before']} -> "
            f"{event['frame_after']} | "
            f"Time {event['timestamp_sec']}s | "
            f"Score {event['cut_score']}"
        )

    print("\n=== TOP 10 TRANSITIONS ===")

    top = sorted(
        scores,
        key=lambda x: x["cut_score"],
        reverse=True
    )[:10]

    for item in top:

        print(
            f"{item['frame_before']} -> "
            f"{item['frame_after']} | "
            f"Score: {item['cut_score']} | "
            f"Histogram: {item['hist_distance']} | "
            f"Pixel: {item['pixel_difference']}"
        )

    # If no candidates are detected, create a review
    # sheet from the strongest transitions.
    review_events = events

    if not review_events:
        print(
            "\nNo transitions passed the threshold. "
            "Creating a top-score review sheet instead."
        )
        review_events = top[:8]

    selected = sorted(
        review_events,
        key=lambda x: x["cut_score"],
        reverse=True
    )[:args.max_contact_events]

    selected.sort(
        key=lambda x: x["frame_after"]
    )

    if selected:

        contact_file = (
            output_dir /
            "camera_cut_contact_sheet.jpg"
        )

        create_contact_sheet(
            source,
            selected,
            contact_file,
            fps
        )

        print(
            "\nContact sheet saved:",
            contact_file
        )

    print("\nCSV saved:", score_file)
    print("Events saved:", event_file)

    print("\n=== PHASE 2.6 CAMERA DETECTION COMPLETE ===")
    print(
        "IMPORTANT: Candidate transitions require "
        "visual verification."
    )


if __name__ == "__main__":
    main()
