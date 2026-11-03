import csv
import cv2
from pathlib import Path
import numpy as np

VIDEO = "data/raw/match.mp4"
CSV_FILE = "outputs/player_tracks_final.csv"

OUTPUT = Path("outputs/diagnostics")
OUTPUT.mkdir(parents=True, exist_ok=True)

events = [
    ("P1", 381, 382),
    ("P2", 536, 537),
    ("P2", 276, 277),
    ("P1", 48, 49),
    ("P1", 541, 542),
]

with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

by_frame = {}

for row in rows:
    frame = int(row["frame"])
    by_frame.setdefault(frame, []).append(row)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open match video")

panel_w = 480
panel_h = 300

sheet = np.zeros(
    (len(events) * panel_h, 4 * panel_w, 3),
    dtype=np.uint8
)

colors = {
    "P1": (0, 255, 0),
    "P2": (0, 165, 255),
}

for event_index, (player, start, end) in enumerate(events):

    selected = [
        start - 1,
        start,
        end,
        end + 1
    ]

    for col, frame_no in enumerate(selected):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.resize(frame, (panel_w, 270))

        for row in by_frame.get(frame_no, []):

            if row["detected"] != "1":
                continue

            p = row["player"]
            color = colors[p]

            sx = panel_w / 1920
            sy = 270 / 1080

            x1 = int(float(row["bbox_x1"]) * sx)
            y1 = int(float(row["bbox_y1"]) * sy)
            x2 = int(float(row["bbox_x2"]) * sx)
            y2 = int(float(row["bbox_y2"]) * sy)

            fx = int(float(row["foot_x_px"]) * sx)
            fy = int(float(row["foot_y_px"]) * sy)

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), color, 2
            )

            cv2.circle(
                frame, (fx, fy), 6, color, -1
            )

            cv2.putText(
                frame,
                f"{p} ID:{row['tracker_id']} "
                f"C:{float(row['confidence']):.2f}",
                (max(5, x1), max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )

        panel = sheet[
            event_index * panel_h:
            (event_index + 1) * panel_h,
            col * panel_w:
            (col + 1) * panel_w
        ]

        panel[:270] = frame

        title = (
            f"{player} jump {start}->{end} | Frame {frame_no}"
        )

        cv2.putText(
            panel, title, (8, 291),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1
        )

cap.release()

destination = OUTPUT / "trajectory_jump_contact_sheet.jpg"

cv2.imwrite(str(destination), sheet)

print("Saved:", destination)
print("Visual trajectory audit frames prepared.")