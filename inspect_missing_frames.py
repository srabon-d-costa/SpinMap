
import cv2
import math
import numpy as np
from pathlib import Path

video = "data/raw/match.mp4"

frames = [
    1, 5,
    60, 75, 89,
    124, 145, 166, 170, 181, 192, 193,
    382, 390, 397,
    547, 555, 563,
    606, 617, 628,
    652, 673, 674, 685, 697, 698, 704
]

cap = cv2.VideoCapture(video)

if not cap.isOpened():
    raise RuntimeError("Cannot open match.mp4")

w, h = 480, 270
label_h = 32
cols = 4
rows = math.ceil(len(frames) / cols)

sheet = np.zeros(
    (rows * (h + label_h), cols * w, 3),
    dtype=np.uint8
)

for i, frame_no in enumerate(frames):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
    ok, frame = cap.read()

    x = (i % cols) * w
    y = (i // cols) * (h + label_h)

    if ok:
        thumb = cv2.resize(frame, (w, h))
        sheet[y:y+h, x:x+w] = thumb

        cv2.putText(
            sheet,
            f"Frame {frame_no}",
            (x + 10, y + h + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

cap.release()

output = Path("outputs/diagnostics/missing_frame_contact_sheet.jpg")
output.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(output), sheet)

print(f"Contact sheet saved: {output}")
