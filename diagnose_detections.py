
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

VIDEO = "data/raw/match.mp4"
MODEL = "yolo26n-pose.pt"

FRAMES = [
    1, 5, 60, 75, 89,
    124, 145, 166, 170, 181,
    192, 193, 382, 390, 397,
    547, 555, 563, 606, 617,
    628, 647, 652, 673, 674,
    685, 697, 698, 704
]

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

output_dir = Path("outputs/diagnostics")
output_dir.mkdir(parents=True, exist_ok=True)

cols = 4
thumb_w, thumb_h = 480, 270
label_h = 32

rows = (len(FRAMES) + cols - 1) // cols

sheet = np.zeros(
    (rows * (thumb_h + label_h), cols * thumb_w, 3),
    dtype=np.uint8
)

for i, frame_no in enumerate(FRAMES):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
    ok, frame = cap.read()

    if not ok:
        print(f"Cannot read frame {frame_no}")
        continue

    h, w = frame.shape[:2]

    result = model.predict(
        frame,
        classes=[0],
        conf=0.10,
        verbose=False
    )[0]

    detections = result.boxes
    points = result.keypoints

    count = 0

    if detections is not None and len(detections) > 0:

        boxes = detections.xyxy.cpu().numpy()
        scores = detections.conf.cpu().numpy()

        poses = (
            points.xy.cpu().numpy()
            if points is not None else None
        )

        kp_conf = (
            points.conf.cpu().numpy()
            if points is not None and points.conf is not None
            else None
        )

        for j, (box, confidence) in enumerate(zip(boxes, scores)):

            x1, y1, x2, y2 = map(float, box)

            box_height = y2 - y1
            height_ok = box_height >= 0.23 * h

            valid_ankles = []

            if poses is not None and poses[j].shape[0] >= 17:

                for k in (15, 16):

                    x, y = poses[j][k]

                    kp_ok = (
                        kp_conf is None
                        or kp_conf[j][k] >= 0.35
                    )

                    if x > 0 and y > 0 and kp_ok:
                        valid_ankles.append((x, y))

            if valid_ankles:

                foot_x = np.mean([p[0] for p in valid_ankles])
                foot_y = np.mean([p[1] for p in valid_ankles])
                method = "ankle"

            else:

                foot_x = (x1 + x2) / 2
                foot_y = y2
                method = "bbox"

            foot_ok = foot_y >= 0.55 * h

            passed = height_ok and foot_ok

            status = "PASS" if passed else "REJECT"

            color = (0, 220, 0) if passed else (0, 0, 255)

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                3
            )

            cv2.circle(
                frame,
                (int(foot_x), int(foot_y)),
                7,
                color,
                -1
            )

            label = f"D{j} {confidence:.2f} {status} {method}"

            cv2.putText(
                frame,
                label,
                (max(5, int(x1)), max(25, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            count += 1

            print(
                f"Frame {frame_no} | D{j} | "
                f"Confidence={confidence:.2f} | "
                f"Height={box_height/h:.2f} | "
                f"Foot=({foot_x:.0f},{foot_y:.0f}) | "
                f"{method} | {status}"
            )

    print(f"Frame {frame_no}: {count} raw detections")
    print("-" * 65)

    thumb = cv2.resize(frame, (thumb_w, thumb_h))

    x = (i % cols) * thumb_w
    y = (i // cols) * (thumb_h + label_h)

    sheet[y:y+thumb_h, x:x+thumb_w] = thumb

    cv2.putText(
        sheet,
        f"Frame {frame_no}",
        (x + 10, y + thumb_h + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

cap.release()

path = output_dir / "raw_detection_audit.jpg"

cv2.imwrite(str(path), sheet)

print(f"\nAudit contact sheet saved: {path}")
