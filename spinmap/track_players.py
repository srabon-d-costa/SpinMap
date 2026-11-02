"""Optional Phase 2 preview: bounding boxes, persistent IDs and pose ankles.

Output is PIXEL SPACE ONLY. Player ground coordinates require separate
floor-plane calibration. People outside the two competitors may also be detected.
"""
import argparse
import time

import cv2

from .common import open_capture, overlay_text, source_arg, fit_frame


def main():
    p = argparse.ArgumentParser(description='SpinMap - optional player tracking preview')
    p.add_argument('--source', type=source_arg, default=0)
    p.add_argument('--model', default='yolo26n-pose.pt',
                   help='Pose checkpoint; first use downloads weights')
    args = p.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit('Install optional packages: pip install -r requirements-player.txt') from e
    model = YOLO(args.model)
    cap = open_capture(args.source)
    last = None
    smoothed_fps = 0
    print('Demo tracking only: confidence/IDs not yet evaluated; feet are in image pixels.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.track(frame, persist=True, tracker='bytetrack.yaml',
                                 classes=[0], verbose=False)[0]
            annotated = result.plot()
            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
                kpts = result.keypoints.xy.cpu().numpy()
                for i, track_id in enumerate(ids):
                    # Standard COCO indices 15/16: left/right ankles.
                    if i >= len(kpts) or kpts.shape[1] < 17:
                        continue
                    ankles = [(float(kpts[i, j, 0]), float(kpts[i, j, 1]))
                              for j in (15, 16) if kpts[i, j, 0] > 0 and kpts[i, j, 1] > 0]
                    if ankles:
                        x = round(sum(a[0] for a in ankles) / len(ankles))
                        y = round(sum(a[1] for a in ankles) / len(ankles))
                        cv2.circle(annotated, (x, y), 7, (0, 255, 255), -1)
                        overlay_text(annotated, f'ID {track_id} ankle-px', x + 8, y, 0.5)
            now = time.perf_counter()
            if last is not None:
                fps = 1 / max(now - last, 1e-6)
                smoothed_fps = fps if not smoothed_fps else .85*smoothed_fps+.15*fps
            last = now
            overlay_text(annotated, f'SpinMap | player preview | FPS {smoothed_fps:.1f} | Q quit')
            display, _ = fit_frame(annotated)
            cv2.imshow('SpinMap - Player Tracking', display)
            if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
