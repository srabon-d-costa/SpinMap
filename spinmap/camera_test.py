"""Phase 1A: Check camera/live video input, optionally save a frame."""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from .common import open_capture, fit_frame, source_arg, overlay_text


def main():
    parser = argparse.ArgumentParser(description="SpinMap - test camera/video feed")
    parser.add_argument('--source', type=source_arg, default=0,
                        help='Camera ID (0, 1, ...) or video filename')
    parser.add_argument('--width', type=int, default=0)
    parser.add_argument('--height', type=int, default=0)
    parser.add_argument('--fps', type=float, default=0)
    args = parser.parse_args()
    cap = open_capture(args.source, args.width, args.height, args.fps)
    print('Camera requested:', args.source)
    print('Device reported:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'x',
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
          '@', cap.get(cv2.CAP_PROP_FPS), 'fps (reported, not measured)')
    prev = None
    fps_smoothed = 0.0
    outdir = Path('outputs')
    outdir.mkdir(exist_ok=True, parents=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Video ended or camera disconnected.')
                break
            now = time.perf_counter()
            if prev is not None:
                inst = 1 / max(now - prev, 1e-6)
                fps_smoothed = 0.85 * fps_smoothed + 0.15 * inst if fps_smoothed else inst
            prev = now
            display, _ = fit_frame(frame)
            overlay_text(display, f'SpinMap | Input FPS: {fps_smoothed:.1f} | S: save frame | Q: quit')
            cv2.imshow('SpinMap - Camera Test', display)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                file = outdir / f'camera_frame_{datetime.now():%Y%m%d_%H%M%S}.png'
                cv2.imwrite(str(file), frame)
                print('Saved', file)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
