"""Phase 1C: Live tabletop rectification. No ball/foot locations are inferred."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .common import open_capture, overlay_text, source_arg, fit_frame


def warp_image(frame, config):
    resolution = config['source_resolution']
    actual_h, actual_w = frame.shape[:2]
    if (actual_w, actual_h) != (resolution['width'], resolution['height']):
        raise RuntimeError(
            f"Calibration expects {resolution['width']}x{resolution['height']} but source is "
            f"{actual_w}x{actual_h}. Recalibrate at this video resolution.")
    H = np.asarray(config['H_image_to_table_pixels'], dtype=np.float64)
    out = config['warp_size_px']
    warped = cv2.warpPerspective(frame, H, (out['width'], out['height']))
    return warped


def main():
    p = argparse.ArgumentParser(description='SpinMap - live top-down table preview')
    p.add_argument('--source', type=source_arg, default=0)
    p.add_argument('--config', default='configs/table_calibration.json')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    cap = open_capture(args.source)
    outdir = Path('outputs')
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Source ended or disconnected.')
                break
            warped = warp_image(frame, config)
            h, w = warped.shape[:2]
            cv2.line(warped, (0, h//2), (w-1, h//2), (255,255,255), 2)
            cv2.line(warped, (w//2, 0), (w//2, h-1), (170,170,170), 1)
            overlay_text(warped, 'SpinMap: table plane only', x=12, y=28)
            display, _ = fit_frame(warped, max_w=620, max_h=760)
            cv2.imshow('SpinMap - Rectified Table (Q quit, S screenshot)', display)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                path = outdir / f'warped_table_{datetime.now():%Y%m%d_%H%M%S}.png'
                cv2.imwrite(str(path), warped)
                print('Saved', path)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
