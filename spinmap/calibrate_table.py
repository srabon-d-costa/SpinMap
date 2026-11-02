"""Phase 1B: Manually label tabletop quadrilateral and save a homography.

Click order: far-left -> far-right -> near-right -> near-left
relative to the two table ends in the camera view. When camera layout differs,
choose one end as the far end and remain consistent with the order.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .common import fit_frame, get_frame, open_capture, overlay_text, source_arg

TABLE_WIDTH_M = 1.525
TABLE_LENGTH_M = 2.74


def estimate_homography(corners, output_size=(610, 1096)):
    if len(corners) != 4:
        raise ValueError('Exactly four ordered table corners are required')
    src = np.asarray(corners, dtype=np.float32)
    w, h = output_size
    dst = np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                     dtype=np.float32)
    # Reject near-degenerate four-point input before saving a misleading warp.
    contour_area = abs(float(cv2.contourArea(src.reshape((-1, 1, 2)))))
    if contour_area < 150:
        raise ValueError('Corner polygon is too small or degenerate')
    H = cv2.getPerspectiveTransform(src, dst)
    if abs(float(np.linalg.det(H))) < 1e-12:
        raise ValueError('Homography is singular; reclick the table corners')
    return H


def main():
    p = argparse.ArgumentParser(description='SpinMap - four-corner tabletop calibration')
    p.add_argument('--source', type=source_arg, default=0)
    p.add_argument('--skip', type=int, default=20,
                   help='Frames to skip before freezing the calibration view')
    p.add_argument('--out', default='configs/table_calibration.json')
    args = p.parse_args()
    cap = open_capture(args.source)
    try:
        frame = get_frame(cap, args.skip)
    finally:
        cap.release()
    full_h, full_w = frame.shape[:2]
    base, scale = fit_frame(frame)
    points = []
    win = 'SpinMap - Click FL, FR, NR, NL (far left/right, near right/left)'
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))

    cv2.setMouseCallback(win, mouse)
    print('Click: FAR-left, FAR-right, NEAR-right, NEAR-left. S save, C clear, Q quit.')
    try:
        while True:
            disp = base.copy()
            display_points = [(round(x * scale), round(y * scale)) for x, y in points]
            names = ['FL', 'FR', 'NR', 'NL']
            for i, (x, y) in enumerate(display_points):
                cv2.circle(disp, (x, y), 6, (0, 255, 255), -1)
                overlay_text(disp, names[i], x + 8, max(22, y - 6), 0.65)
                if i:
                    cv2.line(disp, display_points[i - 1], (x, y), (0, 255, 255), 2)
            if len(display_points) == 4:
                cv2.line(disp, display_points[3], display_points[0], (0, 255, 255), 2)
            overlay_text(disp, f'Corners: {len(points)}/4 | S: save | C: clear | Q: quit')
            cv2.imshow(win, disp)
            key = cv2.waitKey(30) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord('c'):
                points.clear()
            if key == ord('s'):
                if len(points) != 4:
                    print('Please click all 4 corners before saving.')
                    continue
                try:
                    H = estimate_homography(points)
                except ValueError as e:
                    print('Calibration error:', e)
                    continue
                file = Path(args.out)
                file.parent.mkdir(exist_ok=True, parents=True)
                config = {
                    'camera_source': str(args.source),
                    'source_resolution': {'width': full_w, 'height': full_h},
                    'corner_order': ['far_left', 'far_right', 'near_right', 'near_left'],
                    'table_corners_px': [[round(x, 2), round(y, 2)] for x, y in points],
                    'table_dimensions_m': {'width': TABLE_WIDTH_M, 'length': TABLE_LENGTH_M},
                    'warp_size_px': {'width': 610, 'height': 1096},
                    'H_image_to_table_pixels': H.tolist(),
                    'scope': 'tabletop plane only; DO NOT use for player floor positions'
                }
                file.write_text(json.dumps(config, indent=2), encoding='utf-8')
                print('Saved calibration:', file)
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
