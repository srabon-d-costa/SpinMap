"""Shared helpers for camera/video capture and display."""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2


def source_arg(value: str):
    """Interpret a decimal integer as camera index, otherwise as video pathname."""
    return int(value) if value.isdecimal() else value


def open_capture(source, width=0, height=0, fps=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source!r}. Check camera permission or path.")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def get_frame(cap, skip=0):
    for _ in range(max(0, skip)):
        ok, _ = cap.read()
        if not ok:
            raise RuntimeError("Source ended before a calibration frame could be read.")
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read a frame from video/camera.")
    return frame


def fit_frame(frame, max_w=1280, max_h=760):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1:
        return cv2.resize(frame, (round(w * scale), round(h * scale))), scale
    return frame.copy(), 1.0


def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def overlay_text(frame, text, x=12, y=30, scale=0.65):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
