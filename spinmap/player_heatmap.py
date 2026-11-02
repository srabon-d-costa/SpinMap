"""SpinMap Phase 2.3: generate camera-pixel player movement heatmaps.

This module consumes outputs/player_tracks.csv from the Phase 2.2 tracker.
The resulting maps are IMAGE-SPACE visualizations only. They are not floor
coordinates and must not be used to calculate physical distance or speed.

Example:
    python -m spinmap.player_heatmap --csv outputs/player_tracks.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate separate and combined SpinMap player heatmaps."
    )
    parser.add_argument("--csv", default="outputs/player_tracks.csv", help="Phase 2.2 tracking CSV")
    parser.add_argument("--output-dir", default="outputs/heatmaps", help="Directory for PNG outputs")
    parser.add_argument("--width", type=int, default=1920, help="Source video width in pixels")
    parser.add_argument("--height", type=int, default=1080, help="Source video height in pixels")
    parser.add_argument("--sigma", type=float, default=22.0, help="Gaussian blur sigma in pixels")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum detection confidence")
    return parser.parse_args()


def read_positions(csv_path: Path, min_confidence: float) -> dict[str, list[tuple[float, float]]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Tracking CSV not found: {csv_path}")

    required = {"player", "detected", "confidence", "foot_x_px", "foot_y_px"}
    positions: dict[str, list[tuple[float, float]]] = {"P1": [], "P2": []}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(
                "CSV is missing required columns: " + ", ".join(sorted(missing))
                + f". Found columns: {', '.join(reader.fieldnames or [])}"
            )

        for row_number, row in enumerate(reader, start=2):
            player = (row.get("player") or "").strip().upper()
            if player not in positions:
                continue
            if (row.get("detected") or "0").strip() not in {"1", "true", "True"}:
                continue
            try:
                confidence = float(row["confidence"])
                x = float(row["foot_x_px"])
                y = float(row["foot_y_px"])
            except (TypeError, ValueError, KeyError):
                continue
            if confidence < min_confidence or not np.isfinite([x, y]).all():
                continue
            positions[player].append((x, y))

    return positions


def make_heatmap(
    points: list[tuple[float, float]], width: int, height: int, sigma: float, title: str
) -> np.ndarray:
    density = np.zeros((height, width), dtype=np.float32)
    valid_count = 0
    for x, y in points:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < width and 0 <= iy < height:
            density[iy, ix] += 1.0
            valid_count += 1

    if valid_count == 0:
        raise ValueError(f"No valid in-frame points available for {title}.")

    # Smooth point samples into a spatial occupancy density.
    blurred = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_TURBO)

    # Keep zero-density regions visually subdued rather than showing a full-frame color.
    mask = (normalized > 1.0).astype(np.float32)[..., None]
    dark = np.full_like(colored, 28)
    image = (colored.astype(np.float32) * mask + dark.astype(np.float32) * (1.0 - mask)).astype(np.uint8)

    cv2.putText(image, title, (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        image,
        f"Valid ankle observations: {valid_count} | camera-pixel space only",
        (28, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return image


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.sigma <= 0:
        raise SystemExit("width, height, and sigma must be positive.")

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = read_positions(csv_path, args.min_confidence)
    combined = positions["P1"] + positions["P2"]

    outputs: list[tuple[str, list[tuple[float, float]], str]] = [
        ("player_1_heatmap.png", positions["P1"], "SpinMap - Player 1 Movement Density"),
        ("player_2_heatmap.png", positions["P2"], "SpinMap - Player 2 Movement Density"),
        ("combined_heatmap.png", combined, "SpinMap - Combined Player Movement Density"),
    ]

    for filename, points, title in outputs:
        if not points:
            print(f"SKIP {filename}: no valid observations.")
            continue
        image = make_heatmap(points, args.width, args.height, args.sigma, title)
        destination = output_dir / filename
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Could not write heatmap: {destination}")
        print(f"Saved {destination} ({len(points)} observations)")

    print("Note: these are image-space heatmaps. Floor-plane calibration is required")
    print("before interpreting positions as physical court locations or distances.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
