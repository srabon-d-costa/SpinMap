
"""SpinMap Phase 2.5: image-space and normalized ROI movement heatmaps.

Preserves the original camera-pixel heatmaps and additionally generates
approximate normalized ROI heatmaps using a saved homography.

IMPORTANT:
The normalized ROI is not a verified physical court-plane calibration.
Do not interpret it as metres, physical distance, or speed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SpinMap image-space and normalized ROI heatmaps."
    )

    parser.add_argument(
        "--csv",
        default="outputs/player_tracks.csv",
        help="Phase 2.2 tracking CSV"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/heatmaps",
        help="Directory for PNG outputs"
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--sigma", type=float, default=22.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)

    parser.add_argument(
        "--calibration",
        default="configs/floor_calibration.json",
        help="Normalized ROI calibration JSON"
    )

    return parser.parse_args()


def read_positions(
    csv_path: Path,
    min_confidence: float
) -> dict[str, list[tuple[float, float]]]:

    if not csv_path.is_file():
        raise FileNotFoundError(f"Tracking CSV not found: {csv_path}")

    required = {
        "player",
        "detected",
        "confidence",
        "foot_x_px",
        "foot_y_px"
    }

    positions = {"P1": [], "P2": []}

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as handle:

        reader = csv.DictReader(handle)

        fields = set(reader.fieldnames or [])
        missing = required - fields

        if missing:
            raise ValueError(
                "CSV missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:

            player = (row.get("player") or "").strip().upper()

            if player not in positions:
                continue

            detected = (row.get("detected") or "0").strip().lower()

            if detected not in {"1", "true", "yes"}:
                continue

            try:
                confidence = float(row["confidence"])
                x = float(row["foot_x_px"])
                y = float(row["foot_y_px"])

            except (TypeError, ValueError, KeyError):
                continue

            if not np.isfinite([confidence, x, y]).all():
                continue

            if confidence < min_confidence:
                continue

            positions[player].append((x, y))

    return positions


def load_calibration(
    calibration_path: Path,
    image_width: int,
    image_height: int
) -> dict:

    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_path}"
        )

    with calibration_path.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)

    if calibration.get("coordinate_units") != "normalized":
        raise ValueError(
            "This workflow expects normalized calibration."
        )

    if calibration.get("metric_calibrated") is not False:
        raise ValueError(
            "Expected metric_calibrated=False."
        )

    if (
        calibration.get("image_width_px") != image_width
        or calibration.get("image_height_px") != image_height
    ):
        raise ValueError(
            "Calibration image dimensions do not match "
            "the supplied video dimensions."
        )

    H = np.asarray(
        calibration["homography_image_to_floor"],
        dtype=np.float64
    )

    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("Invalid homography matrix.")

    if abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("Homography matrix is singular.")

    target_width = float(calibration["target_width"])
    target_depth = float(calibration["target_depth"])

    map_width = int(calibration["map_width"])
    map_height = int(calibration["map_height"])

    if min(target_width, target_depth, map_width, map_height) <= 0:
        raise ValueError("Invalid normalized ROI dimensions.")

    calibration["_H"] = H
    calibration["_target_width"] = target_width
    calibration["_target_depth"] = target_depth
    calibration["_map_width"] = map_width
    calibration["_map_height"] = map_height

    return calibration


def transform_positions(
    positions: dict[str, list[tuple[float, float]]],
    calibration: dict
) -> tuple[dict, dict]:

    H = calibration["_H"]

    target_width = calibration["_target_width"]
    target_depth = calibration["_target_depth"]

    map_width = calibration["_map_width"]
    map_height = calibration["_map_height"]

    transformed = {"P1": [], "P2": []}
    statistics = {}

    for player, points in positions.items():

        kept = 0
        rejected = 0

        if not points:
            statistics[player] = (kept, rejected)
            continue

        source = np.asarray(
            points,
            dtype=np.float32
        ).reshape(-1, 1, 2)

        projected = cv2.perspectiveTransform(
            source,
            H.astype(np.float64)
        ).reshape(-1, 2)

        for x, y in projected:

            if not np.isfinite([x, y]).all():
                rejected += 1
                continue

            # Keep only positions inside the normalized ROI.
            if not (
                0.0 <= x <= target_width
                and 0.0 <= y <= target_depth
            ):
                rejected += 1
                continue

            # Convert normalized coordinates into heatmap pixels.
            px = (x / target_width) * (map_width - 1)
            py = (y / target_depth) * (map_height - 1)

            transformed[player].append((px, py))
            kept += 1

        statistics[player] = (kept, rejected)

    return transformed, statistics


def make_heatmap(
    points: list[tuple[float, float]],
    width: int,
    height: int,
    sigma: float,
    title: str,
    space_label: str
) -> np.ndarray:

    density = np.zeros(
        (height, width),
        dtype=np.float32
    )

    valid_count = 0

    for x, y in points:

        ix = int(round(x))
        iy = int(round(y))

        if 0 <= ix < width and 0 <= iy < height:
            density[iy, ix] += 1.0
            valid_count += 1

    if valid_count == 0:
        raise ValueError(
            f"No valid observations available for {title}."
        )

    blurred = cv2.GaussianBlur(
        density,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    normalized = cv2.normalize(
        blurred,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    colored = cv2.applyColorMap(
        normalized.astype(np.uint8),
        cv2.COLORMAP_TURBO
    )

    mask = (normalized > 1.0).astype(np.float32)[..., None]

    dark = np.full_like(colored, 28)

    image = (
        colored.astype(np.float32) * mask
        + dark.astype(np.float32) * (1.0 - mask)
    ).astype(np.uint8)

    cv2.putText(
        image,
        title,
        (28, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        f"Valid ankle observations: {valid_count} | {space_label}",
        (28, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return image


def save_heatmap_set(
    positions: dict,
    output_dir: Path,
    width: int,
    height: int,
    sigma: float,
    space_label: str,
    suffix: str = ""
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    combined = positions["P1"] + positions["P2"]

    outputs = [
        (
            f"player_1_heatmap{suffix}.png",
            positions["P1"],
            "SpinMap - Player 1 Movement Density"
        ),
        (
            f"player_2_heatmap{suffix}.png",
            positions["P2"],
            "SpinMap - Player 2 Movement Density"
        ),
        (
            f"combined_heatmap{suffix}.png",
            combined,
            "SpinMap - Combined Player Movement Density"
        )
    ]

    for filename, points, title in outputs:

        if not points:
            print(f"SKIP {filename}: no valid observations.")
            continue

        image = make_heatmap(
            points,
            width,
            height,
            sigma,
            title,
            space_label
        )

        destination = output_dir / filename

        if not cv2.imwrite(str(destination), image):
            raise OSError(
                f"Could not write heatmap: {destination}"
            )

        print(
            f"Saved {destination} ({len(points)} observations)"
        )


def main() -> int:

    args = parse_args()

    if (
        args.width <= 0
        or args.height <= 0
        or args.sigma <= 0
        or args.min_confidence < 0
    ):
        raise SystemExit("Invalid heatmap parameters.")

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== SPINMAP PHASE 2.5 ===")

    positions = read_positions(
        csv_path,
        args.min_confidence
    )

    print(
        f"Tracking observations: "
        f"P1={len(positions['P1'])}, "
        f"P2={len(positions['P2'])}"
    )

    # --------------------------------------------------
    # 1. Original camera-pixel heatmaps
    # --------------------------------------------------

    print("\n[1/2] Generating original image-space heatmaps...")

    save_heatmap_set(
        positions,
        output_dir,
        args.width,
        args.height,
        args.sigma,
        "camera-pixel space only"
    )

    # --------------------------------------------------
    # 2. Normalized ROI heatmaps
    # --------------------------------------------------

    print("\n[2/2] Generating normalized ROI heatmaps...")

    calibration_path = Path(args.calibration)

    if not calibration_path.is_file():

        print(
            "WARNING: Calibration file not found. "
            "Normalized ROI heatmaps skipped."
        )

        return 0

    calibration = load_calibration(
        calibration_path,
        args.width,
        args.height
    )

    normalized_positions, statistics = transform_positions(
        positions,
        calibration
    )

    for player, (kept, rejected) in statistics.items():

        print(
            f"{player}: {kept} retained, "
            f"{rejected} outside normalized ROI"
        )

    normalized_dir = output_dir / "normalized_roi"

    save_heatmap_set(
        normalized_positions,
        normalized_dir,
        calibration["_map_width"],
        calibration["_map_height"],
        args.sigma,
        "approximate normalized camera ROI | NOT metres"
    )

    print("\n=== PHASE 2.5 COMPLETE ===")

    print(f"Original maps: {output_dir}")
    print(f"Normalized ROI maps: {normalized_dir}")

    print(
        "\nIMPORTANT: Normalized ROI coordinates are not "
        "verified physical court coordinates."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
