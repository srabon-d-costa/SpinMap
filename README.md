# SpinMap - Phase 1 Starter Kit

Real-time table tennis player/ball analytics and exploratory, uncertainty-aware
adaptive tactical pressure prediction.

**This package implements Phase 1 tabletop calibration and an OPTIONAL Phase 2
player-tracking preview. It does not yet detect ball bounces, recover player
floor positions, create true heatmaps or implement U-CAPM.**

## Requirements

- Windows / Linux / macOS with Python **3.10+** and a desktop display (OpenCV windows).
- USB camera/webcam or local video file showing a real table tennis table.
- Fixed camera, visible table edges, minimal shake and consistent resolution.
- Later stages benefit from 120-FPS capture and NVIDIA GPU, but Phase 1 runs on CPU.

## Install (Windows PowerShell)

```powershell
cd SpinMap_Starter
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux/macOS use `python3 -m venv .venv` and `source .venv/bin/activate`.

## Step 1 - Test a camera

```powershell
python -m spinmap.camera_test --source 0
```

Or test a video and check the view before calibration:

```powershell
python -m spinmap.camera_test --source "data/raw/match.mp4"
```

`S` saves a full-resolution sample image to `outputs/`; `Q` quits.
For camera settings you may try `--width 1920 --height 1080 --fps 120`;
requested FPS is not always achieved and must be measured.

## Step 2 - Click the four tabletop corners

```powershell
python -m spinmap.calibrate_table --source 0
```

Or, for a file:

```powershell
python -m spinmap.calibrate_table --source "data/raw/match.mp4" --skip 20
```

Click in this consistent order (observed from your chosen camera viewpoint):

1. FAR-LEFT tabletop corner (FL)
2. FAR-RIGHT tabletop corner (FR)
3. NEAR-RIGHT tabletop corner (NR)
4. NEAR-LEFT tabletop corner (NL)

Press `S` to save `configs/table_calibration.json`. `C` clears points; `Q` quits.
Corner selection assumes the camera sees opposing table ends along the
length of the table. For a different angle, consistently pick one end as FAR.
A good camera view shows all four corners with minimal occlusion.

## Step 3 - Validate rectified top-down live table

```powershell
python -m spinmap.view_table --source 0
```

Or use the same video file and resolution you calibrated:

```powershell
python -m spinmap.view_table --source "data/raw/match.mp4"
```

Press `S` to save the rectified table image in `outputs/`.

**Important:** This homography is ONLY for the table plane; do not apply
it to players' feet or to a flying ball. Their 3D/floor geometry differs.
You should see the entire table approximately rectangular with the net at
halfway along its length; reannotate corners if it is distorted.

## Optional - Preview player detection (starts Phase 2)

```powershell
pip install -r requirements-player.txt
python -m spinmap.track_players --source 0
```

Or with prerecorded video:

```powershell
python -m spinmap.track_players --source "data/raw/match.mp4"
```

On first launch Ultralytics may download a pose-model checkpoint. Player
IDs are provisional; spectators can be detected and pose ankles are
IMAGE PIXELS only, not physical floor positions. Don't interpret these
as valid motion speed or real-world heatmaps yet.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## File layout

```
SpinMap_Starter/
  spinmap/camera_test.py        live/video capture and FPS preview
  spinmap/calibrate_table.py    annotated table corners + homography
  spinmap/view_table.py        rectified table preview
  spinmap/track_players.py     optional YOLO pose tracking preview
  configs/                     generated calibration JSON
  data/raw/                    private local videos (do not commit)
  outputs/                     saved frames
  docs/ROADMAP.md              research architecture and validation gates
  tests/test_calibration.py    deterministic geometry tests
```

## Next development tasks

1. Review 20-30 annotated scenes across lighting and camera positions.
2. Complete physical **floor-plane** player calibration.
3. Build a specialized temporal ball detector and independent bounce event model.
4. Only then generate physically meaningful occupancy and landing heatmaps.
5. Add U-CAPM after enough validated per-shot training data exists.

This is a scaffold, not a finished research implementation.
