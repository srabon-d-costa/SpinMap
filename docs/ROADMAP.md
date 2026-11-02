# SpinMap implementation and research roadmap

## Working scientific claim (hypothesis, not proven novelty)

A calibrated, real-time, personalized return-difficulty model combining ball flight,
player motion and explicit predictive uncertainty may outperform landing-only and
position-only baselines in forecasting unsuccessful or weak returns.

## Stages with measurable gates

### P1: Input and tabletop calibration (current pack)
- Camera or prerecorded table-tennis file opens and frames are timestamped.
- Four table corners are annotated and stored.
- A 2.74 m by 1.525 m tabletop is rectified consistently.
- Visually inspect warped end lines/net alignment and saved screenshots.
- **Gate:** Repeatable tabletop preview. No ball-bounce or player-ground claims.

### P2: Both player IDs and floor-plane calibration
- Detect exactly two competitors; reject spectators/referees.
- Pose ankles/foot-ground contact, occlusion handling, persistent IDs.
- Calibrate separate court-floor plane with known floor markers, or a robust
  3D/ground-plane setup. Never use tabletop homography for player feet.
- **Gate:** Evaluate foot error, ID switches and 30 minutes of continuous tracking.

### P3: Ball and event analysis
- Use specialized temporal, small-object detector and short-gap reconstruction.
- Separate racket-hit vs table-bounce vs net-contact events using time sequences.
- Assign side, bounce location, and event confidence; exclude low-confidence events.
- A camera-projected point on table does NOT by itself mean a bounce occurred.
- **Gate:** Validate detected events with independently reviewed annotations.

### P4: Analytics engine
- Time-weighted, ground-calibrated player occupancy heatmaps.
- Confirmed-bounce event density in tabletop coordinates.
- Footwork distance, lateral acceleration, recovery and normalized tactical zones.
- **Gate:** Compare computed movement distances against measured trajectories.

### P5: U-CAPM research
- Train return-failure prediction with fixed prediction timestamp.
- Feature groups: incoming-ball velocity/arrival, receiver state, previous
  player-specific outcomes, confidence/noise indicators.
- Generate conditional return-failure probability on a spatial candidate grid.
- Calibrate probabilities and display confidence/uncertainty separately.
- Abstain on missing/occluded trajectories; avoid overconfident coaching claims.
- **Gate:** Subject-disjoint evaluation; outperform landing-only and reachability baselines.

### P6: Live dashboard and research evaluation
- WebSocket live event pipeline, heatmaps and per-rally summary.
- Evaluate camera-to-display latency (median, p95), actual FPS and coverage.
- Log missing data and processing speed for each module.
- Compare: (A) landing-only (B) landing+receiver location
  (C) +motion/trajectory (D) +player adaptation (E) +uncertainty calibration.
- Test cross-player, cross-session and cross-camera generalization.

## Minimal labeling schema
Each point/rally ID should include:
- video_id, session_id, player_id, timestamps, camera_id
- impact_time, bounce_time, bounce_xy_m, table_side, certainty
- receiver_position_xy_m on floor, velocity_xy_mps, validity
- ball_features from **before prediction time** (prevent data leakage)
- actual return: successful, unsuccessful, weak return (if defined)
- labels verified by two annotators where feasible

## Research limitations to disclose
- Single-view cameras cannot robustly recover true ball 3D velocity or spin.
- Landing prediction from one frame is inadequate; temporal trajectory needed.
- Tactical recommendation without prospective testing is exploratory.
- Pretrained-model use is implementation, not standalone novelty.
- Personalized weakness requires repeated unbiased shot/rally observations.
- Other papers already cover tracking, control areas and shot maps.
