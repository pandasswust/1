# Go1 Gazebo street dataset

## Role in the manuscript

E4 (modality-weight responsiveness), E6 (over-clustering sensitivity), E7
(in-scene calibration and blocked CV), E8 (cross-illumination boundary), and
E10 (closed-loop filtering).

## Provenance

Controlled ROS/Gazebo co-simulation of a Unitree Go1 quadruped with a
forward-facing RGB-D camera. The scene contains a concrete sidewalk, curb and
road, brick facade, yellow tactile paving, a ten-step staircase (0.15 m rise,
0.30 m run), a bollard, street furniture, static and walking pedestrians, and a
human handler. No physical-robot photographs are used.

## Calibration set (351 frames)

| Class | Index | Frames |
|---|---:|---:|
| Flat | 0 | 59 |
| Stairs | 1 | 152 |
| Obstacle | 2 | 85 |
| Tactile | 3 | 55 |

- Feature vector: 24-D = 13 visual (7 photometric/edge statistics + mean R/G/B,
  mean HSV saturation, yellow fraction, red fraction) + 11 depth descriptors
  including `valid_ratio`.
- ROI: normalized image coordinates `(0.15, 0.85) x (0.35, 0.98)`.
- Online rate: about 10 Hz.
- Calibration protocol: temporally blocked 5-fold CV. Each class is sorted by
  timestamp and split into contiguous blocks, so adjacent video frames never
  straddle train/test. Random 5-fold CV is reported only as a leakage
  diagnostic.

## Closed-loop runs

- 3 nominal and 3 degraded 30 s runs.
- 50 Hz filter, 5 Hz exteroceptive pose updates.
- Two 4 s outages: 8-12 s and 18-22 s.
- Degradation window 6-20 s: overexposure, horizontal motion blur, and 50%
  depth-pixel dropout.
- Repeatable kinematic reference trajectory shared by all seven filters.

## Uploaded files

- `data/ahpff_go1_calibration.npz`: 351 x 24 features, labels, timestamps,
  feature names, ROI.
- `data/ahpff_go1_model.joblib`: RW-MVK cluster readout, K = 32.
- `data/ahpff_go1_model_supervised.joblib`: ExtraTrees upper bound.
- `data/ahpff_go1_model.json`: blocked/random CV, confusion matrices, mapping.
- `data/ahpff_go1_predictions*.npz`: online predictions, clusters, weights,
  health qualities.
- `figures/go1*/`, `figures/degradation_runs/`,
  `figures/reliability_runs/`, `figures/cross_illumination/`: trajectories,
  filter JSON, semantic labels, logs, representative figures.
- `ros_ws/worlds/guide_dog_street.world`,
  `ros_ws/worlds/guide_dog_street_shifted.world`,
  `ros_ws/go1_street.launch`, `ros_ws/go1_street_shifted.launch`,
  `ros_ws/ahpff_perception_node.py`, `ros_ws/sapf_gazebo_node.py`.

## Caveats

Single simulated street, not a multi-city benchmark. The repeatable kinematic
trajectory is deliberate: it isolates perception degradation and covariance
gating from low-level locomotion controller divergence. Physical-robot trials
are declared as future work.
