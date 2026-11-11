# Standalone 6-DoF stair-climbing simulation

## Role in the manuscript

E9: fair comparison of seven filters with a controlled, repeatable 30 s stair
trajectory.

## Protocol

- Duration: 30 s; filter rate: 50 Hz; particles: N = 1000.
- Exteroceptive pose: 5 Hz (every 10 filter steps).
- Outages: 8-12 s and 18-22 s (two 4 s blackouts).
- Filters: leg odometry (LO), standard particle filter (SPF), fixed
  large-covariance PF (LPF), additive EKF, SAPF-Pred, SAPF-Oracle, and
  SAPF-Shuffled.
- Seeds: 20 paired seeds; all filters see exactly the same observations,
  outages, and terrain schedule.
- Semantic stream for SAPF-Pred: the 3-class NYU readout at 73.0% sequence
  accuracy.

## Metrics

- 3D position RMSE, vertical Z RMSE, orientation RMSE, stair-segment Z RMSE,
  flat-segment Tr(Q), and final Z error.
- Flat-ground uncertainty accounting: Tr(Q_base) = 1.20e-5 vs
  Tr(Q_LPF) = 46.20e-5, a 38.5x inflation; the selective gate removes 97.4% of
  the accumulated flat-ground process variance.

## Uploaded files

- `figures/sapf_tracking_result.json`
- `figures/sapf_tracking_result.png`
- `112/source_data/figure4_sapf_comparison.csv`
- `scripts/sapf.py`, `scripts/eval_sapf.py`

## Caveats

Synthetic kinematic profile. It is a controlled filter benchmark, not a
hardware durability test.
