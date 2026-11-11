# Degradation and calibration protocols

## NYU asymmetric degradation

- Levels: 0.0 (clean), 1.0, 1.5, 2.0.
- At each non-zero level, 50% of samples per class are visually degraded:
  overexposure (`mean_intensity += 0.25 * level * (1 - mean_intensity)`),
  contrast loss (`std_intensity *= 1 - 0.65 * level`), sharpness loss
  (`laplacian_variance *= 1 - 0.80 * level`), edge/gradient attenuation,
  entropy loss, and Gaussian descriptor noise `N(0, (0.04 * level)^2)`.
- The other 50% are depth-degraded: valid-ratio dropout up to 90%, depth-span
  compression, gradient inflation, and Gaussian descriptor noise.
- Every degraded sample loses exactly one modality. The degraded modality
  assignment is stratified and reproducible from the seed.
- 20 seeds per level; the same seeds are used across all seven methods.

## Go1 closed-loop degradation

- Corruption window: 6-20 s.
- Overexposure, horizontal motion blur, and 50% depth-pixel dropout applied
  simultaneously.
- Baseline and degraded sessions use the same kinematic reference trajectory,
  observation schedule, and outage windows.

## Calibration and blocked CV (Go1-351)

- Per-view scalers, RW-MVK centroids, and majority-vote cluster-to-class maps
  are fitted only on training blocks.
- Blocked 5-fold: sort each class by timestamp, split into five contiguous
  blocks, and assign block b of every class to fold b.
- Random 5-fold is reported only to quantify the leakage-induced optimism.
- The supervised ExtraTrees model is an offline upper bound, never used by the
  online node.

## Label-efficiency protocol (E11)

- For each blocked fold, draw a stratified subset of the training indices at
  fractions 25%, 40%, 55%, 70%, 85%, 100%.
- Repeat five times per fold (25 evaluations per fraction).
- Fit the RW-MVK readout (K = min(32, n_train - 1)) and an ExtraTrees upper
  bound on exactly the same subset; evaluate on the held-out temporal block.
- Report mean +/- s.e.m. across the 25 evaluations.
