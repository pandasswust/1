# NYU Depth V2 verified terrain subset

## Role in the manuscript

E0 (dataset atlas), E1 (clean clustering), E2 (asymmetric degradation),
E3 (qualitative clustering), E5 (reliability sweep), E6 (temperature
sensitivity), and E11 (paired statistics).

## Source and licence

- Dataset: NYU Depth V2, N. Silberman, D. Hoiem, P. Kohli, R. Fergus,
  *Indoor segmentation and support inference from RGBD images*, ECCV 2012.
- Local archive used for extraction: `nyu_depth_v2_labeled.mat` (2.97 GB,
  1449 RGB-D frames, 894 annotated classes).
- The raw archive is **not redistributed** in this package. The extraction
  script `scripts/extract_nyu_expert.py` reproduces every derived file from the
  public archive. Users must follow the original NYU Depth V2 research-use
  terms.

## Verified labels and frame selection

| Class | NYU label id (1-based) | Candidate frames | Selected frames |
|---|---|---:|---:|
| Flat | 11 | 1246 | 17 |
| Stairs | 215 | 17 | 17 |
| Obstacle / chair proxy | 5 | 654 | 17 |

The selection keeps all 17 frames that contain at least 100 annotated stair
pixels in the full labelled set, then draws 17 frames per class so that the
three classes are exactly balanced. A recurring indexing error in prior work
uses 1-based ids 1/42/4 (book/shelves/ceiling); this package uses 11/215/5.

Selected frame indices:

- Flat: 85, 147, 322, 443, 449, 574, 609, 626, 915, 1125, 1134, 1159, 1222,
  1231, 1243, 1316, 1411
- Stairs: 94, 343, 351, 352, 432, 577, 600, 601, 849, 853, 855, 1201, 1244,
  1362, 1405, 1431, 1432
- Obstacle: 199, 277, 421, 886, 988, 1011, 1060, 1083, 1110, 1211, 1262,
  1284, 1293, 1346, 1361, 1398, 1445

## Descriptors

- Visual block (7-D): Canny edge density, Laplacian variance, mean intensity,
  intensity standard deviation, mean gradient, gradient standard deviation,
  8-bin Shannon entropy.
- Depth block (11-D): mean depth, depth standard deviation, mean absolute
  horizontal/vertical Sobel response, mean depth-gradient magnitude,
  depth-gradient standard deviation, 25th/75th percentiles, depth span,
  vertical/horizontal gradient ratio, valid-depth ratio.
- Descriptors are computed inside the ground-truth object mask to isolate
  multi-view fusion quality from segmentation error.

## Degradation protocol

Levels 0.0 (clean), 1.0, 1.5, 2.0. At each non-zero level, half of the samples
per class are visually degraded (overexposure, contrast/sharpness loss,
descriptor noise) and the other half are depth-degraded (pixel dropout,
compressed span, gradient noise), so every degraded sample loses exactly one
modality. Each level is repeated over 20 seeds.

## Uploaded files

- `data/Zv_real.npy`, `data/Zl_real.npy`, `data/Y_real.npy`: standardized
  descriptors and labels.
- `data/Zv_real_raw.npy`, `data/Zl_real_raw.npy`: raw descriptors used for
  health/weight computation.
- `data/nyu_feature_meta.json`: class mapping, frame lists, feature names.
- `112/dataset_samples/nyu/`: 51 RGB JPEGs, 51 metric-depth NPZ arrays, 51 dense
  mask PNGs, 51 overlay JPEGs.
- `112/source_data/figure7_dataset_atlas.csv`: per-frame pixel counts, valid
  depth ratio, mean depth and depth span.
- `figures/prl_robustness_eval_expert.json` and
  `figures/meeting118_per_seed_metrics.csv`: benchmark metrics and per-seed
  results.

## Caveats

This is a verified **pilot** subset, not a large benchmark. Its purpose is
controlled, sample-efficient degradation analysis; large-scale outdoor
validation is declared as future work.
