# Upload manifest and packaging guide

This package is organized so that every manuscript claim can be audited without
searching a flat directory. Paths are relative to the package root.

## 1. Manuscript

- `manuscript/119.tex` - revised manuscript source.
- `manuscript/119_preview.pdf` - compiled 36-page A4 preview (Noto Sans).
- `manuscript/README_119.md` - compile notes and changed sections.
- `manuscript/118会思路_响应与实施报告.md` - point-by-point response.

## 2. Figures

- `figures/pdf/` - vector PDFs, Figure 1-9 and S1-S4.
- `figures/png_1000dpi/` - 1000 dpi RGB PNGs.
- `figures/112_all_figures.pdf` - combined contact sheet.

## 3. Source data (tidy CSV)

- `source_data/figure2_nyu_clustering.csv`
- `source_data/figure3_reliability_sweep.csv`
- `source_data/figure4_sapf_comparison.csv`
- `source_data/figure5_go1_closed_loop.csv`
- `source_data/figure7_dataset_atlas.csv`
- `source_data/figure7_class_color_codes.csv`
- `source_data/figure8_clustering_embeddings.csv`
- `source_data/figure8_method_metrics.csv`
- `source_data/figure8_per_class_recall.csv`
- `source_data/figure8_reliability_weights.csv`
- `source_data/figure9_runtime.csv`
- `source_data/figure9_paired_statistics.csv`
- `source_data/figure9_calibration_curve.csv`
- `source_data/figure9_per_seed_accuracy.csv`
- `source_data/figure9_per_class_recall.csv`
- `source_data/figureS1..S4*.csv`

## 4. Dataset descriptors, models, predictions

- `data/Zv_real.npy`, `data/Zl_real.npy`, `data/Y_real.npy`
- `data/Zv_real_raw.npy`, `data/Zl_real_raw.npy`
- `data/nyu_feature_meta.json`
- `data/ahpff_go1_calibration.npz`
- `data/ahpff_go1_model.joblib`, `data/ahpff_go1_model.json`
- `data/ahpff_go1_model_supervised.joblib`
- `data/ahpff_go1_predictions*.npz`
- `data/ahpff_guidedog_calibration.npz`, `data/ahpff_guidedog_model*`

The raw 2.97 GB `nyu_depth_v2_labeled.mat` is not included. Use
`scripts/extract_nyu_expert.py` with a local copy of the public archive.

## 5. Dataset samples (visual audit)

- `dataset_samples/nyu/` - 51 frames x {RGB JPEG, depth NPZ, dense-mask PNG,
  overlay JPEG}. Filenames encode class and NYU frame index, e.g.
  `Stairs_frame0094_rgb.jpg`.

## 6. Dataset cards and experiment protocols

- `dataset_cards/nyu_depth_v2_verified_subset.md`
- `dataset_cards/go1_street_dataset.md`
- `dataset_cards/standalone_sapf_simulation.md`
- `dataset_cards/cross_illumination.md`
- `experiment_protocols/experiment_matrix.md`
- `experiment_protocols/degradation_and_calibration.md`

## 7. Code

- `scripts/` - clustering, SAPF, figure generation, verification:
  - `ahpff.py`, `sapf.py`, `eval_sapf.py`
  - `extract_nyu_expert.py`, `run_arw_mvc_expert.py`
  - `run_reliability_sweep.py`, `train_ahpff_guidedog.py`
  - `publication_figures.py`, `verify_paper_numbers.py`
  - `meeting118_dataset_atlas.py`
  - `meeting118_clustering_qualitative.py`
  - `meeting118_efficiency_statistics.py`
- `ros_ws/` - perception node, SAPF node, worlds, launch files, run
  scripts.

## 8. Results, logs, checksums

- `results/figures/` - JSON, logs, trajectories, representative PNGs.
- `checksums.sha256` - SHA-256 for all PDF/PNG deliverables and the main data
  tables.
- `MANIFEST.txt` - figure index.
- `README.txt` - technical specification and reproduction commands.

## Minimal reproduction sequence

```bash
# 1. NYU descriptors and benchmark
python3 scripts/extract_nyu_expert.py
python3 scripts/run_arw_mvc_expert.py

# 2. Three new 118 figures and source data
python3 scripts/meeting118_dataset_atlas.py --output 112 --dpi 1000
python3 scripts/meeting118_clustering_qualitative.py --output 112 --dpi 1000
python3 scripts/meeting118_efficiency_statistics.py --output 112 --dpi 1000

# 3. Paper numbers and figure deliverables
python3 scripts/verify_paper_numbers.py --figures 112
```

For the ROS/Gazebo closed loop, see `ros_ws/run_ahpff_calibration_go1.sh`,
`run_ahpff_sapf_go1.sh`, and `run_sensor_degradation_go1_repeats.sh`.
