# RC-119: 118-meeting revision package

Reliability-Aware Multi-View Terrain Clustering and Semantic-Adaptive State
Estimation for Robotic Guide Dogs under Asymmetric Sensor Degradation.

This package implements the 118 meeting checklist point by point:

1. **Experiments** - E0-E11 matrix, including the newly added qualitative
   clustering, paired statistics, runtime, and label-efficiency experiments.
2. **Datasets** - NYU Depth V2 verified 51-frame subset, Go1-351 calibration,
   Go1 closed-loop runs, standalone SAPF, cross-illumination, and external raw
   NYU archive.
3. **Data to upload** - seven categories: descriptors, models, predictions,
   source-data CSVs, RGB/depth/mask samples, code/ROS files, figures/checksums.
4. **Sufficiency** - explicit gap analysis; honest report of the severe
   depth-only tie; declared scale boundary.
5. **Separate clustering comparison** - Figure 8 with common t-SNE maps,
   per-class recall heatmap, and silhouette-versus-accuracy paradox.
6. **Colour visual audit** - Figure 7 with actual RGB, depth, dense masks,
   class colour codes, and Go1 nominal/degraded/terrain-strip panels.
7. **Clear reasoning** - dataset cards, experiment protocols, source data,
   checksums, and a point-by-point response report.

## Start here

- `118会思路_响应与实施报告.md` - Chinese point-by-point response.
- `manuscript/119.tex` - revised manuscript.
- `figures/pdf/` and `figures/png_1000dpi/` - Figure 1-9 and S1-S4.
- `source_data/` - tidy CSV for every data-driven panel.
- `dataset_cards/` and `experiment_protocols/` - data and protocol metadata.
- `dataset_samples/nyu/` - 51 actual NYU RGB/depth/mask/overlay samples.
- `scripts/` - clustering, SAPF, figure generation, and verification.
- `ros_ws/` - perception node, SAPF node, worlds, launch files.
- `upload_manifest.md` - full file inventory and reproduction commands.
- `checksums.sha256` - SHA-256 for the deliverables.
- `118_checklist.csv` - one-row-per-item checklist.
- `meeting_notes/` - original 118 notes and response.
- `reporting_summary.md` - Nature-family reporting summary (data provenance,
  sample sizes, statistics, software, limitations).

## Headline revisions

- NYU clean ACC 71.1%, severe ACC 60.0%; severe macro-recall 60.0% (best).
- Paired Wilcoxon: significant over five of six baselines under severe
  corruption; depth-only is a statistically tied strong control
  (60.0% vs 59.6%, p = 0.797), reported explicitly.
- Per-frame CPU perception path ~0.31 ms vs the 100 ms / 10 Hz budget.
- Calibration readout saturates at 25% of labels: 77.7 vs 77.9% at 100%.
- Selective covariance gating matches LPF stair tracking while cutting
  flat-ground process variance by 97.4%.
