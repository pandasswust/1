# Experiment matrix (E0-E11)

| ID | Experiment | Data and repeats | Metrics | Artifact |
|---|---|---|---|---|
| E0 | Dataset audit and visual atlas | NYU-51; Go1-351 | RGB, depth, dense masks, colour codes | `Figure7_dataset_atlas.pdf`, `dataset_samples/` |
| E1 | Clean multi-view clustering | NYU-51; 20 seeds | ACC, NMI, ARI, silhouette | Table 4, `Figure2_nyu_clustering.pdf` |
| E2 | Asymmetric degradation | NYU-51; 20 seeds x 4 levels | ACC, NMI | Table 5, Figure 2d |
| E3 | Qualitative clustering and per-class error | NYU-51; representative seed + 20-seed recall | t-SNE maps, per-class recall | `Figure8_clustering_qualitative.pdf`, per-class recall table |
| E4 | Modality-weight responsiveness | Go1; four 30 s sessions | q_v, q_d, w_v, w_d | Table 8, `FigureS1_modality_weights.pdf` |
| E5 | Reliability-to-performance sweep | NYU-51; 20 seeds x 2 sweeps | q, w, ACC, NMI | `Figure3_reliability_sweep.pdf` |
| E6 | Hyperparameter sensitivity | NYU-51, Go1-351 | ACC, NMI, macro-F1 | Table 9, `FigureS2_hyperparameter_sensitivity.pdf` |
| E7 | In-scene calibration and blocked CV | Go1-351; 5-fold blocked/random | ACC, recall, confusion | Tables 6-7, `FigureS3_blocked_cv_confusion.pdf` |
| E8 | Cross-illumination boundary | Shifted world; one 30 s run | s_v, w_v, ACC, RMSE | `dataset_cards/cross_illumination.md` |
| E9 | Standalone 6-DoF filtering | 20 seeds; seven filters | RMSE, stair Z RMSE, Tr(Q) | Table 10, `Figure4_sapf_comparison.pdf` |
| E10 | Closed-loop Go1 filtering | 3 nominal + 3 degraded runs | 3D RMSE, online ACC | Table 11, `Figure5_go1_closed_loop.pdf` |
| E11 | Runtime, paired statistics, label efficiency | NYU-51, Go1-351; 5 timing repeats, 20 paired seeds, 25 blocked evaluations | ms, Wilcoxon p, ACC vs labels | `Figure9_efficiency_and_statistics.pdf`, runtime and Wilcoxon tables |

## Why these experiments are sufficient

- E0 makes the actual data visible; E1-E6 test the clustering claim from clean
  accuracy through sample-level degradation, continuous sweeps, and
  hyperparameters; E7-E8 test calibration and domain shift; E9-E10 test the
  downstream estimator in standalone and closed-loop settings; E3 and E11 add
  the previously missing qualitative, statistical, runtime, and label-efficiency
  evidence.
- All comparisons use paired seeds or identical runs. All splits that involve
  time series use contiguous temporal blocks, not random adjacent frames.
- Negative and tied results are reported: severe depth-only is statistically
  tied on ACC; the obstacle class remains the hardest; the cross-illumination
  boundary is explicitly quantified.
