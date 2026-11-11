112 — publication-grade figures for the revised manuscript
===========================================================

Manuscript:
  Reliability-Aware Multi-View Terrain Clustering and Semantic-Adaptive
  State Estimation for Robotic Guide Dogs under Asymmetric Sensor Degradation

Contents
--------
  112.tex                  LaTeX source wired to the figure PDFs.
  pdf/                     One vector PDF per figure (single page).
  png_1000dpi/             One 1000 dpi PNG per figure.
  112_all_figures.pdf      All main and extended-data figures merged in order.
  source_data/             Tidy CSV data used by every data-driven panel.
  checksums.sha256         SHA-256 checksums for the PDF and PNG deliverables.

Figure order
------------
  Figure 1   Methodological roadmap (RA-AHPFF + SAPF).
  Figure 2   NYU Depth V2 clustering benchmark (Table 4/5).
  Figure 3   Reliability-to-performance sweep (Fig. 3 caption).
  Figure 4   Standalone 20-seed SAPF comparison (Table 10).
  Figure 5   Go1 closed-loop X–Z trajectory and terrain readout (Table 11).
  Figure 6   Go1 nominal scene and degraded RGB-D preview.
  Figure 7   Dataset atlas: actual RGB, depth, dense masks, colour codes, Go1 view.
  Figure 8   Stand-alone qualitative clustering comparison and per-class errors.
  Figure 9   Runtime, paired statistics, and calibration-label efficiency.
  Figure S1  Modality weights and health under four corruption modes (Table 8).
  Figure S2  Softmax-temperature and over-clustering sensitivity (Table 9).
  Figure S3  Temporally blocked 5-fold confusion matrices (Tables 6/7).
  Figure S4  Directional process covariance and flat-ground Tr(Q) (Table 3, Prop. 3).

Technical specification
-----------------------
  Font family       Liberation Sans (Arial metric compatible).
  PDF               True vector, embedded TrueType fonts (pdf.fonttype 42).
  PNG               RGB, 1000 dpi, white background, lossless.
  Palette           Okabe–Ito colour-blind-safe palette.
  Panel labels      lower-case bold letters placed outside the axes.
  Nominal widths    183 mm (double column) for all figures.

Reproduce
---------
  python3 scripts/publication_figures.py --output 112 --dpi 1000
  python3 scripts/meeting118_dataset_atlas.py --output 112 --dpi 1000
  python3 scripts/meeting118_clustering_qualitative.py --output 112 --dpi 1000
  python3 scripts/meeting118_efficiency_statistics.py --output 112 --dpi 1000
  python3 scripts/verify_paper_numbers.py --figures 112
  xelatex 119.tex          # current manuscript; requires fontspec + Noto Sans
  # Figure PDFs are also symlinked at the 112 root, so 119.tex compiles directly from 112/.

Canonical values
----------------
  Figure 2 uses the 112 manuscript Tables 4/5 values. Figure 3 reads
  figures/reliability_sweep.json (the values printed in the 112 Fig. 3 caption).
  Figure 4 reads figures/sapf_tracking_result.json (Table 10). Figure 5 reads
  figures/degradation_runs/baseline_1/ (representative 90.3% run; three-run mean
  90.1 ± 1.0%). The source_data/ CSVs make every plotted value explicit.

Notes
-----
  The NYU Table 4/5 numbers were produced with the manuscript-benchmark depth
  setting, which predates the strict r = 0 boundary in Eq. (4). The benchmark
  defaults to this manuscript-compatible setting; pass --strict-depth-boundary
  to run_arw_mvc_expert.py to enforce q_d = w_d = 0 at complete depth dropout.
  The online perception node follows the strict boundary used in Algorithm 2.

118 Meeting Revision Additions
-----------------------------
  Figure 7   Dataset atlas: actual NYU RGB, metric depth, dense annotation
             masks, mask overlays, class colour codes, and Go1 nominal /
             degraded RGB with online-versus-truth terrain strips.
  Figure 8   Stand-alone qualitative clustering comparison: common t-SNE maps
             for clean and severe data, per-class recall heatmap, and the
             silhouette-versus-accuracy compactness paradox.
  Figure 9   CPU latency, paired Wilcoxon significance over 20 seeds, per-seed
             paired gains, and calibration-label efficiency on Go1-351.

  script     scripts/meeting118_dataset_atlas.py
             scripts/meeting118_clustering_qualitative.py
             scripts/meeting118_efficiency_statistics.py

  data       source_data/figure7_*.csv, figure8_*.csv, figure9_*.csv
             dataset_samples/nyu/ (51 RGB/depth/mask/overlay samples)

  cards      dataset_cards/ and experiment_protocols/

Canonical 118 values
--------------------
  Runtime (single-thread CPU): health 0.25 ms, softmax 0.014 ms,
  readout K=3 0.040 ms, readout K=32 0.045 ms, ExtraTrees inference 15.1 ms,
  RW-MVK K=3 calibration 27.8 ms, RW-MVK K=32 calibration 235.9 ms,
  ExtraTrees training 394.2 ms.

  Paired Wilcoxon (20 seeds, Ours - baseline):
    clean: Vision +24.90 pp, Depth +6.08, Fixed +14.51, PCA +2.55,
           Uniform +14.41, Shuffled +4.41 (all p < 0.004).
    severe: Vision +17.55, Fixed +5.59, PCA +10.78, Uniform +5.78,
            Shuffled +3.82 (all p < 0.012); Depth +0.39 (p = 0.797, tied).

  Label efficiency (Go1 blocked CV, mean +/- s.e.m., 25 evaluations):
    25% labels: cluster 77.7 +/- 3.8%, supervised 76.4 +/- 4.4%
    100% labels: cluster 77.9 +/- 3.3%, supervised 80.2 +/- 4.6%
