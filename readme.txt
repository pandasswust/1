Q2_RAS_Project: Reliability-Aware RGB-D Terrain Clustering for Robotic Guide Dogs

This repository accompanies the revised manuscript
"Reliability-Aware Multi-View Terrain Clustering and Semantic-Adaptive State
Estimation for Robotic Guide Dogs under Asymmetric Sensor Degradation".

Revision status
---------------
The previous version used silhouette scores to define modality weights, added PCA
scores from two sensor spaces, and used a supervised ExtraTrees classifier in the
online loop while describing an offline clustering ablation. The revised pipeline
addresses those issues:

- modality weights are computed per sample from sensor-health indicators
  (exposure, contrast, sharpness, entropy, depth valid ratio, depth smoothness);
- the standardized visual and depth blocks are combined by weighted
  concatenation, never by adding PCA scores from different sensor spaces;
- clustering uses reliability-weighted multi-view K-Means (RW-MVK);
- cluster indices are mapped to terrain classes by majority vote on a labelled
  calibration set, and the online node assigns a sample to the nearest cluster
  centroid before publishing the terrain label;
- Silhouette scores are used only for evaluation and optional K diagnostics;
- the SAPF is compared with odometry, a standard PF, a fixed large-covariance PF,
  an EKF, an oracle-label SAPF, and a shuffled-label SAPF;
- the exteroceptive pose is available at 5 Hz with two 4 s outage windows,
  instead of an idealized 50 Hz absolute pose;
- the 112 revision adds Proposition 1 (finite-step monotone RW-MVK),
  Proposition 2 (exponential attenuation of a corrupted view), and
  Proposition 3 (selective covariance gating vs. the elementwise-maximum
  Large-Covariance PF), together with the seven-filter comparison and the
  lambda / over-clustering sensitivity study;
- all manuscript figures, including the Go1 street-scene and degraded-preview
  panels, are generated at NCS/Elsevier quality in the 112/ package, including
  vector PDFs and 1000 dpi PNGs.

The corrected NYU label mapping is:
  Flat:     11
  Stairs:   215
  Obstacle: 5 (chair proxy)

NYU Depth V2 contains 17 annotated stair frames with at least 100 annotated pixels.
The corrected balanced subset therefore has 17 images per class (51 images total).
It is a pilot subset, not a large benchmark.

Repository Structure
--------------------
Q2_RAS_Project/
├── data/
│   ├── nyu_depth_v2_labeled.mat        # External symlink
│   ├── MSRCv1_real.mat                 # External symlink (legacy demo)
│   ├── Zv_real.npy, Zl_real.npy, Y_real.npy
│   │                                    # Standardized NYU descriptors and labels
│   ├── Zv_real_raw.npy, Zl_real_raw.npy # Raw descriptors for health/weight computation
│   ├── nyu_feature_meta.json           # Label ids, frame lists, counts, feature names
│   ├── ahpff_go1_calibration.npz       # 351 Go1 calibration samples (23 features)
│   ├── ahpff_go1_model.joblib          # Reliability-weighted cluster readout (K=32)
│   ├── ahpff_go1_model_supervised.joblib # ExtraTrees upper-bound model
│   ├── ahpff_go1_model.json            # Blocked/random CV, confusion matrices, mapping
│   ├── ahpff_guidedog_calibration.npz  # 338 guide_dog_sim samples (legacy scenario)
│   ├── ahpff_guidedog_model.joblib     # Cluster readout for guide_dog_sim
│   ├── ahpff_go1_predictions.npz       # Online predictions with weights/quality/clusters
│   ├── ahpff_go1_predictions_baseline.npz
│   ├── ahpff_go1_predictions_degraded.npz
│   └── test_run.bag, dpf_dataset.pt     # Legacy files, not used by the revised pipeline
│
├── scripts/
│   ├── ahpff.py                        # Health indicators, softmax weights, RW-MVK,
│   │                                    # cluster mapping, blocked CV, readout
│   ├── extract_nyu_expert.py           # NYU descriptors + valid ratio + raw features
│   ├── run_arw_mvc_expert.py           # NYU clustering benchmark, 20 seeds, 4 levels
│   ├── run_reliability_sweep.py        # One-modality degradation sweep: q, w, ACC, NMI
│   ├── publication_figures.py           # Fig. 1-5 + Extended Data S1-S4, PDF/1000 dpi PNG
│   ├── verify_paper_numbers.py          # Checks tables, propositions, and figure deliverables
│   ├── plot_go1_representative.py      # Representative trajectory and label figures
│   ├── train_ahpff_guidedog.py         # Cluster readout + supervised upper bound
│   ├── sapf.py                         # 6-DoF SAPF and additive EKF baseline
│   ├── eval_sapf.py                    # Fair 20-seed SAPF comparison
│   ├── train_dpf.py, eval_dpf.py       # Legacy 2D DPF demo (optional)
│   ├── adapt_ahdd.py                   # Legacy AHDD demo
│   └── run_msrc_benchmark.py           # Legacy MSRCv1 t-SNE baseline
│
├── ros_ws/
│   ├── ahpff_perception_node.py        # Online cluster readout; publishes C_t and weights
│   ├── sapf_gazebo_node.py             # Online SAPF/SPF/LPF/EKF/oracle/shuffled comparison
│   ├── install_go1_front_camera.py     # Adds the Go1 forward RGB-D camera
│   ├── go1_street.launch               # Go1 + street world
│   ├── go1_street_shifted.launch       # Same geometry, shifted illumination
│   ├── worlds/guide_dog_street.world   # Sidewalk, tactile paving, stairs, street furniture
│   ├── worlds/guide_dog_street_shifted.world # Dim warm light + heavier fog
│   ├── run_ahpff_calibration_go1.sh    # Collect 351 samples and train the cluster readout
│   ├── run_ahpff_sapf_go1.sh           # One closed-loop run
│   ├── run_sensor_degradation_go1.sh   # Baseline vs mixed degradation
│   ├── run_sensor_degradation_go1_repeats.sh # 3 baseline + 3 degraded runs
│   ├── run_reliability_ablation_go1.sh # Baseline/overexposure/blur/depth-dropout weights
│   ├── run_cross_illumination_go1.sh   # Zero-shot shifted-illumination run
│   └── launch_*.sh                     # Convenience launchers
│
├── 112/                                # Publication figure package (PDF + 1000 dpi PNG)
│   ├── 112.tex                          # 112 manuscript source wired to the new figures
│   ├── pdf/                             # Vector figure PDFs
│   ├── png_1000dpi/                     # 1000 dpi PNGs
│   ├── source_data/                     # Tidy CSV data for every panel
│   ├── 112_all_figures.pdf              # Combined main + extended-data figures
│   └── README.txt                       # Figure guide and technical specification
│
├── figures/
│   ├── ahpff_ablation.png              # NYU clustering figure
│   ├── prl_robustness_eval_expert.json # NYU metrics (20 seeds, 4 levels)
│   ├── reliability_sweep.png            # Reliability-to-performance sweep
│   ├── sapf_tracking_result.png        # Standalone 20-seed SAPF comparison
│   ├── sapf_tracking_result.json
│   ├── go1/                            # Latest single closed-loop run
│   ├── degradation_runs/               # 3 baseline + 3 degraded runs and summary.json
│   ├── reliability_runs/               # Modality-specific weight experiment
│   ├── go1_baseline/, go1_degraded/    # Representative repeated-run outputs
│   ├── cross_illumination/             # Zero-shot shifted-illumination run
│   └── guidedog/                       # guide_dog_sim closed-loop outputs
│
├── run_all.sh
└── start_project.sh                    # One-click Go1 closed loop

Quick Start
-----------
Install dependencies:

  pip install numpy scipy scikit-learn matplotlib h5py opencv-python joblib

Run the non-ROS pipeline:

  ./run_all.sh

This extracts corrected NYU features, runs the 20-seed clustering benchmark, runs
the 20-seed fair SAPF comparison, and retrains the Go1 cluster readout from the
existing 351-sample calibration set. To force feature extraction:

  ./run_all.sh --recompute

To build and verify the 112 figure package after the experiments:

  ./run_all.sh --with-figures
  # or, without rerunning the experiments:
  python3 scripts/publication_figures.py --output 112 --dpi 1000
  python3 scripts/verify_paper_numbers.py --figures 112

The ROS/Gazebo experiments require ROS Noetic and the external ~/unitree_ws
workspace with the Go1 description/controllers.

1. Reliability-aware AHPFF front end
------------------------------------
scripts/extract_nyu_expert.py extracts object-masked Canny edge density,
Laplacian variance, intensity/gradient statistics, entropy, and color statistics
(for the online RGB version) plus depth statistics, percentiles, Sobel gradients,
and a valid-pixel ratio. scripts/ahpff.py computes:

  q_v = sigmoid(5*(0.32*exposure + 0.24*contrast + 0.20*sharpness
                   + 0.12*texture + 0.06*entropy + 0.06*edge_density - 0.45))
  q_d = sigmoid(5*(0.50*valid_ratio + 0.25*smoothness
                   + 0.15*spread + 0.10*range - 0.45))
  w_v = exp(lambda*q_v) / (exp(lambda*q_v) + exp(lambda*q_d)),  lambda=4
  w_d = 1 - w_v

The weighted descriptor is [sqrt(w_v)*z_v, sqrt(w_d)*z_d]. RW-MVK minimizes the
weighted within-cluster distance and updates centroids with the same weights.
Clusters are mapped to terrain classes by majority vote on labelled calibration
samples. At run time the nearest centroid gives the terrain label; a five-frame
majority filter suppresses isolated flips.

NYU pilot results (17 images/class, 20 seeds, disjoint visual/depth degradation):

  Clean:  RA-AHPFF 71.1% ACC / 38.3% NMI
  Level 2.0: RA-AHPFF 60.0% ACC / 23.2% NMI (best among compared methods)
  Visual-degraded samples: mean w_v falls from 0.281 to 0.100
  Depth-degraded samples:  mean w_d falls from 0.719 to 0.298

Silhouette-weighted K selected 2 clusters on the 51-image pilot; the benchmark
fixes K=3 to the number of annotated classes. This is reported as a limitation,
not as evidence of a stable unsupervised model order.

2. Go1 cluster readout
----------------------
The Go1 forward camera and depth stream provide 351 calibration samples:
Flat 59, Stairs 152, Obstacle 85, Tactile 55. Training uses K=32 (8 clusters per
class) and a majority cluster-to-terrain mapping.

  Random 5-fold CV:       cluster 91.2%, ExtraTrees 97.7%  (optimistic; adjacent-frame leakage)
  Blocked 5-fold CV:      cluster 77.8%, ExtraTrees 80.9%  (contiguous time blocks)
  Online closed loop:     90.1 +/- 1.0% semantic accuracy (3 runs)
  Mixed degradation:      77.6 +/- 1.6% semantic accuracy (3 runs)

The online model is fitted on the whole calibration scene, so the online accuracy
is an in-scene result and is higher than the blocked CV estimate. It is not a
cross-scene generalization result.

3. Fair SAPF comparison
-----------------------
scripts/eval_sapf.py compares:
  LO, SPF, LPF (fixed large covariance), EKF,
  SAPF-Pred, SAPF-Oracle, SAPF-Shuffled.

Standalone 30 s simulation, 20 seeds, 1000 particles, 5 Hz exteroceptive pose
with two 4 s outages (position RMSE, m):

  LO:            0.3456
  SPF:           0.2647
  LPF:           0.1107
  EKF:           0.1734
  SAPF-Pred:     0.1158
  SAPF-Oracle:   0.1123
  SAPF-Shuffled: 0.1374

The large-covariance PF is a strong baseline. SAPF-Pred is close to it, the
oracle is slightly better, and the shuffled-label control is worse. The paper
does not claim that semantic adaptation alone explains the improvement over SPF.

Go1 closed loop, 3 runs, 5 Hz pose with outages (overall position RMSE, m):

  Baseline: LO 0.2520, SPF 0.1711, LPF 0.0883, EKF 0.1285,
            SAPF 0.0877, Oracle 0.1015, Shuffled 0.1131
  Degraded: LO 0.2592, SPF 0.1726, LPF 0.0937, EKF 0.1267,
            SAPF 0.0958, Oracle 0.1137, Shuffled 0.0999

4. Reliability-weight experiment
--------------------------------
ros_ws/run_reliability_ablation_go1.sh runs baseline, overexposure, motion blur,
and depth dropout sessions and saves mean weights during the 6-20 s window:

  Baseline:        w_v 0.425, w_d 0.575, q_v 0.760, q_d 0.839
  Overexposure:    w_v 0.174, w_d 0.826, q_v 0.442, q_d 0.844
  Motion blur:     w_v 0.338, w_d 0.662, q_v 0.668, q_d 0.843
  Depth dropout:   w_v 0.724, w_d 0.276, q_v 0.761, q_d 0.513

This verifies that the per-sample weights respond to the degraded modality.
The figure is figures/reliability_runs/reliability_weights.png.

5. Reliability-to-performance sweep
-----------------------------------
scripts/run_reliability_sweep.py degrades one modality at a time from clean data
to severe degradation while keeping the other modality clean. It tracks the
health score, the softmax weight, and the clustering accuracy/NMI:

  Visual: q_v 0.630 -> 0.323, w_v 0.281 -> 0.100, ACC 71.3% -> 64.7%
  Depth:  q_d 0.888 -> 0.355, w_d 0.719 -> 0.279, ACC 71.3% -> 59.0%

The output is figures/reliability_sweep.png and figures/reliability_sweep.json.
The depth sweep is not strictly monotonic: after an initial drop, the clean
visual view dominates and the accuracy partially recovers.

6. ROS/Gazebo closed loop
-------------------------
Launch Gazebo (GUI or headless), then run:

  ./start_project.sh                 # one-click, baseline
  ./start_project.sh --degraded      # mixed degradation from 6 to 20 s
  ./start_project.sh --calibrate     # recollect 351 samples and retrain
  REPEATS=3 ros_ws/run_sensor_degradation_go1_repeats.sh
  ros_ws/run_reliability_ablation_go1.sh
  python3 scripts/plot_go1_representative.py \
      --run-dir figures/degradation_runs/baseline_1 --output-dir figures

The perception node publishes /guide_dog/terrain_label, /guide_dog/terrain_confidence,
/guide_dog/visual_weight, and /guide_dog/depth_weight. The SAPF node publishes
/guide_dog/{sapf,spf,odom,ground_truth}/pose and writes JSON/NPZ/PNG outputs.

Important limitations
---------------------
- The NYU study is a 51-image pilot subset; confidence intervals are wide.
- The Go1 locomotion in Gazebo is kinematic, not a physics-based gait controller.
- The exteroceptive pose is simulated with noise at 5 Hz with outages; it is not
  a real visual/LiDAR localization module.
- The cluster readout is trained and tested in the same Gazebo scene; blocked CV
  reduces adjacent-frame leakage but is not cross-scene validation. A zero-shot
  shifted-illumination run on the same geometry drops from 90.1% to 73.3%.
- The large-covariance PF is a strong baseline; the semantic-adaptive filter is
  comparable rather than dramatically better.
- The health coefficients and over-clustering factors are fixed calibration
  parameters, not learned end-to-end.
- No real-world robot deployment is claimed.

Contact
-------
Special Environment Robotics Laboratory, Southwest University of Science and Technology.

Compiling the paper
-------------------
The LaTeX sources remain in two locations:
  - 论文/elsevier_zbm_/figs/manuscript.tex
  - 论文/manuscript.tex

The new 112 figure package is in 112/.  The previous LaTeX layout predates the
112 text; insert the new figure PDFs (or their 1000 dpi PNGs) into the 112
source before the final build.

Use the clean build script so that BibTeX is rerun after citation changes:
  cd 论文/elsevier_zbm_/figs && ./compile.sh
or:
  cd 论文 && ./compile.sh

The script runs pdflatex, bibtex, pdflatex, and pdflatex again. Running only
pdflatex after adding citations leaves the new references unresolved as [?].
The old PDF at 论文/1031article.pdf was renamed to
论文/1031article_stale_old_figures.pdf; rebuild it with compile.sh.

7. 112 publication figure package
----------------------------------
The 112/ directory contains six main figures and four extended-data figures for
the revised manuscript, each as a single-page vector PDF and a 1000 dpi PNG:

  Figure 1   Methodological roadmap (RA-AHPFF + SAPF)
  Figure 2   NYU Depth V2 clustering benchmark (Tables 4/5)
  Figure 3   Reliability-to-performance sweep (Fig. 3 caption)
  Figure 4   Standalone 20-seed SAPF comparison (Table 10)
  Figure 5   Go1 closed-loop X-Z trajectory and terrain readout (Table 11)
  Figure 6   Go1 nominal street scene and degraded RGB-D preview
  Figure S1  Modality weights and health under four corruption modes (Table 8)
  Figure S2  Softmax-temperature and over-clustering sensitivity (Table 9)
  Figure S3  Temporally blocked 5-fold confusion matrices (Tables 6/7)
  Figure S4  Directional process covariance and flat-ground Tr(Q) (Table 3, Prop. 3)

Build and verify:

  python3 scripts/publication_figures.py --output 112 --dpi 1000
  python3 scripts/verify_paper_numbers.py --figures 112

The package uses Liberation Sans (Arial metric compatible), TrueType-embedded
vector PDF, the Okabe-Ito colour-blind-safe palette, and 183 mm double-column
layouts. The combined PDF is 112/112_all_figures.pdf; the packaged archive is
112.zip. Tidy source-data CSVs are in 112/source_data/.  The LaTeX source
112/112.tex includes Figs. 1--5 with \includegraphics and adds the four
extended-data figures S1--S4; compile it with XeLaTeX (fontspec + Noto Sans).

The 112 Tables 4/5 were generated with the manuscript-benchmark depth setting,
which predates the strict r = 0 boundary in Eq. (4). The benchmark therefore
defaults to that manuscript-compatible setting; pass --strict-depth-boundary to
run_arw_mvc_expert.py to enforce q_d = w_d = 0 at complete depth dropout. The
online perception node follows the strict boundary used in Algorithm 2.


8. Zero-shot cross-illumination check
-------------------------------------
The model calibrated on the original bright street scene was tested without
retraining in a modified world with dimmer warm lighting, a darker sky, and
heavier fog (same geometry and trajectory):

  Online semantic accuracy: 73.3% (vs 90.1% in-scene)
  Mean visual weight: 0.203; mean visual health: 0.324
  Flat: never predicted
  SAPF overall RMSE: 0.149 m (vs 0.088 m in-scene)

This is intentionally reported as a negative cross-domain result. It shows that
the in-scene accuracy is not evidence of cross-scene generalization. The files are figures/cross_illumination/,
ros_ws/go1_street_shifted.launch, ros_ws/worlds/guide_dog_street_shifted.world,
and ros_ws/run_cross_illumination_go1.sh. To reproduce, launch the shifted world
and then run ros_ws/run_cross_illumination_go1.sh.


killall -9 gzserver gzclient roscore rosmaster rosout
rm -rf ~/.gazebo/server-*

118 meeting revision (current)
------------------------------
Current manuscript source: 112/119.tex
Compiled preview:           112/119_preview.pdf
New analysis scripts:
  scripts/meeting118_dataset_atlas.py
  scripts/meeting118_clustering_qualitative.py
  scripts/meeting118_efficiency_statistics.py
New figures: Figure7_dataset_atlas, Figure8_clustering_qualitative,
  Figure9_efficiency_and_statistics (PDF + 1000 dpi PNG in 112/).
New data: source_data/figure7_*.csv, figure8_*.csv, figure9_*.csv,
  dataset_samples/nyu/ (51 RGB/depth/mask/overlay samples), dataset_cards/,
  experiment_protocols/.
Categorized deliverable: 119_118meeting_revision.zip
The older 112.zip is the previous figure package and is superseded by the new
119_118meeting_revision.zip for the current manuscript.
