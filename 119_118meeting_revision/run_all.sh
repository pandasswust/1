#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python3}"
RECOMPUTE=0
WITH_DPF=0
WITH_GO1=1
WITH_FIGURES=0
for argument in "$@"; do
  case "$argument" in
    --recompute) RECOMPUTE=1 ;;
    --with-dpf-demo) WITH_DPF=1 ;;
    --without-go1) WITH_GO1=0 ;;
    --with-figures) WITH_FIGURES=1 ;;
    *)
      echo "Unknown option: $argument"
      echo "Usage: ./run_all.sh [--recompute] [--with-dpf-demo] [--without-go1] [--with-figures]"
      exit 1
      ;;
  esac
done

echo "Project: $PROJECT_DIR"
"$PYTHON" - <<'PY'
import importlib.util
import sys

packages = {
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "h5py": "h5py",
    "cv2": "opencv-python",
    "joblib": "joblib",
}
missing = [pip_name for module, pip_name in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    print("Missing dependencies: " + ", ".join(missing))
    print("Install with: pip install " + " ".join(missing))
    sys.exit(1)
PY

if [ "$RECOMPUTE" = "1" ] || [ ! -f data/nyu_feature_meta.json ]; then
  echo "[1/5] Extracting corrected NYU Depth V2 features..."
  "$PYTHON" scripts/extract_nyu_expert.py --min-pixels 100 --max-per-class 200
else
  echo "[1/5] Using existing corrected NYU features (pass --recompute to rebuild)."
fi

echo "[2/5] Running the reliability-aware NYU clustering benchmark..."
"$PYTHON" scripts/run_arw_mvc_expert.py --scheme disjoint --seeds 20 --noise-levels 0 1.0 1.5 2.0

echo "[3/5] Running the reliability-to-performance sweep..."
"$PYTHON" scripts/run_reliability_sweep.py --seeds 20

echo "[4/5] Running the fair 6-DoF SAPF comparison..."
"$PYTHON" scripts/eval_sapf.py --seeds 20 --particles 1000 --clusters 9 --obs-rate 5.0

if [ "$WITH_GO1" = "1" ] && [ -f data/ahpff_go1_calibration.npz ]; then
  echo "[5/5] Training the Go1 cluster readout from the existing calibration set..."
  "$PYTHON" scripts/train_ahpff_guidedog.py \
    --data data/ahpff_go1_calibration.npz \
    --model data/ahpff_go1_model.joblib \
    --meta data/ahpff_go1_model.json \
    --clusters 32 \
    --beta 4.0
else
  echo "[5/5] Skipping the Go1 model (missing data/ahpff_go1_calibration.npz or --without-go1)."
fi

if [ "$WITH_DPF" = "1" ]; then
  echo "[optional] Running the legacy 2D synthetic DPF demo..."
  if "$PYTHON" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"; then
    "$PYTHON" scripts/train_dpf.py
    "$PYTHON" scripts/eval_dpf.py
  else
    echo "torch is not installed; skipping the 2D DPF demo."
  fi
fi

if [ "$WITH_FIGURES" = "1" ]; then
  echo "[figures] Building the 112 vector-PDF and 1000 dpi PNG package..."
  "$PYTHON" scripts/publication_figures.py --output 112 --dpi 1000
  if [ -f data/nyu_feature_meta.json ]; then
    echo "[figures] Building the 118 dataset atlas, qualitative clustering, and sufficiency figures..."
    if [ -f /home/yd/unitree_ws/nyu_depth_v2_labeled.mat ] || [ -f data/nyu_depth_v2_labeled.mat ]; then
      "$PYTHON" scripts/meeting118_dataset_atlas.py --output 112 --dpi 1000
    else
      echo "  NYU .mat archive not found; skipping Figure 7 (dataset atlas)."
    fi
    "$PYTHON" scripts/meeting118_clustering_qualitative.py --output 112 --dpi 1000
    "$PYTHON" scripts/meeting118_efficiency_statistics.py --output 112 --dpi 1000
  fi
  "$PYTHON" scripts/verify_paper_numbers.py --figures 112
fi

echo "Done."
echo "Figures: figures/ahpff_ablation.png, figures/reliability_sweep.png, figures/reliability_runs/reliability_weights.png (if ROS run exists), figures/sapf_tracking_result.png"
echo "Metrics: figures/prl_robustness_eval_expert.json, figures/reliability_sweep.json, figures/sapf_tracking_result.json, data/ahpff_go1_model.json"
if [ "$WITH_FIGURES" = "1" ]; then
  echo "Publication package: 112/, 112.zip, and 119_118meeting_revision.zip"
fi
