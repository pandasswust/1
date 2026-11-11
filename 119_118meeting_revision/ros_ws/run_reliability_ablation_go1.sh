#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/noetic/setup.bash
source ~/unitree_ws/devel/setup.bash

PROJECT_DIR="$HOME/Q2_RAS_Project"
cd "$PROJECT_DIR"
ROOT="$PROJECT_DIR/figures/reliability_runs"
mkdir -p "$ROOT"

run_session () {
  local mode="$1"
  local output_dir="$2"
  local extra_args=("${@:3}")
  mkdir -p "$output_dir"

  python3 ros_ws/ahpff_perception_node.py \
    _mode:=infer \
    _camera_topic:=/camera_front/color/image_raw \
    _depth_topic:=/camera_front/depth/image_raw \
    _vertical_flip:=true \
    _publish_rectified:=false \
    _degradation_mode:="$mode" \
    _model_path:="$PROJECT_DIR/data/ahpff_go1_model.joblib" \
    _prediction_path:="$output_dir/predictions.npz" \
    "${extra_args[@]}" \
    > "$output_dir/perception.log" 2>&1 &
  local perception_pid=$!
  sleep 2

  python3 ros_ws/sapf_gazebo_node.py \
    _scenario:=go1_street \
    _model_name:=go1_gazebo \
    _spawn_environment:=false \
    _publish_joint_commands:=true \
    _use_external_labels:=true \
    _duration:=30.0 \
    _particles:=300 \
    _observation_rate:=5.0 \
    _observation_outages:=8:12,18:22 \
    _seed:=52 \
    _output_dir:="$output_dir"

  kill -INT "$perception_pid" 2>/dev/null || true
  wait "$perception_pid" 2>/dev/null || true
}

run_session none "$ROOT/baseline"
for mode in overexposure blur depth_dropout; do
  run_session "$mode" "$ROOT/$mode" \
    _degradation_start:=6.0 \
    _degradation_end:=20.0 \
    _degradation_strength:=1.0
done

python3 - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
conditions = ["baseline", "overexposure", "blur", "depth_dropout"]
rows = {}
for condition in conditions:
    path = root / condition / "predictions.npz"
    if not path.exists():
        continue
    data = np.load(path, allow_pickle=True)
    wv = data["visual_weights"]
    wd = data["depth_weights"]
    timestamps = data["timestamps"]
    active = (timestamps - timestamps[0] >= 6.0) & (timestamps - timestamps[0] < 20.0)
    rows[condition] = {
        "mean_wv_active": float(np.nanmean(wv[active])) if np.any(active) else float("nan"),
        "mean_wd_active": float(np.nanmean(wd[active])) if np.any(active) else float("nan"),
        "mean_wv_inactive": float(np.nanmean(wv[~active])) if np.any(~active) else float("nan"),
        "mean_wd_inactive": float(np.nanmean(wd[~active])) if np.any(~active) else float("nan"),
        "mean_qv_active": float(np.nanmean(data["visual_quality"][active])) if np.any(active) else float("nan"),
        "mean_qd_active": float(np.nanmean(data["depth_quality"][active])) if np.any(active) else float("nan"),
        "samples": int(timestamps.size),
    }
(root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

fig, axis = plt.subplots(figsize=(8.5, 4.8), dpi=300)
x = np.arange(len(rows))
width = 0.36
labels = list(rows)
wv_values = [rows[name]["mean_wv_active"] for name in labels]
wd_values = [rows[name]["mean_wd_active"] for name in labels]
axis.bar(x - width / 2, wv_values, width, label="Visual weight $w_v$", color="#5DADE2", edgecolor="black", linewidth=0.5)
axis.bar(x + width / 2, wd_values, width, label="Depth weight $w_d$", color="#F1948A", edgecolor="black", linewidth=0.5)
for index, (wv, wd) in enumerate(zip(wv_values, wd_values)):
    axis.annotate(f"{wv:.2f}", xy=(index - width / 2, wv), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    axis.annotate(f"{wd:.2f}", xy=(index + width / 2, wd), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
axis.set_xticks(x)
axis.set_xticklabels(["Baseline", "Overexposure", "Motion blur", "Depth dropout"], fontsize=9)
axis.set_ylabel("Mean modality weight during degradation window", fontsize=10)
axis.set_ylim(0, 1)
axis.set_title("Reliability weights adapt to the degraded modality", fontsize=11, fontweight="bold")
axis.grid(axis="y", linestyle="--", alpha=0.4)
axis.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(root / "reliability_weights.png", bbox_inches="tight")
plt.close(fig)
print(json.dumps(rows, indent=2))
print(f"Saved {root / 'summary.json'} and {root / 'reliability_weights.png'}")
PY
