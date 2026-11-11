#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/noetic/setup.bash
source ~/unitree_ws/devel/setup.bash

PROJECT_DIR="$HOME/Q2_RAS_Project"
cd "$PROJECT_DIR"
REPEATS="${REPEATS:-3}"
ROOT="$PROJECT_DIR/figures/degradation_runs"
mkdir -p "$ROOT"

run_session () {
  local mode="$1"
  local output_dir="$2"
  local prediction_path="$3"
  local log_path="$4"
  local extra_args=("${@:5}")
  mkdir -p "$output_dir"

  python3 ros_ws/ahpff_perception_node.py \
    _mode:=infer \
    _camera_topic:=/camera_front/color/image_raw \
    _depth_topic:=/camera_front/depth/image_raw \
    _vertical_flip:=true \
    _publish_rectified:=true \
    _degradation_mode:="$mode" \
    _model_path:="$PROJECT_DIR/data/ahpff_go1_model.joblib" \
    _prediction_path:="$prediction_path" \
    "${extra_args[@]}" \
    > "$log_path" 2>&1 &
  local perception_pid=$!
  sleep 2

  python3 ros_ws/sapf_gazebo_node.py \
    _scenario:=go1_street \
    _model_name:=go1_gazebo \
    _spawn_environment:=false \
    _publish_joint_commands:=true \
    _use_external_labels:=true \
    _duration:=30.0 \
    _particles:=500 \
    _observation_rate:=5.0 \
    _observation_outages:=8:12,18:22 \
    _seed:=$((42 + repeat)) \
    _output_dir:="$output_dir"

  kill -INT "$perception_pid" 2>/dev/null || true
  wait "$perception_pid" 2>/dev/null || true
}

for repeat in $(seq 1 "$REPEATS"); do
  echo "Baseline repeat $repeat/$REPEATS"
  run_session none "$ROOT/baseline_$repeat" \
    "$ROOT/baseline_$repeat/predictions.npz" \
    "$ROOT/baseline_$repeat/perception.log"
  echo "Degraded repeat $repeat/$REPEATS"
  run_session mixed "$ROOT/degraded_$repeat" \
    "$ROOT/degraded_$repeat/predictions.npz" \
    "$ROOT/degraded_$repeat/perception.log" \
    _degradation_start:=6.0 \
    _degradation_end:=20.0 \
    _degradation_strength:=1.0 \
    _preview_path:="$ROOT/degraded_$repeat/degraded_preview.png"
done

python3 - "$ROOT" "$REPEATS" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
repeats = int(sys.argv[2])
methods = ("LO", "SPF", "LPF", "EKF", "SAPF", "SAPF-Oracle", "SAPF-Shuffled")
summary = {}
for condition in ("baseline", "degraded"):
    rows = []
    accuracy = []
    for index in range(1, repeats + 1):
        path = root / f"{condition}_{index}" / "sapf_gazebo_result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        rows.append(result["metrics"])
        accuracy.append(result["semantic_label_accuracy"])
    aggregate = {}
    for method in methods:
        aggregate[method] = {}
        for key in ("x", "y", "z"):
            values = [row[method]["axis_rmse"][key] for row in rows]
            aggregate[method][key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for key in ("position_rmse", "orientation_rmse", "stairs_z_rmse", "flat_z_rmse", "final_z_error"):
            values = [row[method][key] for row in rows]
            aggregate[method][key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    summary[condition] = {
        "runs": len(rows),
        "semantic_accuracy": {"mean": float(np.mean(accuracy)), "std": float(np.std(accuracy))},
        "metrics": aggregate,
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Saved {root / 'summary.json'}")
for condition in ("baseline", "degraded"):
    data = summary[condition]
    print(f"{condition}: semantic {data['semantic_accuracy']['mean']*100:.1f}% ± {data['semantic_accuracy']['std']*100:.1f}%")
    for method in methods:
        metrics = data["metrics"][method]
        print(
            f"  {method:14s} Z={metrics['z']['mean']:.4f}±{metrics['z']['std']:.4f} "
            f"Overall={metrics['position_rmse']['mean']:.4f}±{metrics['position_rmse']['std']:.4f} "
            f"StairsZ={metrics['stairs_z_rmse']['mean']:.4f} "
            f"Orient={metrics['orientation_rmse']['mean']:.4f}"
        )
PY
