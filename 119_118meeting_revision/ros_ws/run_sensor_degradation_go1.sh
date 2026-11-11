#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/noetic/setup.bash
source ~/unitree_ws/devel/setup.bash

PROJECT_DIR="$HOME/Q2_RAS_Project"
cd "$PROJECT_DIR"
mkdir -p figures/go1_baseline figures/go1_degraded

run_session () {
  local mode="$1"
  local output_dir="$2"
  local prediction_path="$3"
  local log_path="$4"
  local extra_args=("${@:5}")

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
    _output_dir:="$output_dir"

  kill -INT "$perception_pid" 2>/dev/null || true
  wait "$perception_pid" 2>/dev/null || true
}

run_session none "$PROJECT_DIR/figures/go1_baseline" \
  "$PROJECT_DIR/data/ahpff_go1_predictions_baseline.npz" \
  /tmp/ahpff_go1_baseline.log

run_session mixed "$PROJECT_DIR/figures/go1_degraded" \
  "$PROJECT_DIR/data/ahpff_go1_predictions_degraded.npz" \
  /tmp/ahpff_go1_degraded.log \
  _degradation_start:=6.0 \
  _degradation_end:=20.0 \
  _degradation_strength:=1.0 \
  _preview_path:="$PROJECT_DIR/figures/go1_degraded/degraded_preview.png"

python3 - <<'PY'
import json
from pathlib import Path

root = Path.home() / "Q2_RAS_Project"
paths = {
    "baseline": root / "figures/go1_baseline/sapf_gazebo_result.json",
    "degraded": root / "figures/go1_degraded/sapf_gazebo_result.json",
}
data = {name: json.loads(path.read_text()) for name, path in paths.items()}
summary = {}
for name, result in data.items():
    summary[name] = {
        "semantic_accuracy": result["semantic_label_accuracy"],
        "label_counts": result["semantic_label_counts"],
        "ground_truth_counts": result["ground_truth_label_counts"],
        "metrics": result["metrics"],
    }
(root / "figures/sensor_degradation_go1.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("Sensor degradation experiment")
for name in ("baseline", "degraded"):
    result = data[name]
    metrics = result["metrics"]
    print(f"[{name}] semantic accuracy {result['semantic_label_accuracy'] * 100:.1f}%")
    for method in ("LO", "SPF", "LPF", "EKF", "SAPF", "SAPF-Oracle", "SAPF-Shuffled"):
        item = metrics[method]
        print(
            f"  {method:14s} Z={item['axis_rmse']['z']:.4f} "
            f"Overall={item['position_rmse']:.4f} "
            f"Orientation={item['orientation_rmse']:.4f} "
            f"StairsZ={item['stairs_z_rmse']:.4f}"
        )
print(f"Saved: {root / 'figures/sensor_degradation_go1.json'}")
PY
