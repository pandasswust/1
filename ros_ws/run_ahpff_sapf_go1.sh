#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/noetic/setup.bash
source ~/unitree_ws/devel/setup.bash

PROJECT_DIR="$HOME/Q2_RAS_Project"
cd "$PROJECT_DIR"
mkdir -p figures/go1

python3 ros_ws/ahpff_perception_node.py \
  _mode:=infer \
  _camera_topic:=/camera_front/color/image_raw \
  _depth_topic:=/camera_front/depth/image_raw \
  _vertical_flip:=true \
  _publish_rectified:=true \
  _degradation_mode:=none \
  _model_path:="$PROJECT_DIR/data/ahpff_go1_model.joblib" \
  _prediction_path:="$PROJECT_DIR/data/ahpff_go1_predictions.npz" \
  > /tmp/ahpff_go1_infer.log 2>&1 &
PERCEPTION_PID=$!

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
  _output_dir:="$PROJECT_DIR/figures/go1"

kill -INT "$PERCEPTION_PID" 2>/dev/null || true
wait "$PERCEPTION_PID" 2>/dev/null || true

echo "Saved figures/go1 and data/ahpff_go1_predictions.npz"
