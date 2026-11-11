#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/noetic/setup.bash
source ~/unitree_ws/devel/setup.bash

PROJECT_DIR="$HOME/Q2_RAS_Project"
cd "$PROJECT_DIR"
rm -f data/ahpff_guidedog_calibration.npz

python3 ros_ws/ahpff_perception_node.py \
  _mode:=collect \
  _output_path:="$PROJECT_DIR/data/ahpff_guidedog_calibration.npz" \
  > /tmp/ahpff_collect.log 2>&1 &
PERCEPTION_PID=$!

sleep 2

python3 ros_ws/sapf_gazebo_node.py \
  _scenario:=guide_dog \
  _model_name:=guide_dog \
  _spawn_environment:=false \
  _publish_joint_commands:=false \
  _use_external_labels:=false \
  _duration:=30.0 \
  _particles:=300 \
  _output_dir:=/tmp/ahpff_calibration_sapf

kill -INT "$PERCEPTION_PID" 2>/dev/null || true
wait "$PERCEPTION_PID" 2>/dev/null || true

python3 scripts/train_ahpff_guidedog.py --clusters 32 --beta 4.0
