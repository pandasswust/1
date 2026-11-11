#!/usr/bin/env bash
set -euo pipefail

echo "========================================================"
echo "6-DoF SAPF Guide-Dog Gazebo Evaluation"
echo "========================================================"

echo "[1/2] Launching Gazebo and the Go1 model..."
gnome-terminal --tab --title="Gazebo_Env" -- bash -c "source /opt/ros/noetic/setup.bash; source ~/unitree_ws/devel/setup.bash; roslaunch unitree_gazebo normal.launch rname:=go1; exec bash"

echo "Waiting for Gazebo and the joint controllers (~10 s)..."
sleep 10

echo "[2/2] Starting the SAPF Gazebo node..."
gnome-terminal --tab --title="SAPF_Gazebo" -- bash -c "source /opt/ros/noetic/setup.bash; source ~/unitree_ws/devel/setup.bash; python3 ~/Q2_RAS_Project/ros_ws/sapf_gazebo_node.py _duration:=30.0 _output_dir:=/home/yd/Q2_RAS_Project/figures; exec bash"

echo "========================================================"
echo "The SAPF node will run for 30 s and then print the RMSE table."
echo "Outputs in ~/Q2_RAS_Project/figures:"
echo "  - trajectory_xz.png"
echo "  - semantic_labels.png"
echo "  - sapf_gazebo_result.json"
echo "  - sapf_gazebo_trajectory.npz"
echo "========================================================"
