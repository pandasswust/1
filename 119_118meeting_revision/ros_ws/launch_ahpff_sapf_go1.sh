#!/usr/bin/env bash
set -euo pipefail

echo "Installing the Go1 forward camera if needed..."
python3 "$HOME/Q2_RAS_Project/ros_ws/install_go1_front_camera.py"

echo "Launching the Go1 street scene..."
gnome-terminal --tab --title="Gazebo_Go1_Street" -- bash -c "source /opt/ros/noetic/setup.bash; source ~/unitree_ws/devel/setup.bash; roslaunch $HOME/Q2_RAS_Project/ros_ws/go1_street.launch; exec bash"

echo "Waiting for Gazebo and the Go1 controllers (~12 s)..."
sleep 12

gnome-terminal --tab --title="AHPFF_Go1_SAPF" -- bash -c "$HOME/Q2_RAS_Project/ros_ws/run_ahpff_sapf_go1.sh; exec bash"

echo "The AHPFF node publishes /guide_dog/terrain_label; the SAPF node consumes it."
echo "Outputs: figures/go1/ and data/ahpff_go1_predictions.npz"
