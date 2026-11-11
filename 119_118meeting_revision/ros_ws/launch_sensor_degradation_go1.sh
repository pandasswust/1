#!/usr/bin/env bash
set -euo pipefail

echo "Installing the Go1 forward camera if needed..."
python3 "$HOME/Q2_RAS_Project/ros_ws/install_go1_front_camera.py"

echo "Launching the Go1 street scene..."
gnome-terminal --tab --title="Gazebo_Go1_Degradation" -- bash -c "source /opt/ros/noetic/setup.bash; source ~/unitree_ws/devel/setup.bash; roslaunch $HOME/Q2_RAS_Project/ros_ws/go1_street.launch; exec bash"

echo "Waiting for Gazebo and the Go1 controllers (~12 s)..."
sleep 12

gnome-terminal --tab --title="Sensor_Degradation" -- bash -c "$HOME/Q2_RAS_Project/ros_ws/run_sensor_degradation_go1.sh; exec bash"

echo "Baseline: no degradation. Degraded: overexposure + motion blur + depth dropout from 6-20 s."
echo "Results: figures/go1_baseline/, figures/go1_degraded/, figures/sensor_degradation_go1.json"
