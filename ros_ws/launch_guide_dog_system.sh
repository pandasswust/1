#!/usr/bin/env bash
set -euo pipefail

echo "Launching the SAPF guide-dog Gazebo system..."

gnome-terminal --tab --title="Gazebo_Env" -- bash -c "source /opt/ros/noetic/setup.bash; source ~/unitree_ws/devel/setup.bash; roslaunch unitree_gazebo normal.launch rname:=go1; exec bash"

echo "Waiting for Gazebo and the joint controllers (~10 s)..."
sleep 10

gnome-terminal --tab --title="SAPF_Gazebo" -- bash -c "source /opt/ros/noetic/setup.bash; source ~/unitree_ws/devel/setup.bash; python3 ~/Q2_RAS_Project/ros_ws/sapf_gazebo_node.py _duration:=30.0 _output_dir:=/home/yd/Q2_RAS_Project/figures; exec bash"

echo "SAPF node started. It will print the RMSE table after 30 s."
echo "The figure and metrics are written to ~/Q2_RAS_Project/figures."
