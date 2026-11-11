#!/usr/bin/env bash
set -euo pipefail

echo "Launching guide_dog_sim + AHPFF calibration + SAPF label source..."

gnome-terminal --tab --title="Gazebo_GuideDog" -- bash -c "source /opt/ros/noetic/setup.bash; source ~/unitree_ws/devel/setup.bash; roslaunch $HOME/Q2_RAS_Project/ros_ws/guidedog_sim.launch; exec bash"

echo "Waiting for Gazebo and the guide-dog sensor rig (~10 s)..."
sleep 10

gnome-terminal --tab --title="AHPFF_Calibration" -- bash -c "$HOME/Q2_RAS_Project/ros_ws/run_ahpff_calibration.sh; exec bash"

echo "The calibration run records data/ahpff_guidedog_calibration.npz and trains data/ahpff_guidedog_model.joblib."
