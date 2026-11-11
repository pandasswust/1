#!/usr/bin/env bash
set -euo pipefail

echo "Launching guide_dog_sim + AHPFF perception + 6-DoF SAPF..."

gnome-terminal --tab --title="Gazebo_GuideDog" -- bash -c "source /opt/ros/noetic/setup.bash; source ~/unitree_ws/devel/setup.bash; roslaunch $HOME/Q2_RAS_Project/ros_ws/guidedog_sim.launch; exec bash"

echo "Waiting for Gazebo and the guide-dog sensor rig (~10 s)..."
sleep 10

gnome-terminal --tab --title="AHPFF_SAPF" -- bash -c "$HOME/Q2_RAS_Project/ros_ws/run_ahpff_sapf_guidedog.sh; exec bash"

echo "The perception node publishes /guide_dog/terrain_label; the SAPF node consumes it."
echo "Outputs: figures/guidedog/ and data/ahpff_guidedog_predictions.npz"
