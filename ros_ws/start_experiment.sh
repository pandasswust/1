#!/bin/bash
killall -9 gzserver gzclient rosmaster roscore python python3 > /dev/null 2>&1
source /opt/ros/noetic/setup.bash
source ~/unitree_ws/devel/setup.bash

echo "🚀 1. 启动纯净物理世界..."
roslaunch gazebo_ros empty_world.launch paused:=false use_sim_time:=true &
sleep 5

echo "🤖 2. 投放 战术姿态 H1 队长..."
rosrun gazebo_ros spawn_model -file ~/unitree_ws/src/multimodal_control/urdf/robot1_h1_frozen.urdf -urdf -model robot1 -x 0.0 -y 0.0 -z 1.05 &
sleep 2

echo "🐶 3. 投放 战术姿态 Go1 机器狗 (绝不消失)..."
rosrun gazebo_ros spawn_model -file ~/unitree_ws/src/multimodal_control/urdf/robot2_go1_frozen.urdf -urdf -model robot2 -x -1.5 -y 1.0 -z 0.4 &
sleep 3

echo "🧠 4. 挂载 NSB 编队大脑与控制面板..."
rosrun multimodal_control nsb_formation.py _robot_name:=robot2 _leader_name:=robot1 &
rosrun rqt_robot_steering rqt_robot_steering _default_topic:=/robot1/cmd_vel
