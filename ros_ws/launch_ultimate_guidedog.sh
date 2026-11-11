#!/bin/bash
echo "========================================================"
echo "🎬 AHDD-DPF 导盲犬系统：高保真视觉与物理联合仿真"
echo "========================================================"

echo "🧹 [1/3] 正在清理历史残留进程，确保环境纯净..."
killall -9 roscore rosout rosmaster gzserver gzclient 2>/dev/null
sleep 3

echo "▶️ [2/3] 启动 Gazebo 物理引擎与 Go1 机器狗..."
gnome-terminal --tab --title="Gazebo_Env" -- bash -c "source /opt/ros/noetic/setup.bash; source ~/unitree_ws/devel/setup.bash; roslaunch unitree_gazebo normal.launch rname:=go1; exec bash"

echo "⏳ 正在等待 Gazebo 与底层关节控制器完全加载 (约 10 秒)..."
sleep 10

echo "▶️ [3/3] 启动 6-DoF SAPF 滤波与评估..."
gnome-terminal --tab --title="SAPF_Eval" -- bash -c "source /opt/ros/noetic/setup.bash; source ~/unitree_ws/devel/setup.bash; python3 ~/Q2_RAS_Project/ros_ws/sapf_gazebo_node.py _duration:=30.0 _output_dir:=/home/yd/Q2_RAS_Project/figures; exec bash"

echo "========================================================"
echo "✅ 全部启动指令已下发！"
echo "👉 请将目光移至弹出的 Gazebo 窗口，准备见证奇迹！"
echo "========================================================"
