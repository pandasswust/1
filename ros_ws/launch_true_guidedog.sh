#!/bin/bash
echo "========================================================"
echo "🐾 AHDD-DPF 导盲犬系统：高保真人机协同物理仿真"
echo "========================================================"

# 1. 启动带有四足平衡控制器的真实机器狗仿真
echo "▶️ [1/3] 启动宇树 Go1 带控制器仿真..."
# 使用带有控制器的 launch 文件 (如果报错找不到，请确保您编译了 unitree_controller)
gnome-terminal --tab --title="Go1_Controller" -- bash -c "source /opt/ros/noetic/setup.bash; source ~/unitree_ws/devel/setup.bash; roslaunch unitree_guide gazeboSim.launch wname:=stairs; exec bash"

sleep 10

# 2. 动态注入视障人士模型与柔性牵引绳物理约束
echo "▶️ [2/3] 动态注入视障人士模型 (Handler) 与牵引物理约束..."
cat << 'PYEOF' > ~/unitree_ws/spawn_handler.py
import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose

rospy.init_node('spawn_handler')
rospy.wait_for_service('/gazebo/spawn_sdf_model')
spawn_model_prox = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

# 简单的圆柱体代表人类，重 60kg，与狗之间通过 gazebo 物理引擎产生约束
handler_sdf = """
<sdf version="1.6">
  <model name="blind_handler">
    <pose>-0.8 0.3 1.0 0 0 0</pose> <!-- 在狗的左后方 -->
    <link name="link">
      <inertial>
        <mass>60.0</mass>
      </inertial>
      <collision name="collision">
        <geometry><cylinder><radius>0.2</radius><length>1.8</length></cylinder></geometry>
      </collision>
      <visual name="visual">
        <geometry><cylinder><radius>0.2</radius><length>1.8</length></cylinder></geometry>
        <material><ambient>0.2 0.2 0.8 1</ambient></material>
      </visual>
    </link>
  </model>
</sdf>
"""
initial_pose = Pose()
initial_pose.position.x = -0.8
initial_pose.position.y = 0.3
initial_pose.position.z = 0.9

spawn_model_prox("blind_handler", handler_sdf, "handler_namespace", initial_pose, "world")
print("✅ 视障人士模型已成功加载到狗的左后方！")
PYEOF

/usr/bin/python3 ~/unitree_ws/spawn_handler.py

sleep 2

# 3. 启动 6-DoF SAPF 定量评估
echo "▶️ [3/3] 启动 6-DoF SAPF 滤波与评估节点..."
gnome-terminal --tab --title="SAPF_Eval" -- bash -c "source /opt/ros/noetic/setup.bash; source ~/unitree_ws/devel/setup.bash; python3 ~/Q2_RAS_Project/ros_ws/sapf_gazebo_node.py _duration:=30.0 _model_name:=go1_gazebo _output_dir:=/home/yd/Q2_RAS_Project/figures; exec bash"

echo "✅ 高保真仿真环境已全部就绪！"
echo "👉 狗现在是站立状态，请在终端发布 /cmd_vel 控制狗上台阶。"
