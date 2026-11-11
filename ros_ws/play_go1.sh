#!/bin/bash
# 1. 彻底清场
killall -9 gzserver gzclient rosmaster roscore python python3 > /dev/null 2>&1
source /opt/ros/noetic/setup.bash
source ~/unitree_ws/devel/setup.bash

echo "⏳ 准备运行环境..."
mkdir -p ~/unitree_ws/src/multimodal_control/config
mkdir -p ~/unitree_ws/src/multimodal_control/scripts

echo "🛠️ 提取原始模型..."
rosrun xacro xacro ~/unitree_ws/src/unitree_ros/robots/go1_description/xacro/robot.xacro > /tmp/go1_raw.urdf 2>/dev/null

# 2. 符合物理法则的模型净化器
cat << 'PYEOF' > ~/unitree_ws/prepare_urdf.py
import xml.etree.ElementTree as ET

with open('/tmp/go1_raw.urdf', 'r') as f: data = f.read()
data = data[data.find('<'):]
root = ET.fromstring(data)
tree = ET.ElementTree(root)

# 【核心修复】：保留质量，仅剥离腿部碰撞箱，防止触地炸膛
for link in root.findall('.//link'):
    name = link.get('name', '')
    if 'thigh' in name or 'calf' in name or 'foot' in name:
        for coll in link.findall('collision'): link.remove(coll)

# 【核心修复】：关闭躯干重力，让他悬浮 (坚决不用 kinematic)
for link_name in ['base', 'trunk']:
    gz = ET.SubElement(root, 'gazebo', {'reference': link_name})
    ET.SubElement(gz, 'gravity').text = 'false'

# 切换电机控制模式
for trans in root.findall('.//transmission'):
    for hw in trans.findall('.//hardwareInterface'):
        hw.text = 'hardware_interface/PositionJointInterface'

# 净化插件
for gazebo in root.findall('.//gazebo'):
    for plugin in gazebo.findall('plugin'):
        if 'gazebo_ros_control' in plugin.get('filename', ''):
            ns = plugin.find('robotNamespace')
            if ns is not None: ns.text = '/go1'
            else: ET.SubElement(plugin, 'robotNamespace').text = '/go1'
        else:
            gazebo.remove(plugin)

# 注入全向底盘
pg = ET.SubElement(root, 'gazebo')
pl = ET.SubElement(pg, 'plugin', {'name': 'planar_move', 'filename': 'libgazebo_ros_planar_move.so'})
ET.SubElement(pl, 'robotNamespace').text = '/go1'
ET.SubElement(pl, 'commandTopic').text = 'cmd_vel'
ET.SubElement(pl, 'odometryTopic').text = 'odom'
ET.SubElement(pl, 'odometryFrame').text = 'odom'
ET.SubElement(pl, 'odometryRate').text = '50.0'
ET.SubElement(pl, 'robotBaseFrame').text = 'trunk'

tree.write('/tmp/go1_teleop_visual.urdf', xml_declaration=True, encoding='utf-8')
PYEOF

python3 ~/unitree_ws/prepare_urdf.py

# 3. 关节配置
cat << 'CONFIG' > ~/unitree_ws/src/multimodal_control/config/go1_teleop.yaml
go1:
  joint_state_controller:
    type: joint_state_controller/JointStateController
    publish_rate: 50
  leg_controller:
    type: position_controllers/JointGroupPositionController
    joints: [FL_hip_joint, FL_thigh_joint, FL_calf_joint, FR_hip_joint, FR_thigh_joint, FR_calf_joint, RL_hip_joint, RL_thigh_joint, RL_calf_joint, RR_hip_joint, RR_thigh_joint, RR_calf_joint]
CONFIG

# 4. 步态生成器
cat << 'PYTHON_SCRIPT' > ~/unitree_ws/src/multimodal_control/scripts/teleop_animator.py
#!/usr/bin/env python3
import rospy, math
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

class TrotAnimator:
    def __init__(self):
        rospy.init_node('teleop_animator')
        self.pub = rospy.Publisher('/go1/leg_controller/command', Float64MultiArray, queue_size=1)
        self.sub = rospy.Subscriber('/go1/cmd_vel', Twist, self.vel_cb)
        self.is_moving, self.t = False, 0.0

    def vel_cb(self, msg):
        self.is_moving = abs(msg.linear.x) > 0.05 or abs(msg.angular.z) > 0.05

    def run(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.is_moving: self.t += 0.25 
            else: self.t = 0.0    
            phase_A, phase_B = self.t, self.t + math.pi
            thigh_base, calf_base = 0.8, -1.5
            swing_A = math.sin(phase_A) * 0.4 if self.is_moving else 0.0
            swing_B = math.sin(phase_B) * 0.4 if self.is_moving else 0.0
            cmd = Float64MultiArray()
            cmd.data = [
                0.0, thigh_base + swing_A, calf_base - swing_A, 
                0.0, thigh_base + swing_B, calf_base - swing_B, 
                0.0, thigh_base + swing_B, calf_base - swing_B, 
                0.0, thigh_base + swing_A, calf_base - swing_A  
            ]
            self.pub.publish(cmd)
            rate.sleep()

if __name__ == '__main__':
    TrotAnimator().run()
PYTHON_SCRIPT
chmod +x ~/unitree_ws/src/multimodal_control/scripts/teleop_animator.py

# 5. 点火起飞
echo "🌍 启动 Gazebo..."
roslaunch gazebo_ros empty_world.launch paused:=false use_sim_time:=true &
sleep 5

echo "🐶 投放实体悬浮机器狗..."
rosparam load ~/unitree_ws/src/multimodal_control/config/go1_teleop.yaml
rosrun gazebo_ros spawn_model -file /tmp/go1_teleop_visual.urdf -urdf -model go1 -z 0.4 &
sleep 3

echo "⚙️ 激活底层控制器..."
rosrun controller_manager spawner --namespace=/go1 joint_state_controller leg_controller &
sleep 2
python3 ~/unitree_ws/src/multimodal_control/scripts/teleop_animator.py &

echo "=========================================================="
echo "🎮 终极实体步态遥控器已激活！(点选此窗口操作)"
echo "   i : 前进    j : 左转    l : 右转    , : 后退    k : 刹车"
echo "=========================================================="
rosrun teleop_twist_keyboard teleop_twist_keyboard.py _key_timeout:=0.3 cmd_vel:=/go1/cmd_vel
