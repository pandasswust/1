#!/bin/bash
# 1. 彻底清场，干掉所有幽灵进程
killall -9 gzserver gzclient rosmaster roscore rviz python python3 > /dev/null 2>&1
source /opt/ros/noetic/setup.bash
source ~/unitree_ws/devel/setup.bash

echo "🚀 启动多智能体独立命名空间架构..."

mkdir -p ~/unitree_ws/src/multimodal_control/scripts

# 2. 提取并净化模型，涂装高级色
rosrun xacro xacro ~/unitree_ws/src/unitree_ros/robots/go1_description/xacro/robot.xacro > /tmp/go1_raw.urdf 2>/dev/null
cat ~/unitree_ws/src/unitree_ros/robots/h1_description/urdf/h1.urdf > /tmp/h1_raw.urdf 2>/dev/null

cat << 'PYEOF' > ~/unitree_ws/src/multimodal_control/scripts/rviz_urdf_fixer.py
import xml.etree.ElementTree as ET

def fix_urdf(in_path, out_path, color_rgba, animated_joints):
    try:
        with open(in_path, 'r') as f: data = f.read()
        data = data[data.find('<'):]
        root = ET.fromstring(data)
        
        for visual in root.findall('.//visual'):
            mat = visual.find('material')
            if mat is None: mat = ET.SubElement(visual, 'material', {'name': 'custom_color'})
            color = mat.find('color')
            if color is None: ET.SubElement(mat, 'color', {'rgba': color_rgba})
            else: color.set('rgba', color_rgba)
                
        for joint in root.findall('.//joint'):
            if joint.get('type') != 'fixed':
                if joint.get('name') not in animated_joints:
                    joint.set('type', 'fixed')
                    
        tree = ET.ElementTree(root)
        tree.write(out_path, xml_declaration=True, encoding='utf-8')
    except Exception as e: print(f"Error parsing {in_path}: {e}")

h1_anim = ['left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint']
go1_anim = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']

# 队长深空灰，两只护法纯黑
fix_urdf('/tmp/h1_raw.urdf', '/tmp/h1_rviz.urdf', '0.2 0.2 0.2 1.0', h1_anim)
fix_urdf('/tmp/go1_raw.urdf', '/tmp/go1_rviz.urdf', '0.1 0.1 0.1 1.0', go1_anim)
PYEOF
python3 ~/unitree_ws/src/multimodal_control/scripts/rviz_urdf_fixer.py

# 3. 生成基于 Namespace 与 TF Prefix 的顶级 Launch 文件 (彻底根除红字 Error)
cat << 'LAUNCH' > /tmp/squad.launch
<launch>
  <!-- H1 队长 -->
  <group ns="h1">
    <param name="robot_description" textfile="/tmp/h1_rviz.urdf"/>
    <node pkg="robot_state_publisher" type="robot_state_publisher" name="rsp">
      <param name="tf_prefix" value="h1"/>
    </node>
  </group>
  
  <!-- Go1 左护法 -->
  <group ns="dog1">
    <param name="robot_description" textfile="/tmp/go1_rviz.urdf"/>
    <node pkg="robot_state_publisher" type="robot_state_publisher" name="rsp">
      <param name="tf_prefix" value="dog1"/>
    </node>
  </group>

  <!-- Go1 右护法 -->
  <group ns="dog2">
    <param name="robot_description" textfile="/tmp/go1_rviz.urdf"/>
    <node pkg="robot_state_publisher" type="robot_state_publisher" name="rsp">
      <param name="tf_prefix" value="dog2"/>
    </node>
  </group>
</launch>
LAUNCH

# 4. 编写三角战术阵型与步态发生器
cat << 'PYEOF' > ~/unitree_ws/src/multimodal_control/scripts/squad_animator.py
#!/usr/bin/env python3
import rospy, math, tf
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class SquadAnimator:
    def __init__(self):
        rospy.init_node('squad_animator')
        self.h1_pub = rospy.Publisher('/h1/joint_states', JointState, queue_size=1)
        self.d1_pub = rospy.Publisher('/dog1/joint_states', JointState, queue_size=1)
        self.d2_pub = rospy.Publisher('/dog2/joint_states', JointState, queue_size=1)
        self.sub = rospy.Subscriber('/cmd_vel', Twist, self.vel_cb)
        self.br = tf.TransformBroadcaster()
        
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.vx, self.vy, self.vyaw = 0.0, 0.0, 0.0
        self.t = 0.0
        self.last = rospy.Time.now()

    def vel_cb(self, msg):
        self.vx, self.vy, self.vyaw = msg.linear.x, msg.linear.y, msg.angular.z

    def run(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - self.last).to_sec()
            self.last = now

            # 队长平移解算
            self.yaw += self.vyaw * dt
            self.x += (self.vx * math.cos(self.yaw) - self.vy * math.sin(self.yaw)) * dt
            self.y += (self.vx * math.sin(self.yaw) + self.vy * math.cos(self.yaw)) * dt

            q = tf.transformations.quaternion_from_euler(0, 0, self.yaw)

            # 1. 广播 H1 坐标
            self.br.sendTransform((self.x, self.y, 1.05), q, now, "h1/pelvis", "odom")
            
            # 2. 广播 Dog1 坐标 (左后方)
            d1_dx, d1_dy = -0.8, 0.8
            d1_x = self.x + d1_dx * math.cos(self.yaw) - d1_dy * math.sin(self.yaw)
            d1_y = self.y + d1_dx * math.sin(self.yaw) + d1_dy * math.cos(self.yaw)
            self.br.sendTransform((d1_x, d1_y, 0.4), q, now, "dog1/trunk", "odom")

            # 3. 广播 Dog2 坐标 (右后方)
            d2_dx, d2_dy = -0.8, -0.8
            d2_x = self.x + d2_dx * math.cos(self.yaw) - d2_dy * math.sin(self.yaw)
            d2_y = self.y + d2_dx * math.sin(self.yaw) + d2_dy * math.cos(self.yaw)
            self.br.sendTransform((d2_x, d2_y, 0.4), q, now, "dog2/trunk", "odom")

            # 步态动画生成
            is_moving = abs(self.vx) > 0.05 or abs(self.vy) > 0.05 or abs(self.vyaw) > 0.05
            if is_moving: self.t += 8.0 * dt
            else: self.t = 0.0

            sA = math.sin(self.t) * 0.4 if is_moving else 0.0
            sB = math.sin(self.t + math.pi) * 0.4 if is_moving else 0.0
            
            # 两只狗共用同一套动作数据，各自的 robot_state_publisher 会根据命名空间分配 TF
            js_dog = JointState()
            js_dog.header.stamp = now
            js_dog.name = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
            tb, cb = 0.8, -1.5
            js_dog.position = [0.0, tb+sA, cb-sA, 0.0, tb+sB, cb-sB, 0.0, tb+sB, cb-sB, 0.0, tb+sA, cb-sA]
            self.d1_pub.publish(js_dog)
            self.d2_pub.publish(js_dog)

            # 队长独立动作
            hA = math.sin(self.t * 0.5) * 0.5 if is_moving else 0.0
            hB = math.sin(self.t * 0.5 + math.pi) * 0.5 if is_moving else 0.0
            js_h1 = JointState()
            js_h1.header.stamp = now
            js_h1.name = ['left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint']
            js_h1.position = [-hA, -hB, 0.3, 0.3, hA, 0.2, 0.0, hB, 0.2, 0.0]
            self.h1_pub.publish(js_h1)

            rate.sleep()

if __name__ == '__main__':
    try: SquadAnimator().run()
    except rospy.ROSInterruptException: pass
PYEOF
chmod +x ~/unitree_ws/src/multimodal_control/scripts/squad_animator.py

# 5. 生成 3 机位绝对无报错 RViz 配置
cat << 'RVIZCONF' > /tmp/squad_view.rviz
Panels:
  - Class: rviz/Displays
    Name: Displays
Visualization Manager:
  Displays:
    - Class: rviz/Grid
      Name: Grid
      Cell Size: 1
      Plane Cell Count: 20
    - Class: rviz/RobotModel
      Name: H1_Leader
      Robot Description: /h1/robot_description
      TF Prefix: h1
    - Class: rviz/RobotModel
      Name: Go1_Left_Guard
      Robot Description: /dog1/robot_description
      TF Prefix: dog1
    - Class: rviz/RobotModel
      Name: Go1_Right_Guard
      Robot Description: /dog2/robot_description
      TF Prefix: dog2
  Global Options:
    Fixed Frame: odom
RVIZCONF

# 6. 一键点火起飞
echo "🌌 启动多智能体中枢引擎..."
roscore &
sleep 2

echo "🐶 召唤三角战术兵团..."
roslaunch /tmp/squad.launch &
sleep 2

rosrun rviz rviz -d /tmp/squad_view.rviz &
sleep 2

python3 ~/unitree_ws/src/multimodal_control/scripts/squad_animator.py &
sleep 1

echo "=========================================================="
echo "🎮 1主2从 终极遥控器已激活！(点选此终端窗口操作)"
echo "   i : 前进    j : 左转    l : 右转    , : 后退    k : 刹车"
echo "=========================================================="
rosrun teleop_twist_keyboard teleop_twist_keyboard.py _key_timeout:=0.3
