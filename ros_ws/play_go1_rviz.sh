#!/bin/bash
# 1. 彻底清场
killall -9 gzserver gzclient rosmaster roscore rviz python python3 > /dev/null 2>&1
source /opt/ros/noetic/setup.bash
source ~/unitree_ws/devel/setup.bash

echo "🚀 启动纯视觉引擎架构..."
roscore &
sleep 2

# 2. 提取官方高清模型并上传到参数服务器 (修复了 -t 语法错误)
rosrun xacro xacro ~/unitree_ws/src/unitree_ros/robots/go1_description/xacro/robot.xacro > /tmp/go1_rviz.urdf 2>/dev/null
rosparam set robot_description -t /tmp/go1_rviz.urdf

# 3. 编写纯运动学步态发生器
mkdir -p ~/unitree_ws/src/multimodal_control/scripts
cat << 'PYEOF' > ~/unitree_ws/src/multimodal_control/scripts/rviz_animator.py
#!/usr/bin/env python3
import rospy, math, tf
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class RvizDog:
    def __init__(self):
        rospy.init_node('rviz_animator')
        self.js_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.sub = rospy.Subscriber('/cmd_vel', Twist, self.vel_cb)
        self.br = tf.TransformBroadcaster()
        
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.vx, self.vy, self.vyaw = 0.0, 0.0, 0.0
        self.t = 0.0
        self.last_time = rospy.Time.now()

    def vel_cb(self, msg):
        self.vx, self.vy, self.vyaw = msg.linear.x, msg.linear.y, msg.angular.z

    def run(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - self.last_time).to_sec()
            self.last_time = now

            self.yaw += self.vyaw * dt
            self.x += (self.vx * math.cos(self.yaw) - self.vy * math.sin(self.yaw)) * dt
            self.y += (self.vx * math.sin(self.yaw) + self.vy * math.cos(self.yaw)) * dt

            quat = tf.transformations.quaternion_from_euler(0, 0, self.yaw)
            self.br.sendTransform((self.x, self.y, 0.4), quat, now, "base", "odom")

            is_moving = abs(self.vx) > 0.05 or abs(self.vy) > 0.05 or abs(self.vyaw) > 0.05
            if is_moving: self.t += 8.0 * dt
            else: self.t = 0.0

            swing_A = math.sin(self.t) * 0.4 if is_moving else 0.0
            swing_B = math.sin(self.t + math.pi) * 0.4 if is_moving else 0.0

            js = JointState()
            js.header.stamp = now
            js.name = [
                'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
                'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
            ]
            tb, cb = 0.8, -1.5
            js.position = [
                0.0, tb + swing_A, cb - swing_A,
                0.0, tb + swing_B, cb - swing_B,
                0.0, tb + swing_B, cb - swing_B,
                0.0, tb + swing_A, cb - swing_A
            ]
            self.js_pub.publish(js)
            rate.sleep()

if __name__ == '__main__':
    try: RvizDog().run()
    except rospy.ROSInterruptException: pass
PYEOF
chmod +x ~/unitree_ws/src/multimodal_control/scripts/rviz_animator.py

# 4. 启动状态发布器
rosrun robot_state_publisher robot_state_publisher &

# 5. 生成 RViz 显示配置
cat << 'RVIZCONF' > /tmp/go1_view.rviz
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
      Name: RobotModel
      Robot Description: robot_description
  Global Options:
    Fixed Frame: odom
  Tools:
    - Class: rviz/Interact
    - Class: rviz/MoveCamera
RVIZCONF

# 6. 启动 RViz 与键盘监听
echo "🐕 召唤 3D 机械狗..."
rosrun rviz rviz -d /tmp/go1_view.rviz &
sleep 2
python3 ~/unitree_ws/src/multimodal_control/scripts/rviz_animator.py &
sleep 1

echo "=========================================================="
echo "🎮 终极 3D 渲染遥控器已激活！(点选此终端窗口操作)"
echo "   i : 前进    j : 左转    l : 右转    , : 后退    k : 刹车"
echo "=========================================================="
rosrun teleop_twist_keyboard teleop_twist_keyboard.py _key_timeout:=0.3
