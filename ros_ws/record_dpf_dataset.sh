#!/bin/bash
killall -9 gzserver gzclient ign gazebo rosmaster roscore python python3 > /dev/null 2>&1
source /opt/ros/noetic/setup.bash
source ~/unitree_ws/devel/setup.bash

cat << 'PYEOF' > ~/unitree_ws/src/multimodal_control/urdf/prep_ign_sensors.py
import os, xml.etree.ElementTree as ET
os.system('rosrun xacro xacro ~/unitree_ws/src/unitree_ros/robots/go1_description/xacro/robot.xacro > /tmp/go1_raw.urdf 2>/dev/null')
with open('/tmp/go1_raw.urdf', 'r') as f: data = f.read()

abs_path = os.path.expanduser('~/unitree_ws/src/unitree_ros/robots/go1_description')
data = data.replace('package://go1_description', f'file://{abs_path}')
data = data.replace('model://go1_description', f'file://{abs_path}')
data = data[data.find('<'):]
root = ET.fromstring(data)

for parent in root.iter():
    for plugin in parent.findall('plugin'):
        parent.remove(plugin)

plugin_gazebo = ET.SubElement(root, 'gazebo')
plugin_vel = ET.SubElement(plugin_gazebo, 'plugin', {
    'filename': 'libignition-gazebo-velocity-control-system.so',
    'name': 'ignition::gazebo::systems::VelocityControl'
})
ET.SubElement(plugin_vel, 'link_name').text = 'trunk'

plugin_pose = ET.SubElement(plugin_gazebo, 'plugin', {
    'filename': 'libignition-gazebo-pose-publisher-system.so',
    'name': 'ignition::gazebo::systems::PosePublisher'
})
ET.SubElement(plugin_pose, 'publish_link_pose').text = 'false'
ET.SubElement(plugin_pose, 'publish_model_pose').text = 'true'
ET.SubElement(plugin_pose, 'use_pose_vector_msg').text = 'true'
ET.SubElement(plugin_pose, 'update_frequency').text = '20'

lidar_gazebo = ET.SubElement(root, 'gazebo', {'reference': 'trunk'})
sensor = ET.SubElement(lidar_gazebo, 'sensor', {'name': 'lidar_2d', 'type': 'gpu_lidar'})
# 【核心修正】：加上了绝对路径斜杠 /scan，让 Ignition 乖乖发到指定话题
ET.SubElement(sensor, 'topic').text = '/scan'
ET.SubElement(sensor, 'update_rate').text = '10'
ET.SubElement(sensor, 'always_on').text = '1'
ET.SubElement(sensor, 'visualize').text = 'true'
ray = ET.SubElement(sensor, 'ray')
scan = ET.SubElement(ray, 'scan')
horizontal = ET.SubElement(scan, 'horizontal')
ET.SubElement(horizontal, 'samples').text = '360'
ET.SubElement(horizontal, 'resolution').text = '1'
ET.SubElement(horizontal, 'min_angle').text = '-3.14159'
ET.SubElement(horizontal, 'max_angle').text = '3.14159'
range_tag = ET.SubElement(ray, 'range')
ET.SubElement(range_tag, 'min').text = '0.2'
ET.SubElement(range_tag, 'max').text = '10.0'

tree = ET.ElementTree(root)
tree.write('/tmp/go1_ign_sensors.urdf', xml_declaration=True, encoding='utf-8')
PYEOF

python3 ~/unitree_ws/src/multimodal_control/urdf/prep_ign_sensors.py

cat << 'LAUNCH' > ~/unitree_ws/src/multimodal_control/launch/go1_ign_dataset.launch
<launch>
  <include file="$(find ros_ign_gazebo)/launch/ign_gazebo.launch">
    <arg name="ign_args" value="-r empty.sdf"/>
  </include>
  <param name="robot_description" textfile="/tmp/go1_ign_sensors.urdf"/>
  <node pkg="ros_ign_gazebo" type="create" name="spawn_go1" args="-world empty -param robot_description -name go1 -z 0.4"/>
  <node pkg="ros_ign_bridge" type="parameter_bridge" name="ros_ign_bridge"
        args="/model/go1/cmd_vel@geometry_msgs/Twist]ignition.msgs.Twist
              /scan@sensor_msgs/LaserScan[ignition.msgs.LaserScan
              /model/go1/pose@tf2_msgs/TFMessage[ignition.msgs.Pose_V">
    <remap from="/model/go1/cmd_vel" to="/cmd_vel"/>
    <remap from="/model/go1/pose" to="/tf"/>
  </node>
  <node pkg="teleop_twist_keyboard" type="teleop_twist_keyboard.py" name="teleop_keyboard" output="screen" launch-prefix="xterm -e"/>
</launch>
LAUNCH

roslaunch multimodal_control go1_ign_dataset.launch
