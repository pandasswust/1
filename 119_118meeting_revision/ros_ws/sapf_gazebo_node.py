#!/usr/bin/env python3
"""ROS/Gazebo integration of the 6-DoF SAPF.

The node drives the scripted guide-dog trajectory in Gazebo, reads the robot
pose from /gazebo/model_states, simulates noisy leg odometry and a reduced-rate
exteroceptive pose observation (5 Hz by default, with two outage windows), and
runs a Standard PF, a fixed large-covariance PF, an EKF, the semantic-adaptive
SAPF, an oracle-label SAPF, and a shuffled-label SAPF online.  The semantic
label C_t is published on /guide_dog/terrain_label; an RA-AHPFF perception node
can replace the scripted publisher without changing the SAPF interface.

The locomotion is a kinematic animation (the model state is commanded), not a
physics-based stair-climbing controller.  The filter results, however, are
computed online by the real particle filters rather than by linear blending.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROS_PYTHON_PATH = Path("/opt/ros/noetic/lib/python3/dist-packages")
if ROS_PYTHON_PATH.exists() and str(ROS_PYTHON_PATH) not in sys.path:
    sys.path.append(str(ROS_PYTHON_PATH))

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, TransformStamped
from nav_msgs.msg import Path as RosPath
from std_msgs.msg import ColorRGBA, Int32
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

try:
    from unitree_legged_msgs.msg import MotorCmd
except ImportError:
    MotorCmd = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "scripts" / "sapf.py").exists():
            return parent
    return Path("/home/yd/Q2_RAS_Project")


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from sapf import SAPFConfig, SemanticAdaptiveParticleFilter, SixDoFExtendedKalmanFilter, wrap_angle

CLASS_NAMES = {0: "Flat", 1: "Stairs", 2: "Obstacle", 3: "Tactile"}
STATE_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
BASE_OBS_STD = np.array([0.02, 0.02, 0.03, 0.01, 0.01, 0.01])
ODOM_STD = np.array([0.005, 0.005, 0.005, 0.002, 0.002, 0.002])
INITIAL_STD = np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01])
STAIR_START_X = 2.0
STAIR_END_X = 4.4


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Quaternion:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )


def euler_from_quaternion(quaternion: Quaternion) -> np.ndarray:
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def pose_to_state(pose: Pose) -> np.ndarray:
    orientation = euler_from_quaternion(pose.orientation)
    return np.array(
        [pose.position.x, pose.position.y, pose.position.z, *orientation],
        dtype=np.float64,
    )


def state_to_pose(state: np.ndarray) -> Pose:
    return Pose(
        position=Point(x=float(state[0]), y=float(state[1]), z=float(state[2])),
        orientation=quaternion_from_euler(float(state[3]), float(state[4]), float(state[5])),
    )


def wrap_state(state: np.ndarray) -> np.ndarray:
    wrapped = np.array(state, copy=True)
    wrapped[3:6] = wrap_angle(wrapped[3:6])
    return wrapped


def compute_metrics(estimate: np.ndarray, truth: np.ndarray, labels: np.ndarray) -> dict:
    difference = estimate - truth
    difference[:, 3:6] = wrap_angle(difference[:, 3:6])
    axis_rmse = np.sqrt(np.mean(difference**2, axis=0))
    stairs = labels == 1
    flat = labels != 1
    return {
        "axis_rmse": {name: float(value) for name, value in zip(STATE_NAMES, axis_rmse)},
        "position_rmse": float(np.sqrt(np.mean(np.sum(difference[:, :3] ** 2, axis=1)))),
        "orientation_rmse": float(np.sqrt(np.mean(np.sum(difference[:, 3:] ** 2, axis=1)))),
        "stairs_z_rmse": float(np.sqrt(np.mean(difference[stairs, 2] ** 2))) if np.any(stairs) else 0.0,
        "stairs_position_rmse": float(np.sqrt(np.mean(np.sum(difference[stairs, :3] ** 2, axis=1)))) if np.any(stairs) else 0.0,
        "flat_z_rmse": float(np.sqrt(np.mean(difference[flat, 2] ** 2))) if np.any(flat) else 0.0,
        "flat_position_rmse": float(np.sqrt(np.mean(np.sum(difference[flat, :3] ** 2, axis=1)))) if np.any(flat) else 0.0,
        "final_z_error": float(abs(difference[-1, 2])),
    }


class SAPFGazeboNode:
    def __init__(self) -> None:
        rospy.init_node("sapf_gazebo_node", anonymous=False)
        self.duration = float(rospy.get_param("~duration", 30.0))
        self.rate = float(rospy.get_param("~rate", 50.0))
        self.particles = int(rospy.get_param("~particles", 1000))
        self.observation_rate = float(rospy.get_param("~observation_rate", 5.0))
        outage_text = str(rospy.get_param("~observation_outages", "8:12,18:22"))
        self.observation_outages = []
        for item in outage_text.split(","):
            item = item.strip()
            if not item:
                continue
            start_text, end_text = item.split(":")
            self.observation_outages.append((float(start_text), float(end_text)))
        self.model_name = str(rospy.get_param("~model_name", "go1_gazebo"))
        self.human_name = str(rospy.get_param("~human_name", "vivid_blind_handler"))
        self.scenario = str(rospy.get_param("~scenario", "go1_custom")).lower()
        self.spawn_environment = bool(rospy.get_param("~spawn_environment", True))
        self.use_model_states = bool(rospy.get_param("~use_model_states", True))
        self.use_external_labels = bool(rospy.get_param("~use_external_labels", False))
        self.publish_joint_commands = bool(
            rospy.get_param("~publish_joint_commands", self.scenario != "guide_dog")
        )
        if self.scenario in ("guide_dog", "go1_street"):
            self.spawn_environment = False
        self.output_dir = Path(rospy.get_param("~output_dir", str(PROJECT_ROOT / "figures")))
        self.seed = int(rospy.get_param("~seed", 42))
        self.dt = 1.0 / self.rate

        self.rng = np.random.RandomState(self.seed)
        self.spf = SemanticAdaptiveParticleFilter(
            SAPFConfig(num_particles=self.particles, use_semantics=False),
            np.random.RandomState(self.seed + 1),
        )
        self.sapf = SemanticAdaptiveParticleFilter(
            SAPFConfig(num_particles=self.particles, use_semantics=True),
            np.random.RandomState(self.seed + 2),
        )
        maximum_process_std = self.spf.maximum_process_std()
        self.lpf = SemanticAdaptiveParticleFilter(
            SAPFConfig(
                num_particles=self.particles,
                process_std_base=maximum_process_std,
                use_semantics=False,
            ),
            np.random.RandomState(self.seed + 3),
        )
        self.oracle_sapf = SemanticAdaptiveParticleFilter(
            SAPFConfig(num_particles=self.particles, use_semantics=True),
            np.random.RandomState(self.seed + 4),
        )
        self.shuffled_sapf = SemanticAdaptiveParticleFilter(
            SAPFConfig(num_particles=self.particles, use_semantics=True),
            np.random.RandomState(self.seed + 5),
        )
        self.ekf = SixDoFExtendedKalmanFilter(
            self.spf.config.process_std_base,
            np.random.RandomState(self.seed + 6),
        )
        self.observation_period = max(1, int(round(1.0 / max(self.observation_rate, 1e-6) / self.dt)))
        schedule_rng = np.random.RandomState(self.seed + 99)
        schedule = np.asarray(
            [self.terrain_label(index * self.dt) for index in range(int(round(self.duration / self.dt)))],
            dtype=np.int64,
        )
        self.shuffled_schedule = schedule_rng.permutation(schedule)

        self.model_state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        self.sapf_pose_pub = rospy.Publisher("/guide_dog/sapf/pose", PoseStamped, queue_size=10)
        self.spf_pose_pub = rospy.Publisher("/guide_dog/spf/pose", PoseStamped, queue_size=10)
        self.wo_pose_pub = rospy.Publisher("/guide_dog/odom/pose", PoseStamped, queue_size=10)
        self.gt_pose_pub = rospy.Publisher("/guide_dog/ground_truth/pose", PoseStamped, queue_size=10)
        self.label_pub = rospy.Publisher("/guide_dog/terrain_label", Int32, queue_size=10, latch=True)
        self.particle_pub = rospy.Publisher(
            "/guide_dog/sapf/particles", MarkerArray, queue_size=1
        )
        self.gt_path_pub = rospy.Publisher("/guide_dog/ground_truth/path", RosPath, queue_size=1)
        self.sapf_path_pub = rospy.Publisher("/guide_dog/sapf/path", RosPath, queue_size=1)
        self.spf_path_pub = rospy.Publisher("/guide_dog/spf/path", RosPath, queue_size=1)
        self.tf_broadcaster = TransformBroadcaster()
        self.ground_truth_path = RosPath()
        self.ground_truth_path.header.frame_id = "world"
        self.sapf_path = RosPath()
        self.sapf_path.header.frame_id = "world"
        self.spf_path = RosPath()
        self.spf_path.header.frame_id = "world"
        self.external_label: int | None = None
        self.warned_external_label = False
        if self.use_external_labels:
            self.label_sub = rospy.Subscriber(
                "/guide_dog/terrain_label", Int32, self.label_callback, queue_size=1
            )

        self.joint_pubs = {}
        if MotorCmd is not None and self.publish_joint_commands:
            for leg in ("FR", "FL", "RR", "RL"):
                for joint in ("hip", "thigh", "calf"):
                    topic = f"/go1_gazebo/{leg}_{joint}_controller/command"
                    self.joint_pubs[f"{leg}_{joint}"] = rospy.Publisher(topic, MotorCmd, queue_size=10)

        self.latest_model_pose: Pose | None = None
        self.have_model_states = False
        self.model_states_sub = rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.model_states_callback, queue_size=1
        )

        self.step_index = 0
        self.finished = False
        self.initialized = False
        self.previous_truth: np.ndarray | None = None
        self.wo_state: np.ndarray | None = None
        self.time_history: list[float] = []
        self.truth_history: list[np.ndarray] = []
        self.wo_history: list[np.ndarray] = []
        self.spf_history: list[np.ndarray] = []
        self.lpf_history: list[np.ndarray] = []
        self.ekf_history: list[np.ndarray] = []
        self.sapf_history: list[np.ndarray] = []
        self.oracle_history: list[np.ndarray] = []
        self.shuffled_history: list[np.ndarray] = []
        self.label_history: list[int] = []
        self.ground_truth_label_history: list[int] = []

        if self.spawn_environment and self.scenario == "go1_custom":
            self.spawn_environment_models()

        self.timer = rospy.Timer(rospy.Duration(self.dt), self.tick)
        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo(
            "SAPF Gazebo node started: duration=%.1fs, rate=%.0fHz, particles=%d, model=%s",
            self.duration,
            self.rate,
            self.particles,
            self.model_name,
        )

    def model_states_callback(self, message: ModelStates) -> None:
        if self.model_name in message.name:
            index = message.name.index(self.model_name)
            self.latest_model_pose = message.pose[index]
            self.have_model_states = True

    def label_callback(self, message: Int32) -> None:
        self.external_label = int(message.data)

    def terrain_label(self, timestamp: float) -> int:
        if self.scenario == "go1_street":
            x = 0.5 + 0.25 * timestamp
            if 3.0 <= x < 5.0:
                return 2
            if 1.0 <= x < 2.2:
                return 3
            if 5.0 <= x < 8.0:
                return 1
            return 0
        if self.scenario == "guide_dog":
            x = 0.5 + 0.2 * timestamp
            if 3.2 <= x < 3.8:
                return 2
            if x < 4.0:
                return 3
            if x < 4.8:
                return 0
            if x < 5.6:
                return 1
            return 0
        if timestamp < 3.0:
            return 0
        if timestamp < 5.0:
            return 3
        if timestamp < 7.0:
            return 2
        if timestamp < 10.0:
            return 0
        if timestamp < 22.0:
            return 1
        return 0

    def desired_state(self, timestamp: float) -> np.ndarray:
        if self.scenario == "go1_street":
            x = 0.5 + 0.25 * timestamp
            if x < 5.0:
                z = 0.4
            elif x < 8.0:
                step_index = min(int(math.floor((x - 5.0) / 0.3)), 9)
                fraction = (x - 5.0) - step_index * 0.3
                ramp = min(max(fraction / 0.05, 0.0), 1.0)
                z = 0.4 + 0.15 * (step_index + ramp)
            else:
                z = 0.4 + 0.15 * 10.0
            pitch = 0.0
            roll = 0.0
            if 5.0 <= x < 8.0:
                pitch = math.atan2(0.15, 0.3) + 0.02 * math.sin(2.0 * math.pi * 1.2 * timestamp)
                roll = 0.02 * math.sin(2.0 * math.pi * 1.0 * timestamp)
            return np.array([x, 0.0, z, roll, pitch, 0.0], dtype=np.float64)
        if self.scenario == "guide_dog":
            x = 0.5 + 0.2 * timestamp
            step_top = np.interp(
                x,
                [0.0, 4.75, 4.8, 5.15, 5.2, 5.55, 5.6, 7.0],
                [0.0, 0.0, 0.2, 0.2, 0.6, 0.6, 1.0, 1.0],
            )
            z = 0.3 + float(step_top)
            pitch = 0.0
            roll = 0.0
            if 4.8 <= x < 5.6:
                pitch = 0.25 + 0.02 * math.sin(2.0 * math.pi * 1.2 * timestamp)
                roll = 0.02 * math.sin(2.0 * math.pi * 1.0 * timestamp)
            return np.array([x, 0.0, z, roll, pitch, 0.0], dtype=np.float64)

        x = 0.2 * timestamp
        y = 0.0
        if 5.0 <= timestamp < 7.0:
            y = 0.10 * math.sin(math.pi * (timestamp - 5.0) / 2.0)
        if timestamp < 10.0:
            z = 0.40
        elif timestamp < 22.0:
            phase = timestamp - 10.0
            step_index = math.floor(phase)
            fraction = phase - step_index
            ramp = min(max(fraction / 0.20, 0.0), 1.0)
            z = 0.40 + 0.04 * (step_index + ramp)
        else:
            z = 0.40 + 0.04 * 12.0
        pitch = 0.0
        roll = 0.0
        yaw = 0.0
        if 10.0 <= timestamp < 22.0:
            pitch = math.atan2(0.04, 0.2) + 0.010 * math.sin(2.0 * math.pi * 1.2 * timestamp)
            roll = 0.015 * math.sin(2.0 * math.pi * 1.0 * timestamp)
        if 5.0 <= timestamp < 7.0:
            yaw = 0.25 * math.sin(math.pi * (timestamp - 5.0) / 2.0)
            roll = 0.02 * math.sin(2.0 * math.pi * 1.0 * timestamp)
        return np.array([x, y, z, roll, pitch, yaw], dtype=np.float64)

    def observation_active(self, timestamp: float) -> bool:
        if self.step_index % self.observation_period != 0:
            return False
        for start, end in self.observation_outages:
            if start <= timestamp < end:
                return False
        return True

    def observation_std(self, timestamp: float, label: int) -> np.ndarray:
        std = BASE_OBS_STD.copy()
        std[2] = 0.05
        if label == 1:
            std[2] = 0.10
            std[4] = 0.02
        if not self.observation_active(timestamp):
            std[:] = 1e3
        return std

    def odometry_input(self, timestamp: float, label: int) -> tuple[np.ndarray, np.ndarray]:
        bias = np.zeros(6)
        if label == 1:
            bias[2] = 0.001
            bias[4] = 0.0001
        if label == 2:
            bias[0] = 0.0002
            bias[5] = 0.0002
        return bias, ODOM_STD

    def publish_pose(self, publisher: rospy.Publisher, state: np.ndarray) -> None:
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "world"
        message.pose = state_to_pose(state)
        publisher.publish(message)

    def publish_tf(self, state: np.ndarray) -> None:
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "world"
        transform.child_frame_id = "base"
        transform.transform.translation.x = float(state[0])
        transform.transform.translation.y = float(state[1])
        transform.transform.translation.z = float(state[2])
        transform.transform.rotation = quaternion_from_euler(
            float(state[3]), float(state[4]), float(state[5])
        )
        self.tf_broadcaster.sendTransform(transform)

    @staticmethod
    def pose_stamped(state: np.ndarray) -> PoseStamped:
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "world"
        message.pose = state_to_pose(state)
        return message

    def publish_paths(
        self,
        truth: np.ndarray,
        spf_estimate: np.ndarray,
        sapf_estimate: np.ndarray,
    ) -> None:
        self.ground_truth_path.poses.append(self.pose_stamped(truth))
        self.spf_path.poses.append(self.pose_stamped(spf_estimate))
        self.sapf_path.poses.append(self.pose_stamped(sapf_estimate))
        self.ground_truth_path.header.stamp = rospy.Time.now()
        self.spf_path.header.stamp = self.ground_truth_path.header.stamp
        self.sapf_path.header.stamp = self.ground_truth_path.header.stamp
        self.gt_path_pub.publish(self.ground_truth_path)
        self.spf_path_pub.publish(self.spf_path)
        self.sapf_path_pub.publish(self.sapf_path)

    def publish_visualization(
        self,
        label: int,
        spf_estimate: np.ndarray,
        sapf_estimate: np.ndarray,
    ) -> None:
        stamp = rospy.Time.now()
        particles = self.sapf.particles
        weights = self.sapf.weights
        maximum = float(np.max(weights)) if weights.size else 1.0
        strength = np.sqrt(np.clip(weights / (maximum + 1e-12), 0.0, 1.0))

        message = MarkerArray()

        cloud = Marker()
        cloud.header.stamp = stamp
        cloud.header.frame_id = "world"
        cloud.ns = "sapf_particles"
        cloud.id = 0
        cloud.type = Marker.POINTS
        cloud.action = Marker.ADD
        cloud.pose.orientation.w = 1.0
        cloud.scale.x = 0.07
        cloud.scale.y = 0.07
        cloud.lifetime = rospy.Duration(0.0)
        for index in range(particles.shape[0]):
            cloud.points.append(
                Point(
                    x=float(particles[index, 0]),
                    y=float(particles[index, 1]),
                    z=float(particles[index, 2]),
                )
            )
            cloud.colors.append(
                ColorRGBA(
                    r=float(0.15 + 0.85 * strength[index]),
                    g=float(0.35 + 0.35 * strength[index]),
                    b=float(1.0 - 0.90 * strength[index]),
                    a=1.0,
                )
            )
        message.markers.append(cloud)

        sapf_marker = Marker()
        sapf_marker.header.stamp = stamp
        sapf_marker.header.frame_id = "world"
        sapf_marker.ns = "sapf_estimate"
        sapf_marker.id = 1
        sapf_marker.type = Marker.SPHERE
        sapf_marker.action = Marker.ADD
        sapf_marker.pose = state_to_pose(sapf_estimate)
        sapf_marker.scale.x = 0.16
        sapf_marker.scale.y = 0.16
        sapf_marker.scale.z = 0.16
        sapf_marker.color = ColorRGBA(r=0.1, g=1.0, b=0.3, a=1.0)
        sapf_marker.lifetime = rospy.Duration(0.0)
        message.markers.append(sapf_marker)

        spf_marker = Marker()
        spf_marker.header.stamp = stamp
        spf_marker.header.frame_id = "world"
        spf_marker.ns = "spf_estimate"
        spf_marker.id = 2
        spf_marker.type = Marker.SPHERE
        spf_marker.action = Marker.ADD
        spf_marker.pose = state_to_pose(spf_estimate)
        spf_marker.scale.x = 0.09
        spf_marker.scale.y = 0.09
        spf_marker.scale.z = 0.09
        spf_marker.color = ColorRGBA(r=0.2, g=0.5, b=1.0, a=1.0)
        spf_marker.lifetime = rospy.Duration(0.0)
        message.markers.append(spf_marker)

        text = Marker()
        text.header.stamp = stamp
        text.header.frame_id = "world"
        text.ns = "terrain_label"
        text.id = 3
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position = Point(
            x=float(sapf_estimate[0]),
            y=float(sapf_estimate[1]),
            z=float(sapf_estimate[2]) + 0.75,
        )
        text.pose.orientation.w = 1.0
        text.scale.z = 0.22
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.text = f"SAPF: {CLASS_NAMES.get(label, 'Unknown')}"
        text.lifetime = rospy.Duration(0.0)
        message.markers.append(text)

        self.particle_pub.publish(message)

    def publish_model_state(self, timestamp: float, desired: np.ndarray) -> None:
        dog_message = ModelState()
        dog_message.model_name = self.model_name
        dog_message.pose = state_to_pose(desired)
        dog_message.reference_frame = "world"
        self.model_state_pub.publish(dog_message)

        if self.scenario in ("go1_custom", "go1_street"):
            human_message = ModelState()
            human_message.model_name = self.human_name
            human_state = np.array(
                [
                    desired[0] - 0.6,
                    desired[1] + 0.4,
                    desired[2] - 0.4,
                    0.0,
                    0.0,
                    desired[5],
                ],
                dtype=np.float64,
            )
            human_message.pose = state_to_pose(human_state)
            human_message.reference_frame = "world"
            self.model_state_pub.publish(human_message)

        if not self.joint_pubs:
            return

        gait_phase = 15.0 * timestamp
        for leg, phase in zip(("FR", "RL", "FL", "RR"), (gait_phase, gait_phase, gait_phase + math.pi, gait_phase + math.pi)):
            hip = MotorCmd()
            hip.mode = 0x0A
            hip.q = 0.0
            hip.dq = 0.0
            hip.tau = 0.0
            hip.Kp = 100.0
            hip.Kd = 5.0

            thigh = MotorCmd()
            thigh.mode = 0x0A
            thigh.q = 0.8 + 0.5 * math.sin(phase)
            thigh.dq = 0.0
            thigh.tau = 0.0
            thigh.Kp = 300.0
            thigh.Kd = 8.0

            calf = MotorCmd()
            calf.mode = 0x0A
            calf.q = -1.5 - 0.5 * math.cos(phase)
            calf.dq = 0.0
            calf.tau = 0.0
            calf.Kp = 300.0
            calf.Kd = 8.0

            self.joint_pubs[f"{leg}_hip"].publish(hip)
            self.joint_pubs[f"{leg}_thigh"].publish(thigh)
            self.joint_pubs[f"{leg}_calf"].publish(calf)

    def current_truth(self, desired: np.ndarray) -> np.ndarray:
        if self.step_index == 0:
            # Avoid reading a stale pose from a previous run before Gazebo has
            # applied the first commanded model state.
            return desired.copy()
        if self.use_model_states and self.have_model_states and self.latest_model_pose is not None:
            return pose_to_state(self.latest_model_pose)
        return desired.copy()

    def tick(self, event) -> None:
        if self.finished:
            return
        timestamp = self.step_index * self.dt
        if timestamp >= self.duration - 1e-12:
            self.finish()
            return

        desired = self.desired_state(timestamp)
        ground_truth_label = self.terrain_label(timestamp)
        if self.use_external_labels:
            if self.external_label is None:
                if not self.warned_external_label:
                    rospy.logwarn(
                        "use_external_labels is true but no label has arrived; "
                        "falling back to the scripted terrain schedule."
                    )
                    self.warned_external_label = True
                label = ground_truth_label
            else:
                label = self.external_label
            if label not in CLASS_NAMES:
                label = ground_truth_label
        else:
            label = ground_truth_label
            self.label_pub.publish(Int32(label))
        self.publish_model_state(timestamp, desired)

        truth = self.current_truth(desired)
        observation_std = self.observation_std(timestamp, label)

        shuffled_label = int(self.shuffled_schedule[self.step_index % self.shuffled_schedule.size])
        if not self.initialized:
            observation = wrap_state(truth + self.rng.normal(0.0, 1.0, 6) * observation_std)
            self.previous_truth = truth.copy()
            self.wo_state = observation.copy()
            self.spf.initialize(observation, INITIAL_STD)
            self.lpf.initialize(observation, INITIAL_STD)
            self.sapf.initialize(observation, INITIAL_STD)
            self.oracle_sapf.initialize(observation, INITIAL_STD)
            self.shuffled_sapf.initialize(observation, INITIAL_STD)
            self.ekf.initialize(observation, INITIAL_STD)
            spf_estimate = self.spf.estimate()
            lpf_estimate = self.lpf.estimate()
            sapf_estimate = self.sapf.estimate()
            oracle_estimate = self.oracle_sapf.estimate()
            shuffled_estimate = self.shuffled_sapf.estimate()
            ekf_estimate = self.ekf.estimate()
            self.initialized = True
        else:
            assert self.previous_truth is not None
            assert self.wo_state is not None
            truth_delta = wrap_state(truth - self.previous_truth)
            bias, odom_std = self.odometry_input(timestamp, label)
            odom_delta = wrap_state(
                truth_delta + bias + self.rng.normal(0.0, 1.0, 6) * odom_std
            )
            self.wo_state = wrap_state(self.wo_state + odom_delta)
            observation = wrap_state(truth + self.rng.normal(0.0, 1.0, 6) * observation_std)
            spf_estimate = self.spf.step(odom_delta, observation, observation_std, label)
            lpf_estimate = self.lpf.step(odom_delta, observation, observation_std, label)
            sapf_estimate = self.sapf.step(odom_delta, observation, observation_std, label)
            oracle_estimate = self.oracle_sapf.step(
                odom_delta, observation, observation_std, ground_truth_label
            )
            shuffled_estimate = self.shuffled_sapf.step(
                odom_delta, observation, observation_std, shuffled_label
            )
            ekf_estimate = self.ekf.step(odom_delta, observation, observation_std)
            self.previous_truth = truth.copy()

        self.time_history.append(timestamp)
        self.truth_history.append(truth.copy())
        self.wo_history.append(self.wo_state.copy())
        self.spf_history.append(np.array(spf_estimate, copy=True))
        self.lpf_history.append(np.array(lpf_estimate, copy=True))
        self.sapf_history.append(np.array(sapf_estimate, copy=True))
        self.oracle_history.append(np.array(oracle_estimate, copy=True))
        self.shuffled_history.append(np.array(shuffled_estimate, copy=True))
        self.ekf_history.append(np.array(ekf_estimate, copy=True))
        self.label_history.append(label)
        self.ground_truth_label_history.append(ground_truth_label)

        self.publish_pose(self.gt_pose_pub, truth)
        self.publish_pose(self.wo_pose_pub, self.wo_state)
        self.publish_pose(self.spf_pose_pub, spf_estimate)
        self.publish_pose(self.sapf_pose_pub, sapf_estimate)
        self.publish_tf(truth)
        if self.step_index % 2 == 0:
            self.publish_visualization(label, spf_estimate, sapf_estimate)
        if self.step_index % 5 == 0:
            self.publish_paths(truth, spf_estimate, sapf_estimate)
        self.step_index += 1

    def spawn_environment_models(self) -> None:
        try:
            rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=10.0)
            rospy.wait_for_service("/gazebo/delete_model", timeout=10.0)
        except rospy.ROSException:
            rospy.logwarn("Gazebo model services are unavailable; skipping environment spawn.")
            self.spawn_environment = False
            return

        delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        model_names = ["bright_floor", "white_stairs", "tactile_paving", "obstacle_cylinder", "top_platform"]
        model_names += [f"step_{index}" for index in range(12)]
        model_names += ["vivid_blind_handler"]
        for name in model_names:
            try:
                delete_model(name)
            except rospy.ServiceException:
                pass

        floor_sdf = """<sdf version="1.6"><model name="bright_floor"><static>true</static><link name="link"><visual name="visual"><geometry><box><size>50 50 0.01</size></box></geometry><material><script><name>Gazebo/LightGrey</name></script></material></visual></link></model></sdf>"""
        spawn_model("bright_floor", floor_sdf, "env_ns", Pose(position=Point(0, 0, 0.001)), "world")

        tactile_sdf = """<sdf version="1.6"><model name="tactile_paving"><static>true</static><link name="link"><visual name="visual"><geometry><box><size>0.4 1.2 0.02</size></box></geometry><material><script><name>Gazebo/Yellow</name></script></material></visual></link></model></sdf>"""
        spawn_model("tactile_paving", tactile_sdf, "env_ns", Pose(position=Point(0.8, 0.0, 0.011)), "world")

        obstacle_sdf = """<sdf version="1.6"><model name="obstacle_cylinder"><static>true</static><link name="link"><visual name="visual"><geometry><cylinder><radius>0.14</radius><length>1.0</length></cylinder></geometry><material><script><name>Gazebo/Red</name></script></material></visual></link></model></sdf>"""
        spawn_model("obstacle_cylinder", obstacle_sdf, "env_ns", Pose(position=Point(1.2, 0.45, 0.5)), "world")

        for index in range(12):
            height = 0.04 * (index + 1)
            center_x = 2.0 + 0.2 * index + 0.1
            center_z = 0.02 * (index + 1)
            step_sdf = f"""<sdf version="1.6"><model name="step_{index}"><static>true</static><link name="link"><visual name="visual"><geometry><box><size>0.2 2.0 {height}</size></box></geometry><material><script><name>Gazebo/Grey</name></script></material></visual></link></model></sdf>"""
            spawn_model(f"step_{index}", step_sdf, "env_ns", Pose(position=Point(center_x, 0.0, center_z)), "world")

        platform_sdf = """<sdf version="1.6"><model name="top_platform"><static>true</static><link name="link"><visual name="visual"><geometry><box><size>2.4 2.0 0.48</size></box></geometry><material><script><name>Gazebo/Grey</name></script></material></visual></link></model></sdf>"""
        spawn_model("top_platform", platform_sdf, "env_ns", Pose(position=Point(5.6, 0.0, 0.24)), "world")

        human_sdf = """
        <sdf version="1.6"><model name="vivid_blind_handler"><pose>-0.6 0.4 0 0 0 0</pose><link name="link"><inertial><mass>65.0</mass></inertial>
        <visual name="head"><pose>0 0 1.65 0 0 0</pose><geometry><sphere><radius>0.12</radius></sphere></geometry><material><script><name>Gazebo/Orange</name></script></material></visual>
        <visual name="torso"><pose>0 0 1.20 0 0 0</pose><geometry><box><size>0.25 0.4 0.6</size></box></geometry><material><script><name>Gazebo/Black</name></script></material></visual>
        <visual name="left_leg"><pose>0 0.1 0.45 0 0 0</pose><geometry><cylinder><radius>0.08</radius><length>0.9</length></cylinder></geometry><material><script><name>Gazebo/Blue</name></script></material></visual>
        <visual name="right_leg"><pose>0 -0.1 0.45 0 0 0</pose><geometry><cylinder><radius>0.08</radius><length>0.9</length></cylinder></geometry><material><script><name>Gazebo/Blue</name></script></material></visual>
        <visual name="white_cane"><pose>0.4 -0.3 0.5 0 0.785 0</pose><geometry><cylinder><radius>0.015</radius><length>1.2</length></cylinder></geometry><material><script><name>Gazebo/White</name></script></material></visual>
        </link></model></sdf>"""
        spawn_model("vivid_blind_handler", human_sdf, "handler_ns", Pose(position=Point(-0.6, 0.4, 0.0)), "world")
        rospy.loginfo("Gazebo environment and scripted human handler spawned.")

    def finish(self, request_shutdown: bool = True) -> None:
        if self.finished:
            return
        self.finished = True
        if hasattr(self, "timer"):
            self.timer.shutdown()

        if not self.time_history:
            rospy.logwarn("No SAPF samples were recorded; nothing to save.")
            if request_shutdown and not rospy.is_shutdown():
                rospy.signal_shutdown("No samples")
            return

        time_array = np.asarray(self.time_history)
        truth = np.asarray(self.truth_history)
        wo = np.asarray(self.wo_history)
        spf = np.asarray(self.spf_history)
        lpf = np.asarray(self.lpf_history)
        ekf = np.asarray(self.ekf_history)
        sapf = np.asarray(self.sapf_history)
        oracle = np.asarray(self.oracle_history)
        shuffled = np.asarray(self.shuffled_history)
        labels = np.asarray(self.label_history, dtype=np.int64)
        ground_truth_labels = np.asarray(self.ground_truth_label_history, dtype=np.int64)
        semantic_accuracy = float(np.mean(labels == ground_truth_labels)) if labels.size else 0.0
        estimates = {
            "LO": wo,
            "SPF": spf,
            "LPF": lpf,
            "EKF": ekf,
            "SAPF": sapf,
            "SAPF-Oracle": oracle,
            "SAPF-Shuffled": shuffled,
        }
        metrics = {
            name: compute_metrics(estimates[name], truth, ground_truth_labels)
            for name in estimates
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "sapf_gazebo_result.json"
        npz_path = self.output_dir / "sapf_gazebo_trajectory.npz"
        trajectory_path = self.output_dir / "trajectory_xz.png"
        labels_path = self.output_dir / "semantic_labels.png"

        output = {
            "experiment": "ROS/Gazebo 6-DoF SAPF",
            "model_name": self.model_name,
            "duration": self.duration,
            "rate": self.rate,
            "steps": int(time_array.size),
            "particles": self.particles,
            "observation_rate_hz": self.observation_rate,
            "observation_outages_s": [list(item) for item in self.observation_outages],
            "used_model_states": bool(self.have_model_states),
            "semantic_label_source": (
                "external /guide_dog/terrain_label"
                if self.use_external_labels
                else "scripted terrain schedule"
            ),
            "semantic_label_names": {str(key): value for key, value in CLASS_NAMES.items()},
            "semantic_label_accuracy": semantic_accuracy,
            "semantic_label_counts": {
                CLASS_NAMES[key]: int(np.sum(labels == key)) for key in sorted(CLASS_NAMES)
            },
            "ground_truth_label_counts": {
                CLASS_NAMES[key]: int(np.sum(ground_truth_labels == key)) for key in sorted(CLASS_NAMES)
            },
            "metrics": metrics,
            "note": (
                "The locomotion is a kinematic Gazebo animation. Leg odometry and the "
                "exteroceptive pose observation are simulated with noise; the particle "
                "filters run online and are not linear blends. The exteroceptive pose is "
                "available at a reduced rate with two outage windows, so the process "
                "model matters between updates."
            ),
        }
        json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        np.savez(
            npz_path,
            time=time_array,
            truth=truth,
            wo=wo,
            spf=spf,
            lpf=lpf,
            ekf=ekf,
            sapf=sapf,
            oracle=oracle,
            shuffled=shuffled,
            labels=labels,
            ground_truth_labels=ground_truth_labels,
        )

        fig, axis = plt.subplots(figsize=(10, 6), dpi=300)
        axis.plot(truth[:, 0], truth[:, 2], "k-", linewidth=2.5, label="Ground Truth")
        axis.plot(wo[:, 0], wo[:, 2], "r--", linewidth=1.4, label="Leg Odometry")
        axis.plot(spf[:, 0], spf[:, 2], "b:", linewidth=1.8, label="Standard PF")
        axis.plot(lpf[:, 0], lpf[:, 2], "m-.", linewidth=1.4, label="Large-Q PF")
        axis.plot(sapf[:, 0], sapf[:, 2], "g-", linewidth=2.0, label="SAPF-Pred")
        axis.plot(oracle[:, 0], oracle[:, 2], "c--", linewidth=1.4, label="SAPF-Oracle")
        axis.plot(shuffled[:, 0], shuffled[:, 2], color="gray", linestyle=":", linewidth=1.4, label="SAPF-Shuffled")
        if self.scenario == "guide_dog":
            stair_start_x = 4.8
            stair_end_x = 5.6
        elif self.scenario == "go1_street":
            stair_start_x = 5.0
            stair_end_x = 8.0
        else:
            stair_start_x = STAIR_START_X
            stair_end_x = STAIR_END_X
        axis.axvline(stair_start_x, color="gray", linestyle="-.", alpha=0.6)
        axis.axvline(stair_end_x, color="gray", linestyle="-.", alpha=0.6)
        axis.set_xlabel("X Position (m)", fontsize=11)
        axis.set_ylabel("Z Position (m)", fontsize=11)
        axis.set_title("ROS/Gazebo X-Z Trajectory: Fair Filter Comparison", fontsize=13, fontweight="bold")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(loc="upper left", fontsize=7)
        plt.tight_layout()
        plt.savefig(trajectory_path, bbox_inches="tight")
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(10, 3.5), dpi=300)
        axis.step(time_array, labels, where="post", color="darkorange", linewidth=2.0, label="C_t used by Q")
        axis.step(
            time_array,
            ground_truth_labels,
            where="post",
            color="black",
            linewidth=1.2,
            linestyle="--",
            label="Ground truth",
        )
        axis.set_yticks([0, 1, 2, 3])
        axis.set_yticklabels([CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2], CLASS_NAMES[3]])
        axis.set_xlabel("Time (s)", fontsize=11)
        axis.set_ylabel("Semantic label C_t", fontsize=11)
        axis.set_title(
            f"Terrain Labels Used by Q(C_t), accuracy {semantic_accuracy * 100:.1f}%",
            fontsize=13,
            fontweight="bold",
        )
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        plt.savefig(labels_path, bbox_inches="tight")
        plt.close(fig)

        rospy.loginfo("SAPF Gazebo evaluation complete.")
        rospy.loginfo("%-14s %9s %9s %9s %9s %9s", "Method", "X", "Y", "Z", "Overall", "Orient")
        for name in ("LO", "SPF", "LPF", "EKF", "SAPF", "SAPF-Oracle", "SAPF-Shuffled"):
            axis_rmse = metrics[name]["axis_rmse"]
            rospy.loginfo(
                "%-14s %9.4f %9.4f %9.4f %9.4f %9.4f",
                name,
                axis_rmse["x"],
                axis_rmse["y"],
                axis_rmse["z"],
                metrics[name]["position_rmse"],
                metrics[name]["orientation_rmse"],
            )
        rospy.loginfo("Saved: %s", json_path)
        rospy.loginfo("Saved: %s", npz_path)
        rospy.loginfo("Saved: %s", trajectory_path)
        rospy.loginfo("Saved: %s", labels_path)

        if request_shutdown and not rospy.is_shutdown():
            rospy.signal_shutdown("SAPF Gazebo evaluation finished")

    def on_shutdown(self) -> None:
        self.finish(request_shutdown=False)


def main() -> None:
    SAPFGazeboNode()
    rospy.spin()


if __name__ == "__main__":
    main()
