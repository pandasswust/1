#!/usr/bin/env python3
"""RA-AHPFF terrain-perception node for the Go1 and guide_dog_sim Gazebo worlds.

The node extracts hand-crafted visual and depth/LiDAR descriptors from the
forward camera and the depth stream or 2D LiDAR, computes per-sample
sensor-health weights, and either (a) records a labelled calibration set from
/guide_dog/terrain_label or (b) assigns each sample to the nearest centroid of a
reliability-weighted cluster readout.  The mapped terrain class is published on
/guide_dog/terrain_label for the SAPF node; the visual and depth weights are
published on /guide_dog/visual_weight and /guide_dog/depth_weight.

Collect mode:
  python3 ahpff_perception_node.py _mode:=collect \
      _output_path:=/home/yd/Q2_RAS_Project/data/ahpff_guidedog_calibration.npz

Infer mode:
  python3 ahpff_perception_node.py _mode:=infer \
      _model_path:=/home/yd/Q2_RAS_Project/data/ahpff_go1_model.joblib
"""

from __future__ import annotations

import sys
from collections import Counter, deque
from pathlib import Path

ROS_PYTHON_PATH = Path("/opt/ros/noetic/lib/python3/dist-packages")
if ROS_PYTHON_PATH.exists() and str(ROS_PYTHON_PATH) not in sys.path:
    sys.path.append(str(ROS_PYTHON_PATH))

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32, Int32

import message_filters

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from ahpff import compute_quality, modality_weights, predict_cluster_readout


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "scripts" / "sapf.py").exists():
            return parent
    return Path("/home/yd/Q2_RAS_Project")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
CLASS_NAMES = {0: "Flat", 1: "Stairs", 2: "Obstacle", 3: "Tactile"}
VISION_FEATURE_NAMES = [
    "canny_edge_density",
    "laplacian_variance",
    "mean_intensity",
    "std_intensity",
    "mean_gradient",
    "std_gradient",
    "intensity_entropy",
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_saturation",
    "yellow_fraction",
    "red_fraction",
]
LIDAR_FEATURE_NAMES = [
    "range_mean",
    "range_std",
    "range_min",
    "range_max",
    "range_p25",
    "range_p75",
    "range_span",
    "range_mean_abs_diff",
    "range_std_diff",
    "range_max_fraction",
    "valid_ratio",
]
DEPTH_FEATURE_NAMES = [
    "depth_mean",
    "depth_std",
    "depth_min",
    "depth_max",
    "depth_p25",
    "depth_p75",
    "depth_span",
    "depth_mean_abs_sobel_x",
    "depth_mean_abs_sobel_y",
    "depth_mean_gradient",
    "valid_ratio",
]
ROI = (0.15, 0.85, 0.35, 0.98)


def entropy(values: np.ndarray, bins: int = 8) -> float:
    hist, _ = np.histogram(values, bins=bins, range=(0, 256))
    probs = hist.astype(np.float64) / max(int(hist.sum()), 1)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def vision_features(image_rgb: np.ndarray) -> np.ndarray | None:
    height, width = image_rgb.shape[:2]
    x0, x1, y0, y1 = ROI
    crop = image_rgb[int(y0 * height) : int(y1 * height), int(x0 * width) : int(x1 * width)]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(grad_x, grad_y)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    r = crop[:, :, 0].astype(np.float64)
    g = crop[:, :, 1].astype(np.float64)
    b = crop[:, :, 2].astype(np.float64)
    yellow = ((r > 100) & (g > 100) & (b < 100)).mean()
    red = ((r > 100) & (g < 90) & (b < 90)).mean()
    return np.array(
        [
            edges.mean() / 255.0,
            float(laplacian.var()),
            float(gray.mean()) / 255.0,
            float(gray.std()) / 255.0,
            float(grad_mag.mean()) / 255.0,
            float(grad_mag.std()) / 255.0,
            entropy(gray),
            float(r.mean()) / 255.0,
            float(g.mean()) / 255.0,
            float(b.mean()) / 255.0,
            float(hsv[:, :, 1].mean()) / 255.0,
            float(yellow),
            float(red),
        ],
        dtype=np.float64,
    )


def lidar_features(scan: LaserScan) -> np.ndarray | None:
    ranges = np.asarray(scan.ranges, dtype=np.float64)
    if ranges.size == 0:
        return None
    angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment
    forward = np.abs(angles) <= (np.pi / 3.0)
    if int(forward.sum()) < 5:
        forward = np.ones(ranges.size, dtype=bool)
    values = ranges[forward].copy()
    valid_ratio = float(np.mean(np.isfinite(values)))
    if not np.isfinite(values).all():
        values[~np.isfinite(values)] = scan.range_max
    values[values > scan.range_max] = scan.range_max
    values[values < scan.range_min] = scan.range_min
    differences = np.diff(values) if values.size > 1 else np.zeros(1)
    return np.array(
        [
            float(values.mean()),
            float(values.std()),
            float(values.min()),
            float(values.max()),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 75)),
            float(values.max() - values.min()),
            float(np.mean(np.abs(differences))),
            float(np.std(differences)),
            float(np.mean(values >= scan.range_max - 0.05)),
            valid_ratio,
        ],
        dtype=np.float64,
    )


def depth_features(depth: np.ndarray) -> np.ndarray | None:
    height, width = depth.shape[:2]
    x0, x1, y0, y1 = ROI
    crop = depth[int(y0 * height) : int(y1 * height), int(x0 * width) : int(x1 * width)].astype(np.float64)
    if crop.size == 0:
        return None
    valid = np.isfinite(crop) & (crop > 0.0)
    if int(valid.sum()) < 20:
        return None
    values = crop[valid]
    filled = crop.copy()
    filled[~valid] = float(np.median(values))
    sobel_x = cv2.Sobel(filled, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(filled, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(sobel_x, sobel_y)
    return np.array(
        [
            float(values.mean()),
            float(values.std()),
            float(values.min()),
            float(values.max()),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 75)),
            float(values.max() - values.min()),
            float(np.mean(np.abs(sobel_x[valid]))),
            float(np.mean(np.abs(sobel_y[valid]))),
            float(np.mean(grad_mag[valid])),
            float(valid.sum() / max(int(crop.size), 1)),
        ],
        dtype=np.float64,
    )


class AHPFFPerceptionNode:
    def __init__(self) -> None:
        rospy.init_node("ahpff_perception_node", anonymous=False)
        self.mode = str(rospy.get_param("~mode", "infer")).lower()
        self.camera_topic = str(rospy.get_param("~camera_topic", "/guide_dog/camera/image_raw"))
        self.scan_topic = str(rospy.get_param("~scan_topic", "/guide_dog/lidar/scan"))
        self.depth_topic = str(rospy.get_param("~depth_topic", ""))
        self.vertical_flip = bool(rospy.get_param("~vertical_flip", False))
        self.publish_rectified = bool(rospy.get_param("~publish_rectified", False))
        self.degradation_mode = str(rospy.get_param("~degradation_mode", "none")).lower()
        self.degradation_start = float(rospy.get_param("~degradation_start", 0.0))
        self.degradation_end = float(rospy.get_param("~degradation_end", 0.0))
        self.degradation_strength = float(rospy.get_param("~degradation_strength", 1.0))
        self.preview_path = Path(
            rospy.get_param("~preview_path", str(PROJECT_ROOT / "figures" / "go1" / "degraded_preview.png"))
        )
        self.node_start_time = rospy.Time.now().to_sec()
        self.degradation_logged = False
        self.preview_saved = False
        self.label_topic = str(rospy.get_param("~label_topic", "/guide_dog/terrain_label"))
        self.output_path = Path(
            rospy.get_param(
                "~output_path",
                str(PROJECT_ROOT / "data" / "ahpff_guidedog_calibration.npz"),
            )
        )
        self.model_path = Path(
            rospy.get_param(
                "~model_path",
                str(PROJECT_ROOT / "data" / "ahpff_guidedog_model.joblib"),
            )
        )
        self.prediction_path = Path(
            rospy.get_param(
                "~prediction_path",
                str(PROJECT_ROOT / "data" / "ahpff_guidedog_predictions.npz"),
            )
        )
        self.smoothing_window = int(rospy.get_param("~smoothing_window", 5))
        self.bridge = CvBridge()
        self.latest_label: int | None = None
        self.features: list[np.ndarray] = []
        self.labels: list[int] = []
        self.timestamps: list[float] = []
        self.predictions: list[int] = []
        self.confidences: list[float] = []
        self.clusters: list[int] = []
        self.visual_weights: list[float] = []
        self.depth_weights: list[float] = []
        self.visual_quality: list[float] = []
        self.depth_quality: list[float] = []
        self.smoothed_predictions: deque[int] = deque(maxlen=max(1, self.smoothing_window))
        self.model = None
        self.scaler = None
        self.cluster_payload = None
        self.shutting_down = False
        self.feature_names = list(VISION_FEATURE_NAMES) + (
            list(DEPTH_FEATURE_NAMES) if self.depth_topic else list(LIDAR_FEATURE_NAMES)
        )

        if self.mode == "infer":
            self.load_model()
            self.label_pub = rospy.Publisher(
                self.label_topic, Int32, queue_size=1, latch=True
            )
            self.confidence_pub = rospy.Publisher(
                "/guide_dog/terrain_confidence", Float32, queue_size=1
            )
            self.visual_weight_pub = rospy.Publisher(
                "/guide_dog/visual_weight", Float32, queue_size=1
            )
            self.depth_weight_pub = rospy.Publisher(
                "/guide_dog/depth_weight", Float32, queue_size=1
            )
            if self.publish_rectified:
                self.rectified_pub = rospy.Publisher(
                    "/camera_front/color/image_rectified", Image, queue_size=1
                )
        elif self.mode == "collect":
            self.label_sub = rospy.Subscriber(
                self.label_topic, Int32, self.label_callback, queue_size=1
            )
        else:
            raise SystemExit("~mode must be 'collect' or 'infer'.")

        camera_sub = message_filters.Subscriber(self.camera_topic, Image)
        if self.depth_topic:
            depth_sub = message_filters.Subscriber(self.depth_topic, Image)
            synchronizer = message_filters.ApproximateTimeSynchronizer(
                [camera_sub, depth_sub], queue_size=10, slop=0.15
            )
            synchronizer.registerCallback(self.synchronized_depth_callback)
            sensor_label = self.depth_topic
        else:
            scan_sub = message_filters.Subscriber(self.scan_topic, LaserScan)
            synchronizer = message_filters.ApproximateTimeSynchronizer(
                [camera_sub, scan_sub], queue_size=10, slop=0.15
            )
            synchronizer.registerCallback(self.synchronized_scan_callback)
            sensor_label = self.scan_topic
        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo(
            "AHPFF perception node started in %s mode (camera=%s, sensor=%s)",
            self.mode,
            self.camera_topic,
            sensor_label,
        )

    def label_callback(self, message: Int32) -> None:
        self.latest_label = int(message.data)

    def load_model(self) -> None:
        import joblib

        if not self.model_path.exists():
            raise SystemExit(
                f"Missing model {self.model_path}. Run collect mode and "
                "scripts/train_ahpff_guidedog.py first."
            )
        payload = joblib.load(self.model_path)
        model_features = payload.get("feature_names")
        if model_features and list(model_features) != self.feature_names:
            rospy.logwarn(
                "Model feature names do not match the current sensor mode; "
                "check _depth_topic and the training data."
            )
        if payload.get("model_type") == "reliability_weighted_cluster_readout":
            self.cluster_payload = payload
            self.vision_names = [str(name) for name in payload["vision_names"]]
            self.depth_names = [str(name) for name in payload["depth_names"]]
            rospy.loginfo(
                "Loaded reliability-weighted cluster readout from %s (K=%d, beta=%.2f)",
                self.model_path,
                int(payload["n_clusters"]),
                float(payload["beta"]),
            )
        else:
            self.model = payload["model"]
            self.scaler = payload["scaler"]
            rospy.loginfo("Loaded legacy supervised AHPFF model from %s", self.model_path)

    def convert_image(self, image_message: Image) -> np.ndarray | None:
        try:
            image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding="rgb8")
        except Exception as error:
            rospy.logwarn_throttle(5.0, "cv_bridge failed: %s", error)
            return None
        if self.vertical_flip:
            image = np.ascontiguousarray(cv2.flip(image, 0))
        return image

    def degradation_active(self) -> bool:
        if self.mode != "infer" or self.degradation_mode == "none":
            return False
        elapsed = rospy.Time.now().to_sec() - self.node_start_time
        if self.degradation_end <= self.degradation_start:
            return True
        return self.degradation_start <= elapsed < self.degradation_end

    def apply_degradation(
        self, image: np.ndarray, depth: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.degradation_active():
            return image, depth
        if not self.degradation_logged:
            rospy.loginfo(
                "Sensor degradation active: mode=%s start=%.1fs end=%.1fs strength=%.2f",
                self.degradation_mode,
                self.degradation_start,
                self.degradation_end,
                self.degradation_strength,
            )
            self.degradation_logged = True

        mode = self.degradation_mode
        strength = self.degradation_strength
        if mode in ("overexposure", "mixed"):
            image = np.clip(
                image.astype(np.float32) * 1.15 + 100.0 * strength, 0.0, 255.0
            ).astype(np.uint8)
        if mode in ("blur", "mixed"):
            kernel = np.zeros((1, 15), dtype=np.float32)
            kernel[0, :] = 1.0 / 15.0
            image = cv2.filter2D(image, -1, kernel)
        if depth is not None and mode in ("depth_dropout", "mixed"):
            rng = np.random.RandomState()
            mask = rng.rand(*depth.shape) < min(0.5 * strength, 0.95)
            depth = depth.copy()
            depth[mask] = 0.0

        if not self.preview_saved:
            self.preview_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(self.preview_path),
                cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR),
            )
            self.preview_saved = True
        return image, depth

    def synchronized_scan_callback(self, image_message: Image, scan_message: LaserScan) -> None:
        image = self.convert_image(image_message)
        if image is None:
            return
        image, _ = self.apply_degradation(image, None)
        visual = vision_features(image)
        distance = lidar_features(scan_message)
        if visual is None or distance is None:
            return
        self.handle_feature(
            np.concatenate([visual, distance]),
            float(image_message.header.stamp.to_sec()),
        )

    def synchronized_depth_callback(self, image_message: Image, depth_message: Image) -> None:
        image = self.convert_image(image_message)
        if image is None:
            return
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_message, desired_encoding="passthrough")
        except Exception as error:
            rospy.logwarn_throttle(5.0, "cv_bridge depth failed: %s", error)
            return
        depth = np.asarray(depth, dtype=np.float64)
        if self.vertical_flip:
            depth = np.ascontiguousarray(cv2.flip(depth, 0))
        image, depth = self.apply_degradation(image, depth)
        if self.mode == "infer" and self.publish_rectified:
            rectified = self.bridge.cv2_to_imgmsg(np.ascontiguousarray(image), encoding="passthrough")
            rectified.encoding = "rgb8"
            rectified.header = image_message.header
            try:
                self.rectified_pub.publish(rectified)
            except rospy.ROSException:
                return
        visual = vision_features(image)
        distance = depth_features(depth)
        if visual is None or distance is None:
            return
        self.handle_feature(
            np.concatenate([visual, distance]),
            float(image_message.header.stamp.to_sec()),
        )

    def handle_feature(self, feature: np.ndarray, timestamp: float) -> None:
        if self.shutting_down or rospy.is_shutdown():
            return
        if self.mode == "collect":
            if self.latest_label is None:
                return
            self.features.append(feature)
            self.labels.append(int(self.latest_label))
            self.timestamps.append(timestamp)
            if len(self.labels) % 100 == 0:
                rospy.loginfo("Collected %d AHPFF samples", len(self.labels))
            return

        if self.cluster_payload is not None:
            vision_dim = len(self.vision_names)
            raw_vision = feature[None, :vision_dim]
            raw_depth = feature[None, vision_dim:]
            prediction_array, cluster_array, qv, qd, confidence_array = predict_cluster_readout(
                self.cluster_payload,
                raw_vision,
                raw_depth,
                self.vision_names,
                self.depth_names,
                hard_depth_dropout=True,
            )
            prediction = int(prediction_array[0])
            cluster = int(cluster_array[0])
            wv, wd = modality_weights(
                qv, qd, beta=float(self.cluster_payload["beta"]), hard_depth_dropout=True
            )
            weight_v = float(wv[0])
            weight_d = float(wd[0])
            confidence = float(confidence_array[0])
            self.clusters.append(cluster)
            self.visual_quality.append(float(qv[0]))
            self.depth_quality.append(float(qd[0]))
        else:
            scaled = self.scaler.transform(feature[None, :])
            prediction = int(self.model.predict(scaled)[0])
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(scaled)[0]
                confidence = float(np.max(probabilities))
            else:
                confidence = 1.0
            weight_v = float("nan")
            weight_d = float("nan")
        self.smoothed_predictions.append(prediction)
        smoothed = int(Counter(self.smoothed_predictions).most_common(1)[0][0])
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.visual_weights.append(weight_v)
        self.depth_weights.append(weight_d)
        self.timestamps.append(timestamp)
        try:
            self.label_pub.publish(Int32(smoothed))
            self.confidence_pub.publish(Float32(confidence))
            self.visual_weight_pub.publish(Float32(weight_v))
            self.depth_weight_pub.publish(Float32(weight_d))
        except rospy.ROSException:
            return

    def on_shutdown(self) -> None:
        self.shutting_down = True
        if self.mode == "collect":
            if not self.features:
                rospy.logwarn("No calibration samples were collected.")
                return
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                self.output_path,
                features=np.asarray(self.features),
                labels=np.asarray(self.labels, dtype=np.int64),
                timestamps=np.asarray(self.timestamps),
                feature_names=np.asarray(self.feature_names, dtype=object),
                roi=np.asarray(ROI),
            )
            counts = Counter(self.labels)
            rospy.loginfo(
                "Saved %d calibration samples to %s; counts=%s",
                len(self.labels),
                self.output_path,
                {CLASS_NAMES.get(key, str(key)): value for key, value in counts.items()},
            )
            return

        if self.predictions:
            prediction_path = self.prediction_path
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                prediction_path,
                predictions=np.asarray(self.predictions, dtype=np.int64),
                confidences=np.asarray(self.confidences),
                timestamps=np.asarray(self.timestamps),
                feature_names=np.asarray(self.feature_names, dtype=object),
                clusters=np.asarray(self.clusters, dtype=np.int64),
                visual_weights=np.asarray(self.visual_weights),
                depth_weights=np.asarray(self.depth_weights),
                visual_quality=np.asarray(self.visual_quality),
                depth_quality=np.asarray(self.depth_quality),
            )
            rospy.loginfo("Saved %d predictions to %s", len(self.predictions), prediction_path)


def main() -> None:
    AHPFFPerceptionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
