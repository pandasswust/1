#!/usr/bin/env python3
"""Fair comparison of the 6-DoF SAPF on a standalone stair-climbing sequence.

The semantic labels are produced by the reliability-aware cluster readout fitted
on the corrected NYU Depth V2 pilot subset.  The trajectory and sensors are
synthetic, so this script is not a ROS-Gazebo or real-robot experiment.  It
compares odometry, a standard PF, a fixed large-covariance PF, a Kalman filter,
and SAPF with predicted, oracle, and shuffled terrain labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from ahpff import (
    ReliabilityWeightedKMeans,
    compute_quality,
    map_clusters_to_classes,
    predict_cluster_readout,
    modality_weights,
)
from run_arw_mvc_expert import apply_depth_degradation, apply_visual_degradation
from sapf import (
    SAPFConfig,
    SemanticAdaptiveParticleFilter,
    SixDoFExtendedKalmanFilter,
    wrap_angles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"
CLASS_NAMES = ("Flat", "Stairs", "Obstacle")
STATE_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
METHODS = ("LO", "SPF", "LPF", "EKF", "SAPF-Pred", "SAPF-Oracle", "SAPF-Shuffled")
METHOD_LABELS = {
    "LO": "Odometry",
    "SPF": "Standard PF",
    "LPF": "Large-Q PF",
    "EKF": "EKF",
    "SAPF-Pred": "SAPF-Pred",
    "SAPF-Oracle": "SAPF-Oracle",
    "SAPF-Shuffled": "SAPF-Shuffled",
}
BASE_OBS_STD = np.array([0.02, 0.02, 0.03, 0.01, 0.01, 0.01])
BASE_PROCESS_STD = np.array([0.002, 0.002, 0.001, 0.001, 0.001, 0.001])
PROCESS_BOOST = {
    0: np.zeros(6),
    1: np.array(
        [0.0022360679775, 0.0022360679775, 0.0059160797831, 0.0038729833462, 0.0109544511501, 0.0017320508076]
    ),
    2: np.array(
        [0.0118321595662, 0.0118321595662, 0.0017320508076, 0.0059160797831, 0.0017320508076, 0.0109544511501]
    ),
    3: np.array([0.005, 0.005, 0.002, 0.002, 0.002, 0.005]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--step-height", type=float, default=0.06)
    parser.add_argument("--step-period", type=float, default=1.0)
    parser.add_argument("--odom-bias-z", type=float, default=0.001)
    parser.add_argument("--obs-z-flat", type=float, default=0.05)
    parser.add_argument("--obs-z-stairs", type=float, default=0.10)
    parser.add_argument("--obs-rate", type=float, default=5.0)
    parser.add_argument(
        "--obs-outages",
        type=str,
        default="8:12,18:22",
        help="comma-separated start:end exteroceptive-pose outage windows in seconds",
    )
    parser.add_argument("--feature-noise", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--clusters", type=int, default=0, help="0 uses the number of classes")
    parser.add_argument("--base-seed", type=int, default=42)
    return parser.parse_args()


def load_frontend(data_dir: Path, seed: int, beta: float, clusters: int) -> dict:
    metadata = json.loads((data_dir / "nyu_feature_meta.json").read_text(encoding="utf-8"))
    vision_raw = np.load(data_dir / "Zv_real_raw.npy").astype(np.float64)
    depth_raw = np.load(data_dir / "Zl_real_raw.npy").astype(np.float64)
    target = np.load(data_dir / "Y_real.npy").astype(np.int64)
    vision_names = list(metadata["feature_names_vision"])
    depth_names = list(metadata["feature_names_lidar"])
    if vision_raw.shape[0] != target.size or depth_raw.shape[0] != target.size:
        raise SystemExit("Feature and label files are inconsistent.")

    rng = np.random.RandomState(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in np.unique(target):
        class_indices = np.where(target == label)[0].copy()
        rng.shuffle(class_indices)
        n_test = max(1, int(round(class_indices.size * 0.33)))
        test_indices.extend(class_indices[:n_test].tolist())
        train_indices.extend(class_indices[n_test:].tolist())
    train_indices = np.asarray(sorted(train_indices), dtype=np.int64)
    test_indices = np.asarray(sorted(test_indices), dtype=np.int64)

    vision_scaler = StandardScaler().fit(vision_raw[train_indices])
    depth_scaler = StandardScaler().fit(depth_raw[train_indices])
    train_vision = vision_scaler.transform(vision_raw[train_indices])
    train_depth = depth_scaler.transform(depth_raw[train_indices])
    qv_train, qd_train, _ = compute_quality(
        vision_raw[train_indices], depth_raw[train_indices], vision_names, depth_names
    )
    n_classes = int(np.unique(target).size)
    n_clusters = clusters if clusters > 0 else n_classes
    model = ReliabilityWeightedKMeans(
        n_clusters=n_clusters,
        beta=beta,
        n_init=15,
        seed=seed,
    ).fit(train_vision, train_depth, qv_train, qd_train)
    mapping, _ = map_clusters_to_classes(model.labels_, target[train_indices], n_clusters, n_classes)
    payload = {
        "vision_scaler": vision_scaler,
        "depth_scaler": depth_scaler,
        "clusters": model,
        "cluster_to_class": mapping,
        "n_clusters": n_clusters,
        "beta": beta,
    }
    test_prediction, test_cluster, test_qv, test_qd, test_confidence = predict_cluster_readout(
        payload,
        vision_raw[test_indices],
        depth_raw[test_indices],
        vision_names,
        depth_names,
    )
    holdout_accuracy = float(accuracy_score(target[test_indices], test_prediction))
    holdout_confusion = confusion_matrix(target[test_indices], test_prediction, labels=np.arange(n_classes))

    return {
        "payload": payload,
        "vision_raw_test": vision_raw[test_indices],
        "depth_raw_test": depth_raw[test_indices],
        "test_labels": target[test_indices],
        "test_predictions": test_prediction,
        "test_confidence": test_confidence,
        "vision_names": vision_names,
        "depth_names": depth_names,
        "holdout_accuracy": holdout_accuracy,
        "holdout_confusion": holdout_confusion.tolist(),
        "n_train": int(train_indices.size),
        "n_test": int(test_indices.size),
        "class_names": CLASS_NAMES,
        "n_clusters": n_clusters,
        "beta": beta,
    }


def generate_ground_truth(
    duration: float,
    dt: float,
    step_height: float,
    step_period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.arange(0.0, duration, dt)
    speed = 0.2
    yaw = np.zeros_like(time)
    obstacle = (time >= 8.0) & (time < 10.0)
    yaw[obstacle] = 0.25 * np.sin(np.pi * (time[obstacle] - 8.0) / 2.0)
    x = np.cumsum(speed * np.cos(yaw) * dt)
    y = np.cumsum(speed * np.sin(yaw) * dt)
    x -= x[0]
    y -= y[0]

    stair_start = 14.0
    stair_end = 26.0
    stairs = (time >= stair_start) & (time < stair_end)
    n_steps = int(np.floor((stair_end - stair_start) / step_period))
    z = np.full_like(time, 0.40)
    phase = time[stairs] - stair_start
    step_index = np.minimum(np.floor(phase / step_period), n_steps - 1)
    fraction = phase - step_index * step_period
    ramp = np.clip(fraction / 0.20, 0.0, 1.0)
    z[stairs] = 0.40 + step_height * (step_index + ramp)
    z[time >= stair_end] = 0.40 + step_height * n_steps

    pitch = np.zeros_like(time)
    pitch[stairs] = np.arctan2(step_height, step_period * speed) + 0.010 * np.sin(
        2.0 * np.pi * 1.2 * time[stairs]
    )
    roll = np.zeros_like(time)
    roll[stairs] = 0.015 * np.sin(2.0 * np.pi * 1.0 * time[stairs])
    roll[obstacle] = 0.02 * np.sin(2.0 * np.pi * 1.0 * time[obstacle])

    state = wrap_angles(np.column_stack([x, y, z, roll, pitch, yaw]))
    labels = np.zeros(time.size, dtype=np.int64)
    labels[obstacle] = 2
    labels[stairs] = 1
    return time, state, labels


def make_sensors(
    state: np.ndarray,
    labels: np.ndarray,
    time: np.ndarray,
    seed: int,
    odom_bias_z: float,
    obs_z_flat: float,
    obs_z_stairs: float,
    obs_rate: float,
    obs_outages: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = time.size
    true_delta = wrap_angles(np.diff(state, axis=0))
    bias = np.zeros((n - 1, 6))
    stairs_delta = labels[:-1] == 1
    obstacle_delta = labels[:-1] == 2
    bias[stairs_delta, 2] = odom_bias_z
    bias[stairs_delta, 4] = 0.0001
    bias[obstacle_delta, 0] = 0.0002
    bias[obstacle_delta, 5] = 0.0002
    odom_delta = wrap_angles(
        true_delta
        + bias
        + rng.normal(0.0, np.array([0.005, 0.005, 0.005, 0.002, 0.002, 0.002]), size=(n - 1, 6))
    )

    odom = np.zeros_like(state)
    odom[0] = state[0] + rng.normal(0.0, np.array([0.01, 0.01, 0.01, 0.005, 0.005, 0.005]))
    for index in range(1, n):
        odom[index] = wrap_angles(odom[index - 1] + odom_delta[index - 1])

    observation_std = np.tile(BASE_OBS_STD, (n, 1))
    stairs = labels == 1
    observation_std[:, 2] = obs_z_flat
    observation_std[stairs, 2] = obs_z_stairs
    observation_std[stairs, 4] = 0.02

    active = np.zeros(n, dtype=bool)
    period = max(1, int(round(1.0 / max(obs_rate, 1e-6) / (time[1] - time[0]))))
    active[::period] = True
    for start, end in obs_outages:
        active[(time >= start) & (time < end)] = False
    observation_std[~active] = 1e3
    observation = wrap_angles(state + rng.normal(0.0, 1.0, size=state.shape) * observation_std)
    return odom_delta, odom, observation, observation_std


def semantic_predictions(
    frontend: dict,
    true_labels: np.ndarray,
    time: np.ndarray,
    seed: int,
    feature_noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    vision = frontend["vision_raw_test"].copy()
    depth = frontend["depth_raw_test"].copy()
    labels = frontend["test_labels"]
    prediction = np.zeros(true_labels.size, dtype=np.int64)
    qv_history = np.zeros(true_labels.size)
    qd_history = np.zeros(true_labels.size)
    for index, label in enumerate(true_labels):
        candidates = np.where(labels == label)[0]
        if candidates.size == 0:
            prediction[index] = int(label)
            continue
        selected = int(rng.choice(candidates))
        sample_v = vision[selected : selected + 1].copy()
        sample_d = depth[selected : selected + 1].copy()
        scale_v = np.std(vision, axis=0)
        scale_d = np.std(depth, axis=0)
        sample_v += rng.normal(0.0, feature_noise * 0.05, size=sample_v.shape) * scale_v
        sample_d += rng.normal(0.0, feature_noise * 0.05, size=sample_d.shape) * scale_d
        timestamp = float(time[index])
        if 16.0 <= timestamp < 20.0:
            apply_visual_degradation(sample_v, 1.5, rng, frontend["vision_names"])
        if 20.0 <= timestamp < 24.0:
            apply_depth_degradation(sample_d, 1.5, rng, frontend["depth_names"])
        pred, cluster, qv, qd, confidence = predict_cluster_readout(
            frontend["payload"],
            sample_v,
            sample_d,
            frontend["vision_names"],
            frontend["depth_names"],
        )
        prediction[index] = int(pred[0])
        qv_history[index] = float(qv[0])
        qd_history[index] = float(qd[0])
    return prediction, qv_history, qd_history


def run_filter(
    method: str,
    config: SAPFConfig,
    observation: np.ndarray,
    observation_std: np.ndarray,
    odom_delta: np.ndarray,
    semantic_labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int, float]:
    if method == "LO":
        raise ValueError("Odometry is evaluated directly.")
    initial_std = np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01])
    rng = np.random.RandomState(seed)
    if method == "EKF":
        estimator = SixDoFExtendedKalmanFilter(config.process_std_base, rng)
        estimator.initialize(observation[0], initial_std)
        estimates = np.zeros((observation.shape[0], 6))
        estimates[0] = estimator.estimate()
        for index in range(1, observation.shape[0]):
            estimates[index] = estimator.step(
                odom_delta[index - 1], observation[index], observation_std[index]
            )
        return estimates, 0, 0.0

    estimator = SemanticAdaptiveParticleFilter(config, rng)
    estimator.initialize(observation[0], initial_std)
    estimates = np.zeros((observation.shape[0], 6))
    estimates[0] = estimator.estimate()
    for index in range(1, observation.shape[0]):
        estimates[index] = estimator.step(
            odom_delta[index - 1],
            observation[index],
            observation_std[index],
            int(semantic_labels[index - 1]),
        )
    mean_ess = float(np.mean(estimator.ess_history)) if estimator.ess_history else 0.0
    return estimates, estimator.resample_count, mean_ess


def compute_metrics(estimate: np.ndarray, truth: np.ndarray, labels: np.ndarray) -> dict:
    difference = wrap_angles(estimate - truth)
    axis_rmse = np.sqrt(np.mean(difference**2, axis=0))
    position_rmse = float(np.sqrt(np.mean(np.sum(difference[:, :3] ** 2, axis=1))))
    orientation_rmse = float(np.sqrt(np.mean(np.sum(difference[:, 3:] ** 2, axis=1))))
    stairs = labels == 1
    stairs_z_rmse = float(np.sqrt(np.mean(difference[stairs, 2] ** 2)))
    return {
        "axis_rmse": {name: float(value) for name, value in zip(STATE_NAMES, axis_rmse)},
        "position_rmse": position_rmse,
        "orientation_rmse": orientation_rmse,
        "stairs_z_rmse": stairs_z_rmse,
        "final_z_error": float(abs(difference[-1, 2])),
    }


def aggregate(per_seed: list[dict]) -> dict:
    aggregate_metrics: dict[str, dict] = {}
    for method in METHODS:
        method_metrics: dict[str, dict] = {}
        for axis in STATE_NAMES:
            values = [row[method]["axis_rmse"][axis] for row in per_seed]
            method_metrics[axis] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for key in ("position_rmse", "orientation_rmse", "stairs_z_rmse", "final_z_error"):
            values = [row[method][key] for row in per_seed]
            method_metrics[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        aggregate_metrics[method] = method_metrics
    return aggregate_metrics


def plot_results(
    time: np.ndarray,
    truth: np.ndarray,
    estimates: dict[str, np.ndarray],
    aggregate_metrics: dict,
    output_path: Path,
    semantic_accuracy: float,
    seed_count: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=300)
    trajectory_axis = axes[0]
    trajectory_axis.plot(truth[:, 0], truth[:, 2], "k-", linewidth=2.5, label="Ground Truth")
    trajectory_axis.plot(estimates["LO"][:, 0], estimates["LO"][:, 2], "r--", linewidth=1.4, label="Odometry")
    trajectory_axis.plot(estimates["SPF"][:, 0], estimates["SPF"][:, 2], "b:", linewidth=1.8, label="Standard PF")
    trajectory_axis.plot(estimates["LPF"][:, 0], estimates["LPF"][:, 2], "m-.", linewidth=1.5, label="Large-Q PF")
    trajectory_axis.plot(estimates["SAPF-Pred"][:, 0], estimates["SAPF-Pred"][:, 2], "g-", linewidth=2.0, label="SAPF-Pred")
    trajectory_axis.plot(
        estimates["SAPF-Oracle"][:, 0], estimates["SAPF-Oracle"][:, 2], "c--", linewidth=1.5, label="SAPF-Oracle"
    )
    trajectory_axis.plot(
        estimates["SAPF-Shuffled"][:, 0], estimates["SAPF-Shuffled"][:, 2], color="gray", linestyle=":", linewidth=1.5,
        label="SAPF-Shuffled",
    )
    stair_start = time[np.argmax((time >= 14.0) & (time < 26.0))]
    stair_end = time[np.argmax((time >= 26.0))]
    trajectory_axis.axvline(stair_start, color="gray", linestyle="-.", alpha=0.6)
    trajectory_axis.axvline(stair_end, color="gray", linestyle="-.", alpha=0.6)
    trajectory_axis.set_xlabel("X Position (m)", fontsize=11)
    trajectory_axis.set_ylabel("Z Position (m)", fontsize=11)
    trajectory_axis.set_title("X-Z Trajectory", fontsize=12, fontweight="bold")
    trajectory_axis.grid(True, linestyle="--", alpha=0.4)
    trajectory_axis.legend(loc="upper left", fontsize=7)

    bar_axis = axes[1]
    x = np.arange(len(METHODS))
    width = 0.36
    z_values = [aggregate_metrics[method]["z"]["mean"] for method in METHODS]
    z_errors = [aggregate_metrics[method]["z"]["std"] for method in METHODS]
    position_values = [aggregate_metrics[method]["position_rmse"]["mean"] for method in METHODS]
    position_errors = [aggregate_metrics[method]["position_rmse"]["std"] for method in METHODS]
    bar_axis.bar(
        x - width / 2, z_values, width, yerr=z_errors, capsize=3, label="Z RMSE",
        color="#5DADE2", edgecolor="black", linewidth=0.5,
    )
    bar_axis.bar(
        x + width / 2, position_values, width, yerr=position_errors, capsize=3, label="Position RMSE",
        color="#F1948A", edgecolor="black", linewidth=0.5,
    )
    bar_axis.set_xticks(x)
    bar_axis.set_xticklabels([METHOD_LABELS[method] for method in METHODS], rotation=25, ha="right", fontsize=8)
    bar_axis.set_ylabel("RMSE (m)", fontsize=11)
    bar_axis.set_title(
        f"RMSE over {seed_count} seeds; cluster-readout accuracy {semantic_accuracy * 100:.1f}%",
        fontsize=11,
        fontweight="bold",
    )
    bar_axis.grid(axis="y", linestyle="--", alpha=0.4)
    bar_axis.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Fair SAPF comparison on a standalone stair-climbing simulation\n"
        "Synthetic trajectory/sensors; cluster labels from the corrected NYU pilot subset",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frontend = load_frontend(args.data_dir, args.base_seed, args.beta, args.clusters)
    time, truth, true_labels = generate_ground_truth(
        args.duration, args.dt, args.step_height, args.step_period
    )
    obs_outages = []
    for item in args.obs_outages.split(","):
        item = item.strip()
        if not item:
            continue
        start, end = item.split(":")
        obs_outages.append((float(start), float(end)))

    config_spf = SAPFConfig(
        num_particles=args.particles,
        process_std_base=BASE_PROCESS_STD,
        process_std_boost=PROCESS_BOOST,
        use_semantics=False,
    )
    config_sapf = SAPFConfig(
        num_particles=args.particles,
        process_std_base=BASE_PROCESS_STD,
        process_std_boost=PROCESS_BOOST,
        use_semantics=True,
    )
    config_lpf = SAPFConfig(
        num_particles=args.particles,
        process_std_base=np.sqrt(BASE_PROCESS_STD**2 + np.max(np.vstack(list(PROCESS_BOOST.values())), axis=0) ** 2),
        process_std_boost=PROCESS_BOOST,
        use_semantics=False,
    )
    config_ekf = SAPFConfig(
        num_particles=args.particles,
        process_std_base=BASE_PROCESS_STD,
        process_std_boost=PROCESS_BOOST,
        use_semantics=False,
    )

    per_seed: list[dict] = []
    semantic_accuracies: list[float] = []
    weight_rows = []
    resample_counts = {method: [] for method in METHODS if method not in ("LO", "EKF")}
    mean_ess_values = {method: [] for method in METHODS if method not in ("LO", "EKF")}
    representative_estimates: dict[str, np.ndarray] = {}

    for index in range(args.seeds):
        seed = args.base_seed + index
        odom_delta, odom, observation, observation_std = make_sensors(
            truth,
            true_labels,
            time,
            seed,
            args.odom_bias_z,
            args.obs_z_flat,
            args.obs_z_stairs,
            args.obs_rate,
            obs_outages,
        )
        predicted_labels, qv_history, qd_history = semantic_predictions(
            frontend, true_labels, time, seed, args.feature_noise
        )
        semantic_accuracies.append(float(np.mean(predicted_labels == true_labels)))
        wv, wd = modality_weights(qv_history, qd_history, beta=args.beta)
        weight_rows.append(
            {
                "mean_wv": float(np.mean(wv)),
                "mean_wd": float(np.mean(wd)),
                "mean_wv_during_visual_degradation": float(np.mean(wv[(time >= 16.0) & (time < 20.0)])),
                "mean_wd_during_depth_degradation": float(np.mean(wd[(time >= 20.0) & (time < 24.0)])),
            }
        )
        shuffled_labels = predicted_labels.copy()
        np.random.RandomState(seed + 771).shuffle(shuffled_labels)

        estimates = {"LO": odom}
        spf_estimate, spf_resamples, spf_ess = run_filter(
            "SPF", config_spf, observation, observation_std, odom_delta, predicted_labels, seed + 1000
        )
        lpf_estimate, lpf_resamples, lpf_ess = run_filter(
            "LPF", config_lpf, observation, observation_std, odom_delta, predicted_labels, seed + 2000
        )
        ekf_estimate, _, _ = run_filter(
            "EKF", config_ekf, observation, observation_std, odom_delta, predicted_labels, seed + 3000
        )
        sapf_estimate, sapf_resamples, sapf_ess = run_filter(
            "SAPF-Pred", config_sapf, observation, observation_std, odom_delta, predicted_labels, seed + 4000
        )
        oracle_estimate, oracle_resamples, oracle_ess = run_filter(
            "SAPF-Oracle", config_sapf, observation, observation_std, odom_delta, true_labels, seed + 5000
        )
        shuffled_estimate, shuffled_resamples, shuffled_ess = run_filter(
            "SAPF-Shuffled", config_sapf, observation, observation_std, odom_delta, shuffled_labels, seed + 6000
        )
        estimates.update(
            {
                "SPF": spf_estimate,
                "LPF": lpf_estimate,
                "EKF": ekf_estimate,
                "SAPF-Pred": sapf_estimate,
                "SAPF-Oracle": oracle_estimate,
                "SAPF-Shuffled": shuffled_estimate,
            }
        )
        for method, count, ess in (
            ("SPF", spf_resamples, spf_ess),
            ("LPF", lpf_resamples, lpf_ess),
            ("SAPF-Pred", sapf_resamples, sapf_ess),
            ("SAPF-Oracle", oracle_resamples, oracle_ess),
            ("SAPF-Shuffled", shuffled_resamples, shuffled_ess),
        ):
            resample_counts[method].append(count)
            mean_ess_values[method].append(ess)

        row = {method: compute_metrics(estimates[method], truth, true_labels) for method in METHODS}
        per_seed.append(row)
        if index == 0:
            representative_estimates = {method: estimates[method].copy() for method in METHODS}

    aggregate_metrics = aggregate(per_seed)
    semantic_accuracy = float(np.mean(semantic_accuracies))
    output = {
        "dataset": {
            "name": "Standalone 6-DoF stair-climbing simulation",
            "synthetic": True,
            "note": (
                "Synthetic trajectory and sensors. Semantic labels are produced by the "
                "reliability-aware cluster readout on the corrected NYU Depth V2 pilot "
                "subset. This is not ROS-Gazebo and not a real robot run."
            ),
        },
        "frontend": {
            "class_names": list(CLASS_NAMES),
            "holdout_accuracy": frontend["holdout_accuracy"],
            "holdout_confusion": frontend["holdout_confusion"],
            "n_train": frontend["n_train"],
            "n_test": frontend["n_test"],
            "n_clusters": frontend["n_clusters"],
            "beta": frontend["beta"],
            "sequence_semantic_accuracy_mean": semantic_accuracy,
            "sequence_semantic_accuracy_std": float(np.std(semantic_accuracies)),
            "weights": {
                key: float(np.mean([row[key] for row in weight_rows]))
                for key in weight_rows[0]
            },
        },
        "simulation": {
            "state": list(STATE_NAMES),
            "duration": args.duration,
            "dt": args.dt,
            "steps": int(time.size),
            "step_height": args.step_height,
            "step_period": args.step_period,
            "odom_bias_z": args.odom_bias_z,
            "obs_z_flat": args.obs_z_flat,
            "obs_z_stairs": args.obs_z_stairs,
            "obs_rate_hz": args.obs_rate,
            "obs_outages_s": obs_outages,
            "feature_noise": args.feature_noise,
            "particles": args.particles,
            "seeds": args.seeds,
            "true_label_counts": {
                name: int(np.sum(true_labels == index)) for index, name in enumerate(CLASS_NAMES)
            },
            "process_model": "x_t = x_{t-1} + u_t + N(0, Q(C_t)); diagonal Q",
            "observation_model": "z_t = x_t + N(0, R_t)",
            "resampling": "systematic resampling when ESS < 0.5 * N",
            "metrics_definition": (
                "axis RMSE per state component; overall position RMSE = "
                "sqrt(mean(||p_est - p_gt||^2)); orientation RMSE = "
                "sqrt(mean(||angle_error||^2))"
            ),
            "base_process_std": BASE_PROCESS_STD.tolist(),
            "process_boost": {str(key): value.tolist() for key, value in PROCESS_BOOST.items()},
            "resample_counts": {method: [int(value) for value in values] for method, values in resample_counts.items()},
            "mean_ess": {method: float(np.mean(values)) for method, values in mean_ess_values.items()},
        },
        "aggregate": aggregate_metrics,
        "per_seed": [
            {"seed": args.base_seed + index, "semantic_accuracy": semantic_accuracies[index], **row}
            for index, row in enumerate(per_seed)
        ],
    }

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.figure_dir / "sapf_tracking_result.png"
    metrics_path = args.figure_dir / "sapf_tracking_result.json"
    plot_results(
        time, truth, representative_estimates, aggregate_metrics, figure_path, semantic_accuracy, args.seeds
    )
    metrics_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Front-end holdout accuracy: {frontend['holdout_accuracy'] * 100:.1f}%")
    print(f"Sequence semantic accuracy: {semantic_accuracy * 100:.1f}%")
    print(f"Particles: {args.particles}; seeds: {args.seeds}; steps: {time.size}")
    print("Mean RMSE over seeds:")
    header = f"{'Method':14s} " + " ".join(f"{name:>8s}" for name in STATE_NAMES) + f" {'Overall':>9s} {'Orient.':>9s}"
    print(header)
    for method in METHODS:
        values = aggregate_metrics[method]
        row = [values[name]["mean"] for name in STATE_NAMES]
        row.append(values["position_rmse"]["mean"])
        row.append(values["orientation_rmse"]["mean"])
        print(f"{method:14s} " + " ".join(f"{value:8.4f}" for value in row))
    print(f"Saved figure: {figure_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
