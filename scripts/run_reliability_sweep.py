#!/usr/bin/env python3
"""Reliability-to-performance sweep for the RA-AHPFF front end.

This experiment degrades one modality at a time over all samples, from clean
data to severe degradation, and tracks the health score, the softmax weight,
and the clustering accuracy/NMI.  It supplies the missing link between the
sensor degradation, the reliability score, the modality weight, and the
clustering result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from ahpff import (
    ReliabilityWeightedKMeans,
    compute_quality,
    map_clusters_to_classes,
    modality_weights,
)
from run_arw_mvc_expert import apply_depth_degradation, apply_visual_degradation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    return parser.parse_args()


def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    metadata = json.loads((data_dir / "nyu_feature_meta.json").read_text(encoding="utf-8"))
    vision = np.load(data_dir / "Zv_real_raw.npy").astype(np.float64)
    depth = np.load(data_dir / "Zl_real_raw.npy").astype(np.float64)
    target = np.load(data_dir / "Y_real.npy").astype(np.int64)
    return vision, depth, target, list(metadata["feature_names_vision"]), list(metadata["feature_names_lidar"])


def evaluate(
    vision: np.ndarray,
    depth: np.ndarray,
    target: np.ndarray,
    vision_names: list[str],
    depth_names: list[str],
    vision_scaler: StandardScaler,
    depth_scaler: StandardScaler,
    beta: float,
    seed: int,
) -> dict:
    qv, qd, _ = compute_quality(vision, depth, vision_names, depth_names)
    zv = vision_scaler.transform(vision)
    zd = depth_scaler.transform(depth)
    n_classes = int(np.unique(target).size)
    model = ReliabilityWeightedKMeans(
        n_clusters=n_classes,
        beta=beta,
        n_init=10,
        seed=seed,
    ).fit(zv, zd, qv, qd)
    mapping, _ = map_clusters_to_classes(model.labels_, target, n_classes, n_classes)
    prediction = mapping[model.labels_]
    wv, wd = modality_weights(qv, qd, beta=beta)
    return {
        "mean_qv": float(np.mean(qv)),
        "mean_qd": float(np.mean(qd)),
        "mean_wv": float(np.mean(wv)),
        "mean_wd": float(np.mean(wd)),
        "acc": float(np.mean(prediction == target)),
        "nmi": float(normalized_mutual_info_score(target, model.labels_)),
    }


def sweep(
    modality: str,
    vision: np.ndarray,
    depth: np.ndarray,
    target: np.ndarray,
    vision_names: list[str],
    depth_names: list[str],
    vision_scaler: StandardScaler,
    depth_scaler: StandardScaler,
    levels: list[float],
    seeds: int,
    beta: float,
) -> dict:
    rows = []
    for level in levels:
        values = []
        for seed in range(seeds):
            rng = np.random.RandomState(1000 + seed)
            current_vision = vision.copy()
            current_depth = depth.copy()
            if level > 0.0 and modality == "vision":
                apply_visual_degradation(current_vision, level, rng, vision_names)
            if level > 0.0 and modality == "depth":
                apply_depth_degradation(current_depth, level, rng, depth_names)
            values.append(
                evaluate(
                    current_vision,
                    current_depth,
                    target,
                    vision_names,
                    depth_names,
                    vision_scaler,
                    depth_scaler,
                    beta,
                    seed,
                )
            )
        rows.append(
            {
                "level": float(level),
                **{
                    key: {
                        "mean": float(np.mean([row[key] for row in values])),
                        "std": float(np.std([row[key] for row in values])),
                    }
                    for key in values[0]
                },
            }
        )
    return {"modality": modality, "levels": rows}


def plot_results(vision_sweep: dict, depth_sweep: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)

    def plot_reliability(axis, sweep, quality_key, weight_key, title, color):
        levels = [row["level"] for row in sweep["levels"]]
        quality = [row[quality_key]["mean"] for row in sweep["levels"]]
        quality_std = [row[quality_key]["std"] for row in sweep["levels"]]
        weight = [row[weight_key]["mean"] for row in sweep["levels"]]
        weight_std = [row[weight_key]["std"] for row in sweep["levels"]]
        axis.errorbar(levels, quality, yerr=quality_std, marker="o", capsize=3,
                      linewidth=2.0, label=quality_key.replace("mean_", ""), color=color)
        axis.errorbar(levels, weight, yerr=weight_std, marker="s", capsize=3,
                      linewidth=2.0, linestyle="--", label=weight_key.replace("mean_", ""), color="black")
        axis.set_xlabel("Degradation level", fontsize=10)
        axis.set_ylabel("Score / weight", fontsize=10)
        axis.set_ylim(0, 1)
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.legend(loc="best", fontsize=9)

    def plot_performance(axis, sweep, title):
        levels = [row["level"] for row in sweep["levels"]]
        acc = [row["acc"]["mean"] * 100.0 for row in sweep["levels"]]
        acc_std = [row["acc"]["std"] * 100.0 for row in sweep["levels"]]
        nmi = [row["nmi"]["mean"] * 100.0 for row in sweep["levels"]]
        nmi_std = [row["nmi"]["std"] * 100.0 for row in sweep["levels"]]
        axis.errorbar(levels, acc, yerr=acc_std, marker="o", capsize=3,
                      linewidth=2.2, label="ACC", color="#2ECC71")
        axis.errorbar(levels, nmi, yerr=nmi_std, marker="s", capsize=3,
                      linewidth=2.0, linestyle="--", label="NMI", color="#F39C12")
        axis.set_xlabel("Degradation level", fontsize=10)
        axis.set_ylabel("Clustering score (%)", fontsize=10)
        axis.set_ylim(0, 100)
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.legend(loc="best", fontsize=9)

    plot_reliability(axes[0, 0], vision_sweep, "mean_qv", "mean_wv",
                     "(a) Visual degradation: reliability and weight", "#3498DB")
    plot_performance(axes[0, 1], vision_sweep, "(b) Visual degradation: clustering")
    plot_reliability(axes[1, 0], depth_sweep, "mean_qd", "mean_wd",
                     "(c) Depth degradation: reliability and weight", "#E74C3C")
    plot_performance(axes[1, 1], depth_sweep, "(d) Depth degradation: clustering")

    fig.suptitle(
        "Reliability-to-performance sweep on the corrected NYU Depth V2 pilot subset\n"
        "One modality is degraded at a time; the other modality remains clean",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    vision, depth, target, vision_names, depth_names = load_data(args.data_dir)
    vision_scaler = StandardScaler().fit(vision)
    depth_scaler = StandardScaler().fit(depth)
    levels = sorted(float(level) for level in args.levels)

    vision_sweep = sweep(
        "vision", vision, depth, target, vision_names, depth_names,
        vision_scaler, depth_scaler, levels, args.seeds, args.beta,
    )
    depth_sweep = sweep(
        "depth", vision, depth, target, vision_names, depth_names,
        vision_scaler, depth_scaler, levels, args.seeds, args.beta,
    )
    output = {
        "dataset": {
            "name": "NYU Depth V2 corrected pilot subset",
            "samples": int(target.size),
            "classes": int(np.unique(target).size),
        },
        "protocol": {
            "degradation": "one modality degraded over all samples; the other modality clean",
            "levels": levels,
            "seeds": args.seeds,
            "beta": args.beta,
            "clusters": int(np.unique(target).size),
        },
        "vision_sweep": vision_sweep,
        "depth_sweep": depth_sweep,
    }
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.figure_dir / "reliability_sweep.png"
    metrics_path = args.figure_dir / "reliability_sweep.json"
    plot_results(vision_sweep, depth_sweep, figure_path)
    metrics_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Visual degradation sweep")
    for row in vision_sweep["levels"]:
        print(
            f"  level {row['level']:.2f}: q_v={row['mean_qv']['mean']:.3f} "
            f"w_v={row['mean_wv']['mean']:.3f} ACC={row['acc']['mean']*100:.1f} "
            f"NMI={row['nmi']['mean']*100:.1f}"
        )
    print("Depth degradation sweep")
    for row in depth_sweep["levels"]:
        print(
            f"  level {row['level']:.2f}: q_d={row['mean_qd']['mean']:.3f} "
            f"w_d={row['mean_wd']['mean']:.3f} ACC={row['acc']['mean']*100:.1f} "
            f"NMI={row['nmi']['mean']*100:.1f}"
        )
    print(f"Saved figure: {figure_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
