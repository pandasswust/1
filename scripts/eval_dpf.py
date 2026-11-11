#!/usr/bin/env python3
"""Evaluate the 2D DPF demo on the synthetic eight-figure dataset.

This script is a path-fixed, reproducible proof-of-concept.  It is NOT the
6-DoF SAPF described in the paper: the dataset is synthetic, the state is 2D,
and there are no semantic labels or Q(C_t) adaptation.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_dpf import DPFFusionNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"
SEED = 42


def main() -> None:
    warnings.filterwarnings("ignore")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset_path = DATA_DIR / "dpf_dataset.pt"
    if not dataset_path.exists():
        raise SystemExit(f"Missing {dataset_path}. Run scripts/train_dpf.py first.")
    dataset = torch.load(dataset_path, weights_only=False)
    poses, scans = dataset["poses"], dataset["scans"]

    model = DPFFusionNet()
    weights_path = DATA_DIR / "dpf.pth"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"Loaded trained weights from {weights_path}")
    else:
        print(
            "WARNING: no trained weights found at data/dpf.pth; evaluating an "
            "untrained network. Run scripts/train_dpf.py first."
        )
    model.eval()

    num_particles = 100
    estimated_trajectory = []
    prior_trajectory = []
    true_trajectory = []

    print("Running the 2D DPF demo over the synthetic sequence.")
    for t in range(1, min(600, poses.shape[0])):
        true_pose = poses[t]
        prev_pose = poses[t - 1]
        scan = scans[t]
        particles = prev_pose + torch.randn(num_particles, 2) * 0.3
        with torch.no_grad():
            weights = model(scan, particles, prev_pose)
        estimated_pose = torch.sum(weights.unsqueeze(1) * particles, dim=0)
        estimated_trajectory.append(estimated_pose.numpy())
        prior_trajectory.append(prev_pose.numpy())
        true_trajectory.append(true_pose.numpy())

    estimated_trajectory = np.asarray(estimated_trajectory)
    prior_trajectory = np.asarray(prior_trajectory)
    true_trajectory = np.asarray(true_trajectory)

    rmse = float(np.sqrt(np.mean(np.sum((estimated_trajectory - true_trajectory) ** 2, axis=1))))
    prior_rmse = float(np.sqrt(np.mean(np.sum((prior_trajectory - true_trajectory) ** 2, axis=1))))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIG_DIR / "dpf_tracking_result.png"
    metrics_path = FIG_DIR / "dpf_tracking_result.json"

    plt.figure(figsize=(10, 8), dpi=150)
    plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], "k--", linewidth=2, label="Ground Truth")
    plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], "r-", linewidth=1.5, alpha=0.8, label="2D DPF Demo")
    blind_zone_1 = true_trajectory[149:199]
    blind_zone_2 = true_trajectory[399:449]
    plt.scatter(blind_zone_1[:, 0], blind_zone_1[:, 1], c="gray", s=30, alpha=0.5, label="Synthetic Blind Zone")
    plt.scatter(blind_zone_2[:, 0], blind_zone_2[:, 1], c="gray", s=30, alpha=0.5)
    plt.title(
        "2D DPF Demo on Synthetic Eight-Figure Data\n"
        "Not the paper's 6-DoF stair-climbing SAPF experiment",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("X Position (m)", fontsize=12)
    plt.ylabel("Y Position (m)", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()

    output = {
        "dataset": str(dataset_path),
        "synthetic": True,
        "state_dimension": 2,
        "note": (
            "Synthetic eight-figure trajectory with fake 360-D scans. This is a "
            "2D proof-of-concept, not the paper's 6-DoF SAPF or a real robot run."
        ),
        "rmse_2d": rmse,
        "prior_rmse_2d": prior_rmse,
        "trained_weights_loaded": weights_path.exists(),
    }
    metrics_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"2D RMSE: {rmse:.4f} m; previous-pose baseline RMSE: {prior_rmse:.4f} m")
    print(f"Saved figure: {figure_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
