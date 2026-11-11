#!/usr/bin/env python3
"""Plot representative Go1 closed-loop figures from a recorded run.

Only four trajectories are drawn (ground truth, leg odometry, standard PF,
large-covariance PF, and SAPF-Pred) so that the figure matches the fair
comparison table without an overcrowded legend.  The semantic-label figure
explicitly identifies the run as a representative run and reports the
three-run mean in the title.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "figures" / "degradation_runs" / "baseline_1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures")
    parser.add_argument("--mean-accuracy", type=float, default=0.901)
    parser.add_argument("--std-accuracy", type=float, default=0.010)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.run_dir / "sapf_gazebo_trajectory.npz")
    result = json.loads((args.run_dir / "sapf_gazebo_result.json").read_text(encoding="utf-8"))
    accuracy = float(result["semantic_label_accuracy"])
    time = data["time"]
    truth = data["truth"]
    spf = data["spf"]
    lpf = data["lpf"]
    sapf = data["sapf"]
    labels = data["labels"]
    ground_truth = data["ground_truth_labels"]
    class_names = ["Flat", "Stairs", "Obstacle", "Tactile"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(10, 5.8), dpi=300)
    axis.plot(truth[:, 0], truth[:, 2], "k-", linewidth=2.8, label="Ground truth")
    axis.plot(spf[:, 0], spf[:, 2], "b:", linewidth=2.2, label="Standard PF (SPF)")
    axis.plot(lpf[:, 0], lpf[:, 2], "m-.", linewidth=1.8, label="Large-Q PF (LPF)")
    axis.plot(sapf[:, 0], sapf[:, 2], "g-", linewidth=2.4, label="SAPF-Pred")
    axis.axvline(5.0, color="gray", linestyle="-.", alpha=0.55)
    axis.axvline(8.0, color="gray", linestyle="-.", alpha=0.55)
    axis.set_xlabel("X position (m)", fontsize=11)
    axis.set_ylabel("Z position (m)", fontsize=11)
    axis.set_title("Representative Go1 closed-loop trajectory", fontsize=12, fontweight="bold")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    trajectory_path = args.output_dir / "trajectory_lpf_sapf.png"
    plt.savefig(trajectory_path, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 3.6), dpi=300)
    axis.step(time, labels, where="post", color="darkorange", linewidth=2.0, label="C_t used by Q")
    axis.step(time, ground_truth, where="post", color="black", linewidth=1.3, linestyle="--", label="Ground truth")
    axis.set_yticks([0, 1, 2, 3])
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Time (s)", fontsize=11)
    axis.set_ylabel("Terrain label C_t", fontsize=11)
    axis.set_title(
        "Representative run: online accuracy "
        f"{accuracy * 100:.1f}% (three-run mean {args.mean_accuracy * 100:.1f} +/- {args.std_accuracy * 100:.1f}%)",
        fontsize=11,
        fontweight="bold",
    )
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    labels_path = args.output_dir / "semantic_labels_representative.png"
    plt.savefig(labels_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {trajectory_path}")
    print(f"Saved {labels_path}")
    print(f"Representative run accuracy: {accuracy * 100:.1f}%")
    print(f"Three-run mean: {args.mean_accuracy * 100:.1f} +/- {args.std_accuracy * 100:.1f}%")


if __name__ == "__main__":
    main()
