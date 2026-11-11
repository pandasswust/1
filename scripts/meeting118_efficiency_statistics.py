#!/usr/bin/env python3
"""Meeting-118 sufficiency experiments: runtime, paired statistics, calibration scaling.

This script closes the experiment-sufficiency gap identified in the 118 review:

1. per-seed accuracy for all seven clustering methods on the verified NYU subset
   (clean and severe level 2.0, 20 seeds), with paired Wilcoxon signed-rank
   tests against the proposed method;
2. CPU-only latency for health scoring, softmax gating, nearest-centroid
   readout, RW-MVK calibration and the supervised ExtraTrees upper bound;
3. a label-efficiency curve on the 351-frame Go1 calibration set using the
   temporally blocked protocol, comparing the unsupervised cluster readout with
   the supervised upper bound as the number of calibration labels increases.

Outputs
-------
112/pdf/Figure9_efficiency_and_statistics.pdf
112/png_1000dpi/Figure9_efficiency_and_statistics_1000dpi.png
figures/meeting118_per_seed_metrics.csv
figures/meeting118_statistics.json
112/source_data/figure9_runtime.csv
112/source_data/figure9_paired_statistics.csv
112/source_data/figure9_calibration_curve.csv
112/source_data/figure9_per_seed_accuracy.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ahpff import (  # noqa: E402
    ReliabilityWeightedKMeans,
    blocked_stratified_folds,
    compute_quality,
    fit_cluster_readout,
    modality_weights,
    predict_cluster_readout,
)
from publication_figures import MM, PALETTE, apply_style, save_figure, write_csv  # noqa: E402
from run_arw_mvc_expert import (  # noqa: E402
    METHOD_LABELS,
    SEED,
    evaluate_method,
    inject_degradation,
    load_data,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "112"
SEVERE_LEVEL = 2.0
BETA = 4.0
METHOD_SHORT = {
    "Vision only": "Vision",
    "Depth only": "Depth",
    "Fixed concat": "Fixed concat",
    "PCA average": "PCA avg",
    "Uniform weighted": "Uniform",
    "Shuffled reliability": "Shuffled",
    "RA-AHPFF (Ours)": "Ours",
}
CALIBRATION_FRACTIONS = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
CALIBRATION_REPEATS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--per-seed-only", action="store_true")
    return parser.parse_args()


def per_seed_nyu(seeds: int, seed_base: int) -> tuple[dict, list[dict]]:
    vision_raw, depth_raw, target, metadata = load_data(PROJECT_ROOT / "data")
    vision_names = list(metadata["feature_names_vision"])
    depth_names = list(metadata["feature_names_lidar"])
    n_classes = int(np.unique(target).size)
    vision_scaler = StandardScaler().fit(vision_raw)
    depth_scaler = StandardScaler().fit(depth_raw)
    per_seed: dict[str, dict[str, list[float]]] = {
        level: {name: [] for name in METHOD_LABELS} for level in ("clean", "severe")
    }
    rows: list[dict] = []
    for index in range(seeds):
        seed = seed_base + index
        for level_name, level in (("clean", 0.0), ("severe", SEVERE_LEVEL)):
            degraded_vision, degraded_depth, qv, qd, _ = inject_degradation(
                vision_raw,
                depth_raw,
                target,
                vision_names,
                depth_names,
                level,
                seed,
                "disjoint",
                BETA,
                depth_hard_zero=False,
            )
            zv = vision_scaler.transform(degraded_vision)
            zd = depth_scaler.transform(degraded_depth)
            for name in METHOD_LABELS:
                metrics, prediction, _ = evaluate_method(
                    name,
                    zv,
                    zd,
                    qv,
                    qd,
                    target,
                    n_classes,
                    seed,
                    BETA,
                    hard_depth_dropout=False,
                )
                per_seed[level_name][name].append(float(metrics["acc"]))
                rows.append(
                    {
                        "level": level_name,
                        "seed": seed,
                        "method": name,
                        "accuracy": float(metrics["acc"]),
                        "nmi": float(metrics["nmi"]),
                        "ari": float(metrics["ari"]),
                        "recall_flat": float(metrics["recall"]["0"]),
                        "recall_stairs": float(metrics["recall"]["1"]),
                        "recall_obstacle": float(metrics["recall"]["2"]),
                    }
                )
    return per_seed, rows


def paired_statistics(per_seed: dict) -> list[dict]:
    rows: list[dict] = []
    ours = np.asarray(per_seed["severe"]["RA-AHPFF (Ours)"])
    for level_name in ("clean", "severe"):
        ours = np.asarray(per_seed[level_name]["RA-AHPFF (Ours)"])
        for name in METHOD_LABELS:
            if name == "RA-AHPFF (Ours)":
                continue
            baseline = np.asarray(per_seed[level_name][name])
            difference = ours - baseline
            if np.allclose(difference, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = wilcoxon(ours, baseline, alternative="two-sided", zero_method="wilcox")
            rows.append(
                {
                    "level": level_name,
                    "baseline": name,
                    "ours_mean": float(np.mean(ours)),
                    "baseline_mean": float(np.mean(baseline)),
                    "mean_gain": float(np.mean(difference)),
                    "median_gain": float(np.median(difference)),
                    "wins": int(np.sum(difference > 0)),
                    "losses": int(np.sum(difference < 0)),
                    "ties": int(np.sum(difference == 0)),
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant_0p05": bool(p_value < 0.05),
                }
            )
    return rows


def benchmark_runtime() -> list[dict]:
    vision_raw, depth_raw, target, metadata = load_data(PROJECT_ROOT / "data")
    vision_names = list(metadata["feature_names_vision"])
    depth_names = list(metadata["feature_names_lidar"])
    vision_scaler = StandardScaler().fit(vision_raw)
    depth_scaler = StandardScaler().fit(depth_raw)
    qv, qd, _ = compute_quality(vision_raw, depth_raw, vision_names, depth_names)
    zv = vision_scaler.transform(vision_raw)
    zd = depth_scaler.transform(depth_raw)

    calibration = np.load(PROJECT_ROOT / "data" / "ahpff_go1_calibration.npz", allow_pickle=True)
    features = calibration["features"]
    labels = calibration["labels"]
    go_vision_names = list(calibration["feature_names"][:13])
    go_depth_names = list(calibration["feature_names"][13:])
    go_vision = StandardScaler().fit_transform(features[:, :13])
    go_depth = StandardScaler().fit_transform(features[:, 13:])
    go_qv, go_qd, _ = compute_quality(features[:, :13], features[:, 13:], go_vision_names, go_depth_names)

    rows: list[dict] = []

    def timed(label: str, repeats: int, calls: int, operation) -> np.ndarray:
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(calls):
                operation()
            samples.append((time.perf_counter() - start) / calls * 1000.0)
        values = np.asarray(samples)
        rows.append(
            {
                "operation": label,
                "mean_ms": float(np.mean(values)),
                "std_ms": float(np.std(values)),
                "repeats": repeats,
                "calls_per_repeat": calls,
            }
        )
        return values

    timed("Health scores (per frame)", 5, 200, lambda: compute_quality(vision_raw[:1], depth_raw[:1], vision_names, depth_names))
    timed("Softmax reliability weights (per frame)", 5, 500, lambda: modality_weights(qv[:1], qd[:1], beta=BETA))

    nyu_model = ReliabilityWeightedKMeans(n_clusters=3, beta=BETA, n_init=10, seed=SEED).fit(
        zv, zd, qv, qd
    )
    timed("Nearest-centroid readout K=3 (per frame)", 5, 200, lambda: nyu_model.predict(zv[:1], zd[:1], qv[:1], qd[:1]))
    go_model = ReliabilityWeightedKMeans(n_clusters=32, beta=BETA, n_init=10, seed=SEED).fit(
        go_vision, go_depth, go_qv, go_qd
    )
    timed("Nearest-centroid readout K=32 (per frame)", 5, 200, lambda: go_model.predict(go_vision[:1], go_depth[:1], go_qv[:1], go_qd[:1]))

    extra = ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=1).fit(features, labels)
    timed("ExtraTrees inference (per frame)", 5, 200, lambda: extra.predict(features[:1]))

    timed(
        "RW-MVK calibration K=3, n_init=10 (51 frames)",
        3,
        1,
        lambda: ReliabilityWeightedKMeans(n_clusters=3, beta=BETA, n_init=10, seed=SEED).fit(
            zv, zd, qv, qd
        ),
    )
    timed(
        "RW-MVK calibration K=32, n_init=10 (351 frames)",
        3,
        1,
        lambda: ReliabilityWeightedKMeans(n_clusters=32, beta=BETA, n_init=10, seed=SEED).fit(
            go_vision, go_depth, go_qv, go_qd
        ),
    )
    timed(
        "ExtraTrees training, 300 trees (351 frames)",
        2,
        1,
        lambda: ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=1).fit(features, labels),
    )
    return rows


def calibration_curve() -> list[dict]:
    calibration = np.load(PROJECT_ROOT / "data" / "ahpff_go1_calibration.npz", allow_pickle=True)
    features = calibration["features"]
    labels = calibration["labels"]
    timestamps = calibration["timestamps"]
    vision_names = list(calibration["feature_names"][:13])
    depth_names = list(calibration["feature_names"][13:])
    vision = features[:, :13]
    depth = features[:, 13:]
    n_classes = int(np.unique(labels).size)
    folds = blocked_stratified_folds(labels, timestamps, n_splits=5)
    rows: list[dict] = []
    rng = np.random.RandomState(SEED)
    for fraction in CALIBRATION_FRACTIONS:
        cluster_scores: list[float] = []
        supervised_scores: list[float] = []
        for fold_index, (train_indices, test_indices) in enumerate(folds):
            for repeat in range(CALIBRATION_REPEATS):
                subset = []
                for label in np.unique(labels[train_indices]):
                    candidates = train_indices[labels[train_indices] == label]
                    take = max(3, int(round(fraction * candidates.size)))
                    take = min(take, candidates.size)
                    subset.extend(rng.choice(candidates, size=take, replace=False).tolist())
                subset = np.asarray(sorted(subset), dtype=np.int64)
                if subset.size <= n_classes:
                    continue
                n_clusters = min(32, max(4, subset.size - 1))
                payload = fit_cluster_readout(
                    vision[subset],
                    depth[subset],
                    labels[subset],
                    n_clusters,
                    n_classes,
                    vision_names,
                    depth_names,
                    beta=BETA,
                    seed=SEED + fold_index * 97 + repeat,
                    n_init=5,
                )
                prediction, _, _, _, _ = predict_cluster_readout(
                    payload,
                    vision[test_indices],
                    depth[test_indices],
                    vision_names,
                    depth_names,
                    hard_depth_dropout=False,
                )
                cluster_scores.append(float(accuracy_score(labels[test_indices], prediction)))
                supervised = ExtraTreesClassifier(
                    n_estimators=150,
                    random_state=SEED + fold_index * 97 + repeat,
                    n_jobs=-1,
                ).fit(vision[subset], labels[subset])
                supervised_scores.append(float(accuracy_score(labels[test_indices], supervised.predict(vision[test_indices]))))
        rows.append(
            {
                "fraction": fraction,
                "cluster_mean": float(np.mean(cluster_scores)),
                "cluster_std": float(np.std(cluster_scores)),
                "cluster_sem": float(np.std(cluster_scores) / np.sqrt(max(1, len(cluster_scores)))),
                "supervised_mean": float(np.mean(supervised_scores)),
                "supervised_std": float(np.std(supervised_scores)),
                "supervised_sem": float(np.std(supervised_scores) / np.sqrt(max(1, len(supervised_scores)))),
                "n_evaluations": len(cluster_scores),
            }
        )
    return rows


def build_figure(
    output: Path,
    runtime_rows: list[dict],
    stat_rows: list[dict],
    curve_rows: list[dict],
    per_seed_rows: list[dict],
    dpi: int,
) -> tuple[Path, Path, Path]:
    apply_style()
    fig = plt.figure(figsize=(183 * MM, 118 * MM))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.13,
        right=0.985,
        top=0.91,
        bottom=0.17,
        wspace=0.42,
        hspace=0.55,
    )

    runtime_axis = fig.add_subplot(grid[0, 0])
    inference = [
        row for row in runtime_rows
        if "training" not in row["operation"].lower() and "calibration" not in row["operation"].lower()
    ]
    training = [
        row for row in runtime_rows
        if "training" in row["operation"].lower() or "calibration" in row["operation"].lower()
    ]
    order = inference + training
    values = np.asarray([row["mean_ms"] for row in order])
    errors = np.asarray([row["std_ms"] for row in order])
    colors = [PALETTE["blue"] if "calibration" in row["operation"].lower() or "training" in row["operation"].lower() else PALETTE["green"] for row in order]
    y = np.arange(len(order))[::-1]
    runtime_axis.barh(y, values, xerr=errors, color=colors, edgecolor="#333333", linewidth=0.4, height=0.62, capsize=1.5)
    runtime_axis.set_xscale("log")
    runtime_axis.set_yticks(y)
    runtime_axis.set_yticklabels(
        [row["operation"].replace(" (per frame)", "\n(per frame)").replace(", n_init", "\nn_init") for row in order],
        fontsize=4.8,
    )
    for position, value in zip(y, values):
        runtime_axis.text(value * 1.18, position, f"{value:.2f} ms", va="center", fontsize=4.5, color="#222222")
    runtime_axis.axvline(100.0, color=PALETTE["vermillion"], linestyle="--", linewidth=0.7)
    runtime_axis.text(100.0 / 1.12, len(order) - 0.35, "100 ms\n(10 Hz budget)", ha="right", va="top", fontsize=4.4, color=PALETTE["vermillion"])
    runtime_axis.set_xlabel("Wall-clock time (ms, log scale)", fontsize=6.0, labelpad=1)
    runtime_axis.set_title("(a) CPU-only latency", fontsize=6.4, pad=3)
    runtime_axis.tick_params(axis="x", labelsize=5.2, length=1.8, width=0.4)
    runtime_axis.grid(axis="x", color="#E6E6E6", linewidth=0.4)
    runtime_axis.set_axisbelow(True)

    curve_axis = fig.add_subplot(grid[0, 1])
    fractions = np.asarray([row["fraction"] for row in curve_rows]) * 100.0
    cluster_mean = np.asarray([row["cluster_mean"] for row in curve_rows]) * 100.0
    cluster_std = np.asarray([row.get("cluster_sem", row["cluster_std"]) for row in curve_rows]) * 100.0
    supervised_mean = np.asarray([row["supervised_mean"] for row in curve_rows]) * 100.0
    supervised_std = np.asarray([row.get("supervised_sem", row["supervised_std"]) for row in curve_rows]) * 100.0
    curve_axis.errorbar(
        fractions,
        cluster_mean,
        yerr=cluster_std,
        marker="o",
        markersize=3.2,
        linewidth=1.2,
        capsize=2.0,
        color=PALETTE["vermillion"],
        label="RA-AHPFF cluster readout",
    )
    curve_axis.errorbar(
        fractions,
        supervised_mean,
        yerr=supervised_std,
        marker="s",
        markersize=3.0,
        linewidth=1.0,
        capsize=2.0,
        color=PALETTE["grey"],
        label="ExtraTrees upper bound",
    )
    curve_axis.set_xlabel("Calibration labels used (% of training folds)", fontsize=6.0, labelpad=1)
    curve_axis.set_ylabel("Blocked-CV accuracy (%)", fontsize=6.0, labelpad=1)
    curve_axis.set_title("(b) Label efficiency on Go1 (351 frames)", fontsize=6.4, pad=3)
    curve_axis.grid(color="#E6E6E6", linewidth=0.45)
    curve_axis.set_axisbelow(True)
    curve_axis.legend(loc="lower right", fontsize=4.9, frameon=False)
    curve_axis.tick_params(labelsize=5.3, length=1.8, width=0.4)

    stats_axis = fig.add_subplot(grid[1, 0])
    baselines = [name for name in METHOD_LABELS if name != "RA-AHPFF (Ours)"]
    clean_p = []
    severe_p = []
    for name in baselines:
        clean = next(row for row in stat_rows if row["level"] == "clean" and row["baseline"] == name)
        severe = next(row for row in stat_rows if row["level"] == "severe" and row["baseline"] == name)
        clean_p.append(-np.log10(max(clean["p_value"], 1e-12)))
        severe_p.append(-np.log10(max(severe["p_value"], 1e-12)))
    x = np.arange(len(baselines))
    width = 0.38
    stats_axis.bar(x - width / 2, clean_p, width, color=PALETTE["sky"], edgecolor="#333333", linewidth=0.35, label="Clean")
    stats_axis.bar(x + width / 2, severe_p, width, color=PALETTE["vermillion"], edgecolor="#333333", linewidth=0.35, label="Severe")
    stats_axis.axhline(-np.log10(0.05), color="#555555", linestyle="--", linewidth=0.6)
    stats_axis.text(len(baselines) - 0.55, -np.log10(0.05) + 0.12, "p = 0.05", ha="right", va="bottom", fontsize=4.4)
    stats_axis.set_xticks(x)
    stats_axis.set_xticklabels([METHOD_SHORT[name] for name in baselines], rotation=25, ha="right", fontsize=5.0)
    stats_axis.set_ylabel("$-\\log_{10}$ Wilcoxon $p$", fontsize=6.0, labelpad=1)
    stats_axis.set_ylim(0.0, 7.0)
    stats_axis.set_title("(c) Paired significance vs. Ours (20 seeds)", fontsize=6.4, pad=3)
    stats_axis.legend(loc="upper right", fontsize=4.6, frameon=False, ncol=2)
    stats_axis.grid(axis="y", color="#E6E6E6", linewidth=0.45)
    stats_axis.set_axisbelow(True)
    stats_axis.tick_params(labelsize=5.2, length=1.8, width=0.4)

    gain_axis = fig.add_subplot(grid[1, 1])
    rng = np.random.RandomState(SEED)
    severe_rows = [row for row in per_seed_rows if row["level"] == "severe"]
    clean_rows = [row for row in per_seed_rows if row["level"] == "clean"]
    for offset, (label, rows, color) in enumerate(
        (("Clean", clean_rows, PALETTE["sky"]), ("Severe", severe_rows, PALETTE["vermillion"]))
    ):
        for index, name in enumerate(baselines):
            ours = {row["seed"]: row["accuracy"] for row in rows if row["method"] == "RA-AHPFF (Ours)"}
            baseline = {row["seed"]: row["accuracy"] for row in rows if row["method"] == name}
            gains = np.asarray([(ours[seed] - baseline[seed]) * 100.0 for seed in sorted(ours)])
            position = index + (offset - 0.5) * 0.34
            jitter = rng.uniform(-0.07, 0.07, size=gains.size)
            gain_axis.scatter(
                position + jitter,
                gains,
                s=5,
                color=color,
                alpha=0.55,
                edgecolor="none",
                zorder=2,
            )
            gain_axis.plot([position - 0.14, position + 0.14], [np.median(gains), np.median(gains)], color="#222222", linewidth=0.9, zorder=3)
    gain_axis.axhline(0.0, color="#444444", linewidth=0.5)
    gain_axis.set_xticks(np.arange(len(baselines)))
    gain_axis.set_xticklabels([METHOD_SHORT[name] for name in baselines], rotation=25, ha="right", fontsize=5.0)
    gain_axis.set_ylabel("Ours $-$ baseline accuracy (pp)", fontsize=6.0, labelpad=1)
    gain_axis.set_title("(d) Per-seed paired gain", fontsize=6.4, pad=3)
    gain_axis.grid(axis="y", color="#E6E6E6", linewidth=0.45)
    gain_axis.set_axisbelow(True)
    gain_axis.tick_params(labelsize=5.2, length=1.8, width=0.4)
    gain_axis.scatter([], [], s=10, color=PALETTE["sky"], label="Clean")
    gain_axis.scatter([], [], s=10, color=PALETTE["vermillion"], label="Severe")
    gain_axis.legend(loc="lower right", fontsize=4.6, frameon=False, ncol=2)

    for axis, label in ((runtime_axis, "a"), (curve_axis, "b"), (stats_axis, "c"), (gain_axis, "d")):
        axis.text(-0.16, 1.08, label, transform=axis.transAxes, fontsize=8, fontweight="bold", va="top")

    fig.text(
        0.5,
        0.02,
        "All timings are single-thread CPU measurements (Python 3.8, 2.4 GHz class x86-64). "
        "The 10 Hz perception budget is 100 ms per RGB-D frame; calibration is an offline in-scene step. "
        "Panel (b) error bars are s.e.m. over 25 blocked evaluations.",
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="#444444",
    )
    pdf_path, png_path = save_figure(fig, "Figure9_efficiency_and_statistics", output, dpi)
    stats_json = PROJECT_ROOT / "figures" / "meeting118_statistics.json"
    stats_json.write_text(
        json.dumps(
            {
                "runtime": runtime_rows,
                "paired_statistics": stat_rows,
                "calibration_curve": curve_rows,
                "per_seed_seeds": sorted({row["seed"] for row in per_seed_rows}),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return pdf_path, png_path, stats_json


def write_per_seed_tables(per_seed_rows: list[dict]) -> None:
    write_csv(
        PROJECT_ROOT / "figures" / "meeting118_per_seed_metrics.csv",
        ["level", "seed", "method", "accuracy", "nmi", "ari", "recall_flat", "recall_stairs", "recall_obstacle"],
        [
            [
                row["level"],
                row["seed"],
                row["method"],
                f"{row['accuracy']:.8f}",
                f"{row['nmi']:.8f}",
                f"{row['ari']:.8f}",
                f"{row['recall_flat']:.8f}",
                f"{row['recall_stairs']:.8f}",
                f"{row['recall_obstacle']:.8f}",
            ]
            for row in per_seed_rows
        ],
    )
    rows = []
    for level_name in ("clean", "severe"):
        for name in METHOD_LABELS:
            selected = [row for row in per_seed_rows if row["level"] == level_name and row["method"] == name]
            rows.append(
                [
                    level_name,
                    name,
                    f"{np.mean([row['recall_flat'] for row in selected]):.6f}",
                    f"{np.std([row['recall_flat'] for row in selected]):.6f}",
                    f"{np.mean([row['recall_stairs'] for row in selected]):.6f}",
                    f"{np.std([row['recall_stairs'] for row in selected]):.6f}",
                    f"{np.mean([row['recall_obstacle'] for row in selected]):.6f}",
                    f"{np.std([row['recall_obstacle'] for row in selected]):.6f}",
                ]
            )
    write_csv(
        PROJECT_ROOT / "figures" / "meeting118_per_class_recall.csv",
        ["level", "method", "flat_mean", "flat_std", "stairs_mean", "stairs_std", "obstacle_mean", "obstacle_std"],
        rows,
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    per_seed, per_seed_rows = per_seed_nyu(args.seeds, args.seed)
    write_per_seed_tables(per_seed_rows)
    if args.per_seed_only:
        print("Wrote per-seed metrics and per-class recall summaries.")
        return
    stat_rows = paired_statistics(per_seed)
    runtime_rows = [] if args.skip_runtime else benchmark_runtime()
    curve_rows = [] if args.skip_calibration else calibration_curve()
    if not curve_rows:
        curve_rows = [
            {
                "fraction": fraction,
                "cluster_mean": float("nan"),
                "cluster_std": float("nan"),
                "supervised_mean": float("nan"),
                "supervised_std": float("nan"),
                "n_evaluations": 0,
            }
            for fraction in CALIBRATION_FRACTIONS
        ]
    pdf_path, png_path, stats_json = build_figure(
        args.output, runtime_rows, stat_rows, curve_rows, per_seed_rows, args.dpi
    )
    write_csv(
        args.output / "source_data" / "figure9_per_seed_accuracy.csv",
        ["level", "seed", "method", "accuracy"],
        [
            [row["level"], row["seed"], row["method"], f"{row['accuracy']:.8f}"]
            for row in per_seed_rows
        ],
    )
    write_csv(
        args.output / "source_data" / "figure9_runtime.csv",
        ["operation", "mean_ms", "std_ms", "repeats", "calls_per_repeat"],
        [
            [
                row["operation"],
                f"{row['mean_ms']:.6f}",
                f"{row['std_ms']:.6f}",
                row["repeats"],
                row["calls_per_repeat"],
            ]
            for row in runtime_rows
        ],
    )
    write_csv(
        args.output / "source_data" / "figure9_paired_statistics.csv",
        [
            "level",
            "baseline",
            "ours_mean",
            "baseline_mean",
            "mean_gain",
            "median_gain",
            "wins",
            "losses",
            "ties",
            "wilcoxon_statistic",
            "p_value",
            "significant_0p05",
        ],
        [
            [
                row["level"],
                row["baseline"],
                f"{row['ours_mean']:.8f}",
                f"{row['baseline_mean']:.8f}",
                f"{row['mean_gain']:.8f}",
                f"{row['median_gain']:.8f}",
                row["wins"],
                row["losses"],
                row["ties"],
                f"{row['wilcoxon_statistic']:.6f}",
                f"{row['p_value']:.8f}",
                row["significant_0p05"],
            ]
            for row in stat_rows
        ],
    )
    write_csv(
        args.output / "source_data" / "figure9_calibration_curve.csv",
        ["fraction", "cluster_mean", "cluster_std", "cluster_sem", "supervised_mean", "supervised_std", "supervised_sem", "n_evaluations"],
        [
            [
                f"{row['fraction']:.4f}",
                f"{row['cluster_mean']:.8f}",
                f"{row['cluster_std']:.8f}",
                f"{row.get('cluster_sem', float('nan')):.8f}",
                f"{row['supervised_mean']:.8f}",
                f"{row['supervised_std']:.8f}",
                f"{row.get('supervised_sem', float('nan')):.8f}",
                row["n_evaluations"],
            ]
            for row in curve_rows
        ],
    )
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")
    print(f"Saved statistics: {stats_json}")
    for row in stat_rows:
        if row["level"] == "severe":
            print(
                f"  severe vs {row['baseline']:22s} gain {row['mean_gain'] * 100:5.2f} pp, "
                f"p={row['p_value']:.2e}"
            )


if __name__ == "__main__":
    main()
