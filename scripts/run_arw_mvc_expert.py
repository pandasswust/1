#!/usr/bin/env python3
"""Reliability-aware AHPFF clustering benchmark on the NYU Depth V2 pilot set.

The revised pipeline uses per-sample visual/depth health indicators to compute
softmax modality weights and performs reliability-weighted multi-view K-Means
(RW-MVK) on a weighted concatenation of the standardized visual and depth
descriptor blocks.  Cluster indices are converted to terrain classes by a
majority-vote calibration step.  Silhouette scores are used only for evaluation
and for optional model selection, never to define the fusion weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ahpff import (
    ReliabilityWeightedKMeans,
    compute_quality,
    map_clusters_to_classes,
    modality_weights,
    select_k_silhouette,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"
SEED = 42
METHOD_LABELS = (
    "Vision only",
    "Depth only",
    "Fixed concat",
    "PCA average",
    "Uniform weighted",
    "Shuffled reliability",
    "RA-AHPFF (Ours)",
)
NOISE_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(NOISE_LEVELS))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--scheme",
        choices=("disjoint", "overlap"),
        default="disjoint",
        help="disjoint: each degraded sample loses exactly one modality; overlap: independent 50%% modality corruption",
    )
    parser.add_argument(
        "--strict-depth-boundary",
        action="store_true",
        help=(
            "enforce q_d = 0 when the valid-depth ratio is zero (Eq. 4). "
            "The default reproduces the 112 manuscript Tables 4/5, whose "
            "benchmark predates this online boundary."
        ),
    )
    return parser.parse_args()


def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    metadata_path = data_dir / "nyu_feature_meta.json"
    if not metadata_path.exists():
        raise SystemExit(f"Missing {metadata_path}. Run scripts/extract_nyu_expert.py first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vision_raw_path = data_dir / "Zv_real_raw.npy"
    depth_raw_path = data_dir / "Zl_real_raw.npy"
    if vision_raw_path.exists() and depth_raw_path.exists():
        vision = np.load(vision_raw_path).astype(np.float64)
        depth = np.load(depth_raw_path).astype(np.float64)
    else:
        vision = np.load(data_dir / "Zv_real.npy").astype(np.float64)
        depth = np.load(data_dir / "Zl_real.npy").astype(np.float64)
        print("WARNING: raw descriptors are missing; health indicators will be approximate.")
    target = np.load(data_dir / "Y_real.npy").astype(np.int64)
    return vision, depth, target, metadata


def apply_visual_degradation(
    vision: np.ndarray,
    level: float,
    rng: np.random.RandomState,
    names: list[str] | tuple[str, ...],
) -> None:
    index = {name: position for position, name in enumerate(names)}
    overexposure = 0.25 * level
    mean = vision[:, index["mean_intensity"]]
    vision[:, index["mean_intensity"]] = np.clip(mean + overexposure * (1.0 - mean), 0.0, 1.0)
    vision[:, index["std_intensity"]] *= max(0.0, 1.0 - 0.65 * level)
    vision[:, index["laplacian_variance"]] *= max(0.0, 1.0 - 0.80 * level)
    vision[:, index["canny_edge_density"]] *= max(0.0, 1.0 - 0.55 * level)
    vision[:, index["mean_gradient"]] *= max(0.0, 1.0 - 0.65 * level)
    vision[:, index["std_gradient"]] *= max(0.0, 1.0 - 0.65 * level)
    vision[:, index["intensity_entropy"]] *= max(0.0, 1.0 - 0.45 * level)
    vision += rng.normal(0.0, 0.04 * level, size=vision.shape)
    vision[:, index["mean_intensity"]] = np.clip(vision[:, index["mean_intensity"]], 0.0, 1.0)
    vision[:, index["std_intensity"]] = np.clip(vision[:, index["std_intensity"]], 0.0, 1.0)
    for key in ("laplacian_variance", "canny_edge_density", "mean_gradient", "std_gradient", "intensity_entropy"):
        vision[:, index[key]] = np.clip(vision[:, index[key]], 0.0, None)


def apply_depth_degradation(
    depth: np.ndarray,
    level: float,
    rng: np.random.RandomState,
    names: list[str] | tuple[str, ...],
) -> None:
    index = {name: position for position, name in enumerate(names)}
    dropout = min(0.90, 0.55 * level)
    depth[:, index["valid_ratio"]] *= 1.0 - dropout
    depth[:, index["depth_std"]] *= 1.0 + 0.20 * level
    depth[:, index["depth_span"]] *= max(0.0, 1.0 - 0.20 * level)
    depth[:, index["depth_mean_gradient"]] *= 1.0 + 0.40 * level
    depth += rng.normal(0.0, 0.04 * level, size=depth.shape)
    depth[:, index["valid_ratio"]] = np.clip(depth[:, index["valid_ratio"]], 0.0, 1.0)
    for key in ("depth_std", "depth_span", "depth_mean_gradient"):
        depth[:, index[key]] = np.clip(depth[:, index[key]], 0.0, None)


def inject_degradation(
    vision: np.ndarray,
    depth: np.ndarray,
    target: np.ndarray,
    vision_names: list[str] | tuple[str, ...],
    depth_names: list[str] | tuple[str, ...],
    level: float,
    seed: int,
    args_scheme: str = "disjoint",
    beta: float = 4.0,
    depth_hard_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    visual = vision.copy()
    sampled_depth = depth.copy()
    n = target.size
    rng = np.random.RandomState(seed)
    if level <= 0.0:
        visual_mask = np.zeros(n, dtype=bool)
        depth_mask = np.zeros(n, dtype=bool)
    elif args_scheme == "disjoint":
        visual_mask = np.zeros(n, dtype=bool)
        depth_mask = np.zeros(n, dtype=bool)
        for label in np.unique(target):
            indices = np.where(target == label)[0]
            rng.shuffle(indices)
            split = indices.size // 2
            visual_mask[indices[:split]] = True
            depth_mask[indices[split:]] = True
    else:
        visual_mask = rng.rand(n) < 0.5
        depth_mask = rng.rand(n) < 0.5
    if np.any(visual_mask):
        visual_subset = visual[visual_mask].copy()
        apply_visual_degradation(visual_subset, level, rng, vision_names)
        visual[visual_mask] = visual_subset
    if np.any(depth_mask):
        depth_subset = sampled_depth[depth_mask].copy()
        apply_depth_degradation(depth_subset, level, rng, depth_names)
        sampled_depth[depth_mask] = depth_subset
    qv, qd, diagnostics = compute_quality(
        visual, sampled_depth, vision_names, depth_names, depth_hard_zero=depth_hard_zero
    )
    wv, wd = modality_weights(qv, qd, beta=beta)
    info = {
        "visual_degraded_indices": visual_mask.tolist(),
        "depth_degraded_indices": depth_mask.tolist(),
        "quality_diagnostics": diagnostics,
        "mean_qv": float(np.mean(qv)),
        "mean_qd": float(np.mean(qd)),
        "mean_wv": float(np.mean(wv)),
        "mean_wd": float(np.mean(wd)),
        "mean_wv_visual_degraded": float(np.mean(wv[visual_mask])) if np.any(visual_mask) else float("nan"),
        "mean_wv_depth_degraded": float(np.mean(wv[depth_mask])) if np.any(depth_mask) else float("nan"),
        "mean_wd_visual_degraded": float(np.mean(wd[visual_mask])) if np.any(visual_mask) else float("nan"),
        "mean_wd_depth_degraded": float(np.mean(wd[depth_mask])) if np.any(depth_mask) else float("nan"),
        "mean_wv_both_degraded": float(np.mean(wv[visual_mask & depth_mask])) if np.any(visual_mask & depth_mask) else float("nan"),
        "mean_wd_both_degraded": float(np.mean(wd[visual_mask & depth_mask])) if np.any(visual_mask & depth_mask) else float("nan"),
        "mean_wv_clean": float(np.mean(wv[~visual_mask & ~depth_mask])) if np.any(~visual_mask & ~depth_mask) else float("nan"),
        "mean_wd_clean": float(np.mean(wd[~visual_mask & ~depth_mask])) if np.any(~visual_mask & ~depth_mask) else float("nan"),
    }
    return visual, sampled_depth, qv, qd, info


def evaluate_method(
    name: str,
    zv: np.ndarray,
    zd: np.ndarray,
    qv: np.ndarray,
    qd: np.ndarray,
    target: np.ndarray,
    n_classes: int,
    seed: int,
    beta: float,
    hard_depth_dropout: bool = False,
) -> tuple[dict, np.ndarray, np.ndarray]:
    if name == "Vision only":
        clusters = KMeans(n_clusters=n_classes, random_state=seed, n_init=20).fit_predict(zv)
    elif name == "Depth only":
        clusters = KMeans(n_clusters=n_classes, random_state=seed, n_init=20).fit_predict(zd)
    elif name == "Fixed concat":
        clusters = KMeans(n_clusters=n_classes, random_state=seed, n_init=20).fit_predict(
            np.hstack([zv, zd])
        )
    elif name == "PCA average":
        latent_dim = min(8, zv.shape[1], zd.shape[1], target.size - 1)
        vp = StandardScaler().fit_transform(
            PCA(n_components=latent_dim, random_state=seed).fit_transform(zv)
        )
        dp = StandardScaler().fit_transform(
            PCA(n_components=latent_dim, random_state=seed).fit_transform(zd)
        )
        clusters = KMeans(n_clusters=n_classes, random_state=seed, n_init=20).fit_predict(
            0.5 * vp + 0.5 * dp
        )
    elif name in ("Uniform weighted", "Shuffled reliability", "RA-AHPFF (Ours)"):
        if name == "Uniform weighted":
            use_v = np.full(qv.size, 0.5)
            use_d = np.full(qd.size, 0.5)
        elif name == "Shuffled reliability":
            rng = np.random.RandomState(seed + 991)
            use_v = qv[rng.permutation(qv.size)]
            use_d = qd[rng.permutation(qd.size)]
        else:
            use_v = qv
            use_d = qd
        model = ReliabilityWeightedKMeans(
            n_clusters=n_classes,
            beta=beta,
            n_init=10,
            seed=seed,
        ).fit(zv, zd, use_v, use_d, hard_depth_dropout=hard_depth_dropout)
        clusters = model.labels_
    else:
        raise ValueError(name)

    mapping, _ = map_clusters_to_classes(clusters, target, n_classes, n_classes)
    prediction = mapping[clusters]
    accuracy = float(np.mean(prediction == target))
    nmi = float(normalized_mutual_info_score(target, clusters))
    ari = float(adjusted_rand_score(target, clusters))
    fused = np.hstack([zv, zd]) if name not in ("Vision only", "Depth only") else (zv if name == "Vision only" else zd)
    try:
        silhouette = float(silhouette_score(fused, clusters))
    except Exception:
        silhouette = float("nan")
    recall = {}
    for label in range(n_classes):
        mask = target == label
        recall[str(label)] = float(np.mean(prediction[mask] == label)) if np.any(mask) else float("nan")
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, predicted in zip(target, prediction):
        confusion[int(true_label), int(predicted)] += 1
    metrics = {
        "acc": accuracy,
        "nmi": nmi,
        "ari": ari,
        "silhouette": silhouette,
        "recall": recall,
        "confusion": confusion.tolist(),
        "cluster_counts": np.bincount(clusters, minlength=n_classes).tolist(),
    }
    return metrics, prediction, clusters


def bootstrap_accuracy(prediction: np.ndarray, target: np.ndarray, n_resamples: int, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    values = []
    n = target.size
    for _ in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        values.append(float(np.mean(prediction[indices] == target[indices])))
    values = np.asarray(values)
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
    }


def plot_results(aggregate: dict, weights: dict, output_path: Path, n_per_class: int) -> None:
    methods = list(METHOD_LABELS)
    short_labels = [
        "Vision",
        "Depth",
        "Fixed concat",
        "PCA avg",
        "Uniform",
        "Shuffled",
        "RA-AHPFF",
    ]
    levels = sorted(aggregate)
    clean_level = min(levels)
    severe_level = max(levels)
    x = np.arange(len(methods))
    width = 0.38
    clean_color = "#5DADE2"
    severe_color = "#F1948A"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    def grouped_metric(axis, key, scale, ylabel, title):
        clean_values = [aggregate[clean_level][name][key]["mean"] * scale for name in methods]
        severe_values = [aggregate[severe_level][name][key]["mean"] * scale for name in methods]
        clean_errors = [aggregate[clean_level][name][key]["std"] * scale for name in methods]
        severe_errors = [aggregate[severe_level][name][key]["std"] * scale for name in methods]
        axis.bar(x - width / 2, clean_values, width, yerr=clean_errors, capsize=3,
                 label="Clean", color=clean_color, edgecolor="black", linewidth=0.5)
        axis.bar(x + width / 2, severe_values, width, yerr=severe_errors, capsize=3,
                 label=f"Severe (level {severe_level:.1f})", color=severe_color,
                 edgecolor="black", linewidth=0.5)
        axis.set_xticks(x)
        axis.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=9)
        axis.set_ylabel(ylabel, fontsize=10)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.grid(axis="y", linestyle="--", alpha=0.4)
        axis.legend(loc="upper right", fontsize=9)

    grouped_metric(axes[0, 0], "acc", 100.0, "Clustering accuracy (%)", "(a) Accuracy")
    grouped_metric(axes[0, 1], "nmi", 100.0, "NMI (%)", "(b) Normalized mutual information")
    grouped_metric(axes[1, 0], "ari", 100.0, "ARI (%)", "(c) Adjusted Rand index")

    axis = axes[1, 1]
    for name in methods:
        values = [aggregate[level][name]["acc"]["mean"] * 100.0 for level in levels]
        errors = [aggregate[level][name]["acc"]["std"] * 100.0 for level in levels]
        is_ours = name == "RA-AHPFF (Ours)"
        axis.errorbar(
            levels,
            values,
            yerr=errors,
            marker="o",
            markersize=5 if is_ours else 3,
            linewidth=3.0 if is_ours else 1.2,
            capsize=3,
            color="black" if is_ours else None,
            label=short_labels[methods.index(name)],
            zorder=5 if is_ours else 2,
        )
    axis.set_xlabel("Degradation level", fontsize=10)
    axis.set_ylabel("Clustering accuracy (%)", fontsize=10)
    axis.set_title("(d) Accuracy versus degradation severity", fontsize=11, fontweight="bold")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(loc="lower left", fontsize=7, ncol=2)

    fig.suptitle(
        "RA-AHPFF clustering on the corrected NYU Depth V2 pilot subset\n"
        f"17 images per class, 20 seeds, K=3, per-sample reliability weights",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def sum_mean_std(values: list[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def main() -> None:
    args = parse_args()
    vision_raw, depth_raw, target, metadata = load_data(args.data_dir)
    vision_names = list(metadata.get("feature_names_vision", []))
    depth_names = list(metadata.get("feature_names_lidar", []))
    if len(vision_names) != vision_raw.shape[1] or len(depth_names) != depth_raw.shape[1]:
        raise SystemExit("Feature metadata does not match the descriptor arrays.")
    n_classes = int(np.unique(target).size)
    n_per_class = int(metadata.get("balanced_per_class", np.bincount(target).min()))

    vision_scaler = StandardScaler().fit(vision_raw)
    depth_scaler = StandardScaler().fit(depth_raw)
    clean_vision = vision_scaler.transform(vision_raw)
    clean_depth = depth_scaler.transform(depth_raw)
    clean_qv, clean_qd, _ = compute_quality(vision_raw, depth_raw, vision_names, depth_names)
    selected_k, k_scores = select_k_silhouette(
        clean_vision, clean_depth, clean_qv, clean_qd, beta=args.beta, seed=args.seed
    )

    aggregate: dict[float, dict[str, dict]] = {}
    weight_summary: dict[float, dict] = {}
    representative_predictions: dict[str, np.ndarray] = {}
    for level in args.noise_levels:
        level = float(level)
        per_method: dict[str, dict[str, list[float]]] = {
            name: {key: [] for key in ("acc", "nmi", "ari", "silhouette")} for name in METHOD_LABELS
        }
        weight_rows = []
        for seed_index in range(args.seeds):
            seed = args.seed + seed_index
            degraded_vision_raw, degraded_depth_raw, qv, qd, info = inject_degradation(
                vision_raw,
                depth_raw,
                target,
                vision_names,
                depth_names,
                level,
                seed,
                args.scheme,
                args.beta,
                depth_hard_zero=args.strict_depth_boundary,
            )
            degraded_vision = vision_scaler.transform(degraded_vision_raw)
            degraded_depth = depth_scaler.transform(degraded_depth_raw)
            weight_rows.append(info)
            for name in METHOD_LABELS:
                metrics, prediction, clusters = evaluate_method(
                    name,
                    degraded_vision,
                    degraded_depth,
                    qv,
                    qd,
                    target,
                    n_classes,
                    seed,
                    args.beta,
                    hard_depth_dropout=args.strict_depth_boundary,
                )
                for key in ("acc", "nmi", "ari", "silhouette"):
                    per_method[name][key].append(metrics[key])
                if seed_index == 0 and level == 1.5:
                    representative_predictions[name] = prediction
        aggregate[level] = {
            name: {key: sum_mean_std(values) for key, values in per_method[name].items()}
            for name in METHOD_LABELS
        }
        weight_summary[level] = {
            key: float(np.nanmean([row[key] for row in weight_rows]))
            for key in (
                "mean_qv",
                "mean_qd",
                "mean_wv",
                "mean_wd",
                "mean_wv_visual_degraded",
                "mean_wv_depth_degraded",
                "mean_wd_visual_degraded",
                "mean_wd_depth_degraded",
                "mean_wv_both_degraded",
                "mean_wd_both_degraded",
                "mean_wv_clean",
                "mean_wd_clean",
            )
        }

    bootstrap_level = 1.5 if 1.5 in aggregate else max(aggregate)
    bootstrap = {}
    for name, prediction in representative_predictions.items():
        bootstrap[name] = bootstrap_accuracy(prediction, target, args.bootstrap, args.seed)

    output = {
        "dataset": {
            "name": "NYU Depth V2 corrected pilot subset",
            "class_names": metadata.get("class_names"),
            "class_ids": metadata.get("nyu_label_ids"),
            "samples_per_class": metadata.get("samples_per_class"),
            "balanced_per_class": n_per_class,
            "samples_total": int(target.size),
            "note": metadata.get("note"),
        },
        "method": {
            "name": "RA-AHPFF",
            "fusion": "per-sample softmax reliability weights + weighted concatenation",
            "clustering": "reliability-weighted multi-view K-Means",
            "cluster_mapping": "majority vote on labelled calibration samples",
            "beta": args.beta,
            "silhouette_role": "evaluation and optional K selection only",
            "selected_k_by_silhouette": selected_k,
            "silhouette_by_k": {str(key): value for key, value in k_scores.items()},
            "k_used_for_benchmark": n_classes,
        },
        "degradation_model": {
            "visual": "overexposure + reduced contrast/sharpness + feature noise",
            "depth": "random dropout (valid_ratio) + range/statistic corruption + feature noise",
            "scheme": args.scheme,
            "fraction_visual_degraded": 0.5,
            "fraction_depth_degraded": 0.5,
            "levels": list(args.noise_levels),
            "depth_zero_boundary": "strict" if args.strict_depth_boundary else "manuscript-benchmark",
        },
        "seeds": {"base": args.seed, "count": args.seeds},
        "aggregate": {str(level): values for level, values in aggregate.items()},
        "weights": {str(level): values for level, values in weight_summary.items()},
        "bootstrap": {"level": bootstrap_level, "resamples": args.bootstrap, "accuracy": bootstrap},
    }

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.figure_dir / "ahpff_ablation.png"
    metrics_path = args.figure_dir / "prl_robustness_eval_expert.json"
    plot_results(aggregate, weight_summary, figure_path, n_per_class)
    metrics_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Dataset: {metadata.get('class_names')}, n={target.size}, n/class={n_per_class}")
    print(f"Silhouette-selected K={selected_k}; scores={ {k: round(v, 3) for k, v in k_scores.items()} }")
    print(
        f"Benchmark K={n_classes}, beta={args.beta}, seeds={args.seeds}, "
        f"depth boundary={'strict' if args.strict_depth_boundary else 'manuscript-benchmark'}"
    )
    for level in sorted(aggregate):
        print(f"[degradation level {level:.1f}]")
        for name in METHOD_LABELS:
            values = aggregate[level][name]
            print(
                f"  {name:22s} ACC={values['acc']['mean'] * 100:5.1f}±{values['acc']['std'] * 100:4.1f} "
                f"NMI={values['nmi']['mean'] * 100:5.1f} ARI={values['ari']['mean']:.3f}"
            )
        weights = weight_summary[level]
        print(
            f"  weights: clean w_v={weights['mean_wv_clean']:.3f}, "
            f"visual-degraded w_v={weights['mean_wv_visual_degraded']:.3f}, "
            f"depth-degraded w_d={weights['mean_wd_depth_degraded']:.3f}"
        )
    print(f"Bootstrap (level {bootstrap_level}):")
    for name, values in bootstrap.items():
        print(
            f"  {name:22s} ACC={values['mean'] * 100:.1f}% "
            f"CI95=[{values['ci95_low'] * 100:.1f}, {values['ci95_high'] * 100:.1f}]"
        )
    print(f"Saved figure: {figure_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
