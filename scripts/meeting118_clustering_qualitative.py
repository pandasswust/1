#!/usr/bin/env python3
"""Meeting-118 qualitative clustering comparison figure.

The review asked for a stand-alone visual comparison of the clustering
experiments, not only aggregate bars.  This script reproduces the seven
benchmark methods on the verified 51-frame NYU Depth V2 subset, computes a
common two-dimensional t-SNE map for the clean and severely degraded feature
spaces, and draws every method's predictions on that common map.  Marker shape
encodes the true terrain class and colour encodes the predicted class, so a
misclassification appears as a shape/colour mismatch.

Outputs
-------
112/pdf/Figure8_clustering_qualitative.pdf
112/png_1000dpi/Figure8_clustering_qualitative_1000dpi.png
112/source_data/figure8_clustering_embeddings.csv
112/source_data/figure8_method_metrics.csv
112/source_data/figure8_per_class_recall.csv
112/source_data/figure8_reliability_weights.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from publication_figures import MM, METHOD_COLORS, PALETTE, apply_style, save_figure, write_csv  # noqa: E402
from run_arw_mvc_expert import (  # noqa: E402
    METHOD_LABELS,
    SEED,
    evaluate_method,
    inject_degradation,
    load_data,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "112"
SHORT = {
    "Vision only": "Vision",
    "Depth only": "Depth",
    "Fixed concat": "Fixed concat",
    "PCA average": "PCA avg",
    "Uniform weighted": "Uniform",
    "Shuffled reliability": "Shuffled",
    "RA-AHPFF (Ours)": "Ours",
}
CLASS_COLORS = ("#009E73", "#D55E00", "#0072B2")
CLASS_MARKERS = ("o", "^", "s")
CLASS_NAMES = ("Flat", "Stairs", "Obstacle")
SEVERE_LEVEL = 2.0
BETA = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--level", type=float, default=SEVERE_LEVEL)
    return parser.parse_args()


def standardise(
    vision_raw: np.ndarray,
    depth_raw: np.ndarray,
    target: np.ndarray,
    metadata: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vision_names = list(metadata["feature_names_vision"])
    depth_names = list(metadata["feature_names_lidar"])
    vision_scaler = StandardScaler().fit(vision_raw)
    depth_scaler = StandardScaler().fit(depth_raw)
    degraded_vision_raw, degraded_depth_raw, qv, qd, info = inject_degradation(
        vision_raw,
        depth_raw,
        target,
        vision_names,
        depth_names,
        SEVERE_LEVEL,
        SEED,
        "disjoint",
        BETA,
        depth_hard_zero=False,
    )
    vision = vision_scaler.transform(degraded_vision_raw)
    depth = depth_scaler.transform(degraded_depth_raw)
    return vision, depth, qv, qd, info


def embed(vision: np.ndarray, depth: np.ndarray, seed: int) -> np.ndarray:
    fused = np.hstack([vision, depth])
    return TSNE(
        n_components=2,
        perplexity=12,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(fused)


def draw_embedding(
    axis: plt.Axes,
    coordinates: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    title: str,
    metrics: dict | None = None,
) -> None:
    for true_class, marker in enumerate(CLASS_MARKERS):
        for predicted_class, color in enumerate(CLASS_COLORS):
            mask = (target == true_class) & (prediction == predicted_class)
            if not np.any(mask):
                continue
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=16,
                marker=marker,
                facecolor=color,
                edgecolor="#222222",
                linewidth=0.35,
                alpha=0.92,
                zorder=3,
            )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=6.2, pad=2)
    for spine in axis.spines.values():
        spine.set_linewidth(0.45)
        spine.set_edgecolor("#555555")
    if metrics is not None:
        axis.text(
            0.02,
            0.02,
            f"ACC {metrics['acc'] * 100:.1f}%\nNMI {metrics['nmi'] * 100:.1f}%",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=4.9,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.78),
        )


def main() -> None:
    args = parse_args()
    vision_raw, depth_raw, target, metadata = load_data(PROJECT_ROOT / "data")
    vision_names = list(metadata["feature_names_vision"])
    depth_names = list(metadata["feature_names_lidar"])
    n_classes = int(np.unique(target).size)

    clean_scaler_v = StandardScaler().fit(vision_raw)
    clean_scaler_d = StandardScaler().fit(depth_raw)
    clean_vision = clean_scaler_v.transform(vision_raw)
    clean_depth = clean_scaler_d.transform(depth_raw)
    clean_qv, clean_qd, _ = _quality(vision_raw, depth_raw, vision_names, depth_names)
    clean_embedding = embed(clean_vision, clean_depth, args.seed)

    degraded = standardise(vision_raw, depth_raw, target, metadata)
    severe_vision, severe_depth, severe_qv, severe_qd, info = degraded
    severe_embedding = embed(severe_vision, severe_depth, args.seed)

    clean_metrics = {}
    clean_predictions = {}
    for name in METHOD_LABELS:
        metrics, prediction, clusters = evaluate_method(
            name,
            clean_vision,
            clean_depth,
            clean_qv,
            clean_qd,
            target,
            n_classes,
            args.seed,
            BETA,
            hard_depth_dropout=False,
        )
        clean_metrics[name] = metrics
        clean_predictions[name] = prediction

    severe_metrics = {}
    severe_predictions = {}
    for name in METHOD_LABELS:
        metrics, prediction, clusters = evaluate_method(
            name,
            severe_vision,
            severe_depth,
            severe_qv,
            severe_qd,
            target,
            n_classes,
            args.seed,
            BETA,
            hard_depth_dropout=False,
        )
        severe_metrics[name] = metrics
        severe_predictions[name] = prediction

    apply_style()
    fig = plt.figure(figsize=(183 * MM, 152 * MM))
    left, right = 7.0 * MM, 3.0 * MM
    top, bottom = 8.0 * MM, 20.0 * MM
    n_cols, n_rows = 4, 3
    gap_x, gap_y = 2.5 * MM, 4.0 * MM
    panel_width = (183 * MM - left - right - (n_cols - 1) * gap_x) / n_cols
    panel_height = (152 * MM - top - bottom - (n_rows - 1) * gap_y) / n_rows

    def add_panel(row: int, column: int, draw) -> plt.Axes:
        x = left + column * (panel_width + gap_x)
        y = (152 * MM - top) - (row + 1) * panel_height - row * gap_y
        axis = fig.add_axes([x / (183 * MM), y / (152 * MM), panel_width / (183 * MM), panel_height / (152 * MM)])
        draw(axis)
        return axis

    letters = iter("abcdefghijkl")
    axis = add_panel(
        0,
        0,
        lambda ax: draw_embedding(ax, clean_embedding, target, target, "Ground truth (clean)"),
    )
    axis.text(-0.11, 1.05, next(letters), transform=axis.transAxes, fontsize=8, fontweight="bold", va="top")
    axis = add_panel(
        0,
        1,
        lambda ax: draw_embedding(
            ax,
            clean_embedding,
            clean_predictions["RA-AHPFF (Ours)"],
            target,
            "Ours (clean)",
            clean_metrics["RA-AHPFF (Ours)"],
        ),
    )
    axis.text(-0.11, 1.05, next(letters), transform=axis.transAxes, fontsize=8, fontweight="bold", va="top")
    axis = add_panel(
        0,
        2,
        lambda ax: draw_embedding(ax, severe_embedding, target, target, "Ground truth (severe)"),
    )
    axis.text(-0.11, 1.05, next(letters), transform=axis.transAxes, fontsize=8, fontweight="bold", va="top")
    axis = add_panel(
        0,
        3,
        lambda ax: draw_embedding(
            ax,
            severe_embedding,
            severe_predictions["RA-AHPFF (Ours)"],
            target,
            "Ours (severe)",
            severe_metrics["RA-AHPFF (Ours)"],
        ),
    )
    axis.text(-0.11, 1.05, next(letters), transform=axis.transAxes, fontsize=8, fontweight="bold", va="top")

    row_two = ("Vision only", "Depth only", "Fixed concat", "PCA average")
    for column, name in enumerate(row_two):
        axis = add_panel(
            1,
            column,
            lambda ax, name=name: draw_embedding(
                ax,
                severe_embedding,
                severe_predictions[name],
                target,
                f"{SHORT[name]} (severe)",
                severe_metrics[name],
            ),
        )
        axis.text(-0.11, 1.05, next(letters), transform=axis.transAxes, fontsize=8, fontweight="bold", va="top")

    row_three = ("Uniform weighted", "Shuffled reliability")
    for column, name in enumerate(row_three):
        axis = add_panel(
            2,
            column,
            lambda ax, name=name: draw_embedding(
                ax,
                severe_embedding,
                severe_predictions[name],
                target,
                f"{SHORT[name]} (severe)",
                severe_metrics[name],
            ),
        )
        axis.text(-0.11, 1.05, next(letters), transform=axis.transAxes, fontsize=8, fontweight="bold", va="top")

    recall_axis = add_panel(2, 2, lambda ax: None)
    recall_matrix = np.array(
        [[severe_metrics[name]["recall"][str(index)] for index in range(n_classes)] for name in METHOD_LABELS]
    )
    image = recall_axis.imshow(recall_matrix, vmin=0.0, vmax=1.0, cmap="YlGn", aspect="auto")
    recall_axis.set_xticks(range(n_classes))
    recall_axis.set_xticklabels(CLASS_NAMES, fontsize=5.6)
    recall_axis.set_yticks(range(len(METHOD_LABELS)))
    recall_axis.set_yticklabels([SHORT[name] for name in METHOD_LABELS], fontsize=5.2)
    for row in range(recall_matrix.shape[0]):
        for column in range(recall_matrix.shape[1]):
            recall_axis.text(
                column,
                row,
                f"{recall_matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=4.7,
                color="white" if recall_matrix[row, column] > 0.55 else "#222222",
            )
    recall_axis.set_title("Per-class recall (severe)", fontsize=6.2, pad=2)
    recall_axis.tick_params(length=1.5, width=0.4)
    for spine in recall_axis.spines.values():
        spine.set_linewidth(0.4)
        spine.set_edgecolor("#555555")
    recall_axis.text(-0.16, 1.05, next(letters), transform=recall_axis.transAxes, fontsize=8, fontweight="bold", va="top")

    silhouette_axis = add_panel(2, 3, lambda ax: None)
    for name in METHOD_LABELS:
        silhouette = severe_metrics[name]["silhouette"]
        accuracy = severe_metrics[name]["acc"]
        color = METHOD_COLORS[name]
        silhouette_axis.scatter(accuracy * 100.0, silhouette, s=22, color=color, edgecolor="#222222", linewidth=0.35, zorder=3)
        offsets = {
            "Vision": (4, 2),
            "Depth": (-22, 4),
            "PCA avg": (-24, -2),
            "Ours": (-18, 4),
        }
        if SHORT[name] in offsets:
            silhouette_axis.annotate(
                SHORT[name],
                (accuracy * 100.0, silhouette),
                xytext=offsets[SHORT[name]],
                textcoords="offset points",
                fontsize=4.8,
                color="#222222",
            )
    silhouette_axis.set_xlabel("Clustering accuracy (%)", fontsize=6.0, labelpad=1)
    silhouette_axis.set_yticks([])
    silhouette_axis.tick_params(axis="x", labelsize=5.2)
    silhouette_axis.tick_params(labelsize=5.4, length=1.8, width=0.4, pad=1.5)
    silhouette_axis.set_title("Silhouette vs ACC", fontsize=6.2, pad=2)
    silhouette_axis.grid(color="#E6E6E6", linewidth=0.45)
    silhouette_axis.set_axisbelow(True)
    silhouette_axis.axvspan(40, 50, color="#F6D5D5", alpha=0.35, zorder=0)
    silhouette_axis.text(46.8, 0.405, "high silhouette\nlow accuracy", fontsize=4.4, color="#8B1A1A", ha="center")
    silhouette_axis.text(0.98, 0.96, "y: silhouette\nx: accuracy\nother methods\nunlabelled", transform=silhouette_axis.transAxes, ha="right", va="top", fontsize=4.1, color="#777777")
    silhouette_axis.text(-0.16, 1.05, next(letters), transform=silhouette_axis.transAxes, fontsize=8, fontweight="bold", va="top")

    color_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="#222222", markersize=4.2, label=f"Predicted {name}")
        for name, color in zip(CLASS_NAMES, CLASS_COLORS)
    ]
    shape_handles = [
        Line2D([0], [0], marker=marker, color="none", markerfacecolor="#BBBBBB", markeredgecolor="#222222", markersize=4.2, label=f"True {name}")
        for name, marker in zip(CLASS_NAMES, CLASS_MARKERS)
    ]
    fig.legend(
        handles=color_handles + shape_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=6,
        frameon=False,
        fontsize=5.5,
        handlelength=0.9,
        columnspacing=1.4,
    )
    fig.text(
        0.5,
        0.006,
        "Common t-SNE maps of the clean and severely degraded NYU representations. Colour = predicted terrain, shape = true terrain; "
        "shape/colour mismatches are clustering errors. Severe corruption: each sample loses exactly one modality (level 2.0).",
        ha="center",
        va="bottom",
        fontsize=5.4,
        color="#444444",
    )

    pdf_path, png_path = save_figure(fig, "Figure8_clustering_qualitative", args.output, args.dpi)

    embedding_rows = []
    for index in range(target.size):
        embedding_rows.append(
            [
                index,
                CLASS_NAMES[int(target[index])],
                info["visual_degraded_indices"].count(index) > 0,
                info["depth_degraded_indices"].count(index) > 0,
                f"{clean_embedding[index, 0]:.6f}",
                f"{clean_embedding[index, 1]:.6f}",
                f"{severe_embedding[index, 0]:.6f}",
                f"{severe_embedding[index, 1]:.6f}",
                *[CLASS_NAMES[int(severe_predictions[name][index])] for name in METHOD_LABELS],
            ]
        )
    write_csv(
        args.output / "source_data" / "figure8_clustering_embeddings.csv",
        [
            "sample_index",
            "true_class",
            "visual_degraded",
            "depth_degraded",
            "tsne_clean_x",
            "tsne_clean_y",
            "tsne_severe_x",
            "tsne_severe_y",
            *[f"pred_{SHORT[name].replace(' ', '_')}" for name in METHOD_LABELS],
        ],
        embedding_rows,
    )
    metric_rows = []
    for level_name, metrics_by_method in (("clean", clean_metrics), ("severe", severe_metrics)):
        for name in METHOD_LABELS:
            metrics = metrics_by_method[name]
            metric_rows.append(
                [
                    level_name,
                    name,
                    f"{metrics['acc'] * 100:.3f}",
                    f"{metrics['nmi'] * 100:.3f}",
                    f"{metrics['ari']:.6f}",
                    f"{metrics['silhouette']:.6f}",
                ]
            )
    write_csv(
        args.output / "source_data" / "figure8_method_metrics.csv",
        ["level", "method", "accuracy_pct", "nmi_pct", "ari", "silhouette"],
        metric_rows,
    )
    recall_rows = []
    for name in METHOD_LABELS:
        for index, class_name in enumerate(CLASS_NAMES):
            recall_rows.append(
                [
                    "severe",
                    name,
                    class_name,
                    f"{severe_metrics[name]['recall'][str(index)]:.6f}",
                ]
            )
    write_csv(
        args.output / "source_data" / "figure8_per_class_recall.csv",
        ["level", "method", "class", "recall"],
        recall_rows,
    )
    write_csv(
        args.output / "source_data" / "figure8_reliability_weights.csv",
        ["sample_index", "true_class", "degraded_modality", "q_visual", "q_depth", "w_visual", "w_depth"],
        [
            [
                index,
                CLASS_NAMES[int(target[index])],
                "visual" if info["visual_degraded_indices"].count(index) > 0 else "depth",
                f"{severe_qv[index]:.6f}",
                f"{severe_qd[index]:.6f}",
                _weight(severe_qv[index], severe_qd[index])[0],
                _weight(severe_qv[index], severe_qd[index])[1],
            ]
            for index in range(target.size)
        ],
    )
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")


def _quality(
    vision_raw: np.ndarray,
    depth_raw: np.ndarray,
    vision_names: list[str],
    depth_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    from ahpff import compute_quality

    return compute_quality(vision_raw, depth_raw, vision_names, depth_names)


def _weight(qv: float, qd: float) -> tuple[str, str]:
    import numpy as np

    wv = 1.0 / (1.0 + np.exp(4.0 * (qd - qv)))
    return f"{wv:.6f}", f"{1.0 - wv:.6f}"


if __name__ == "__main__":
    main()
