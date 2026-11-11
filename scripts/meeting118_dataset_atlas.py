#!/usr/bin/env python3
"""Meeting-118 dataset atlas: RGB, depth, dense masks and colour codes.

This script answers the review request to show the actual data used by the
manuscript rather than only metric bars:

* RGB images, metric depth maps and densely annotated NYU Depth V2 masks for
  the three verified terrain classes (Flat = 11, Stairs = 215,
  Obstacle = 5, 1-based NYU ids);
* a shared class colour-code legend used by every new figure;
* the Go1 Gazebo street RGB view, its degraded counterpart, and the online
  terrain-readout colour strip against the ground-truth schedule;
* the selected 51 NYU frames are exported as RGB JPEG, metric-depth NPZ,
  dense-mask PNG and overlay PNG for the upload bundle.

Outputs
-------
112/pdf/Figure7_dataset_atlas.pdf
112/png_1000dpi/Figure7_dataset_atlas_1000dpi.png
112/source_data/figure7_dataset_atlas.csv
112/source_data/figure7_class_color_codes.csv
112/dataset_samples/nyu/...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from publication_figures import MM, PALETTE, apply_style, save_figure, write_csv  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "112"
DEFAULT_MAT = Path("/home/yd/unitree_ws/nyu_depth_v2_labeled.mat")

CLASS_STYLE = {
    "Flat": {"color": "#009E73", "nyu_id": 11},
    "Stairs": {"color": "#D55E00", "nyu_id": 215},
    "Obstacle": {"color": "#0072B2", "nyu_id": 5},
}
GO1_STYLE = {
    0: {"name": "Flat", "color": "#009E73"},
    1: {"name": "Stairs", "color": "#D55E00"},
    2: {"name": "Obstacle", "color": "#0072B2"},
    3: {"name": "Tactile", "color": "#E69F00"},
}
OTHER_COLOR = "#BDBDBD"
BACKGROUND_COLOR = "#141414"
DEPTH_MIN = 0.5
DEPTH_MAX = 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=1000)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data" / "nyu_feature_meta.json")
    parser.add_argument("--export-samples", action="store_true", default=True)
    return parser.parse_args()


def load_frame(handle: h5py.File, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.transpose(np.asarray(handle["images"][index]), (2, 1, 0))
    depth = np.asarray(handle["depths"][index]).T
    labels = np.asarray(handle["labels"][index]).T
    return rgb, depth, labels


def colorise_depth(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0.0
    clipped = np.clip(depth, DEPTH_MIN, DEPTH_MAX)
    normalised = (clipped - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
    rgb = plt.get_cmap("inferno")(normalised)[..., :3]
    rgb = (rgb * 255.0).astype(np.uint8)
    rgb[~valid] = np.array([214, 214, 214], dtype=np.uint8)
    return rgb


def colorise_labels(labels: np.ndarray) -> np.ndarray:
    mask = np.zeros((*labels.shape, 3), dtype=np.uint8)
    mask[...] = np.array([20, 20, 20], dtype=np.uint8)
    annotated = labels > 0
    mask[annotated] = np.array([189, 189, 189], dtype=np.uint8)
    for style in CLASS_STYLE.values():
        mask[labels == style["nyu_id"]] = np.array(
            tuple(int(style["color"][i : i + 2], 16) for i in (1, 3, 5)), dtype=np.uint8
        )
    return mask


def overlay_mask(rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
    target = np.zeros(rgb.shape[:2], dtype=bool)
    color_layer = np.zeros_like(rgb, dtype=np.float64)
    for style in CLASS_STYLE.values():
        selected = masks == style["nyu_id"]
        target |= selected
        color_layer[selected] = np.array(
            [int(style["color"][i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float64
        )
    mixed = rgb.astype(np.float64) * 0.55
    mixed[target] = 0.25 * rgb[target].astype(np.float64) + 0.75 * color_layer[target]
    return np.clip(mixed, 0, 255).astype(np.uint8)


def terrain_strip(axis: plt.Axes, time: np.ndarray, labels: np.ndarray, title: str) -> None:
    strip = np.zeros((1, labels.size, 3), dtype=np.float64)
    for cls, style in GO1_STYLE.items():
        strip[0, labels == cls] = np.array(
            [int(style["color"][i : i + 2], 16) / 255.0 for i in (1, 3, 5)]
        )
    axis.imshow(strip, aspect="auto", interpolation="nearest", extent=(time[0], time[-1], 0, 1))
    axis.set_yticks([])
    axis.set_xlabel("Time (s)", labelpad=2)
    axis.set_title(title, fontsize=6.4, pad=2)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.tick_params(axis="x", labelsize=5.2)


def draw_legend(fig: plt.Figure, y: float, include_go1: bool = True) -> None:
    handles = []
    for name, style in CLASS_STYLE.items():
        handles.append(
            Patch(
                facecolor=style["color"],
                edgecolor="#333333",
                linewidth=0.35,
                label=f"{name} (NYU id {style['nyu_id']})",
            )
        )
    if include_go1:
        handles.append(
            Patch(facecolor=GO1_STYLE[3]["color"], edgecolor="#333333", linewidth=0.35, label="Tactile (Go1 class 3)")
        )
    handles.append(Patch(facecolor=OTHER_COLOR, edgecolor="#333333", linewidth=0.35, label="Other annotated NYU class"))
    handles.append(Patch(facecolor=BACKGROUND_COLOR, edgecolor="#333333", linewidth=0.35, label="Unlabelled / background"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=7,
        frameon=False,
        fontsize=5.9,
        handlelength=1.1,
        columnspacing=1.2,
    )


def export_samples(
    handle: h5py.File,
    selected: dict[str, list[int]],
    destination: Path,
) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for class_name, frames in selected.items():
        for frame in frames:
            rgb, depth, labels = load_frame(handle, frame)
            mask = colorise_labels(labels)
            overlay = overlay_mask(rgb, labels)
            stem = f"{class_name}_frame{frame:04d}"
            Image.fromarray(rgb).save(destination / f"{stem}_rgb.jpg", quality=92, subsampling=1)
            Image.fromarray(mask).save(destination / f"{stem}_mask.png")
            Image.fromarray(overlay).save(destination / f"{stem}_overlay.jpg", quality=92, subsampling=1)
            valid = depth > 0
            np.savez_compressed(
                destination / f"{stem}_depth_m.npz",
                depth_m=depth.astype(np.float32),
                valid=valid,
            )
            rows.append(
                {
                    "class": class_name,
                    "nyu_frame": frame,
                    "nyu_label_id": CLASS_STYLE[class_name]["nyu_id"],
                    "class_pixels": int(np.sum(labels == CLASS_STYLE[class_name]["nyu_id"])),
                    "annotated_pixels": int(np.sum(labels > 0)),
                    "valid_depth_ratio": float(np.mean(valid)),
                    "depth_mean_m": float(depth[valid].mean()) if np.any(valid) else float("nan"),
                    "depth_span_m": float(depth[valid].max() - depth[valid].min()) if np.any(valid) else float("nan"),
                }
            )
    return rows


def build_figure(
    handle: h5py.File,
    selected: dict[str, list[int]],
    output: Path,
    dpi: int,
    metadata: dict,
) -> tuple[Path, Path]:
    apply_style()
    fig = plt.figure(figsize=(183 * MM, 166 * MM))
    left = 21.0 * MM
    right = 3.0 * MM
    top = 9.0 * MM
    n_cols = 4
    width = 38.1 * MM
    height = width * 0.75
    gap_x = 2.0 * MM
    gap_y = 4.2 * MM
    row_labels = ("Flat", "Stairs", "Obstacle")
    column_titles = ("RGB image", "Metric depth (m)", "Dense annotation mask", "Mask-overlay comparison")

    for column, title in enumerate(column_titles):
        x = left + column * (width + gap_x)
        fig.text((x + width / 2) / (183 * MM), 0.985, title, ha="center", va="top", fontsize=6.7, fontweight="bold")

    y_cursor = 0.955
    for row, class_name in enumerate(row_labels):
        frame = selected[class_name][0]
        rgb, depth, labels = load_frame(handle, frame)
        panels = [rgb, colorise_depth(depth), colorise_labels(labels), overlay_mask(rgb, labels)]
        y = y_cursor - (row + 1) * (height / (166 * MM)) - row * (gap_y / (166 * MM))
        depth_axis = None
        for column, panel in enumerate(panels):
            x = left + column * (width + gap_x)
            axis = fig.add_axes([x / (183 * MM), y, width / (183 * MM), height / (166 * MM)])
            axis.imshow(panel, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.45)
                spine.set_edgecolor("#555555")
            if column == 1:
                depth_axis = axis
        fig.text(
            (left - 2.0 * MM) / (183 * MM),
            y + height / (166 * MM) / 2,
            f"{class_name}\nNYU #{frame}",
            ha="right",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            linespacing=1.2,
        )
        if row == 2 and depth_axis is not None:
            inset = depth_axis.inset_axes([0.45, 0.03, 0.50, 0.07])
            scalar = ScalarMappable(norm=Normalize(DEPTH_MIN, DEPTH_MAX), cmap="inferno")
            inset_bar = fig.colorbar(scalar, cax=inset, orientation="horizontal")
            inset_bar.ax.tick_params(labelsize=4.8, length=1.0, width=0.35, pad=0.8)
            inset_bar.set_ticks([1, 2, 3, 4, 5, 6])
            inset_bar.outline.set_linewidth(0.35)
            inset.set_facecolor("white")
            inset.set_zorder(5)

    bottom_of_grid = y_cursor - 3 * (height / (166 * MM)) - 2 * (gap_y / (166 * MM))
    nominal = np.asarray(Image.open(PROJECT_ROOT / "figures" / "go1" / "front_camera_view.png").convert("RGB"))
    degraded = np.asarray(
        Image.open(PROJECT_ROOT / "figures" / "go1_degraded_live" / "degraded_preview.png").convert("RGB")
    )
    panel_height = 800
    panel_width = int(round(panel_height * nominal.shape[1] / nominal.shape[0]))
    nominal = nominal[:panel_height, :panel_width]
    degraded = degraded[:panel_height, :panel_width]
    go1_height = width * (panel_height / panel_width) / (166 * MM)
    go1_top = bottom_of_grid - 4.5 * MM / (166 * MM) - go1_height

    for column, (title, image) in enumerate(
        (("Go1 street: nominal RGB", nominal), ("Go1 street: degraded RGB", degraded))
    ):
        x = left + column * (width + gap_x)
        axis = fig.add_axes([x / (183 * MM), go1_top, width / (183 * MM), go1_height])
        axis.imshow(image, interpolation="bilinear")
        axis.set_xticks([])
        axis.set_yticks([])
        if column == 0:
            axis.add_patch(
                Rectangle(
                    (0.15 * image.shape[1], 0.35 * image.shape[0]),
                    0.70 * image.shape[1],
                    0.63 * image.shape[0],
                    fill=False,
                    edgecolor="#FFFFFF",
                    linewidth=0.8,
                    linestyle="--",
                )
            )
        for spine in axis.spines.values():
            spine.set_linewidth(0.45)
            spine.set_edgecolor("#555555")
        axis.set_title(title, fontsize=6.4, pad=2)

    trajectory = np.load(PROJECT_ROOT / "figures" / "go1_live" / "sapf_gazebo_trajectory.npz")
    time = np.asarray(trajectory["time"])
    predicted = np.asarray(trajectory["labels"])
    truth = np.asarray(trajectory["ground_truth_labels"])
    strip_x = left + 2 * (width + gap_x)
    strip_width = width * 2 + gap_x
    strip_height = 10.0 * MM / (166 * MM)
    axis_pred = fig.add_axes([strip_x / (183 * MM), go1_top + strip_height + 3.0 * MM / (166 * MM), strip_width / (183 * MM), strip_height])
    terrain_strip(axis_pred, time, predicted, r"Online readout $\hat{C}_t$ (10 Hz)")
    axis_pred.tick_params(axis="x", bottom=False, labelbottom=False)
    axis_pred.set_xlabel("")
    axis_truth = fig.add_axes([strip_x / (183 * MM), go1_top, strip_width / (183 * MM), strip_height])
    terrain_strip(axis_truth, time, truth, "Ground-truth terrain schedule")

    fig.text(
        0.5,
        0.012,
        "NYU Depth V2 verified subset (17 frames per class; 51 frames). Depth is shown in metres; grey = invalid depth. "
        "Dense masks use the class colour code defined in the legend.",
        ha="center",
        va="bottom",
        fontsize=5.6,
        color="#444444",
    )
    draw_legend(fig, 0.033)

    pdf_path, png_path = save_figure(fig, "Figure7_dataset_atlas", output, dpi)
    return pdf_path, png_path


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    selected = metadata["selected_frames"]
    if not args.mat.exists():
        raise SystemExit(f"Missing NYU Depth V2 mat file: {args.mat}")

    with h5py.File(args.mat, "r") as handle:
        rows = export_samples(handle, selected, args.output / "dataset_samples" / "nyu")
        pdf_path, png_path = build_figure(handle, selected, args.output, args.dpi, metadata)

    write_csv(
        args.output / "source_data" / "figure7_dataset_atlas.csv",
        [
            "class",
            "nyu_frame",
            "nyu_label_id",
            "class_pixels",
            "annotated_pixels",
            "valid_depth_ratio",
            "depth_mean_m",
            "depth_span_m",
        ],
        [
            [
                row["class"],
                row["nyu_frame"],
                row["nyu_label_id"],
                row["class_pixels"],
                row["annotated_pixels"],
                f"{row['valid_depth_ratio']:.6f}",
                f"{row['depth_mean_m']:.6f}",
                f"{row['depth_span_m']:.6f}",
            ]
            for row in rows
        ],
    )
    write_csv(
        args.output / "source_data" / "figure7_class_color_codes.csv",
        ["dataset", "class_index", "class_name", "hex_color", "annotation_id"],
        [
            ["NYU Depth V2", 0, "Flat", CLASS_STYLE["Flat"]["color"], CLASS_STYLE["Flat"]["nyu_id"]],
            ["NYU Depth V2", 1, "Stairs", CLASS_STYLE["Stairs"]["color"], CLASS_STYLE["Stairs"]["nyu_id"]],
            ["NYU Depth V2", 2, "Obstacle", CLASS_STYLE["Obstacle"]["color"], CLASS_STYLE["Obstacle"]["nyu_id"]],
            ["Go1 street", 0, "Flat", GO1_STYLE[0]["color"], 0],
            ["Go1 street", 1, "Stairs", GO1_STYLE[1]["color"], 1],
            ["Go1 street", 2, "Obstacle", GO1_STYLE[2]["color"], 2],
            ["Go1 street", 3, "Tactile", GO1_STYLE[3]["color"], 3],
        ],
    )
    print(f"Exported {len(rows)} NYU samples to {args.output / 'dataset_samples' / 'nyu'}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")


if __name__ == "__main__":
    main()
