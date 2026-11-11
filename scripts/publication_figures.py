#!/usr/bin/env python3
"""Publication-grade figures for the revised 112 manuscript.

The script writes every main and extended-data figure as a vector PDF and as a
1000 dpi PNG, together with tidy source-data CSVs and a combined PDF.  It uses
the values reported in the 112 manuscript (Tables 4--11 and the Fig. 3 caption)
as the canonical paper data, and reads the raw trajectory / sweep / filter
outputs produced by the repository for the panels that benefit from per-seed or
time-resolved data.

Usage
-----
    python3 scripts/publication_figures.py
    python3 scripts/publication_figures.py --output 112 --dpi 1000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, LogLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "112"
MM = 1.0 / 25.4

# ---------------------------------------------------------------------------
# Canonical data transcribed from the 112 manuscript
# ---------------------------------------------------------------------------

METHODS = (
    "Vision only",
    "Depth only",
    "Fixed concat",
    "PCA average",
    "Uniform weighted",
    "Shuffled reliability",
    "RA-AHPFF (Ours)",
)
SHORT = ("Vision", "Depth", "Fixed concat", "PCA avg", "Uniform", "Shuffled", "RA-AHPFF")
METHOD_COLORS = {
    "Vision only": "#56B4E9",
    "Depth only": "#0072B2",
    "Fixed concat": "#7F7F7F",
    "PCA average": "#CC79A7",
    "Uniform weighted": "#999999",
    "Shuffled reliability": "#E69F00",
    "RA-AHPFF (Ours)": "#D55E00",
}

# Table 4 (clean and severe, mean +/- s.d.) and Table 5 (progressive means).
PAPER_CLEAN_ACC = (46.2, 65.0, 56.6, 68.5, 56.7, 66.7, 71.1)
PAPER_CLEAN_ACC_SD = (3.1, 0.7, 2.0, 3.4, 1.7, 3.6, 2.1)
PAPER_CLEAN_NMI = (8.0, 33.2, 24.7, 35.8, 24.8, 33.9, 38.3)
PAPER_CLEAN_NMI_SD = (2.0, 1.1, 4.1, 3.5, 3.6, 3.3, 2.4)
PAPER_SEVERE_ACC = (42.5, 59.6, 54.4, 49.2, 54.2, 56.2, 60.0)
PAPER_SEVERE_ACC_SD = (4.0, 3.5, 5.2, 6.0, 5.3, 3.3, 4.7)
PAPER_SEVERE_NMI = (5.5, 27.1, 19.1, 11.2, 18.7, 21.2, 23.2)
PAPER_SEVERE_NMI_SD = (4.0, 4.5, 5.0, 6.8, 5.5, 4.2, 5.1)
PAPER_SEVERE_ARI = (-0.005, 0.194, 0.121, 0.060, 0.121, 0.148, 0.195)
PAPER_SEVERE_ARI_SD = (0.022, 0.038, 0.057, 0.052, 0.058, 0.059, 0.069)
PAPER_SEVERE_SIL = (0.432, 0.345, 0.246, 0.104, 0.248, 0.215, 0.208)
PAPER_CLEAN_ARI = (0.033, 0.228, 0.175, 0.290, 0.177, 0.265, 0.310)
PAPER_CLEAN_ARI_SD = (0.022, 0.022, 0.028, 0.044, 0.027, 0.040, 0.031)
LEVELS = (0.0, 1.0, 1.5, 2.0)
PAPER_ACC_LEVELS = (
    (46.2, 44.9, 44.7, 42.5),
    (65.0, 63.5, 61.0, 59.6),
    (56.6, 59.3, 55.9, 54.4),
    (68.5, 55.2, 52.5, 49.2),
    (56.7, 59.6, 55.4, 54.2),
    (66.7, 60.1, 59.1, 56.2),
    (71.1, 60.4, 60.2, 60.0),
)
PAPER_ACC_LEVEL_SD = (
    (3.1, 3.2, 3.5, 4.0),
    (0.7, 2.5, 3.4, 3.5),
    (2.0, 3.1, 4.1, 5.2),
    (3.4, 8.5, 6.6, 6.0),
    (1.7, 3.4, 5.2, 5.3),
    (3.6, 5.0, 4.3, 3.3),
    (2.1, 3.1, 4.7, 4.7),
)

# Table 8: mean weights and health qualities in the 6--20 s window.
WEIGHT_CONDITIONS = ("Baseline", "Overexposure", "Motion blur", "Depth dropout")
WEIGHT_WV = (0.425, 0.174, 0.338, 0.724)
WEIGHT_WD = (0.575, 0.826, 0.662, 0.276)
WEIGHT_QV = (0.760, 0.442, 0.668, 0.761)
WEIGHT_QD = (0.839, 0.844, 0.843, 0.513)

# Table 9: hyperparameter sensitivity.
LAMBDA_VALUES = (0, 1, 2, 4, 8)
LAMBDA_ACC = (54.2, 56.8, 58.6, 60.0, 59.4)
LAMBDA_NMI = (18.7, 20.4, 22.1, 23.2, 22.8)
ROC_VALUES = (1, 2, 4, 8)
ROC_ACC = (61.5, 68.9, 74.4, 77.8)
ROC_F1 = (0.54, 0.63, 0.70, 0.734)

PALETTE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "grey": "#666666",
    "light_grey": "#D9D9D9",
    "ink": "#222222",
}

STAGE_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")

# ---------------------------------------------------------------------------
# Style and helpers
# ---------------------------------------------------------------------------


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 5.9,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Liberation Sans",
            "mathtext.it": "Liberation Sans:italic",
            "mathtext.bf": "Liberation Sans:bold",
            "mathtext.sf": "Liberation Sans",
            "mathtext.default": "regular",
        }
    )


def panel_label(fig: plt.Figure, axis: plt.Axes, letter: str, xoff: float = -0.052, yoff: float = 0.012) -> None:
    position = axis.get_position()
    fig.text(
        position.x0 + xoff,
        position.y1 + yoff,
        letter,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="#111111",
    )


def save_figure(fig: plt.Figure, stem: str, output: Path, dpi: int) -> tuple[Path, Path]:
    pdf_path = output / "pdf" / f"{stem}.pdf"
    png_path = output / "png_1000dpi" / f"{stem}_1000dpi.png"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.015, facecolor="white")
    fig.savefig(
        png_path,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.015,
        facecolor="white",
        pil_kwargs={"compress_level": 6},
    )
    plt.close(fig)
    with Image.open(png_path) as image:
        if image.mode != "RGB":
            image.convert("RGB").save(
                png_path,
                format="PNG",
                dpi=(dpi, dpi),
                compress_level=6,
            )
    return pdf_path, png_path


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _style_axis(axis: plt.Axes) -> None:
    axis.tick_params(which="both", direction="out", color="#444444")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _light_grid(axis: plt.Axes, axis_name: str = "x") -> None:
    axis.grid(axis=axis_name, color="#E6E6E6", linewidth=0.45, zorder=0)
    axis.set_axisbelow(True)


def _bold_tick(axis: plt.Axes, index: int, color: str = "#111111") -> None:
    labels = axis.get_yticklabels()
    if labels:
        labels[index].set_fontweight("bold")
        labels[index].set_color(color)


# ---------------------------------------------------------------------------
# Figure 1: methodological roadmap
# ---------------------------------------------------------------------------


def _add_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
    title_size: float = 6.2,
    body_size: float = 5.35,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.0,rounding_size=1.1",
        facecolor="white",
        edgecolor=color,
        linewidth=0.55,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height - 1.85,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color="#111111",
        zorder=3,
    )
    if body:
        axis.text(
            x + width / 2,
            y + height / 2 - 0.95,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color="#222222",
            linespacing=1.18,
            zorder=3,
        )


def figure_roadmap(output: Path, dpi: int) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(183 * MM, 108 * MM))
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.axis("off")

    left = (1.6, 26.2, 50.8, 75.4)
    width = 21.9
    panel_y = 13.2
    panel_h = 72.8
    header_y = 88.8
    header_h = 6.4
    stage_titles = ("1  SENSING", "2  HEALTH GATING", "3  RW-MVK + CALIBRATION", "4  SAPF BACK END")
    stage_subtitles = (
        "observable descriptors",
        "circularity-free weights",
        "unsupervised clusters to terrain",
        "semantic covariance gating",
    )

    # Back-end / front-end brace labels.
    axis.text(
        1.6 + (3 * width + 2 * (left[1] - left[0] - width)) / 2,
        97.3,
        "Perception front end  (RA-AHPFF)",
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color="#333333",
    )
    axis.text(
        left[3] + width / 2,
        97.3,
        "State-estimation back end  (SAPF)",
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color="#333333",
    )

    for stage, (x, color, title, subtitle) in enumerate(zip(left, STAGE_COLORS, stage_titles, stage_subtitles)):
        panel = FancyBboxPatch(
            (x, panel_y),
            width,
            panel_h,
            boxstyle="round,pad=0.0,rounding_size=1.4",
            facecolor=color,
            edgecolor=color,
            linewidth=0.55,
            alpha=0.055,
            zorder=0,
        )
        axis.add_patch(panel)
        header = FancyBboxPatch(
            (x, header_y),
            width,
            header_h,
            boxstyle="round,pad=0.0,rounding_size=1.3",
            facecolor=color,
            edgecolor=color,
            linewidth=0.45,
            zorder=2,
        )
        axis.add_patch(header)
        axis.text(
            x + width / 2,
            header_y + header_h / 2 + 0.55,
            title,
            ha="center",
            va="center",
            fontsize=6.15,
            fontweight="bold",
            color="white",
            zorder=3,
        )
        axis.text(
            x + width / 2,
            header_y - 1.55,
            subtitle,
            ha="center",
            va="center",
            fontsize=5.15,
            style="italic",
            color="#555555",
            zorder=3,
        )
        if stage < 3:
            axis.add_patch(
                FancyArrowPatch(
                    (x + width + 0.25, header_y + header_h / 2),
                    (left[stage + 1] - 0.25, header_y + header_h / 2),
                    arrowstyle="-|>",
                    mutation_scale=6.5,
                    linewidth=0.75,
                    color="#777777",
                    zorder=3,
                )
            )

    boxes = (
        # Stage 1
        (
            (
                "Synchronized RGB-D",
                "$X_t^{(v)}$ RGB  +  $X_t^{(d)}$ depth\n10 Hz online stream",
            ),
            (
                "Foreground ROI",
                "navigable path ahead\n$(0.15,0.85)\\times(0.35,0.98)$",
            ),
            (
                "Physical descriptors",
                "$d_v=7$ NYU / $13$ online\n$d_d=11$; per-view $\\tilde z_t$",
            ),
        ),
        # Stage 2
        (
            (
                "Visual health $s_v$",
                "$0.32e+0.24c+0.20h$\n$+0.12g+0.06\\epsilon+0.06\\kappa$",
            ),
            (
                "Depth health $s_d$",
                "$0.50r+0.25\\omega$\n$+0.15\\rho+0.10\\gamma$",
            ),
            (
                "Logistic reliability",
                "$q_m=\\sigma(5(s_m-0.45))$\n$q_d=0$ if $r=0$",
            ),
            (
                "Softmax gate",
                "$w_v=\\frac{e^{\\lambda q_v}}{e^{\\lambda q_v}+e^{\\lambda q_d}}$\n$\\lambda=4$",
            ),
        ),
        # Stage 3
        (
            (
                "Weighted product space",
                "$[\\sqrt{w_v}\\,\\tilde z^{(v)},\\sqrt{w_d}\\,\\tilde z^{(d)}]$\northogonal blocks",
            ),
            (
                "Per-sample RW-MVK",
                "$K=r_{oc}C$\nweighted centroids",
            ),
            (
                "Majority calibration",
                "$m_k=\\arg\\max_c\\sum I[y_i=c]$\nlabels name clusters only",
            ),
            (
                "Online readout",
                "nearest centroid\n$\\hat C_t$ + 5-frame filter",
            ),
        ),
        # Stage 4
        (
            (
                "Semantic gate",
                "$Q(\\hat C_t)=\\mathrm{diag}(\\sigma_{base}^2+\\sigma_{\\hat C_t}^2)$",
            ),
            (
                "Propagation",
                "50 Hz, 500 particles\nleg odometry $u_t$",
            ),
            (
                "Intermittent update",
                "5 Hz pose; 4 s outages\n8--12 s and 18--22 s",
            ),
            (
                "6-DoF output",
                "$\\hat x_t=[p_x,p_y,p_z,\\phi,\\theta,\\psi]$\nselective inflation",
            ),
        ),
    )

    row_bottoms = ((67.5, 49.2, 30.9), (67.5, 49.2, 30.9, 15.6))
    box_height = 13.2
    for stage, (x, color) in enumerate(zip(left, STAGE_COLORS)):
        stage_boxes = boxes[stage]
        bottoms = row_bottoms[1] if len(stage_boxes) == 4 else row_bottoms[0]
        for row, (title, body) in enumerate(stage_boxes):
            _add_box(axis, x + 1.45, bottoms[row], width - 2.9, box_height, title, body, color)
            if row < len(stage_boxes) - 1:
                axis.add_patch(
                    FancyArrowPatch(
                        (x + width / 2, bottoms[row] - 0.15),
                        (x + width / 2, bottoms[row + 1] + box_height + 0.15),
                        arrowstyle="-|>",
                        mutation_scale=5.8,
                        linewidth=0.65,
                        color=color,
                        zorder=1,
                    )
                )

    # Design note fills the lower part of the sensing stage without implying a
    # sequential processing step.
    note = FancyBboxPatch(
        (left[0] + 1.45, 15.6),
        width - 2.9,
        13.2,
        boxstyle="round,pad=0.0,rounding_size=1.1",
        facecolor="#FAFAFA",
        edgecolor=STAGE_COLORS[0],
        linewidth=0.45,
        linestyle=(0, (3, 2)),
        alpha=0.85,
        zorder=2,
    )
    axis.add_patch(note)
    axis.text(
        left[0] + width / 2,
        26.9,
        "Design constraints",
        ha="center",
        va="top",
        fontsize=6.0,
        fontweight="bold",
        color="#111111",
        zorder=3,
    )
    axis.text(
        left[0] + width / 2,
        22.1,
        "no learned embedding\nCPU-only  ·  10 Hz",
        ha="center",
        va="center",
        fontsize=5.35,
        color="#333333",
        linespacing=1.2,
        zorder=3,
    )

    # Bottom claims band.
    band = FancyBboxPatch(
        (1.6, 1.15),
        96.8,
        9.7,
        boxstyle="round,pad=0.0,rounding_size=1.2",
        facecolor="#F4F4F4",
        edgecolor="#CFCFCF",
        linewidth=0.5,
        zorder=0,
    )
    axis.add_patch(band)
    axis.text(
        3.0,
        7.5,
        "Circularity-free",
        ha="left",
        va="center",
        fontsize=5.9,
        fontweight="bold",
        color="#222222",
        zorder=3,
    )
    axis.text(
        3.0,
        3.7,
        "Weights come from observable sensor health before clustering; PCA coordinates are never added across sensor spaces.",
        ha="left",
        va="center",
        fontsize=5.35,
        color="#333333",
        zorder=3,
    )
    axis.text(
        59.5,
        7.5,
        "Selective gating",
        ha="left",
        va="center",
        fontsize=5.9,
        fontweight="bold",
        color="#222222",
        zorder=3,
    )
    axis.text(
        59.5,
        3.7,
        "Matches stair tracking while cutting flat-ground process variance by 97.4% (LPF inflates it 38.5x).",
        ha="left",
        va="center",
        fontsize=5.35,
        color="#333333",
        zorder=3,
    )

    return save_figure(fig, "Figure1_method_roadmap", output, dpi)


# ---------------------------------------------------------------------------
# Figure 2: NYU clustering benchmark
# ---------------------------------------------------------------------------


def figure_nyu_clustering(output: Path, dpi: int) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 150 * MM))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.925, bottom=0.085, wspace=0.30, hspace=0.46)

    y = np.arange(len(METHODS))
    bar_h = 0.34
    error_style = {"ecolor": "#333333", "elinewidth": 0.55, "capsize": 1.7, "capthick": 0.55}

    def horizontal_pair(
        axis: plt.Axes,
        clean: Sequence[float],
        clean_sd: Sequence[float],
        severe: Sequence[float],
        severe_sd: Sequence[float],
        title: str,
        xlabel: str,
        xlim: tuple[float, float],
        zero_line: bool = False,
    ) -> None:
        axis.barh(
            y - bar_h / 2,
            clean,
            height=bar_h,
            xerr=clean_sd,
            color=PALETTE["sky"],
            edgecolor="#333333",
            linewidth=0.35,
            error_kw=error_style,
            label="Clean",
            zorder=2,
        )
        axis.barh(
            y + bar_h / 2,
            severe,
            height=bar_h,
            xerr=severe_sd,
            color=PALETTE["vermillion"],
            edgecolor="#333333",
            linewidth=0.35,
            error_kw=error_style,
            label="Severe (level 2.0)",
            zorder=2,
        )
        axis.set_yticks(y)
        axis.set_yticklabels(SHORT)
        axis.invert_yaxis()
        axis.set_xlim(*xlim)
        axis.set_xlabel(xlabel)
        axis.set_title(title, loc="left", pad=3)
        axis.tick_params(axis="y", length=0)
        _light_grid(axis, "x")
        _style_axis(axis)
        if zero_line:
            axis.axvline(0.0, color="#444444", linewidth=0.55, zorder=1)
        _bold_tick(axis, len(SHORT) - 1, PALETTE["vermillion"])

    horizontal_pair(
        axes[0, 0],
        PAPER_CLEAN_ACC,
        PAPER_CLEAN_ACC_SD,
        PAPER_SEVERE_ACC,
        PAPER_SEVERE_ACC_SD,
        "Clustering accuracy",
        "Accuracy (%)",
        (0, 80),
    )
    horizontal_pair(
        axes[0, 1],
        PAPER_CLEAN_NMI,
        PAPER_CLEAN_NMI_SD,
        PAPER_SEVERE_NMI,
        PAPER_SEVERE_NMI_SD,
        "Normalized mutual information",
        "NMI (%)",
        (0, 50),
    )
    horizontal_pair(
        axes[1, 0],
        [value * 100.0 for value in PAPER_CLEAN_ARI],
        [value * 100.0 for value in PAPER_CLEAN_ARI_SD],
        [value * 100.0 for value in PAPER_SEVERE_ARI],
        [value * 100.0 for value in PAPER_SEVERE_ARI_SD],
        "Adjusted Rand index",
        "ARI (%)",
        (-5, 40),
        zero_line=True,
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["sky"], edgecolor="#333333", linewidth=0.35),
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["vermillion"], edgecolor="#333333", linewidth=0.35),
    ]
    fig.legend(
        legend_handles,
        ["Clean", "Severe (level 2.0)"],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.9,
        fontsize=6.0,
    )

    # Panel (d): accuracy versus degradation level.
    axis = axes[1, 1]
    line_styles = {
        "Vision only": ("#56B4E9", 0.9, 2.5),
        "Depth only": ("#0072B2", 1.1, 3.0),
        "Fixed concat": ("#7F7F7F", 0.8, 2.4),
        "PCA average": ("#CC79A7", 1.0, 2.8),
        "Uniform weighted": ("#999999", 0.8, 2.4),
        "Shuffled reliability": ("#E69F00", 1.0, 3.0),
        "RA-AHPFF (Ours)": (PALETTE["vermillion"], 1.8, 4.4),
    }
    for index, method in enumerate(METHODS):
        color, linewidth, marker_size = line_styles[method]
        axis.errorbar(
            LEVELS,
            PAPER_ACC_LEVELS[index],
            yerr=PAPER_ACC_LEVEL_SD[index],
            color=color,
            linewidth=linewidth,
            marker="o",
            markersize=marker_size,
            markerfacecolor=color,
            markeredgecolor="white" if method == "RA-AHPFF (Ours)" else color,
            markeredgewidth=0.5,
            elinewidth=0.5,
            capsize=1.2,
            label=SHORT[index],
            zorder=5 if method == "RA-AHPFF (Ours)" else 2,
        )
    axis.set_xticks(LEVELS)
    axis.set_xticklabels(["0\n(clean)", "1.0", "1.5", "2.0\n(severe)"])
    axis.set_xlim(-0.08, 2.45)
    axis.set_ylim(38, 76)
    axis.set_xlabel("Degradation level")
    axis.set_ylabel("Clustering accuracy (%)")
    axis.set_title("Accuracy versus degradation severity", loc="left", pad=3)
    _light_grid(axis, "y")
    _style_axis(axis)

    direct_labels = (
        ("RA-AHPFF (Ours)", "RA-AHPFF", 61.0, PALETTE["vermillion"], True),
        ("Depth only", "Depth", 58.6, "#0072B2", False),
        ("Shuffled reliability", "Shuffled", 56.2, "#E69F00", False),
        ("PCA average", "PCA", 49.2, "#CC79A7", False),
        ("Vision only", "Vision", 42.5, "#56B4E9", False),
    )
    for method, text, value, color, bold in direct_labels:
        axis.text(
            2.05,
            value,
            text,
            ha="left",
            va="center",
            fontsize=5.2,
            color=color,
            fontweight="bold" if bold else "normal",
        )
    axis.text(
        2.05,
        54.0,
        "Fixed/Uniform",
        ha="left",
        va="center",
        fontsize=4.7,
        color="#666666",
    )

    panel_label(fig, axes[0, 0], "a")
    panel_label(fig, axes[0, 1], "b")
    panel_label(fig, axes[1, 0], "c")
    panel_label(fig, axes[1, 1], "d")

    write_csv(
        output / "source_data" / "figure2_nyu_clustering.csv",
        (
            "method",
            "clean_acc",
            "clean_acc_sd",
            "severe_acc",
            "severe_acc_sd",
            "clean_nmi",
            "clean_nmi_sd",
            "severe_nmi",
            "severe_nmi_sd",
            "clean_ari_pct",
            "severe_ari_pct",
            "severe_silhouette",
        ),
        zip(
            METHODS,
            PAPER_CLEAN_ACC,
            PAPER_CLEAN_ACC_SD,
            PAPER_SEVERE_ACC,
            PAPER_SEVERE_ACC_SD,
            PAPER_CLEAN_NMI,
            PAPER_CLEAN_NMI_SD,
            PAPER_SEVERE_NMI,
            PAPER_SEVERE_NMI_SD,
            [value * 100.0 for value in PAPER_CLEAN_ARI],
            [value * 100.0 for value in PAPER_SEVERE_ARI],
            PAPER_SEVERE_SIL,
        ),
    )
    return save_figure(fig, "Figure2_nyu_clustering", output, dpi)


# ---------------------------------------------------------------------------
# Figure 3: reliability-to-performance sweep
# ---------------------------------------------------------------------------


def figure_reliability_sweep(output: Path, dpi: int) -> tuple[Path, Path]:
    data_path = PROJECT_ROOT / "figures" / "reliability_sweep.json"
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}; run scripts/run_reliability_sweep.py first.")
    data = read_json(data_path)

    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 142 * MM))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.925, bottom=0.085, wspace=0.30, hspace=0.42)

    def get(sweep: dict, key: str, component: str) -> np.ndarray:
        return np.asarray([row[key][component] for row in sweep["levels"]], dtype=float)

    def reliability_panel(
        axis: plt.Axes,
        sweep: dict,
        quality_key: str,
        weight_key: str,
        title: str,
        start_label: str,
        end_label: str,
    ) -> None:
        levels = np.asarray([row["level"] for row in sweep["levels"]], dtype=float)
        q_mean = get(sweep, quality_key, "mean")
        q_sd = get(sweep, quality_key, "std")
        w_mean = get(sweep, weight_key, "mean")
        w_sd = get(sweep, weight_key, "std")
        axis.fill_between(levels, q_mean - q_sd, q_mean + q_sd, color=PALETTE["blue"], alpha=0.16, linewidth=0)
        axis.fill_between(levels, w_mean - w_sd, w_mean + w_sd, color=PALETTE["orange"], alpha=0.16, linewidth=0)
        axis.plot(levels, q_mean, color=PALETTE["blue"], linewidth=1.3, marker="o", markersize=2.8, label=start_label)
        axis.plot(
            levels,
            w_mean,
            color=PALETTE["orange"],
            linewidth=1.3,
            marker="s",
            markersize=2.6,
            linestyle="--",
            label=end_label,
        )
        axis.set_ylim(0, 1.0)
        axis.set_xlim(-0.05, 2.05)
        axis.set_xlabel("Degradation level")
        axis.set_ylabel("Health score / modality weight")
        axis.set_title(title, loc="left", pad=3)
        _light_grid(axis, "y")
        _style_axis(axis)
        axis.legend(loc="lower left", frameon=False, handlelength=1.3, labelspacing=0.35)
        axis.annotate(
            f"{q_mean[0]:.3f}",
            xy=(levels[0], q_mean[0]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=5.0,
            color=PALETTE["blue"],
        )
        axis.annotate(
            f"{q_mean[-1]:.3f}",
            xy=(levels[-1], q_mean[-1]),
            xytext=(-18, 5),
            textcoords="offset points",
            fontsize=5.0,
            color=PALETTE["blue"],
        )
        axis.annotate(
            f"{w_mean[0]:.3f}",
            xy=(levels[0], w_mean[0]),
            xytext=(4, -9),
            textcoords="offset points",
            fontsize=5.0,
            color="#9A5A00",
        )
        axis.annotate(
            f"{w_mean[-1]:.3f}",
            xy=(levels[-1], w_mean[-1]),
            xytext=(-18, -9),
            textcoords="offset points",
            fontsize=5.0,
            color="#9A5A00",
        )

    def performance_panel(axis: plt.Axes, sweep: dict, title: str, shift_note: str | None = None) -> None:
        levels = np.asarray([row["level"] for row in sweep["levels"]], dtype=float)
        acc = 100.0 * get(sweep, "acc", "mean")
        acc_sd = 100.0 * get(sweep, "acc", "std")
        nmi = 100.0 * get(sweep, "nmi", "mean")
        nmi_sd = 100.0 * get(sweep, "nmi", "std")
        axis.fill_between(levels, acc - acc_sd, acc + acc_sd, color=PALETTE["green"], alpha=0.16, linewidth=0)
        axis.fill_between(levels, nmi - nmi_sd, nmi + nmi_sd, color=PALETTE["purple"], alpha=0.14, linewidth=0)
        axis.plot(levels, acc, color=PALETTE["green"], linewidth=1.4, marker="o", markersize=2.8, label="ACC")
        axis.plot(
            levels,
            nmi,
            color=PALETTE["purple"],
            linewidth=1.35,
            marker="s",
            markersize=2.5,
            linestyle="--",
            label="NMI",
        )
        axis.set_ylim(0, 80)
        axis.set_xlim(-0.05, 2.05)
        axis.set_xlabel("Degradation level")
        axis.set_ylabel("Clustering score (%)")
        axis.set_title(title, loc="left", pad=3)
        _light_grid(axis, "y")
        _style_axis(axis)
        axis.legend(loc="lower left", frameon=False, handlelength=1.3, labelspacing=0.35)
        axis.annotate(
            f"{acc[-1]:.1f}",
            xy=(levels[-1], acc[-1]),
            xytext=(-16, 5),
            textcoords="offset points",
            fontsize=5.2,
            color=PALETTE["green"],
            fontweight="bold",
        )
        axis.annotate(
            f"{nmi[-1]:.1f}",
            xy=(levels[-1], nmi[-1]),
            xytext=(-16, -10),
            textcoords="offset points",
            fontsize=5.2,
            color=PALETTE["purple"],
            fontweight="bold",
        )
        if shift_note:
            axis.text(
                0.98,
                0.97,
                shift_note,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=5.2,
                color="#555555",
                style="italic",
            )

    reliability_panel(
        axes[0, 0],
        data["vision_sweep"],
        "mean_qv",
        "mean_wv",
        "Visual stream: reliability and weight",
        "$q_v$",
        "$w_v$",
    )
    performance_panel(axes[0, 1], data["vision_sweep"], "Visual stream: clustering performance")
    reliability_panel(
        axes[1, 0],
        data["depth_sweep"],
        "mean_qd",
        "mean_wd",
        "Depth stream: reliability and weight",
        "$q_d$",
        "$w_d$",
    )
    performance_panel(
        axes[1, 1],
        data["depth_sweep"],
        "Depth stream: clustering performance",
        "clean visual view dominates",
    )

    # Shade the depth recovery region.
    axes[1, 0].axvspan(1.0, 2.05, color="#F2F2F2", alpha=0.55, linewidth=0, zorder=0)
    axes[1, 1].axvspan(1.0, 2.05, color="#F2F2F2", alpha=0.55, linewidth=0, zorder=0)
    axes[1, 1].annotate(
        "partial recovery",
        xy=(1.78, 60.2),
        xytext=(1.18, 49.0),
        fontsize=5.3,
        color="#555555",
        arrowprops=dict(arrowstyle="-|>", color="#777777", linewidth=0.55, shrinkA=0, shrinkB=2),
    )

    panel_label(fig, axes[0, 0], "a")
    panel_label(fig, axes[0, 1], "b")
    panel_label(fig, axes[1, 0], "c")
    panel_label(fig, axes[1, 1], "d")

    rows = []
    for sweep_name, sweep in (("visual", data["vision_sweep"]), ("depth", data["depth_sweep"])):
        for row in sweep["levels"]:
            rows.append(
                (
                    sweep_name,
                    row["level"],
                    row["mean_qv"]["mean"],
                    row["mean_qv"]["std"],
                    row["mean_wv"]["mean"],
                    row["mean_wv"]["std"],
                    row["mean_qd"]["mean"],
                    row["mean_qd"]["std"],
                    row["mean_wd"]["mean"],
                    row["mean_wd"]["std"],
                    row["acc"]["mean"] * 100.0,
                    row["acc"]["std"] * 100.0,
                    row["nmi"]["mean"] * 100.0,
                    row["nmi"]["std"] * 100.0,
                )
            )
    write_csv(
        output / "source_data" / "figure3_reliability_sweep.csv",
        (
            "sweep",
            "level",
            "qv_mean",
            "qv_sd",
            "wv_mean",
            "wv_sd",
            "qd_mean",
            "qd_sd",
            "wd_mean",
            "wd_sd",
            "acc_pct",
            "acc_sd_pct",
            "nmi_pct",
            "nmi_sd_pct",
        ),
        rows,
    )
    return save_figure(fig, "Figure3_reliability_sweep", output, dpi)


# ---------------------------------------------------------------------------
# Figure 4: standalone SAPF comparison
# ---------------------------------------------------------------------------


def figure_sapf_comparison(output: Path, dpi: int) -> tuple[Path, Path]:
    data_path = PROJECT_ROOT / "figures" / "sapf_tracking_result.json"
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}; run scripts/eval_sapf.py first.")
    data = read_json(data_path)
    per_seed = data["per_seed"]

    order = ("LO", "SPF", "LPF", "EKF", "SAPF-Shuffled", "SAPF-Oracle", "SAPF-Pred")
    labels = {
        "LO": "Odometry",
        "SPF": "Standard PF",
        "LPF": "Large-Q PF",
        "EKF": "EKF",
        "SAPF-Shuffled": "SAPF-Shuffled",
        "SAPF-Oracle": "SAPF-Oracle",
        "SAPF-Pred": "SAPF-Pred (ours)",
    }
    colors = {
        "LO": "#8C8C8C",
        "SPF": "#56B4E9",
        "LPF": "#0072B2",
        "EKF": "#CC79A7",
        "SAPF-Shuffled": "#E69F00",
        "SAPF-Oracle": "#009E73",
        "SAPF-Pred": PALETTE["vermillion"],
    }
    metrics = (
        ("position_rmse", "Position RMSE (m)", "Position error", (0.0, 0.55)),
        ("z", "Z RMSE (m)", "Vertical error", (0.0, 0.55)),
        ("stairs_z_rmse", "Stair Z RMSE (m)", "Stair vertical error", (0.0, 0.55)),
    )

    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 92 * MM), sharey=True)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.875, bottom=0.17, wspace=0.18)
    rng = np.random.RandomState(7)
    y_positions = np.arange(len(order))

    for axis, (metric, xlabel, title, xlim) in zip(axes, metrics):
        for index, method in enumerate(order):
            if metric == "z":
                values = np.asarray([row[method]["axis_rmse"]["z"] for row in per_seed], dtype=float)
            else:
                values = np.asarray([row[method][metric] for row in per_seed], dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values))
            jitter = np.clip(rng.normal(0.0, 0.045, size=values.size), -0.14, 0.14)
            axis.scatter(
                values,
                np.full(values.size, index) + jitter,
                s=2.0,
                color="#B8B8B8",
                alpha=0.75,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                [max(mean - std, 0.0), mean + std],
                [index, index],
                color="#707070",
                linewidth=0.85,
                solid_capstyle="butt",
                zorder=2,
            )
            axis.scatter(
                [mean],
                [index],
                s=26,
                color=colors[method],
                edgecolor="#222222",
                linewidth=0.35,
                zorder=3,
            )
            axis.text(
                xlim[1] * 0.985,
                index,
                f"{mean:.3f}",
                ha="right",
                va="center",
                fontsize=4.9,
                color="#333333",
                zorder=4,
            )
        axis.set_xlim(*xlim)
        axis.set_xlabel(xlabel)
        axis.set_title(title, loc="left", pad=3)
        axis.set_yticks(y_positions)
        if axis is axes[0]:
            axis.set_yticklabels([labels[method] for method in order])
            axis.tick_params(axis="y", length=0)
            _bold_tick(axis, len(order) - 1, PALETTE["vermillion"])
        else:
            axis.tick_params(axis="y", length=0)
        axis.invert_yaxis()
        _light_grid(axis, "x")
        _style_axis(axis)

    axes[0].text(
        0.985,
        0.04,
        "Flat-ground Tr(Q)\nLPF 46.2×10$^{-5}$ (38.5×)\nSAPF 1.20×10$^{-5}$ (−97.4%)",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color="#333333",
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#D0D0D0", linewidth=0.45),
        zorder=6,
    )
    fig.text(
        0.155,
        0.018,
        "Dots: individual seeds (n = 20).  Bar: mean ± s.d.  The proposed filter is highlighted.",
        fontsize=5.2,
        color="#666666",
    )

    panel_label(fig, axes[0], "a")
    panel_label(fig, axes[1], "b")
    panel_label(fig, axes[2], "c")

    rows = []
    for method in order:
        rows.append(
            (
                method,
                data["aggregate"][method]["position_rmse"]["mean"],
                data["aggregate"][method]["position_rmse"]["std"],
                data["aggregate"][method]["z"]["mean"],
                data["aggregate"][method]["z"]["std"],
                data["aggregate"][method]["stairs_z_rmse"]["mean"],
                data["aggregate"][method]["stairs_z_rmse"]["std"],
            )
        )
    write_csv(
        output / "source_data" / "figure4_sapf_comparison.csv",
        ("method", "position_rmse_mean", "position_rmse_sd", "z_rmse_mean", "z_rmse_sd", "stairs_z_rmse_mean", "stairs_z_rmse_sd"),
        rows,
    )
    return save_figure(fig, "Figure4_sapf_comparison", output, dpi)


# ---------------------------------------------------------------------------
# Figure 5: Go1 closed-loop trajectory and semantic readout
# ---------------------------------------------------------------------------


def figure_go1_closed_loop(output: Path, dpi: int) -> tuple[Path, Path]:
    data_path = PROJECT_ROOT / "figures" / "degradation_runs" / "baseline_1" / "sapf_gazebo_trajectory.npz"
    result_path = PROJECT_ROOT / "figures" / "degradation_runs" / "baseline_1" / "sapf_gazebo_result.json"
    if not data_path.exists() or not result_path.exists():
        raise SystemExit(f"Missing {data_path} or {result_path}; run the Go1 baseline experiment first.")
    data = np.load(data_path)
    result = read_json(result_path)
    time = data["time"]
    truth = data["truth"]
    spf = data["spf"]
    lpf = data["lpf"]
    sapf = data["sapf"]
    labels = data["labels"]
    ground_truth = data["ground_truth_labels"]
    accuracy = float(result["semantic_label_accuracy"])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(183 * MM, 128 * MM),
        gridspec_kw={"height_ratios": (1.25, 1.0)},
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.955, bottom=0.075, hspace=0.30)

    def x_at(timestamp: float) -> float:
        index = int(np.searchsorted(time, timestamp, side="left"))
        return float(truth[min(index, time.size - 1), 0])

    outages = ((8.0, 12.0), (18.0, 22.0))

    # Panel (a): X-Z trajectory.
    axis = axes[0]
    for start, end in outages:
        axis.axvline(x_at(start), color="#9A9A9A", linewidth=0.65, linestyle=(0, (2.2, 2.2)), zorder=1)
        axis.axvline(x_at(end), color="#9A9A9A", linewidth=0.65, linestyle=(0, (2.2, 2.2)), zorder=1)
    stair_indices = np.where(ground_truth == 1)[0]
    if stair_indices.size:
        axis.axvspan(
            float(truth[stair_indices[0], 0]),
            float(truth[stair_indices[-1], 0]),
            color=PALETTE["orange"],
            alpha=0.07,
            linewidth=0,
            zorder=0,
        )
        axis.text(
            float(np.mean(truth[stair_indices, 0])),
            0.16,
            "stairs\n1.5 m climb",
            ha="center",
            va="bottom",
            fontsize=5.0,
            color="#8A5A00",
            style="italic",
        )
    for start, end in outages:
        axis.text(
            0.5 * (x_at(start) + x_at(end)),
            2.03,
            "outage",
            ha="center",
            va="top",
            fontsize=5.0,
            color="#666666",
        )

    axis.plot(truth[:, 0], truth[:, 2], color="#111111", linewidth=1.25, label="Ground truth", zorder=5)
    axis.plot(spf[:, 0], spf[:, 2], color=PALETTE["sky"], linewidth=1.0, linestyle=":", label="SPF", zorder=3)
    axis.plot(lpf[:, 0], lpf[:, 2], color=PALETTE["blue"], linewidth=0.95, linestyle="-.", label="LPF", zorder=3)
    axis.plot(sapf[:, 0], sapf[:, 2], color=PALETTE["vermillion"], linewidth=1.35, label="SAPF-Pred", zorder=4)
    axis.set_xlim(-0.05, 8.15)
    axis.set_ylim(0.0, 2.08)
    axis.set_xlabel("X position (m)")
    axis.set_ylabel("Z position (m)")
    axis.set_title("Representative Go1 trajectory on the X–Z plane", loc="left", pad=4)
    _light_grid(axis, "y")
    _style_axis(axis)
    axis.legend(loc="upper left", frameon=False, handlelength=1.8, labelspacing=0.3)

    worst = int(np.argmax((np.abs(spf[:, 2] - truth[:, 2])) * (ground_truth == 1)))
    axis.annotate(
        "SPF lags on the stairs",
        xy=(spf[worst, 0], min(spf[worst, 2], 2.02)),
        xytext=(5.25, 1.62),
        fontsize=5.3,
        color=PALETTE["blue"],
        fontweight="bold",
        zorder=10,
        annotation_clip=False,
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["blue"], linewidth=0.55, shrinkA=0, shrinkB=2),
    )

    # Panel (b): terrain readout.
    axis = axes[1]
    for start, end in outages:
        axis.axvspan(start, end, color="#BFBFBF", alpha=0.17, linewidth=0, zorder=0)
    axis.step(time, ground_truth, where="post", color="#111111", linewidth=0.9, linestyle="--", label="Ground-truth schedule")
    axis.step(time, labels, where="post", color=PALETTE["vermillion"], linewidth=1.25, label="Online readout $\\hat C_t$")
    wrong = labels != ground_truth
    axis.vlines(time[wrong], -0.62, -0.40, color=PALETTE["vermillion"], linewidth=0.35, alpha=0.65, zorder=3)
    axis.set_yticks([0, 1, 2, 3])
    axis.set_yticklabels(["Flat", "Stairs", "Obstacle", "Tactile"])
    axis.set_ylim(-0.78, 3.38)
    axis.set_xlim(0.0, float(time[-1]))
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Terrain label $C_t$")
    axis.set_title("Time-aligned terrain readout", loc="left", pad=4)
    _light_grid(axis, "x")
    _style_axis(axis)
    axis.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.5, columnspacing=0.9)
    axis.text(
        11.0,
        3.28,
        f"Representative run: {accuracy * 100:.1f}%\nThree-run mean: 90.1 ± 1.0%",
        ha="left",
        va="top",
        fontsize=5.3,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#D0D0D0", linewidth=0.45),
        zorder=6,
    )
    axis.text(
        29.5,
        -0.72,
        "red ticks: misclassified frames",
        ha="right",
        va="bottom",
        fontsize=4.8,
        color=PALETTE["vermillion"],
    )

    panel_label(fig, axes[0], "a")
    panel_label(fig, axes[1], "b")

    step = 5
    rows = zip(
        time[::step],
        truth[::step, 0],
        truth[::step, 2],
        spf[::step, 0],
        spf[::step, 2],
        lpf[::step, 0],
        lpf[::step, 2],
        sapf[::step, 0],
        sapf[::step, 2],
        labels[::step],
        ground_truth[::step],
    )
    write_csv(
        output / "source_data" / "figure5_go1_closed_loop.csv",
        ("time_s", "x_truth", "z_truth", "x_spf", "z_spf", "x_lpf", "z_lpf", "x_sapf", "z_sapf", "label", "ground_truth"),
        rows,
    )
    return save_figure(fig, "Figure5_go1_closed_loop", output, dpi)


# ---------------------------------------------------------------------------
# Figure 6: Go1 scene and degraded preview
# ---------------------------------------------------------------------------


def figure_go1_scene(output: Path, dpi: int) -> tuple[Path, Path]:
    nominal_path = PROJECT_ROOT / "论文" / "go1_scene.png"
    degraded_path = PROJECT_ROOT / "论文" / "degraded_preview.png"
    if not nominal_path.exists():
        nominal_path = PROJECT_ROOT / "figures" / "go1" / "front_camera_view.png"
    if not degraded_path.exists():
        degraded_path = PROJECT_ROOT / "figures" / "degradation_runs" / "degraded_2" / "degraded_preview.png"
    if not nominal_path.exists() or not degraded_path.exists():
        raise SystemExit(f"Missing scene images: {nominal_path} or {degraded_path}")

    with Image.open(nominal_path) as image:
        nominal = np.asarray(image.convert("RGB"))[:620]
    with Image.open(degraded_path) as image:
        degraded = np.asarray(image.convert("RGB"))[:620]

    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 84 * MM))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.90, bottom=0.035, wspace=0.025)

    def annotation(axis: plt.Axes, text: str, xy: tuple[float, float], xytext: tuple[float, float], color: str) -> None:
        axis.annotate(
            text,
            xy=xy,
            xycoords="axes fraction",
            xytext=xytext,
            textcoords="axes fraction",
            fontsize=5.4,
            color="#111111",
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, linewidth=0.45, alpha=0.92),
            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=0.55, shrinkA=1, shrinkB=2),
            zorder=8,
        )

    axis = axes[0]
    axis.imshow(nominal, interpolation="lanczos")
    axis.set_title("Nominal rectified forward view", loc="left", fontsize=7.0, pad=3)
    axis.axis("off")
    annotation(axis, "tactile paving", (0.47, 0.29), (0.06, 0.55), "#8A5A00")
    annotation(axis, "bollard", (0.57, 0.27), (0.62, 0.55), "#8A1C1C")
    annotation(axis, "sidewalk", (0.27, 0.10), (0.05, 0.22), "#444444")

    axis = axes[1]
    axis.imshow(degraded, interpolation="lanczos")
    axis.set_title("Degraded preview (6–20 s)", loc="left", fontsize=7.0, pad=3)
    axis.axis("off")
    annotation(axis, "overexposure", (0.25, 0.55), (0.05, 0.20), "#D55E00")
    annotation(axis, "horizontal blur", (0.84, 0.44), (0.55, 0.18), "#555555")

    panel_label(fig, axes[0], "a", xoff=-0.012, yoff=0.005)
    panel_label(fig, axes[1], "b", xoff=-0.012, yoff=0.005)
    fig.text(
        0.015,
        0.015,
        "Gazebo Go1 street scene, not a physical-robot photograph. The degraded view applies overexposure and horizontal motion blur; the depth stream loses 50% of its pixels during the same window.",
        fontsize=4.8,
        color="#666666",
    )
    return save_figure(fig, "Figure6_go1_scene_and_degradation", output, dpi)


# ---------------------------------------------------------------------------
# Extended Data Figure 1: modality weights and health
# ---------------------------------------------------------------------------


def figure_modality_weights(output: Path, dpi: int) -> tuple[Path, Path]:
    data_path = PROJECT_ROOT / "figures" / "reliability_runs" / "summary.json"
    if data_path.exists():
        raw = read_json(data_path)
        keys = ("baseline", "overexposure", "blur", "depth_dropout")
        wv = [raw[key]["mean_wv_active"] for key in keys]
        wd = [raw[key]["mean_wd_active"] for key in keys]
        qv = [raw[key]["mean_qv_active"] for key in keys]
        qd = [raw[key]["mean_qd_active"] for key in keys]
    else:
        wv, wd, qv, qd = list(WEIGHT_WV), list(WEIGHT_WD), list(WEIGHT_QV), list(WEIGHT_QD)

    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 78 * MM), gridspec_kw={"width_ratios": (1.0, 1.05)})
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.18, wspace=0.30)

    y = np.arange(len(WEIGHT_CONDITIONS))
    axis = axes[0]
    axis.barh(y, wv, color=PALETTE["sky"], edgecolor="#333333", linewidth=0.4, height=0.62, label="$w_v$ visual")
    axis.barh(y, wd, left=wv, color=PALETTE["orange"], edgecolor="#333333", linewidth=0.4, height=0.62, label="$w_d$ depth")
    for index in range(len(y)):
        axis.text(wv[index] / 2, y[index], f"{wv[index]:.2f}", ha="center", va="center", fontsize=5.4, color="#123A4D")
        axis.text(wv[index] + wd[index] / 2, y[index], f"{wd[index]:.2f}", ha="center", va="center", fontsize=5.4, color="#5A3C00")
    axis.set_yticks(y)
    axis.set_yticklabels(WEIGHT_CONDITIONS)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.0)
    axis.set_xlabel("Modality weight (sum = 1)")
    axis.set_title("Per-sample softmax weights", loc="left", pad=3)
    axis.tick_params(axis="y", length=0)
    _light_grid(axis, "x")
    _style_axis(axis)
    axis = axes[1]
    width = 0.34
    axis.barh(y - width / 2, qv, height=width, color=PALETTE["blue"], edgecolor="#333333", linewidth=0.35, label="$q_v$ visual health")
    axis.barh(y + width / 2, qd, height=width, color=PALETTE["green"], edgecolor="#333333", linewidth=0.35, label="$q_d$ depth health")
    for index in range(len(y)):
        axis.text(qv[index] + 0.012, y[index] - width / 2, f"{qv[index]:.2f}", va="center", fontsize=5.2, color="#0B4A6F")
        axis.text(qd[index] + 0.012, y[index] + width / 2, f"{qd[index]:.2f}", va="center", fontsize=5.2, color="#005C43")
    axis.axvline(0.45, color="#777777", linewidth=0.55, linestyle=":")
    axis.text(0.465, -0.42, "0.45", fontsize=4.6, color="#666666", ha="left", va="bottom")
    axis.set_yticks(y)
    axis.set_yticklabels(WEIGHT_CONDITIONS)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.0)
    axis.set_xlabel("Health quality")
    axis.set_title("Modality-specific health response", loc="left", pad=3)
    axis.tick_params(axis="y", length=0)
    _light_grid(axis, "x")
    _style_axis(axis)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["sky"], edgecolor="#333333", linewidth=0.3),
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["orange"], edgecolor="#333333", linewidth=0.3),
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["blue"], edgecolor="#333333", linewidth=0.3),
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["green"], edgecolor="#333333", linewidth=0.3),
    ]
    fig.legend(
        legend_handles,
        ["$w_v$ visual", "$w_d$ depth", "$q_v$ visual health", "$q_d$ depth health"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=4,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.1,
        fontsize=5.8,
    )

    panel_label(fig, axes[0], "a")
    panel_label(fig, axes[1], "b")
    fig.text(
        0.16,
        0.015,
        "Corruption active during 6–20 s in the Go1 street scene; weights are computed per frame, not set globally.",
        fontsize=5.2,
        color="#666666",
    )

    write_csv(
        output / "source_data" / "figureS1_modality_weights.csv",
        ("condition", "wv", "wd", "qv", "qd"),
        zip(WEIGHT_CONDITIONS, wv, wd, qv, qd),
    )
    return save_figure(fig, "FigureS1_modality_weights", output, dpi)


# ---------------------------------------------------------------------------
# Extended Data Figure 2: hyperparameter sensitivity
# ---------------------------------------------------------------------------


def figure_hyperparameters(output: Path, dpi: int) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 82 * MM), gridspec_kw={"width_ratios": (1.0, 1.0)})
    fig.subplots_adjust(left=0.085, right=0.915, top=0.86, bottom=0.18, wspace=0.48)

    axis = axes[0]
    color_acc = PALETTE["green"]
    color_nmi = PALETTE["purple"]
    axis.plot(LAMBDA_VALUES, LAMBDA_ACC, color=color_acc, linewidth=1.4, marker="o", markersize=3.0, label="ACC")
    axis.set_xlabel("Softmax temperature $\\lambda$")
    axis.set_ylabel("Clustering accuracy (%)", color=color_acc)
    axis.tick_params(axis="y", colors=color_acc)
    axis.set_ylim(52, 62)
    axis.set_xticks(LAMBDA_VALUES)
    _light_grid(axis, "y")
    _style_axis(axis)
    axis.axvspan(3.6, 4.4, color="#F2F2F2", alpha=0.8, linewidth=0, zorder=0)
    axis.text(4.0, 52.4, "selected", ha="center", va="bottom", fontsize=5.0, color="#666666")
    axis.annotate(
        f"{LAMBDA_ACC[3]:.1f}",
        xy=(4, LAMBDA_ACC[3]),
        xytext=(-4, 6),
        textcoords="offset points",
        fontsize=5.2,
        color=color_acc,
        fontweight="bold",
    )
    twin = axis.twinx()
    twin.plot(LAMBDA_VALUES, LAMBDA_NMI, color=color_nmi, linewidth=1.25, linestyle="--", marker="s", markersize=2.6, label="NMI")
    twin.set_ylabel("NMI (%)", color=color_nmi)
    twin.tick_params(axis="y", colors=color_nmi)
    twin.set_ylim(17, 25)
    twin.spines["top"].set_visible(False)
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color(color_nmi)
    twin.spines["left"].set_visible(False)
    axis.set_title("Gating temperature", loc="left", pad=3)
    handles = [
        Line2D([0], [0], color=color_acc, marker="o", linewidth=1.2, markersize=2.8, label="ACC"),
        Line2D([0], [0], color=color_nmi, marker="s", linestyle="--", linewidth=1.2, markersize=2.6, label="NMI"),
    ]
    axis.legend(handles=handles, loc="lower right", frameon=False, handlelength=1.4)

    axis = axes[1]
    axis.plot(ROC_VALUES, ROC_ACC, color=PALETTE["blue"], linewidth=1.4, marker="o", markersize=3.0, label="ACC")
    axis.axhline(80.9, color="#777777", linewidth=0.6, linestyle=":")
    axis.text(1.02, 81.4, "ExtraTrees upper bound 80.9%", fontsize=4.8, color="#666666")
    axis.set_xlabel("Over-clustering ratio $r_{oc}$ (K = $r_{oc}$C)")
    axis.set_ylabel("Blocked-CV accuracy (%)", color=PALETTE["blue"])
    axis.tick_params(axis="y", colors=PALETTE["blue"])
    axis.set_xticks(ROC_VALUES)
    axis.set_ylim(58, 84)
    _light_grid(axis, "y")
    _style_axis(axis)
    axis.axvspan(7.2, 8.8, color="#F2F2F2", alpha=0.8, linewidth=0, zorder=0)
    axis.annotate(
        f"{ROC_ACC[-1]:.1f}",
        xy=(8, ROC_ACC[-1]),
        xytext=(-10, 6),
        textcoords="offset points",
        fontsize=5.2,
        color=PALETTE["blue"],
        fontweight="bold",
    )
    axis.set_xlim(0.6, 8.9)
    twin = axis.twinx()
    twin.plot(ROC_VALUES, ROC_F1, color=PALETTE["orange"], linewidth=1.25, linestyle="--", marker="s", markersize=2.6, label="Macro-F1")
    twin.set_ylabel("Macro-F1", color=PALETTE["orange"])
    twin.tick_params(axis="y", colors=PALETTE["orange"])
    twin.set_ylim(0.48, 0.80)
    twin.spines["top"].set_visible(False)
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color(PALETTE["orange"])
    twin.spines["left"].set_visible(False)
    axis.set_title("Over-clustering ratio", loc="left", pad=3)
    handles = [
        Line2D([0], [0], color=PALETTE["blue"], marker="o", linewidth=1.2, markersize=2.8, label="ACC"),
        Line2D([0], [0], color=PALETTE["orange"], marker="s", linestyle="--", linewidth=1.2, markersize=2.6, label="Macro-F1"),
    ]
    axis.legend(handles=handles, loc="lower right", frameon=False, handlelength=1.4)

    panel_label(fig, axes[0], "a")
    panel_label(fig, axes[1], "b")

    write_csv(
        output / "source_data" / "figureS2_hyperparameters.csv",
        ("parameter", "value", "accuracy_pct", "secondary_metric"),
        [(f"lambda={value}", value, LAMBDA_ACC[index], LAMBDA_NMI[index]) for index, value in enumerate(LAMBDA_VALUES)]
        + [(f"roc={value}", value, ROC_ACC[index], ROC_F1[index]) for index, value in enumerate(ROC_VALUES)],
    )
    return save_figure(fig, "FigureS2_hyperparameter_sensitivity", output, dpi)


# ---------------------------------------------------------------------------
# Extended Data Figure 3: blocked cross-validation confusion matrices
# ---------------------------------------------------------------------------


def figure_blocked_cv(output: Path, dpi: int) -> tuple[Path, Path]:
    data_path = PROJECT_ROOT / "data" / "ahpff_go1_model.json"
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}; run scripts/train_ahpff_guidedog.py first.")
    data = read_json(data_path)
    cluster = np.asarray(data["blocked_confusion_matrix"], dtype=float)
    supervised = np.asarray(data["supervised_blocked_confusion_matrix"], dtype=float)
    class_names = ("Flat", "Stairs", "Obstacle", "Tactile")
    cluster_acc = float(data["blocked_cv_accuracy"])
    supervised_acc = float(data["blocked_cv_accuracy_supervised"])

    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 82 * MM))
    fig.subplots_adjust(left=0.10, right=0.92, top=0.84, bottom=0.19, wspace=0.28)

    for axis, matrix, title, accuracy in (
        (axes[0], cluster, "Cluster readout (RA-AHPFF)", cluster_acc),
        (axes[1], supervised, "Supervised ExtraTrees (upper bound)", supervised_acc),
    ):
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
        image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = int(matrix[row, column])
                color = "white" if normalized[row, column] > 0.55 else "#222222"
                axis.text(column, row, str(value), ha="center", va="center", fontsize=6.2, color=color)
        axis.set_xticks(range(len(class_names)))
        axis.set_yticks(range(len(class_names)))
        axis.set_xticklabels(class_names, rotation=30, ha="right")
        axis.set_yticklabels(
            [f"{name} ({matrix[index].sum() and matrix[index, index] / matrix[index].sum():.2f})" for index, name in enumerate(class_names)]
        )
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(f"{title}\nblocked CV accuracy {accuracy * 100:.1f}%", loc="left", pad=4, fontsize=6.7)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(length=0)

    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Row-normalised fraction", fontsize=6.2)
    colorbar.ax.tick_params(labelsize=5.4, width=0.4, length=2)
    colorbar.outline.set_linewidth(0.4)
    fig.text(
        0.10,
        0.025,
        "Temporally blocked 5-fold CV on 351 Go1 samples (K = 32, $r_{oc}$ = 8); chance = 25%.",
        fontsize=5.2,
        color="#666666",
    )

    panel_label(fig, axes[0], "a")
    panel_label(fig, axes[1], "b")

    write_csv(
        output / "source_data" / "figureS3_cluster_confusion.csv",
        ("true_class",) + class_names,
        [(class_names[index],) + tuple(int(value) for value in cluster[index]) for index in range(len(class_names))],
    )
    write_csv(
        output / "source_data" / "figureS3_extratrees_confusion.csv",
        ("true_class",) + class_names,
        [(class_names[index],) + tuple(int(value) for value in supervised[index]) for index in range(len(class_names))],
    )
    return save_figure(fig, "FigureS3_blocked_cv_confusion", output, dpi)


# ---------------------------------------------------------------------------
# Extended Data Figure 4: covariance gating
# ---------------------------------------------------------------------------


def figure_covariance_gating(output: Path, dpi: int) -> tuple[Path, Path]:
    base = np.asarray([0.002, 0.002, 0.001, 0.001, 0.001, 0.001])
    boosts = {
        "Flat": np.zeros(6),
        "Stairs": np.asarray([0.0022360679775, 0.0022360679775, 0.0059160797831, 0.0038729833462, 0.0109544511501, 0.0017320508076]),
        "Obstacle": np.asarray([0.0118321595662, 0.0118321595662, 0.0017320508076, 0.0059160797831, 0.0017320508076, 0.0109544511501]),
        "Tactile": np.asarray([0.005, 0.005, 0.002, 0.002, 0.002, 0.005]),
    }
    labels = ("x", "y", "z", "roll", "pitch", "yaw")
    rows = []
    values = []
    for name, boost in boosts.items():
        std = np.sqrt(base**2 + boost**2)
        values.append(std)
        rows.append((name,) + tuple(std))
    values = np.vstack(values)
    maximum = np.max(values, axis=0)
    row_names = list(boosts.keys())

    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 86 * MM), gridspec_kw={"width_ratios": (1.55, 1.0)})
    fig.subplots_adjust(left=0.12, right=0.98, top=0.85, bottom=0.20, wspace=0.36)

    axis = axes[0]
    normalized = values / maximum[None, :]
    image = axis.imshow(normalized, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if normalized[row, column] > 0.62 else "#222222"
            axis.text(column, row, f"{values[row, column] * 1000:.2f}", ha="center", va="center", fontsize=4.7, color=color)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_yticks(range(len(row_names)))
    axis.set_yticklabels(row_names)
    axis.set_xlabel("State component")
    axis.set_title("Directional process-noise standard deviation (×10$^{-3}$)", loc="left", pad=3)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    colorbar.set_label("Fraction of maximum", fontsize=5.6)
    colorbar.ax.tick_params(labelsize=4.8, width=0.35, length=1.8)
    colorbar.outline.set_linewidth(0.35)

    axis = axes[1]
    methods = ("SPF", "SAPF-Pred", "SAPF-Shuffled", "LPF")
    traces = (1.20, 1.20, 19.80, 46.20)
    colors = ("#56B4E9", PALETTE["vermillion"], PALETTE["orange"], PALETTE["blue"])
    y = np.arange(len(methods))
    axis.barh(y, traces, color=colors, edgecolor="#333333", linewidth=0.4, height=0.58, log=True)
    for index, value in enumerate(traces):
        axis.text(value * 1.08, index, f"{value:.2f}", va="center", ha="left", fontsize=5.4, color="#333333")
    axis.set_yticks(y)
    axis.set_yticklabels(methods)
    axis.invert_yaxis()
    axis.set_xlim(0.9, 120)
    axis.set_xlabel("Flat-ground Tr(Q) (×10$^{-5}$)")
    axis.set_title("Covariance inflation on flat ground", loc="left", pad=3)
    axis.tick_params(axis="y", length=0)
    _light_grid(axis, "x")
    _style_axis(axis)
    axis.text(
        0.98,
        0.97,
        "LPF / SAPF = 38.5×\nflat Tr(Q) −97.4%",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=5.3,
        color="#333333",
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D0D0D0", linewidth=0.4),
    )

    panel_label(fig, axes[0], "a")
    panel_label(fig, axes[1], "b")

    write_csv(
        output / "source_data" / "figureS4_process_std.csv",
        ("class",) + labels,
        rows,
    )
    write_csv(
        output / "source_data" / "figureS4_flat_trace.csv",
        ("method", "flat_trace_x1e5"),
        zip(methods, traces),
    )
    return save_figure(fig, "FigureS4_covariance_gating", output, dpi)


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def combine_pdfs(output: Path, stems: Sequence[str]) -> Path:
    pdfs = [output / "pdf" / f"{stem}.pdf" for stem in stems]
    destination = output / "112_all_figures.pdf"
    if shutil.which("pdfunite"):
        subprocess.run(["pdfunite", *[str(path) for path in pdfs], str(destination)], check=True)
    else:
        try:
            from PyPDF2 import PdfMerger
        except Exception as exc:  # pragma: no cover
            raise SystemExit("Install pdfunite or PyPDF2 to merge the figure PDFs.") from exc
        merger = PdfMerger()
        for path in pdfs:
            merger.append(str(path))
        merger.write(str(destination))
        merger.close()
    return destination


def write_readme(output: Path, stems: Sequence[str]) -> None:
    lines = [
        "112 — publication-grade figures for the revised manuscript",
        "===========================================================",
        "",
        "Manuscript:",
        "  Reliability-Aware Multi-View Terrain Clustering and Semantic-Adaptive",
        "  State Estimation for Robotic Guide Dogs under Asymmetric Sensor Degradation",
        "",
        "Contents",
        "--------",
        "  112.tex                  LaTeX source wired to the figure PDFs.",
        "  pdf/                     One vector PDF per figure (single page).",
        "  png_1000dpi/             One 1000 dpi PNG per figure.",
        "  112_all_figures.pdf      All main and extended-data figures merged in order.",
        "  source_data/             Tidy CSV data used by every data-driven panel.",
        "  checksums.sha256         SHA-256 checksums for the PDF and PNG deliverables.",
        "",
        "Figure order",
        "------------",
        "  Figure 1   Methodological roadmap (RA-AHPFF + SAPF).",
        "  Figure 2   NYU Depth V2 clustering benchmark (Table 4/5).",
        "  Figure 3   Reliability-to-performance sweep (Fig. 3 caption).",
        "  Figure 4   Standalone 20-seed SAPF comparison (Table 10).",
        "  Figure 5   Go1 closed-loop X–Z trajectory and terrain readout (Table 11).",
        "  Figure 6   Go1 nominal scene and degraded RGB-D preview.",
        "  Figure S1  Modality weights and health under four corruption modes (Table 8).",
        "  Figure S2  Softmax-temperature and over-clustering sensitivity (Table 9).",
        "  Figure S3  Temporally blocked 5-fold confusion matrices (Tables 6/7).",
        "  Figure S4  Directional process covariance and flat-ground Tr(Q) (Table 3, Prop. 3).",
        "",
        "Technical specification",
        "-----------------------",
        "  Font family       Liberation Sans (Arial metric compatible).",
        "  PDF               True vector, embedded TrueType fonts (pdf.fonttype 42).",
        "  PNG               RGB, 1000 dpi, white background, lossless.",
        "  Palette           Okabe–Ito colour-blind-safe palette.",
        "  Panel labels      lower-case bold letters placed outside the axes.",
        "  Nominal widths    183 mm (double column) for all figures.",
        "",
        "Reproduce",
        "---------",
        "  python3 scripts/publication_figures.py --output 112 --dpi 1000",
        "  python3 scripts/verify_paper_numbers.py --figures 112",
        "  xelatex 112.tex          # requires fontspec + Noto Sans",
        "",
        "Canonical values",
        "----------------",
        "  Figure 2 uses the 112 manuscript Tables 4/5 values. Figure 3 reads",
        "  figures/reliability_sweep.json (the values printed in the 112 Fig. 3 caption).",
        "  Figure 4 reads figures/sapf_tracking_result.json (Table 10). Figure 5 reads",
        "  figures/degradation_runs/baseline_1/ (representative 90.3% run; three-run mean",
        "  90.1 ± 1.0%). The source_data/ CSVs make every plotted value explicit.",
        "",
        "Notes",
        "-----",
        "  The NYU Table 4/5 numbers were produced with the manuscript-benchmark depth",
        "  setting, which predates the strict r = 0 boundary in Eq. (4). The benchmark",
        "  defaults to this manuscript-compatible setting; pass --strict-depth-boundary",
        "  to run_arw_mvc_expert.py to enforce q_d = w_d = 0 at complete depth dropout.",
        "  The online perception node follows the strict boundary used in Algorithm 2.",
        "",
    ]
    (output / "README.txt").write_text("\n".join(lines), encoding="utf-8")

    manifest = output / "MANIFEST.txt"
    manifest.write_text("\n".join(stems) + "\n", encoding="utf-8")


def write_checksums(output: Path) -> None:
    lines = []
    for path in sorted(list((output / "pdf").glob("*.pdf")) + list((output / "png_1000dpi").glob("*.png"))):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output)}")
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_zip(output: Path) -> Path:
    archive = output.parent / f"{output.name}.zip"
    if archive.exists():
        archive.unlink()
    subprocess.run(["zip", "-rq", str(archive), output.name], cwd=output.parent, check=True)
    return archive


def build(output: Path, dpi: int, make_zip: bool = True) -> None:
    apply_style()
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("pdf", "png_1000dpi", "source_data"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    builders = (
        ("Figure1_method_roadmap", figure_roadmap),
        ("Figure2_nyu_clustering", figure_nyu_clustering),
        ("Figure3_reliability_sweep", figure_reliability_sweep),
        ("Figure4_sapf_comparison", figure_sapf_comparison),
        ("Figure5_go1_closed_loop", figure_go1_closed_loop),
        ("Figure6_go1_scene_and_degradation", figure_go1_scene),
        ("FigureS1_modality_weights", figure_modality_weights),
        ("FigureS2_hyperparameter_sensitivity", figure_hyperparameters),
        ("FigureS3_blocked_cv_confusion", figure_blocked_cv),
        ("FigureS4_covariance_gating", figure_covariance_gating),
    )
    stems = []
    for stem, builder in builders:
        pdf_path, png_path = builder(output, dpi)
        stems.append(stem)
        print(f"  {stem}: {pdf_path.name}, {png_path.name}")

    combined = combine_pdfs(output, stems)
    write_readme(output, stems)
    write_checksums(output)
    print(f"  combined: {combined.name}")
    if make_zip:
        archive = create_zip(output)
        print(f"  archive: {archive}")
    print(f"Output directory: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=1000)
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build(arguments.output, arguments.dpi, make_zip=not arguments.no_zip)
