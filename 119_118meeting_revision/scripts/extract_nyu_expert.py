#!/usr/bin/env python3
"""Extract object-masked physical descriptors from NYU Depth V2.

NYU Depth V2 label ids are 1-based and follow the order of the ``names``
cell array.  The previous implementation used ids 1/42/4, which correspond to
book/shelves/ceiling rather than flat/stairs/obstacle.  This version uses the
verified ids: floor=11, stairs=215, chair=5 (obstacle proxy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MAT = DATA_DIR / "nyu_depth_v2_labeled.mat"

CLASS_SPECS = (
    ("Flat", 11),
    ("Stairs", 215),
    ("Obstacle", 5),
)

VISION_FEATURE_NAMES = (
    "canny_edge_density",
    "laplacian_variance",
    "mean_intensity",
    "std_intensity",
    "mean_gradient",
    "std_gradient",
    "intensity_entropy",
)

LIDAR_FEATURE_NAMES = (
    "depth_mean",
    "depth_std",
    "depth_mean_abs_sobel_x",
    "depth_mean_abs_sobel_y",
    "depth_mean_gradient",
    "depth_std_gradient",
    "depth_p25",
    "depth_p75",
    "depth_span",
    "depth_gradient_y_x_ratio",
    "valid_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--min-pixels", type=int, default=100)
    parser.add_argument("--max-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def entropy(values: np.ndarray, bins: int = 8) -> float:
    hist, _ = np.histogram(values, bins=bins, range=(0, 255))
    probs = hist.astype(np.float64) / max(int(hist.sum()), 1)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def vision_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    count = int(mask.sum())
    if count == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(grad_x, grad_y)
    values = gray[mask]
    return np.array(
        [
            edges[mask].mean() / 255.0,
            float(laplacian[mask].var()),
            float(values.mean()) / 255.0,
            float(values.std()) / 255.0,
            float(grad_mag[mask].mean()) / 255.0,
            float(grad_mag[mask].std()) / 255.0,
            entropy(values),
        ],
        dtype=np.float64,
    )


def lidar_features(depth: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    valid = mask & np.isfinite(depth) & (depth > 0)
    if int(valid.sum()) < 1:
        return None
    sobel_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(sobel_x, sobel_y)
    values = depth[valid]
    abs_x = float(np.abs(sobel_x[valid]).mean())
    abs_y = float(np.abs(sobel_y[valid]).mean())
    return np.array(
        [
            float(values.mean()),
            float(values.std()),
            abs_x,
            abs_y,
            float(grad_mag[valid].mean()),
            float(grad_mag[valid].std()),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 75)),
            float(values.max() - values.min()),
            abs_y / (abs_x + 1e-6),
            float(valid.sum() / max(int(mask.sum()), 1)),
        ],
        dtype=np.float64,
    )


def scan_candidates(labels: h5py.Dataset, specs: tuple[tuple[str, int], ...], min_pixels: int) -> dict[str, list[tuple[int, int]]]:
    id_to_name = {label_id: name for name, label_id in specs}
    candidates: dict[str, list[tuple[int, int]]] = {name: [] for name, _ in specs}
    for frame_index in range(labels.shape[0]):
        frame = np.asarray(labels[frame_index])
        values, counts = np.unique(frame, return_counts=True)
        frame_counts = dict(zip(values.tolist(), counts.tolist()))
        for label_id, name in id_to_name.items():
            count = int(frame_counts.get(label_id, 0))
            if count >= min_pixels:
                candidates[name].append((frame_index, count))
    return candidates


def select_frames(
    candidates: dict[str, list[tuple[int, int]]],
    specs: tuple[tuple[str, int], ...],
    max_per_class: int,
    seed: int,
) -> tuple[dict[str, list[tuple[int, int]]], int]:
    available = {name: len(candidates[name]) for name, _ in specs}
    if min(available.values()) == 0:
        missing = [name for name, count in available.items() if count == 0]
        raise SystemExit(f"No frames found for classes: {missing}. Lower --min-pixels.")
    n_per_class = min(max_per_class, min(available.values()))
    rng = np.random.RandomState(seed)
    selected: dict[str, list[tuple[int, int]]] = {}
    for name, _ in specs:
        pool = candidates[name]
        if len(pool) > n_per_class:
            chosen_indices = np.sort(rng.choice(len(pool), size=n_per_class, replace=False))
            selected[name] = [pool[int(index)] for index in chosen_indices]
        else:
            selected[name] = list(pool)
    return selected, n_per_class


def main() -> None:
    args = parse_args()
    if not args.mat.exists():
        raise SystemExit(f"NYU Depth V2 file not found: {args.mat}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.mat, "r") as handle:
        labels = handle["labels"]
        candidates = scan_candidates(labels, CLASS_SPECS, args.min_pixels)
        selected, n_per_class = select_frames(candidates, CLASS_SPECS, args.max_per_class, args.seed)

        vision_rows: list[np.ndarray] = []
        lidar_rows: list[np.ndarray] = []
        targets: list[int] = []
        used_frames: dict[str, list[int]] = {}

        for class_index, (name, label_id) in enumerate(CLASS_SPECS):
            used_frames[name] = []
            for frame_index, _ in selected[name]:
                image = np.asarray(handle["images"][frame_index]).transpose(2, 1, 0).astype(np.uint8)
                depth = np.asarray(handle["depths"][frame_index]).T.astype(np.float32)
                mask = np.asarray(handle["labels"][frame_index]).T == label_id
                view_features = vision_features(image, mask)
                depth_features = lidar_features(depth, mask)
                if view_features is None or depth_features is None:
                    continue
                vision_rows.append(view_features)
                lidar_rows.append(depth_features)
                targets.append(class_index)
                used_frames[name].append(int(frame_index))

    if not vision_rows:
        raise SystemExit("No valid samples were extracted. Check the label ids and masks.")

    vision_raw = np.asarray(vision_rows, dtype=np.float64)
    lidar_raw = np.asarray(lidar_rows, dtype=np.float64)
    vision = StandardScaler().fit_transform(vision_raw).astype(np.float32)
    lidar = StandardScaler().fit_transform(lidar_raw).astype(np.float32)
    target = np.asarray(targets, dtype=np.int64)

    np.save(args.out_dir / "Zv_real.npy", vision)
    np.save(args.out_dir / "Zl_real.npy", lidar)
    np.save(args.out_dir / "Zv_real_raw.npy", vision_raw)
    np.save(args.out_dir / "Zl_real_raw.npy", lidar_raw)
    np.save(args.out_dir / "Y_real.npy", target)

    class_counts = {
        name: int(np.sum(target == index)) for index, (name, _) in enumerate(CLASS_SPECS)
    }
    metadata = {
        "source_mat": str(args.mat),
        "class_names": [name for name, _ in CLASS_SPECS],
        "nyu_label_ids": {name: label_id for name, label_id in CLASS_SPECS},
        "candidate_frames": {name: len(candidates[name]) for name, _ in CLASS_SPECS},
        "selected_frames": used_frames,
        "samples_per_class": class_counts,
        "samples_total": int(target.size),
        "balanced_per_class": int(n_per_class),
        "min_pixels": int(args.min_pixels),
        "max_per_class": int(args.max_per_class),
        "seed": int(args.seed),
        "feature_names_vision": list(VISION_FEATURE_NAMES),
        "feature_names_lidar": list(LIDAR_FEATURE_NAMES),
        "dim_vision": int(vision.shape[1]),
        "dim_lidar": int(lidar.shape[1]),
        "raw_feature_files": {
            "vision": "Zv_real_raw.npy",
            "depth": "Zl_real_raw.npy",
        },
        "note": (
            "NYU Depth V2 label ids are 1-based. Stairs are rare (17 annotated frames "
            "with at least 100 annotated pixels in the full labeled set); the subset is "
            "balanced to the smallest class (17 images per class, 51 images total). "
            "Results on this subset are pilot-scale, not a large benchmark."
        ),
    }
    with (args.out_dir / "nyu_feature_meta.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    print(f"Extracted {target.size} samples: {class_counts}")
    print(f"Vision features: {vision.shape}, LiDAR features: {lidar.shape}")
    if n_per_class < 50:
        print(
            "WARNING: the balanced subset has fewer than 50 samples per class because "
            "NYU Depth V2 contains very few annotated stairs frames. Report this as a "
            "pilot study or replace the stairs data with a dedicated terrain dataset."
        )
    print(f"Saved features and metadata to {args.out_dir}")


if __name__ == "__main__":
    main()
