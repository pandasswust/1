#!/usr/bin/env python3
"""Train the reliability-aware AHPFF cluster readout from a Gazebo calibration set.

The online label is no longer produced by a supervised classifier alone.  The
descriptor bank is clustered with reliability-weighted multi-view K-Means; each
cluster is then mapped to a terrain class by majority vote over the labelled
calibration samples.  At run time a new sample is assigned to the nearest
cluster centroid in the reliability-weighted descriptor space, and the terrain
class of that cluster is published.  An ExtraTrees model is also fitted as a
supervised upper bound for comparison, but it is not used by the online node by
default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from ahpff import (
    blocked_stratified_folds,
    fit_cluster_readout,
    map_clusters_to_classes,
    random_stratified_folds,
    ReliabilityWeightedKMeans,
    compute_quality,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CLASS_NAMES = {0: "Flat", 1: "Stairs", 2: "Obstacle", 3: "Tactile"}
VISION_FEATURE_NAMES = (
    "canny_edge_density",
    "laplacian_variance",
    "mean_intensity",
    "std_intensity",
    "mean_gradient",
    "std_gradient",
    "intensity_entropy",
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_saturation",
    "yellow_fraction",
    "red_fraction",
)
DEPTH_FEATURE_NAMES = (
    "depth_mean",
    "depth_std",
    "depth_min",
    "depth_max",
    "depth_p25",
    "depth_p75",
    "depth_span",
    "depth_mean_abs_sobel_x",
    "depth_mean_abs_sobel_y",
    "depth_mean_gradient",
    "valid_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "ahpff_guidedog_calibration.npz")
    parser.add_argument("--model", type=Path, default=DATA_DIR / "ahpff_guidedog_model.joblib")
    parser.add_argument("--meta", type=Path, default=DATA_DIR / "ahpff_guidedog_model.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--clusters", type=int, default=0, help="0 sets K to the number of terrain classes")
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def split_feature_columns(
    features: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    vision_names = [name for name in VISION_FEATURE_NAMES if name in name_to_index]
    depth_names = [name for name in DEPTH_FEATURE_NAMES if name in name_to_index]
    if not vision_names or not depth_names:
        raise SystemExit("Could not identify visual and depth feature columns.")
    if len(vision_names) + len(depth_names) != features.shape[1]:
        known = set(vision_names) | set(depth_names)
        extra = [name for name in feature_names if name not in known]
        if extra:
            depth_names = depth_names + extra
    vision_indices = [name_to_index[name] for name in vision_names]
    depth_indices = [name_to_index[name] for name in depth_names]
    return features[:, vision_indices], features[:, depth_indices], vision_names, depth_names


def ensure_valid_ratio(
    features: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    if "valid_ratio" not in feature_names:
        features = np.hstack([features, np.ones((features.shape[0], 1), dtype=np.float64)])
        feature_names = feature_names + ["valid_ratio"]
    return features, feature_names


def cluster_cv(
    vision: np.ndarray,
    depth: np.ndarray,
    target: np.ndarray,
    vision_names: list[str],
    depth_names: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_clusters: int,
    n_classes: int,
    beta: float,
    seed: int,
) -> dict:
    predictions = np.full(target.size, -1, dtype=np.int64)
    fold_rows = []
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        vision_scaler = StandardScaler().fit(vision[train_indices])
        depth_scaler = StandardScaler().fit(depth[train_indices])
        train_vision = vision_scaler.transform(vision[train_indices])
        test_vision = vision_scaler.transform(vision[test_indices])
        train_depth = depth_scaler.transform(depth[train_indices])
        test_depth = depth_scaler.transform(depth[test_indices])
        train_qv, train_qd, _ = compute_quality(
            vision[train_indices], depth[train_indices], vision_names, depth_names
        )
        test_qv, test_qd, _ = compute_quality(
            vision[test_indices], depth[test_indices], vision_names, depth_names
        )
        model = ReliabilityWeightedKMeans(
            n_clusters=n_clusters,
            beta=beta,
            n_init=15,
            seed=seed + fold_index,
        ).fit(train_vision, train_depth, train_qv, train_qd)
        train_clusters = model.labels_
        mapping, _ = map_clusters_to_classes(
            train_clusters, target[train_indices], n_clusters, n_classes
        )
        test_clusters = model.predict(test_vision, test_depth, test_qv, test_qd)
        predictions[test_indices] = mapping[test_clusters]
        fold_rows.append(
            {
                "fold": fold_index,
                "train": int(train_indices.size),
                "test": int(test_indices.size),
                "accuracy": float(np.mean(mapping[test_clusters] == target[test_indices])),
                "class_counts_test": {
                    CLASS_NAMES.get(key, str(key)): int(np.sum(target[test_indices] == key))
                    for key in sorted(set(int(value) for value in target[test_indices]))
                },
            }
        )
    present = np.arange(n_classes)
    report = classification_report(
        target,
        predictions,
        labels=present,
        target_names=[CLASS_NAMES.get(key, str(key)) for key in present],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "predictions": predictions,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(target, predictions, labels=present).tolist(),
        "confusion_labels": [CLASS_NAMES.get(key, str(key)) for key in present],
        "folds": fold_rows,
    }


def supervised_cv(
    vision: np.ndarray,
    depth: np.ndarray,
    target: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> dict:
    features = np.hstack([vision, depth])
    predictions = np.full(target.size, -1, dtype=np.int64)
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        scaler = StandardScaler().fit(features[train_indices])
        classifier = ExtraTreesClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=seed + fold_index,
            n_jobs=1,
        )
        classifier.fit(scaler.transform(features[train_indices]), target[train_indices])
        predictions[test_indices] = classifier.predict(scaler.transform(features[test_indices]))
    present = np.arange(int(np.max(target)) + 1)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "predictions": predictions,
        "classification_report": classification_report(
            target,
            predictions,
            labels=present,
            target_names=[CLASS_NAMES.get(key, str(key)) for key in present],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(target, predictions, labels=present).tolist(),
        "confusion_labels": [CLASS_NAMES.get(key, str(key)) for key in present],
    }


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Missing calibration data: {args.data}")
    data = np.load(args.data, allow_pickle=True)
    features = data["features"].astype(np.float64)
    labels = data["labels"].astype(np.int64)
    timestamps = data["timestamps"].astype(np.float64) if "timestamps" in data else np.arange(labels.size, dtype=np.float64)
    feature_names = [str(name) for name in data["feature_names"]] if "feature_names" in data else []
    if not feature_names:
        if features.shape[1] == len(VISION_FEATURE_NAMES) + len(DEPTH_FEATURE_NAMES) - 1:
            feature_names = list(VISION_FEATURE_NAMES) + list(DEPTH_FEATURE_NAMES[:-1])
        else:
            feature_names = list(VISION_FEATURE_NAMES) + list(DEPTH_FEATURE_NAMES)
    features, feature_names = ensure_valid_ratio(features, feature_names)
    roi = data["roi"].tolist() if "roi" in data else None

    present = sorted(int(value) for value in np.unique(labels))
    counts = {CLASS_NAMES.get(key, str(key)): int(np.sum(labels == key)) for key in present}
    print(f"Samples: {features.shape[0]}, features: {features.shape[1]}, counts: {counts}")
    if min(counts.values()) < 2:
        raise SystemExit("Each class needs at least two samples for calibration.")

    vision, depth, vision_names, depth_names = split_feature_columns(features, feature_names)
    n_classes = max(present) + 1
    n_clusters = args.clusters if args.clusters > 0 else n_classes

    blocked_folds = blocked_stratified_folds(labels, timestamps, n_splits=min(args.folds, min(counts.values())))
    random_folds = random_stratified_folds(labels, n_splits=min(args.folds, min(counts.values())), seed=args.seed)
    blocked_result = cluster_cv(
        vision, depth, labels, vision_names, depth_names, blocked_folds,
        n_clusters, n_classes, args.beta, args.seed,
    )
    random_result = cluster_cv(
        vision, depth, labels, vision_names, depth_names, random_folds,
        n_clusters, n_classes, args.beta, args.seed,
    )
    supervised_blocked = supervised_cv(vision, depth, labels, blocked_folds, args.seed)
    supervised_random = supervised_cv(vision, depth, labels, random_folds, args.seed)

    payload = fit_cluster_readout(
        vision,
        depth,
        labels,
        n_clusters,
        n_classes,
        vision_names,
        depth_names,
        beta=args.beta,
        seed=args.seed,
        n_init=20,
    )
    payload.update(
        {
            "model_type": "reliability_weighted_cluster_readout",
            "class_names": CLASS_NAMES,
            "feature_names": feature_names,
            "vision_names": vision_names,
            "depth_names": depth_names,
            "roi": roi,
            "cv": {key: value for key, value in blocked_result.items() if key != "predictions"},
            "random_cv": {key: value for key, value in random_result.items() if key != "predictions"},
            "supervised_cv": {key: value for key, value in supervised_blocked.items() if key != "predictions"},
            "supervised_random_cv": {key: value for key, value in supervised_random.items() if key != "predictions"},
            "split_protocol": {
                "blocked": "each class sorted by timestamp and divided into contiguous blocks; block b assigned to fold b",
                "random": "stratified random k-fold (optimistic; adjacent frames may cross folds)",
            },
        }
    )
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model)

    supervised_features = np.hstack([vision, depth])
    supervised_scaler = StandardScaler().fit(supervised_features)
    supervised_classifier = ExtraTreesClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=1,
    ).fit(supervised_scaler.transform(supervised_features), labels)
    supervised_model = {
        "model_type": "supervised_extratrees_baseline",
        "scaler": supervised_scaler,
        "model": supervised_classifier,
        "class_names": CLASS_NAMES,
        "feature_names": feature_names,
        "roi": roi,
    }
    supervised_path = args.model.with_name(args.model.stem + "_supervised.joblib")
    joblib.dump(supervised_model, supervised_path)

    metadata = {
        "data": str(args.data),
        "model": str(args.model),
        "supervised_model": str(args.model.with_name(args.model.stem + "_supervised.joblib")),
        "model_type": payload["model_type"],
        "samples": int(features.shape[0]),
        "features": int(features.shape[1]),
        "class_counts": counts,
        "class_names": CLASS_NAMES,
        "feature_names": feature_names,
        "vision_feature_names": vision_names,
        "depth_feature_names": depth_names,
        "n_clusters": int(n_clusters),
        "beta": float(args.beta),
        "cluster_to_class": payload["cluster_to_class"].tolist(),
        "cluster_class_counts": payload["cluster_class_counts"].tolist(),
        "blocked_cv_accuracy": blocked_result["accuracy"],
        "blocked_cv_accuracy_supervised": supervised_blocked["accuracy"],
        "random_cv_accuracy": random_result["accuracy"],
        "random_cv_accuracy_supervised": supervised_random["accuracy"],
        "blocked_classification_report": blocked_result["classification_report"],
        "blocked_confusion_matrix": blocked_result["confusion_matrix"],
        "supervised_blocked_classification_report": supervised_blocked["classification_report"],
        "supervised_blocked_confusion_matrix": supervised_blocked["confusion_matrix"],
        "blocked_folds": blocked_result["folds"],
        "random_folds": {
            "accuracy": random_result["accuracy"],
            "classification_report": random_result["classification_report"],
            "confusion_matrix": random_result["confusion_matrix"],
        },
        "split_protocol": payload["split_protocol"],
        "quality_diagnostics": payload["quality_diagnostics"],
        "roi": roi,
        "seed": args.seed,
    }
    args.meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Cluster readout blocked CV: {blocked_result['accuracy'] * 100:.2f}%")
    print(f"Cluster readout random CV:  {random_result['accuracy'] * 100:.2f}%")
    print(f"ExtraTrees blocked CV:     {supervised_blocked['accuracy'] * 100:.2f}%")
    print(f"ExtraTrees random CV:      {supervised_random['accuracy'] * 100:.2f}%")
    print("Blocked classification report:")
    print(classification_report(
        labels,
        blocked_result["predictions"],
        labels=np.arange(n_classes),
        target_names=[CLASS_NAMES.get(key, str(key)) for key in range(n_classes)],
        zero_division=0,
    ))
    print(f"Saved model: {args.model}")
    print(f"Saved metadata: {args.meta}")


if __name__ == "__main__":
    main()
