"""Reliability-aware AHPFF utilities.

This module implements the revised front end used by the paper:

* per-sample visual and depth health indicators (exposure, contrast,
  sharpness, entropy, depth validity, depth smoothness);
* softmax modality weights derived from those health indicators;
* reliability-weighted multi-view K-Means (RW-MVK) operating on a
  weighted concatenation of the two standardized descriptor blocks;
* cluster-to-terrain calibration, so that the online label is produced by
  assigning a sample to a cluster centroid and then reading out the terrain
  class of that cluster.

Silhouette scores are deliberately *not* used to construct the weights.  They
are only used as an external clustering-quality metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

EPS = 1e-9


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _feature_dict(features: np.ndarray, names: list[str] | tuple[str, ...]) -> dict[str, float]:
    values = np.asarray(features, dtype=np.float64).reshape(-1)
    return {str(name): float(values[index]) for index, name in enumerate(names) if index < values.size}


def visual_health(features: np.ndarray, names: list[str] | tuple[str, ...]) -> dict[str, float]:
    values = _feature_dict(features, names)
    mean_intensity = float(np.clip(values.get("mean_intensity", 0.5), 0.0, 1.0))
    std_intensity = float(np.clip(values.get("std_intensity", 0.0), 0.0, 1.0))
    laplacian = max(float(values.get("laplacian_variance", 0.0)), 0.0)
    entropy = max(float(values.get("intensity_entropy", 0.0)), 0.0)
    gradient = max(float(values.get("mean_gradient", 0.0)), 0.0)
    edges = max(float(values.get("canny_edge_density", 0.0)), 0.0)

    exposure = float(np.clip(1.0 - 2.0 * abs(mean_intensity - 0.5), 0.0, 1.0))
    contrast = float(np.clip(std_intensity / 0.28, 0.0, 1.0))
    sharpness = float(laplacian / (laplacian + 40.0))
    texture = float(np.clip(gradient / 0.20, 0.0, 1.0))
    entropy_score = float(np.clip(entropy / 3.0, 0.0, 1.0))
    edge_score = float(np.clip(edges / 0.08, 0.0, 1.0))

    score = (
        0.32 * exposure
        + 0.24 * contrast
        + 0.20 * sharpness
        + 0.12 * texture
        + 0.06 * entropy_score
        + 0.06 * edge_score
    )
    quality = sigmoid(5.0 * (score - 0.45))
    return {
        "quality": quality,
        "score": score,
        "exposure": exposure,
        "contrast": contrast,
        "sharpness": sharpness,
        "texture": texture,
        "entropy": entropy_score,
        "edges": edge_score,
    }


def depth_health(
    features: np.ndarray,
    names: list[str] | tuple[str, ...],
    valid_ratio: float | None = None,
    hard_zero: bool = True,
) -> dict[str, float]:
    values = _feature_dict(features, names)
    if valid_ratio is None:
        valid_ratio = values.get("valid_ratio", 1.0)
    valid = float(np.clip(valid_ratio, 0.0, 1.0))
    if valid <= 0.0 and hard_zero:
        return {
            "quality": 0.0,
            "score": 0.0,
            "valid_ratio": 0.0,
            "smoothness": 0.0,
            "spread": 0.0,
            "range": 0.0,
            "relative_spread": 0.0,
        }

    mean_depth = max(float(values.get("depth_mean", values.get("range_mean", 1.0))), 1e-6)
    std_depth = max(float(values.get("depth_std", values.get("range_std", 0.0))), 0.0)
    span = max(float(values.get("depth_span", values.get("range_span", 0.0))), 0.0)
    gradient = max(
        float(
            values.get(
                "depth_mean_gradient",
                values.get("range_mean_abs_diff", 0.0),
            )
        ),
        0.0,
    )

    relative_spread = float(np.clip(span / (mean_depth + EPS), 0.0, 2.0))
    spread_score = float(np.clip(relative_spread / 1.2, 0.0, 1.0))
    noise_ratio = gradient / (mean_depth + EPS)
    smoothness = float(1.0 / (1.0 + 6.0 * noise_ratio))
    range_score = float(1.0 - np.exp(-mean_depth / 0.6))

    score = 0.50 * valid + 0.25 * smoothness + 0.15 * spread_score + 0.10 * range_score
    quality = sigmoid(5.0 * (score - 0.45))
    return {
        "quality": quality,
        "score": score,
        "valid_ratio": valid,
        "smoothness": smoothness,
        "spread": spread_score,
        "range": range_score,
        "relative_spread": relative_spread,
    }


def compute_quality(
    vision: np.ndarray,
    depth: np.ndarray,
    vision_names: list[str] | tuple[str, ...],
    depth_names: list[str] | tuple[str, ...],
    depth_hard_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n = vision.shape[0]
    qv = np.zeros(n, dtype=np.float64)
    qd = np.zeros(n, dtype=np.float64)
    v_rows = []
    d_rows = []
    for index in range(n):
        v = visual_health(vision[index], vision_names)
        d = depth_health(depth[index], depth_names, hard_zero=depth_hard_zero)
        qv[index] = v["quality"]
        qd[index] = d["quality"]
        v_rows.append(v)
        d_rows.append(d)
    diagnostics = {
        "vision": {
            key: float(np.mean([row[key] for row in v_rows]))
            for key in v_rows[0]
        },
        "depth": {
            key: float(np.mean([row[key] for row in d_rows]))
            for key in d_rows[0]
        },
    }
    return qv, qd, diagnostics


def modality_weights(
    qv: np.ndarray,
    qd: np.ndarray,
    beta: float = 4.0,
    hard_depth_dropout: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Temperature-scaled softmax reliability gating.

    By default this is exactly Eq. (5), which is the gate used by the offline
    NYU benchmark and the reported sweep.  ``hard_depth_dropout=True`` applies
    the additional Algorithm 2 rule ``wd = 0, wv = 1`` when
    ``compute_quality(..., depth_hard_zero=True)`` has set ``qd = 0`` for a
    frame with no valid depth pixel.
    """
    qv = np.asarray(qv, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    logits_v = beta * qv
    logits_d = beta * qd
    maximum = np.maximum(logits_v, logits_d)
    ev = np.exp(logits_v - maximum)
    ed = np.exp(logits_d - maximum)
    wv = ev / (ev + ed + EPS)
    wd = 1.0 - wv
    invalid_depth = (qd <= 0.0) if hard_depth_dropout else np.zeros(qd.shape, dtype=bool)
    if np.any(invalid_depth):
        wv = np.asarray(wv, dtype=np.float64).copy()
        wd = np.asarray(wd, dtype=np.float64).copy()
        wv[invalid_depth] = 1.0
        wd[invalid_depth] = 0.0
    return wv, wd


@dataclass
class ReliabilityWeightedKMeans:
    """K-Means with per-sample, per-view reliability weights.

    The two views are standardized and divided by sqrt(dimension) before
    clustering.  For a sample i and cluster k the assignment distance is

        w_vi ||z_vi - c_vk||^2 + w_di ||z_di - c_dk||^2.

    The centroid update uses the same weights, i.e. c_vk is a weighted mean of
    the visual descriptors assigned to cluster k.  This is the weighted-
    concatenation fusion rule without adding PCA coordinates from different
    sensor spaces.
    """

    n_clusters: int = 3
    beta: float = 4.0
    n_init: int = 20
    max_iter: int = 300
    tol: float = 1e-6
    seed: int = 42
    centroids_vision_: np.ndarray | None = None
    centroids_depth_: np.ndarray | None = None
    labels_: np.ndarray | None = None
    inertia_: float = float("inf")
    n_iter_: int = 0
    history_: list[float] = field(default_factory=list)

    def _scale_views(self, vision: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # The descriptor blocks are standardized per view before this point.
        # No additional dimension normalization is applied: doing so would
        # implicitly down-weight the more informative depth block.
        return np.asarray(vision, dtype=np.float64), np.asarray(depth, dtype=np.float64)

    def _distances(self, zv: np.ndarray, zd: np.ndarray, wv: np.ndarray, wd: np.ndarray) -> np.ndarray:
        assert self.centroids_vision_ is not None and self.centroids_depth_ is not None
        dv = ((zv[:, None, :] - self.centroids_vision_[None, :, :]) ** 2).sum(axis=2)
        dd = ((zd[:, None, :] - self.centroids_depth_[None, :, :]) ** 2).sum(axis=2)
        return wv[:, None] * dv + wd[:, None] * dd

    def _init_centroids(self, zv: np.ndarray, zd: np.ndarray, wv: np.ndarray, wd: np.ndarray, rng: np.random.RandomState) -> None:
        fused = np.hstack([np.sqrt(wv)[:, None] * zv, np.sqrt(wd)[:, None] * zd])
        labels = KMeans(
            n_clusters=self.n_clusters,
            random_state=int(rng.randint(0, 2**31 - 1)),
            n_init=1,
            max_iter=100,
        ).fit_predict(fused)
        self.centroids_vision_ = np.vstack(
            [
                np.average(zv[labels == k], axis=0, weights=wv[labels == k] + EPS)
                if np.any(labels == k)
                else zv[rng.randint(zv.shape[0])]
                for k in range(self.n_clusters)
            ]
        )
        self.centroids_depth_ = np.vstack(
            [
                np.average(zd[labels == k], axis=0, weights=wd[labels == k] + EPS)
                if np.any(labels == k)
                else zd[rng.randint(zd.shape[0])]
                for k in range(self.n_clusters)
            ]
        )

    def fit(
        self,
        vision: np.ndarray,
        depth: np.ndarray,
        qv: np.ndarray,
        qd: np.ndarray,
        hard_depth_dropout: bool = False,
    ) -> "ReliabilityWeightedKMeans":
        zv_raw, zd_raw = self._scale_views(vision, depth)
        wv, wd = modality_weights(
            qv, qd, beta=self.beta, hard_depth_dropout=hard_depth_dropout
        )
        rng = np.random.RandomState(self.seed)
        best = None
        for _ in range(max(1, self.n_init)):
            self._init_centroids(zv_raw, zd_raw, wv, wd, rng)
            previous = None
            history: list[float] = []
            for iteration in range(self.max_iter):
                distances = self._distances(zv_raw, zd_raw, wv, wd)
                labels = np.argmin(distances, axis=1)
                inertia = float(np.min(distances, axis=1).sum())
                history.append(inertia)
                if previous is not None and abs(previous - inertia) <= self.tol * max(1.0, abs(previous)):
                    break
                previous = inertia
                new_v = []
                new_d = []
                for cluster in range(self.n_clusters):
                    mask = labels == cluster
                    if not np.any(mask):
                        farthest = int(np.argmax(np.min(distances, axis=1)))
                        new_v.append(zv_raw[farthest].copy())
                        new_d.append(zd_raw[farthest].copy())
                        continue
                    new_v.append(np.average(zv_raw[mask], axis=0, weights=wv[mask] + EPS))
                    new_d.append(np.average(zd_raw[mask], axis=0, weights=wd[mask] + EPS))
                self.centroids_vision_ = np.vstack(new_v)
                self.centroids_depth_ = np.vstack(new_d)
            distances = self._distances(zv_raw, zd_raw, wv, wd)
            labels = np.argmin(distances, axis=1)
            inertia = float(np.min(distances, axis=1).sum())
            if best is None or inertia < best[0]:
                best = (
                    inertia,
                    self.centroids_vision_.copy(),
                    self.centroids_depth_.copy(),
                    labels.copy(),
                    iteration + 1,
                    history.copy(),
                )
        assert best is not None
        (
            self.inertia_,
            self.centroids_vision_,
            self.centroids_depth_,
            self.labels_,
            self.n_iter_,
            self.history_,
        ) = best
        return self

    def predict(
        self,
        vision: np.ndarray,
        depth: np.ndarray,
        qv: np.ndarray,
        qd: np.ndarray,
        hard_depth_dropout: bool = False,
    ) -> np.ndarray:
        zv_raw, zd_raw = self._scale_views(vision, depth)
        wv, wd = modality_weights(
            qv, qd, beta=self.beta, hard_depth_dropout=hard_depth_dropout
        )
        return np.argmin(self._distances(zv_raw, zd_raw, wv, wd), axis=1).astype(np.int64)

    def cluster_distances(
        self,
        vision: np.ndarray,
        depth: np.ndarray,
        qv: np.ndarray,
        qd: np.ndarray,
        hard_depth_dropout: bool = False,
    ) -> np.ndarray:
        zv_raw, zd_raw = self._scale_views(vision, depth)
        wv, wd = modality_weights(
            qv, qd, beta=self.beta, hard_depth_dropout=hard_depth_dropout
        )
        return self._distances(zv_raw, zd_raw, wv, wd)


def map_clusters_to_classes(
    cluster_labels: np.ndarray,
    target: np.ndarray,
    n_clusters: int,
    n_classes: int,
    fallback: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each cluster to a terrain class using labelled calibration samples.

    Majority vote is used, so several clusters may map to the same class.  If a
    cluster has no labelled samples it is mapped to the global majority class.
    """
    counts = np.zeros((n_clusters, n_classes), dtype=np.int64)
    for cluster, label in zip(cluster_labels, target):
        if 0 <= int(label) < n_classes:
            counts[int(cluster), int(label)] += 1
    mapping = np.zeros(n_clusters, dtype=np.int64)
    if fallback is None:
        fallback = int(np.argmax(np.bincount(target, minlength=n_classes)))
    for cluster in range(n_clusters):
        if counts[cluster].sum() > 0:
            mapping[cluster] = int(np.argmax(counts[cluster]))
        else:
            mapping[cluster] = int(fallback)
    return mapping, counts


def cluster_metrics(
    vision: np.ndarray,
    depth: np.ndarray,
    qv: np.ndarray,
    qd: np.ndarray,
    model: ReliabilityWeightedKMeans,
    target: np.ndarray,
    mapping: np.ndarray,
    n_classes: int,
) -> dict:
    clusters = model.predict(vision, depth, qv, qd)
    prediction = mapping[clusters]
    accuracy = float(accuracy_score(target, prediction))
    nmi = float(normalized_mutual_info_score(target, clusters))
    ari = float(adjusted_rand_score(target, clusters))
    try:
        silhouette = float(silhouette_score(np.hstack([vision, depth]), clusters))
    except Exception:
        silhouette = float("nan")
    recall = {}
    for label in range(n_classes):
        mask = target == label
        recall[str(label)] = float(np.mean(prediction[mask] == label)) if np.any(mask) else float("nan")
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, predicted in zip(target, prediction):
        confusion[int(true_label), int(predicted)] += 1
    return {
        "acc": accuracy,
        "nmi": nmi,
        "ari": ari,
        "silhouette": silhouette,
        "recall": recall,
        "confusion": confusion.tolist(),
        "cluster_counts": np.bincount(clusters, minlength=model.n_clusters).tolist(),
    }


def blocked_stratified_folds(
    target: np.ndarray,
    timestamps: np.ndarray,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous time-block folds, stratified by class.

    Each class is sorted by timestamp and divided into n_splits contiguous
    blocks.  Block b of every class is assigned to fold b.  This keeps samples
    from the same temporal neighbourhood on the same side of the split and
    reduces adjacent-frame leakage relative to random k-fold CV.
    """
    target = np.asarray(target)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    n_splits = max(2, int(n_splits))
    fold_blocks: list[list[int]] = [[] for _ in range(n_splits)]
    for label in np.unique(target):
        indices = np.where(target == label)[0]
        indices = indices[np.argsort(timestamps[indices])]
        chunks = np.array_split(indices, n_splits)
        for fold, chunk in enumerate(chunks):
            fold_blocks[fold].extend(int(index) for index in chunk)
    all_indices = np.arange(target.size)
    folds = []
    for fold in range(n_splits):
        test_indices = np.asarray(sorted(fold_blocks[fold]), dtype=np.int64)
        if test_indices.size == 0:
            continue
        mask = np.ones(target.size, dtype=bool)
        mask[test_indices] = False
        train_indices = all_indices[mask]
        folds.append((train_indices, test_indices))
    return folds


def random_stratified_folds(
    target: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(target.size), target))


def fit_cluster_readout(
    vision: np.ndarray,
    depth: np.ndarray,
    target: np.ndarray,
    n_clusters: int,
    n_classes: int,
    vision_names: list[str] | tuple[str, ...],
    depth_names: list[str] | tuple[str, ...],
    beta: float = 4.0,
    seed: int = 42,
    n_init: int = 20,
) -> dict:
    """Fit standardization, reliability-weighted clustering and cluster mapping.

    Standardization is fitted on the supplied training samples.  The returned
    payload is also used by the ROS perception node.
    """
    vision_scaler = StandardScaler().fit(vision)
    depth_scaler = StandardScaler().fit(depth)
    zv = vision_scaler.transform(vision)
    zd = depth_scaler.transform(depth)
    qv, qd, diagnostics = compute_quality(vision, depth, vision_names, depth_names)
    model = ReliabilityWeightedKMeans(
        n_clusters=n_clusters,
        beta=beta,
        n_init=n_init,
        seed=seed,
    ).fit(zv, zd, qv, qd)
    mapping, counts = map_clusters_to_classes(model.labels_, target, n_clusters, n_classes)
    return {
        "vision_scaler": vision_scaler,
        "depth_scaler": depth_scaler,
        "clusters": model,
        "cluster_to_class": mapping,
        "cluster_class_counts": counts,
        "quality_diagnostics": diagnostics,
        "n_clusters": n_clusters,
        "beta": beta,
        "vision_names": list(vision_names),
        "depth_names": list(depth_names),
    }


def predict_cluster_readout(
    payload: dict,
    vision: np.ndarray,
    depth: np.ndarray,
    vision_names: list[str] | tuple[str, ...] | None = None,
    depth_names: list[str] | tuple[str, ...] | None = None,
    hard_depth_dropout: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict terrain classes from a fitted cluster-readout payload.

    Returns (class_prediction, cluster, qv, qd, confidence).
    """
    vision = np.atleast_2d(np.asarray(vision, dtype=np.float64))
    depth = np.atleast_2d(np.asarray(depth, dtype=np.float64))
    if vision_names is None:
        vision_names = [f"v{index}" for index in range(vision.shape[1])]
    if depth_names is None:
        depth_names = [f"d{index}" for index in range(depth.shape[1])]
    zv = payload["vision_scaler"].transform(vision)
    zd = payload["depth_scaler"].transform(depth)
    qv, qd, _ = compute_quality(vision, depth, vision_names, depth_names)
    model = payload["clusters"]
    clusters = model.predict(zv, zd, qv, qd, hard_depth_dropout=hard_depth_dropout)
    mapping = payload["cluster_to_class"]
    prediction = mapping[clusters]
    distances = model.cluster_distances(zv, zd, qv, qd, hard_depth_dropout=hard_depth_dropout)
    shifted = distances - distances.min(axis=1, keepdims=True)
    probabilities = np.exp(-shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    confidence = probabilities[np.arange(probabilities.shape[0]), clusters]
    return prediction.astype(np.int64), clusters.astype(np.int64), qv, qd, confidence.astype(np.float64)


def select_k_silhouette(
    vision: np.ndarray,
    depth: np.ndarray,
    qv: np.ndarray,
    qd: np.ndarray,
    k_min: int = 2,
    k_max: int = 6,
    beta: float = 4.0,
    seed: int = 42,
) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    zv = StandardScaler().fit_transform(vision)
    zd = StandardScaler().fit_transform(depth)
    for k in range(k_min, k_max + 1):
        model = ReliabilityWeightedKMeans(n_clusters=k, beta=beta, n_init=10, seed=seed + k)
        model.fit(zv, zd, qv, qd)
        wv, wd = modality_weights(qv, qd, beta=beta)
        fused = np.hstack(
            [
                np.sqrt(wv)[:, None] * zv / np.sqrt(zv.shape[1]),
                np.sqrt(wd)[:, None] * zd / np.sqrt(zd.shape[1]),
            ]
        )
        try:
            scores[k] = float(silhouette_score(fused, model.labels_))
        except Exception:
            scores[k] = float("nan")
    best = max(scores, key=lambda key: scores[key] if np.isfinite(scores[key]) else -np.inf)
    return int(best), scores
