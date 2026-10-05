"""Unsupervised phase mapping pipeline.

raw I(q) map ─► q-crop ─► SNIP background ─► foreground mask
            ─► features (sqrt, exclude texture-sensitive peaks, L2 norm)
            ─► PCA ─► KMeans / Gaussian mixture ─► phase map + mean patterns
            └► (optional) NMF on peak patterns ─► component maps
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from .io import XRDMap
from .preprocess import build_features, foreground_mask, q_mask, reference_background, snip_baseline

BACKGROUND = -1


@dataclass
class PhaseMapResult:
    """Everything produced by :meth:`PhaseMapper.fit`."""

    xmap: XRDMap                 # the (q-cropped) input map
    peaks: np.ndarray            # background-subtracted patterns (n_points, n_q)
    foreground: np.ndarray       # bool (n_points,)
    threshold: float             # foreground threshold (integrated significant counts)
    feature_q: np.ndarray        # bool (n_q,), q bins used as features
    embedding: np.ndarray        # PCA scores of foreground points
    explained_variance: np.ndarray
    labels: np.ndarray           # (n_points,), BACKGROUND for empty points
    probabilities: Optional[np.ndarray] = None   # GMM soft assignment (n_fg, k)
    k_scan: Dict[int, float] = field(default_factory=dict)
    nmf_maps: Optional[np.ndarray] = None        # (n_points, n_nmf), NaN outside fg
    nmf_patterns: Optional[np.ndarray] = None    # (n_nmf, n_q)

    @property
    def n_clusters(self) -> int:
        return int(self.labels.max()) + 1

    @property
    def q(self) -> np.ndarray:
        return self.xmap.q

    def label_image(self) -> np.ndarray:
        return self.xmap.to_image(self.labels)

    def cluster_sizes(self) -> np.ndarray:
        return np.bincount(self.labels[self.labels >= 0], minlength=self.n_clusters)

    def mean_patterns(self) -> np.ndarray:
        """Average background-subtracted pattern of each cluster (k, n_q)."""
        return np.stack([self.peaks[self.labels == c].mean(0) for c in range(self.n_clusters)])

    def summary(self) -> str:
        lines = [
            f"map {self.xmap.shape[0]}x{self.xmap.shape[1]}, {self.xmap.intensity.shape[1]} q-bins "
            f"({self.q.min():.2f}-{self.q.max():.2f} 1/A)",
            f"foreground: {self.foreground.sum()} / {self.foreground.size} points "
            f"(threshold {self.threshold:.0f} counts)",
            "PCA explained variance (first 5): "
            + ", ".join(f"{v:.3f}" for v in self.explained_variance[:5]),
        ]
        if self.k_scan:
            lines.append("model selection: " + ", ".join(f"k={k}: {s:.3f}" for k, s in self.k_scan.items()))
        for c, n in enumerate(self.cluster_sizes()):
            lines.append(f"  cluster {c}: {n} points")
        return "\n".join(lines)


class PhaseMapper:
    """Configurable unsupervised phase-mapping pipeline.

    Parameters
    ----------
    q_range : (qmin, qmax)
        q window kept for analysis (drop beam-stop shadow / noisy high q).
    exclude_q : list of (qmin, qmax)
        Windows removed from the *features* (still kept in outputs). Useful for
        strongly texture-dependent reflections shared by all phases, e.g. the
        00l peak of layered oxides.
    snip_iterations : int
        SNIP half-width in q bins.
    n_components : int
        Number of PCA components used for clustering.
    n_clusters : int or sequence of int
        Fixed k, or candidates for automatic selection (silhouette for
        KMeans, BIC for GMM).
    method : "kmeans" | "gmm"
    n_nmf : int
        If > 0, additionally factorise foreground peak patterns with NMF.
    background : "snip" | "reference"
        ``snip``: mathematical baseline per pattern (default).
        ``reference``: subtract a scaled median pattern of the sample-free
        points (found with a first SNIP pass), which also removes narrow but
        spatially uniform signals from cell windows, electrolyte or foils that
        SNIP would keep as "peaks"; a light SNIP then removes what is left.
    """

    def __init__(
        self,
        q_range: Tuple[float, float] = (0.0, np.inf),
        exclude_q: Sequence[Tuple[float, float]] = (),
        snip_iterations: int = 40,
        foreground_threshold: Optional[float] = None,
        transform: str = "sqrt",
        n_components: int = 10,
        n_clusters=4,
        method: str = "kmeans",
        n_nmf: int = 0,
        silhouette_sample: int = 5000,
        random_state: int = 0,
        background: str = "snip",
    ):
        if method not in ("kmeans", "gmm"):
            raise ValueError("method must be 'kmeans' or 'gmm'")
        self.q_range = q_range
        self.exclude_q = list(exclude_q)
        self.snip_iterations = snip_iterations
        self.foreground_threshold = foreground_threshold
        self.transform = transform
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.method = method
        self.n_nmf = n_nmf
        self.silhouette_sample = silhouette_sample
        self.random_state = random_state
        if background not in ("snip", "reference"):
            raise ValueError("background must be 'snip' or 'reference'")
        self.background = background

    # ------------------------------------------------------------------ #
    def _cluster(self, Z: np.ndarray, k: int):
        if self.method == "kmeans":
            model = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(Z)
            return model.labels_, None, model
        model = GaussianMixture(n_components=k, covariance_type="full", n_init=3,
                                random_state=self.random_state).fit(Z)
        return model.predict(Z), model.predict_proba(Z), model

    def _score(self, Z, labels, model) -> float:
        if self.method == "gmm":
            return float(model.bic(Z))  # lower is better
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(Z), min(self.silhouette_sample, len(Z)), replace=False)
        if len(np.unique(labels[idx])) < 2:
            return -1.0
        return float(silhouette_score(Z[idx], labels[idx]))  # higher is better

    @staticmethod
    def _relabel_by_size(labels: np.ndarray, proba: Optional[np.ndarray]):
        order = np.argsort(-np.bincount(labels))
        remap = np.empty_like(order)
        remap[order] = np.arange(order.size)
        return remap[labels], (proba[:, order] if proba is not None else None)

    # ------------------------------------------------------------------ #
    def fit(self, xmap: XRDMap, points: Optional[np.ndarray] = None) -> PhaseMapResult:
        """``points`` (bool per scan point) restricts the analysis to a region,
        e.g. one particle; the background reference still uses the whole map."""
        xmap = xmap.crop_q(*self.q_range)
        baseline = snip_baseline(xmap.intensity, self.snip_iterations)
        peaks = np.clip(xmap.intensity - baseline, 0, None)
        fg, thr = foreground_mask(peaks, self.foreground_threshold, baseline=baseline)
        self.reference_ = None
        if self.background == "reference":
            resid, scale, ref = reference_background(xmap.intensity, ~fg)
            light = snip_baseline(np.clip(resid, 0, None), self.snip_iterations)
            peaks = np.clip(resid - light, 0, None)
            self.reference_ = {"pattern": ref, "scale": scale, "points": ~fg}
        if points is not None:
            fg = fg & np.asarray(points, bool)
        if fg.sum() < 10:
            raise RuntimeError("fewer than 10 foreground points; lower the threshold")
        keep = q_mask(xmap.q, self.exclude_q)
        X = build_features(peaks[fg], keep, self.transform)

        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=self.random_state)
        Z = pca.fit_transform(X)

        candidates: List[int] = [self.n_clusters] if np.isscalar(self.n_clusters) else list(self.n_clusters)
        scan: Dict[int, float] = {}
        best = None
        for k in candidates:
            labels, proba, model = self._cluster(Z, int(k))
            score = self._score(Z, labels, model) if len(candidates) > 1 else np.nan
            scan[int(k)] = score
            better = best is None or (score < best[0] if self.method == "gmm" else score > best[0])
            if better:
                best = (score, labels, proba)
        _, labels_fg, proba = best
        labels_fg, proba = self._relabel_by_size(labels_fg, proba)

        labels = np.full(xmap.n_points, BACKGROUND, dtype=int)
        labels[fg] = labels_fg

        result = PhaseMapResult(
            xmap=xmap, peaks=peaks, foreground=fg, threshold=thr, feature_q=keep,
            embedding=Z, explained_variance=pca.explained_variance_ratio_,
            labels=labels, probabilities=proba,
            k_scan=scan if len(candidates) > 1 else {},
        )
        if self.n_nmf > 0:
            nmf = NMF(n_components=self.n_nmf, init="nndsvda", max_iter=1000,
                      random_state=self.random_state)
            W = nmf.fit_transform(peaks[fg][:, keep])
            maps = np.full((xmap.n_points, self.n_nmf), np.nan)
            maps[fg] = W
            H = np.zeros((self.n_nmf, xmap.q.size))
            H[:, keep] = nmf.components_
            result.nmf_maps, result.nmf_patterns = maps, H
        return result
