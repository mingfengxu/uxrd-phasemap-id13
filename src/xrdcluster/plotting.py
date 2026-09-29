"""Figures for phase-mapping results."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .cluster import PhaseMapResult

# Okabe-Ito, colour-blind safe; background drawn in light grey
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]
BG_COLOR = "#EEEEEE"


def cluster_cmap(k: int) -> ListedColormap:
    return ListedColormap([BG_COLOR] + [PALETTE[i % len(PALETTE)] for i in range(k)])


def plot_label_map(result: PhaseMapResult, ax=None, title: str = "Cluster map"):
    ax = ax or plt.gca()
    k = result.n_clusters
    img = result.label_image() + 1  # background -> 0
    ax.imshow(img, origin="lower", cmap=cluster_cmap(k), vmin=-0.5, vmax=k + 0.5,
              interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def plot_mean_patterns(result: PhaseMapResult, ax=None, offset: float = 1.1,
                       xlim: Optional[Sequence[float]] = None,
                       markers: Optional[Dict[str, Sequence[float]]] = None, legend: bool = True):
    """Stacked, max-normalised mean pattern of every cluster.

    ``markers`` draws labelled vertical lines, e.g. {"P2 102": 2.766}.
    """
    ax = ax or plt.gca()
    q = result.q
    means = result.mean_patterns()
    sizes = result.cluster_sizes()
    sel = result.feature_q if xlim is None else (q >= xlim[0]) & (q <= xlim[1])
    for c, m in enumerate(means):
        norm = m[sel & result.feature_q].max() or 1.0
        ax.plot(q, m / norm + c * offset, color=PALETTE[c % len(PALETTE)], lw=1,
                label=f"cluster {c} (n={sizes[c]})")
    # shade q windows that were excluded from the clustering features
    excl = ~result.feature_q
    if excl.any():
        edges = np.flatnonzero(np.diff(np.r_[0, excl.astype(int), 0]))
        for lo, hi in zip(edges[::2], edges[1::2]):
            ax.axvspan(q[lo], q[hi - 1], color="grey", alpha=0.15, lw=0)
    if markers:
        trans = ax.get_xaxis_transform()  # x in data units, y in axes units
        for name, qm in markers.items():
            ax.axvline(qm, color="grey", ls=":", lw=0.8)
            ax.text(qm, 0.98, name, transform=trans, rotation=90, va="top", ha="right",
                    fontsize=7, color="grey")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(-0.1, len(means) * offset + 0.3)
    ax.set_xlabel(r"q ($\AA^{-1}$)")
    ax.set_ylabel("normalised intensity (offset)")
    if legend:
        ax.legend(fontsize=7, loc="upper right", frameon=False)
    return ax


def plot_embedding(result: PhaseMapResult, ax=None, color=None, cmap="viridis", label=None):
    """PC1 vs PC2 scatter of foreground points, coloured by cluster or a value."""
    ax = ax or plt.gca()
    Z = result.embedding
    if color is None:
        lab = result.labels[result.foreground]
        c = [PALETTE[i % len(PALETTE)] for i in lab]
        ax.scatter(Z[:, 0], Z[:, 1], s=1, c=c, rasterized=True)
    else:
        sc = ax.scatter(Z[:, 0], Z[:, 1], s=1, c=color, cmap=cmap, rasterized=True)
        plt.colorbar(sc, ax=ax, label=label, shrink=0.8)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    return ax


def overview_figure(result: PhaseMapResult, markers=None, zoom=None, value_map=None,
                    value_label: str = "", suptitle: str = ""):
    """2x2 overview: cluster map, mean patterns, zoom on patterns, and either a
    reference value map (e.g. ROI phase fraction) or the PCA embedding."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plot_label_map(result, axes[0, 0])
    plot_mean_patterns(result, axes[0, 1], markers=markers)
    plot_mean_patterns(result, axes[1, 1], xlim=zoom or (result.q.min(), result.q.max()), markers=markers)
    axes[1, 1].set_title("zoom")
    if value_map is not None:
        img = result.xmap.to_image(np.where(result.foreground, value_map, np.nan))
        im = axes[1, 0].imshow(img, origin="lower", cmap="coolwarm", interpolation="nearest")
        plt.colorbar(im, ax=axes[1, 0], shrink=0.8, label=value_label)
        axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
        axes[1, 0].set_title(value_label)
    else:
        plot_embedding(result, axes[1, 0])
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig
