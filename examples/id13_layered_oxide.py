"""Phase mapping of an ESRF ID13 scanning micro-XRD map of a P2/P3 layered oxide.

Usage:
    python examples/id13_layered_oxide.py path/to/dq_*.h5 [--out results_private]

The raw HDF5 files are not part of this repository.

What it does
  1. loads the ewoks/pyFAI integrated map (one I(q) per scan point)
  2. SNIP background removal and noise-aware particle masking
  3. PCA + KMeans on sqrt-scaled, L2-normalised peak patterns, excluding the
     texture-dominated 00l reflection shared by P2 and P3
  4. automatic choice of k by silhouette score
  5. validation against the classical ROI approach: P2/(P2+P3) marker-peak
     fraction per pixel vs. cluster assignment
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xrdcluster import PhaseMapper, load_ewoks_h5, phase_fraction
from xrdcluster.plotting import PALETTE, plot_embedding, plot_label_map, plot_mean_patterns

# Marker reflections (q in 1/A) that belong to only one polytype, from the
# hexagonal cells P2 (a=2.88, c=11.1 A) and P3 (a=2.88, c=16.8 A).
MARKERS = {"P2": [2.766, 3.040], "P3": [2.635, 3.146]}
LINES = {"P3 012": 2.635, "P2 102": 2.766, "P2 103": 3.040, "P3 015": 3.146}
BASAL = (1.0, 1.25)  # P2 002 / P3 003, strongly affected by crystallite orientation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--shape", type=int, nargs=2)
    ap.add_argument("--out", default="results_private")
    ap.add_argument("-k", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.h5))[0]

    xmap = load_ewoks_h5(a.h5, shape=a.shape)
    mapper = PhaseMapper(q_range=(0.9, 6.5), exclude_q=[BASAL], snip_iterations=40,
                         n_components=10, n_clusters=a.k, method="kmeans", n_nmf=3)
    res = mapper.fit(xmap)
    print(res.summary())

    frac = phase_fraction(res.peaks, res.q, MARKERS)["P2"]
    fg = res.foreground
    stats = {}
    for c in range(res.n_clusters):
        sel = res.labels == c
        stats[c] = dict(n=int(sel.sum()),
                        p2_fraction_median=float(np.nanmedian(frac[sel])),
                        p2_fraction_iqr=[float(v) for v in np.nanpercentile(frac[sel], [25, 75])])
        print(f"cluster {c}: n={stats[c]['n']:6d}  median P2/(P2+P3) = {stats[c]['p2_fraction_median']:.2f}")
    r = np.corrcoef(res.embedding[:, 0], np.nan_to_num(frac[fg]))[0, 1]
    print(f"corr(PC1, ROI P2 fraction) = {r:.2f}")

    # --------------------------------------------------------------- figure
    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])
    ax_map, ax_frac, ax_pat = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax_emb, ax_box, ax_zoom = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])

    plot_label_map(res, ax_map, title=f"KMeans clusters (k={res.n_clusters})")
    img = res.xmap.to_image(np.where(fg, frac, np.nan))
    im = ax_frac.imshow(img, origin="lower", cmap="coolwarm", vmin=0, vmax=0.7, interpolation="nearest")
    fig.colorbar(im, ax=ax_frac, shrink=0.75, label="P2 / (P2 + P3)")
    ax_frac.set_title("ROI reference: marker-peak ratio")
    ax_frac.set_xticks([]); ax_frac.set_yticks([])
    plot_mean_patterns(res, ax_pat)
    ax_pat.set_title("cluster mean patterns (grey: excluded 00l)")
    plot_mean_patterns(res, ax_zoom, xlim=(2.4, 3.3), markers=LINES, legend=False)
    ax_zoom.set_title("zoom on P2 / P3 marker reflections")
    plot_embedding(res, ax_emb, color=np.nan_to_num(frac[fg]), cmap="coolwarm", label="P2 fraction")
    ax_emb.set_title(f"PCA embedding (r(PC1, P2 frac) = {r:.2f})")
    data = [frac[res.labels == c][~np.isnan(frac[res.labels == c])] for c in range(res.n_clusters)]
    bp = ax_box.boxplot(data, showfliers=False, patch_artist=True)
    for c, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[c % len(PALETTE)]); patch.set_alpha(0.6)
    ax_box.set_xticklabels([str(c) for c in range(res.n_clusters)])
    ax_box.set_xlabel("cluster"); ax_box.set_ylabel("P2 / (P2 + P3)")
    ax_box.set_title("ROI ratio per cluster")
    fig.suptitle(f"{stem}  ({res.xmap.shape[0]}x{res.xmap.shape[1]} points, "
                 f"{int(fg.sum())} on particles)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"{stem}_phase_map.png"), dpi=150)

    # NMF figure: identify which component is P2-like / P3-like from marker peaks
    H, W = res.nmf_patterns, res.nmf_maps
    def marker_height(h, qs):
        return sum(h[np.argmin(abs(res.q - qm))] for qm in qs)
    score = np.array([marker_height(h, MARKERS["P2"]) - marker_height(h, MARKERS["P3"]) for h in H]) / H.max(1)
    i_p2, i_p3 = int(np.argmax(score)), int(np.argmin(score))
    names = {i_p2: "P2-like", i_p3: "P3-like"}
    k = H.shape[0]
    fig2, ax2 = plt.subplots(2, k + 1, figsize=(4 * (k + 1), 7.5))
    for j in range(k):
        m = res.xmap.to_image(W[:, j])
        ax2[0, j].imshow(m, origin="lower", cmap="magma", vmax=np.nanpercentile(m, 99), interpolation="nearest")
        ax2[0, j].set_title(f"NMF {j}: {names.get(j, 'other')}")
        ax2[0, j].set_xticks([]); ax2[0, j].set_yticks([])
        ax2[1, j].plot(res.q, H[j] / H[j].max(), lw=0.8, color=PALETTE[j])
        for name, qm in LINES.items():
            ax2[1, j].axvline(qm, color="grey", ls=":", lw=0.7)
        ax2[1, j].set_xlim(2.2, 3.5); ax2[1, j].set_xlabel(r"q ($\AA^{-1}$)")
    # colour-blind-safe blend: orange = P2-like share, blue = P3-like share
    Wf = np.nan_to_num(W / np.nansum(W, 1, keepdims=True))
    share = Wf[:, i_p2] / np.clip(Wf[:, i_p2] + Wf[:, i_p3], 1e-9, None)
    c_p2, c_p3 = np.array([0.835, 0.369, 0.0]), np.array([0.0, 0.447, 0.698])
    rgb = np.ones((res.xmap.n_points, 3))
    rgb[fg] = share[fg, None] * c_p2 + (1 - share[fg, None]) * c_p3
    ax2[0, k].imshow(rgb.reshape(*res.xmap.shape, 3), origin="lower", interpolation="nearest")
    ax2[0, k].set_title("NMF share: orange = P2, blue = P3"); ax2[0, k].set_xticks([]); ax2[0, k].set_yticks([])
    x = share[fg]
    ok = ~np.isnan(frac[fg])
    r_nmf = np.corrcoef(x[ok], frac[fg][ok])[0, 1]
    ax2[1, k].hexbin(frac[fg][ok], x[ok], gridsize=40, cmap="Greys", mincnt=1)
    ax2[1, k].set_xlabel("ROI  P2/(P2+P3)"); ax2[1, k].set_ylabel("NMF  P2/(P2+P3)")
    ax2[1, k].set_title(f"NMF vs ROI, r = {r_nmf:.2f}")
    fig2.tight_layout()
    fig2.savefig(os.path.join(a.out, f"{stem}_nmf.png"), dpi=130)
    print(f"corr(NMF P2 share, ROI P2 fraction) = {r_nmf:.2f}")

    # silhouette scan
    if res.k_scan:
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ks = sorted(res.k_scan)
        ax3.plot(ks, [res.k_scan[k] for k in ks], "o-", color=PALETTE[0])
        ax3.set_xlabel("k"); ax3.set_ylabel("silhouette score"); fig3.tight_layout()
        fig3.savefig(os.path.join(a.out, f"{stem}_kscan.png"), dpi=130)

    with open(os.path.join(a.out, f"{stem}_summary.json"), "w") as fh:
        json.dump({"clusters": stats, "k_scan": res.k_scan, "corr_pc1_p2fraction": r,
                   "corr_nmf_p2share_roi": r_nmf,
                   "foreground_points": int(fg.sum()), "shape": list(res.xmap.shape)}, fh, indent=2)


if __name__ == "__main__":
    main()
