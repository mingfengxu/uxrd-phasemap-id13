"""Benchmark on a synthetic map with known ground truth (no beamline data needed).

    python examples/synthetic_demo.py            # writes docs/figures/synthetic_demo.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from uxrd_phasemap_id13 import PhaseMapper, make_synthetic_map
from uxrd_phasemap_id13.plotting import cluster_cmap, plot_mean_patterns

xmap, truth = make_synthetic_map(shape=(100, 100), n_particles=90, impurity_particles=4,
                                 texture=0.8, seed=7)

results = {}
for name, excl in [("all q", []), ("00l excluded", [(1.0, 1.25)])]:
    res = PhaseMapper(exclude_q=excl, n_clusters=[2, 3, 4, 5], snip_iterations=30).fit(xmap)
    both = res.foreground & (truth >= 0)
    ari = adjusted_rand_score(truth[both], res.labels[both])
    results[name] = (res, ari)
    print(f"{name:14s} k={res.n_clusters}  ARI={ari:.3f}  silhouette scan={res.k_scan}")

res, ari = results["00l excluded"]
fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
ax[0].imshow(truth.reshape(xmap.shape) + 1, origin="lower", cmap=cluster_cmap(3),
             vmin=-0.5, vmax=3.5, interpolation="nearest")
ax[0].set_title("ground truth (P2, P3, impurity)"); ax[0].set_xticks([]); ax[0].set_yticks([])
ax[1].plot(xmap.q, xmap.intensity[np.flatnonzero(truth == 0)[5]], lw=0.7, color="k")
ax[1].set_title("one raw pattern (Poisson noise + background)"); ax[1].set_xlabel(r"q ($\AA^{-1}$)")
# colour each recovered cluster like the ground-truth phase it overlaps most (display only)
match = {c: np.bincount(truth[(res.labels == c) & (truth >= 0)], minlength=3).argmax()
         for c in range(res.n_clusters)}
shown = np.array([match.get(l, -1) if l >= 0 else -1 for l in res.labels])
ax[2].imshow(shown.reshape(xmap.shape) + 1, origin="lower", cmap=cluster_cmap(3),
             vmin=-0.5, vmax=3.5, interpolation="nearest")
ax[2].set_title(f"recovered (k auto = {res.n_clusters}, ARI = {ari:.2f})")
ax[2].set_xticks([]); ax[2].set_yticks([])
plot_mean_patterns(res, ax[3])
ax[3].set_title(f"cluster means; with 00l kept ARI = {results['all q'][1]:.2f}")
fig.tight_layout()
os.makedirs("docs/figures", exist_ok=True)
fig.savefig("docs/figures/synthetic_demo.png", dpi=130)
