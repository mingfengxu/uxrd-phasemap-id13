"""Command line interface: ``xrdcluster map.h5 -k 3 4 5 --out results/``."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _parse_window(text: str):
    lo, hi = (float(v) for v in text.split(":"))
    return lo, hi


def main(argv=None):
    p = argparse.ArgumentParser(description="Unsupervised phase mapping of scanning micro-XRD data")
    p.add_argument("h5", help="ewoks/pyFAI integrated HDF5 file")
    p.add_argument("--shape", type=int, nargs=2, metavar=("ROWS", "COLS"), help="map shape (default: square)")
    p.add_argument("--qrange", type=_parse_window, default=(0.9, 6.5), help="qmin:qmax in 1/A (default 0.9:6.5)")
    p.add_argument("--exclude", type=_parse_window, action="append", default=[],
                   help="q window qmin:qmax removed from features; repeatable")
    p.add_argument("-k", "--clusters", type=int, nargs="+", default=[4], help="k, or several k to auto-select")
    p.add_argument("--method", choices=["kmeans", "gmm"], default="kmeans")
    p.add_argument("--pca", type=int, default=10, help="number of PCA components")
    p.add_argument("--nmf", type=int, default=0, help="number of NMF components (0 = skip)")
    p.add_argument("--snip", type=int, default=40, help="SNIP iterations (q bins)")
    p.add_argument("--out", default="results", help="output directory")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    from .cluster import PhaseMapper
    from .io import load_ewoks_h5
    from .plotting import overview_figure

    xmap = load_ewoks_h5(a.h5, shape=a.shape)
    mapper = PhaseMapper(q_range=a.qrange, exclude_q=a.exclude, snip_iterations=a.snip,
                         n_components=a.pca, n_clusters=a.clusters if len(a.clusters) > 1 else a.clusters[0],
                         method=a.method, n_nmf=a.nmf)
    res = mapper.fit(xmap)
    print(res.summary())

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.h5))[0]
    np.savetxt(os.path.join(a.out, f"{stem}_labels.txt"), res.label_image(), fmt="%d")
    np.savetxt(os.path.join(a.out, f"{stem}_cluster_means.csv"),
               np.column_stack([res.q, res.mean_patterns().T]), delimiter=",",
               header="q," + ",".join(f"cluster{c}" for c in range(res.n_clusters)), comments="")
    with open(os.path.join(a.out, f"{stem}_summary.json"), "w") as fh:
        json.dump({"shape": list(res.xmap.shape), "k_scan": res.k_scan,
                   "cluster_sizes": res.cluster_sizes().tolist(),
                   "foreground_points": int(res.foreground.sum()),
                   "explained_variance": res.explained_variance.tolist()}, fh, indent=2)
    overview_figure(res, suptitle=stem).savefig(os.path.join(a.out, f"{stem}_overview.png"), dpi=150)
    print(f"results written to {a.out}/")


if __name__ == "__main__":
    main()
