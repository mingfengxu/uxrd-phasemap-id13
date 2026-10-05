"""Nano-beam scan of a cycled electrode particle: reference background, NMF, lattice map.

    python examples/nanobeam_particle.py PROCESSED_0001.h5 [--out results_private/nanobeam]

Written for an ID13 nano-branch map of a cycled O'3-NaNiO2 electrode in an in-situ
cell (glassy-carbon window, GF/A separator, carbonate electrolyte, Al current
collector), but the steps are generic:

1. **Reference background**: patterns without Bragg peaks give a measured cell
   background (window, electrolyte, separator, foil). A scaled copy is subtracted
   from every pattern, which removes narrow uniform peaks that SNIP would keep.
2. **Particle selection**: pixels with a strong basal (00l) layered-oxide peak,
   cleaned with a morphological opening, largest connected region.
3. **NMF on the particle only**, with the Al current-collector reflections excluded.
4. **Lattice map**: a per-pixel Gaussian fit of the basal reflection gives its d-spacing
   with sub-bin precision. With a ~50 nm beam each pixel sees one or a few
   crystallites, so continuous d shifts (Na content, strain) appear as domains.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.decomposition import NMF

from uxrd_phasemap_id13 import (foreground_mask, load_ewoks_h5, peak_position_map,
                                reference_background, snip_baseline)

AL = [2.687, 3.103, 4.389, 5.146, 5.375, 6.206]           # fcc Al, a = 4.0495 A
BASAL_WINDOW = (1.10, 1.22)                                 # 00l of O-type and P-type NaxNiO2
P3_FIT = (1.095, 1.175)                                     # P3-type basal reflection
O_TYPE = (1.183, 1.205)                                     # O-type basal reflection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--out", default="results_private/nanobeam")
    ap.add_argument("--nmf", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    xmap = load_ewoks_h5(a.h5).crop_q(0.9, 6.5)
    q, I = xmap.q, xmap.intensity.astype(float)
    base = snip_baseline(I, 40)
    fg, _ = foreground_mask(np.clip(I - base, 0, None), baseline=base)
    resid, scale, ref = reference_background(I, ~fg)
    P = np.clip(resid - snip_baseline(np.clip(resid, 0, None), 40), 0, None)
    print(f"{(~fg).sum()} sample-free reference points; scale factor 1-99 %: "
          f"{np.percentile(scale, 1):.3f}-{np.percentile(scale, 99):.3f}")

    basal = P[:, (q > BASAL_WINDOW[0]) & (q < BASAL_WINDOW[1])].sum(1)
    thr = np.median(basal) + 10 * 1.4826 * np.median(np.abs(basal - np.median(basal)))
    img = ndimage.binary_dilation(ndimage.binary_opening(xmap.to_image(basal > thr)), iterations=3)
    lab, n = ndimage.label(img)
    part = (lab == 1 + int(np.argmax(ndimage.sum(img, lab, range(1, n + 1))))).ravel()
    print(f"particle: {part.sum()} pixels")

    feat = np.ones(q.size, bool)
    for c in AL:
        feat &= ~((q > c - 0.025) & (q < c + 0.025))
    nmf = NMF(a.nmf, init="nndsvda", max_iter=3000, random_state=0)
    W = nmf.fit_transform(P[part][:, feat])
    H = np.zeros((a.nmf, q.size)); H[:, feat] = nmf.components_

    fit = peak_position_map(P, q, *P3_FIT, points=part, min_height=15)
    ok = np.isfinite(fit["d"])
    o_amp = P[:, (q > O_TYPE[0]) & (q < O_TYPE[1])].sum(1)
    summary = {"particle_pixels": int(part.sum()), "fitted_pixels": int(ok.sum()),
               "p3_d_median": float(np.nanmedian(fit["d"])),
               "p3_d_5_95": [float(v) for v in np.nanpercentile(fit["d"], [5, 95])],
               "p3_q_err_median": float(np.nanmedian(fit["q0_err"])),
               "nmf_shares": [float(v) for v in W.sum(0) / W.sum()]}
    print(json.dumps(summary, indent=2))

    rows, cols = np.where(part.reshape(xmap.shape))
    crop = lambda v: v.reshape(xmap.shape)[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    k = a.nmf
    fig, ax = plt.subplots(2, k + 1, figsize=(4.2 * (k + 1), 8))
    for j in range(k):
        m = np.full(xmap.n_points, np.nan); m[part] = W[:, j]
        ax[0, j].imshow(crop(m), origin="lower", cmap="magma", interpolation="nearest")
        ax[0, j].set_title(f"NMF {j} ({summary['nmf_shares'][j] * 100:.0f} %)")
        ax[1, j].plot(q, H[j] / H[j].max(), lw=0.7, color=f"C{j}")
        ins = ax[1, j].inset_axes([0.45, 0.45, 0.5, 0.5])
        ins.plot(q, H[j] / H[j].max(), lw=0.8, color=f"C{j}"); ins.set_xlim(1.08, 1.25)
        ins.tick_params(labelsize=6); ins.set_title("basal region", fontsize=7)
        ax[1, j].set_xlabel(r"q ($\AA^{-1}$)")
    d = np.where(ok, fit["d"], np.nan)
    im = ax[0, k].imshow(crop(d), origin="lower", cmap="coolwarm", interpolation="nearest",
                         vmin=np.nanpercentile(d, 3), vmax=np.nanpercentile(d, 97))
    fig.colorbar(im, ax=ax[0, k], label="P3 d(003) (Å)"); ax[0, k].set_title("P3 interlayer spacing")
    om = np.where(part, o_amp, np.nan)
    im = ax[1, k].imshow(crop(om), origin="lower", cmap="Blues", interpolation="nearest",
                         vmin=0, vmax=np.nanpercentile(om, 99.5))
    fig.colorbar(im, ax=ax[1, k]); ax[1, k].set_title("O-type basal (q≈1.19)")
    for axx in list(ax[0]) + [ax[1, k]]:
        axx.set_xticks([]); axx.set_yticks([])
    fig.suptitle(os.path.basename(a.h5) + " — particle analysis")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "particle_analysis.png"), dpi=110)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
