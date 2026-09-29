# xrdcluster: unsupervised phase mapping for scanning micro-XRD

[![tests](https://github.com/<your-username>/xrdmap-cluster/actions/workflows/tests.yml/badge.svg)](https://github.com/<your-username>/xrdmap-cluster/actions)

In a synchrotron scanning micro-diffraction experiment you record one 1D pattern
`I(q)` at every beam position, which quickly adds up to tens of thousands of patterns
per map. The classical analysis integrates a few hand-picked reflections (region-of-interest
maps). That works when you already know which phases to look for, but it ignores
most of the pattern and misses anything unexpected.

`xrdcluster` is a small, tested Python package that works on **every pattern in the map
without supervision**:

```
raw I(q) map ─► q-crop ─► SNIP background ─► noise-aware particle mask
            ─► features: sqrt (variance-stabilising) + L2 normalisation
               (texture-dominated reflections can be excluded)
            ─► PCA ─► KMeans / Gaussian mixture (k chosen by silhouette / BIC)
            └► NMF on peak patterns ─► non-negative "phase" patterns + abundance maps
```

It reads the HDF5 files written by the ESRF **ewoks / pyFAI** integration pipeline
(e.g. ID13) directly, and generic arrays through `XRDMap`.

## Results

### 1. Synthetic benchmark (known ground truth)

Two layered-oxide polytypes (P2, P3) and a minor impurity. The peak positions are
calculated from their hexagonal cells. The simulation adds a sloping amorphous
background, Poisson noise, particle-thickness variation and a randomly re-weighted
00l reflection to mimic crystallite orientation.

![synthetic](docs/figures/synthetic_demo.png)

| features | k chosen by silhouette | adjusted Rand index |
|---|---|---|
| all q | 2 (impurity merged away) | 0.97 |
| 00l window excluded | **3** | **1.00** |

When the orientation-dependent basal reflection dominates the feature vector, the
small impurity population is lost. Removing that window recovers all three phases.
`python examples/synthetic_demo.py` reproduces this.

### 2. Real data: P2/P3 Na-layered oxide, ESRF ID13 (Eiger 4M, λ = 0.954 Å)

Scanning micro-XRD maps of Li-substituted Na-layered-oxide particles embedded in a
matrix. Each map has 200 × 200 or 250 × 250 points, and each point has 1400 q bins.

**NMF finds the two polytypes without being told what they are.** With three
components, one component carries the P3 reflections (012, 015), one the P2
reflections (102, 103), and the third collects everything else (matrix, minor
phase(s)). Their relative abundance agrees with the classical marker-peak ratio
P2/(P2+P3) at **r = 0.91 on both maps**. The overlay shows P2-rich rims around
P3-rich particle cores.

![nmf](docs/figures/id13_map1_nmf.png)

**KMeans clusters** (k = 4, 00l excluded) separate particle cores from rims. The
ROI P2 fraction rises from a median of 0.13 in the core cluster to about 0.35 in the
rim clusters, and PC1 alone correlates with the ROI ratio at |r| = 0.74.

![clusters](docs/figures/id13_map1_clusters.png)

**What the model-selection score says.** The silhouette score peaks at k = 2 (0.34)
and then stays flat (0.21–0.23 for k = 3–6). The pattern space is a *continuum* of
P2/P3 mixtures, not a set of discrete phases, so hard clustering is a coarse
description. The soft, additive NMF model fits the physics better, and that is why
both are provided.

On the second map a 24-pixel cluster appears whose pattern contains none of the
layered-oxide reflections. The ROI analysis cannot see a phase like this because
nobody defined an ROI for it.

### Limitations / honest notes

* Intensities are affected by crystallite orientation (spotty rings in micro-XRD). The
  ROI ratio and the NMF share are **intensity** ratios, not weight fractions.
* Particle edges have lower signal-to-noise. Part of a core/rim contrast can come from
  partial-volume effects, so rim assignments should be checked against the raw 2D frames.
* SNIP needs a window wider than the Bragg peaks but narrower than the amorphous
  background (`snip_iterations`). The default (40 bins ≈ 0.2 Å⁻¹ here) was checked visually.

## Install & use

```bash
pip install -e ".[dev]"
pytest                     # 13 tests, synthetic data only, ~5 s
```

Command line:

```bash
xrdcluster dq_scan.h5 --qrange 0.9:6.5 --exclude 1.0:1.25 -k 2 3 4 5 6 --nmf 3 --out results/
```

Python:

```python
from xrdcluster import load_ewoks_h5, PhaseMapper, phase_fraction

xmap = load_ewoks_h5("dq_scan.h5")                     # shape inferred for square maps
res = PhaseMapper(q_range=(0.9, 6.5), exclude_q=[(1.0, 1.25)],
                  n_clusters=[2, 3, 4, 5], n_nmf=3).fit(xmap)
print(res.summary())
res.label_image()        # (rows, cols) cluster map, -1 = no particle
res.mean_patterns()      # average background-free pattern per cluster
res.nmf_patterns         # non-negative component patterns
```

The full real-data analysis, including the ROI cross-check, is in
[`examples/id13_layered_oxide.py`](examples/id13_layered_oxide.py). A walk-through
on synthetic data is in [`examples/quickstart.ipynb`](examples/quickstart.ipynb).

## Layout

```
src/xrdcluster/
  io.py          ewoks/pyFAI HDF5 reader, XRDMap container
  preprocess.py  vectorised SNIP baseline, Otsu + Poisson-noise particle mask, features
  cluster.py     PhaseMapper pipeline (PCA, KMeans/GMM, k selection, NMF)
  roi.py         ROI peak areas with local linear background, marker-peak phase ratio
  synthetic.py   physically-motivated synthetic maps with ground truth
  plotting.py    colour-blind-safe figures
  cli.py         command line entry point
tests/           unit + end-to-end tests (CI via GitHub Actions)
```

Beamline data are not included (`*.h5` is git-ignored).

## Author

Mingfeng Xu, University of Bayreuth. The data were measured at ESRF ID13
(proposal in1187). MIT licence.
