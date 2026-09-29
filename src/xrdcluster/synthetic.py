"""Synthetic scanning-XRD maps with known ground truth (tests & demos).

Particles of two layered-oxide-like hexagonal phases (and an optional minor
impurity) are dropped on a 2D grid. Every scan point receives

    I(q) = background(q) + sum_phase  thickness * texture * pattern_phase(q)

followed by Poisson noise. Peak positions come from hexagonal d-spacings
so the patterns look like real powder patterns, not random spikes.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .io import XRDMap


def hexagonal_q(a: float, c: float, hkls: List[Tuple[int, int, int]]) -> np.ndarray:
    """q = 2*pi/d for a hexagonal cell."""
    h, k, l = np.array(hkls, dtype=float).T
    inv_d2 = 4.0 / 3.0 * (h**2 + h * k + k**2) / a**2 + l**2 / c**2
    return 2 * np.pi * np.sqrt(inv_d2)


# Illustrative reflection lists (positions from the cells, intensities rough).
PHASES: Dict[str, dict] = {
    # P2-type, P6_3/mmc, a~2.88 A, c~11.1 A
    "P2": dict(a=2.885, c=11.12,
               hkl=[(0, 0, 2), (0, 0, 4), (1, 0, 0), (1, 0, 2), (1, 0, 3), (1, 0, 4), (1, 0, 6), (1, 1, 0), (1, 1, 2)],
               I=[100, 12, 15, 45, 30, 20, 10, 12, 8]),
    # P3-type, R3m, a~2.88 A, c~16.8 A (hexagonal setting, -h+k+l=3n)
    "P3": dict(a=2.880, c=16.80,
               hkl=[(0, 0, 3), (0, 0, 6), (1, 0, 1), (0, 1, 2), (1, 0, 4), (0, 1, 5), (1, 0, 7), (1, 1, 0), (1, 1, 3)],
               I=[100, 10, 12, 40, 8, 35, 12, 12, 8]),
    # minor impurity with unrelated reflections
    "impurity": dict(a=5.20, c=7.10,
                     hkl=[(1, 0, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0), (1, 1, 2), (2, 0, 2), (2, 1, 1)],
                     I=[60, 100, 40, 30, 50, 25, 20]),
}


def pseudo_voigt(q: np.ndarray, center: float, fwhm: float, eta: float = 0.5) -> np.ndarray:
    x = (q - center) / (fwhm / 2)
    return eta / (1 + x**2) + (1 - eta) * np.exp(-np.log(2) * x**2)


def phase_pattern(q: np.ndarray, name: str, fwhm: float = 0.015) -> np.ndarray:
    ph = PHASES[name]
    qs = hexagonal_q(ph["a"], ph["c"], ph["hkl"])
    y = np.zeros_like(q)
    for qc, amp in zip(qs, ph["I"]):
        y += amp * pseudo_voigt(q, qc, fwhm)
    return y


def make_synthetic_map(
    shape: Tuple[int, int] = (80, 80),
    n_particles: int = 60,
    radius: Tuple[float, float] = (2.5, 6.0),
    phases=("P2", "P3"),
    impurity_particles: int = 3,
    q_range: Tuple[float, float] = (0.8, 5.0),
    n_q: int = 800,
    counts: float = 20.0,
    texture: float = 0.6,
    seed: int = 0,
):
    """Return ``(XRDMap, truth)``; ``truth`` is an int array, -1 = empty,
    otherwise the index into ``list(phases) + ['impurity']``. The projected
    particle thickness per point is stored in ``xmap.meta['thickness']``."""
    rng = np.random.default_rng(seed)
    rows, cols = shape
    q = np.linspace(*q_range, n_q)
    names = list(phases) + (["impurity"] if impurity_particles else [])
    patterns = np.stack([phase_pattern(q, n) for n in names])
    patterns /= patterns.max(1, keepdims=True)

    truth = np.full(shape, -1, dtype=int)
    thickness = np.zeros(shape)
    yy, xx = np.mgrid[0:rows, 0:cols]
    kinds = list(rng.integers(0, len(phases), n_particles)) + [len(phases)] * impurity_particles
    for kind in kinds:
        r = rng.uniform(*radius) * (0.5 if kind == len(phases) else 1.0)
        cy, cx = rng.uniform(0, rows), rng.uniform(0, cols)
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        inside = d2 < r**2
        truth[inside] = kind
        thickness[inside] = np.sqrt(np.clip(1 - d2[inside] / r**2, 0, 1))

    # smooth amorphous + air background
    bg = 3.0 + 6.0 * np.exp(-((q - 1.6) / 0.6) ** 2) + 1.5 / q
    I = np.tile(bg, (rows * cols, 1)) * rng.uniform(0.9, 1.1, (rows * cols, 1))
    t, lab = thickness.ravel(), truth.ravel()
    basal = np.zeros(n_q, dtype=bool)
    basal[(q > 1.0) & (q < 1.25)] = True  # 00l region of layered phases
    for i in np.flatnonzero(lab >= 0):
        pat = patterns[lab[i]].copy()
        # crystallite orientation effect: random re-weighting of the 00l peak
        pat[basal] *= rng.lognormal(0, texture)
        I[i] += counts * t[i] * pat * 10
    I = rng.poisson(np.clip(I, 0, None)).astype(np.float32)
    xmap = XRDMap(I, q, shape, {"source": "synthetic", "phases": names, "thickness": t})
    return xmap, truth.ravel()
