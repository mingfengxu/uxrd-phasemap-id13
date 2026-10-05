"""Classical region-of-interest (ROI) peak maps, used as a physics baseline."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


def roi_area(intensity: np.ndarray, q: np.ndarray, center: float, half_width: float = 0.02,
             bg_width: float = 0.02) -> np.ndarray:
    """Integrated peak area with a local linear background.

    The background is a straight line through the mean intensity of two
    side windows ``[center-hw-bg, center-hw]`` and ``[center+hw, center+hw+bg]``.
    Works on raw or already baseline-subtracted patterns.
    """
    q = np.asarray(q)
    I = np.atleast_2d(intensity)
    peak = (q >= center - half_width) & (q <= center + half_width)
    left = (q >= center - half_width - bg_width) & (q < center - half_width)
    right = (q > center + half_width) & (q <= center + half_width + bg_width)
    if not peak.any():
        raise ValueError(f"ROI at q={center} is outside the q range")
    if left.any() and right.any():
        ql, qr = q[left].mean(), q[right].mean()
        yl, yr = I[:, left].mean(1), I[:, right].mean(1)
        slope = (yr - yl) / (qr - ql)
        bg = yl[:, None] + slope[:, None] * (q[peak][None, :] - ql)
    else:
        bg = 0.0
    area = np.clip(I[:, peak] - bg, 0, None).sum(1)
    return area if np.ndim(intensity) == 2 else area[0]


def phase_fraction(intensity: np.ndarray, q: np.ndarray, markers: Dict[str, Iterable[float]],
                   half_width: float = 0.02) -> Dict[str, np.ndarray]:
    """Relative marker-peak fraction per phase, e.g. P2/(P2+P3).

    ``markers`` maps a phase name to q positions of reflections that belong
    only to that phase. Returns per-point fractions summing to 1 (NaN where
    no marker intensity is found). Note: this is an *intensity* ratio, not a
    weight fraction – texture and structure factors are ignored.
    """
    sums = {name: sum(roi_area(intensity, q, c, half_width) for c in centers)
            for name, centers in markers.items()}
    total = sum(sums.values())
    with np.errstate(divide="ignore", invalid="ignore"):
        return {name: np.where(total > 0, s / total, np.nan) for name, s in sums.items()}


def window_sum(intensity: np.ndarray, q: np.ndarray, qmin: float, qmax: float) -> np.ndarray:
    """Plain sum of intensity over ``qmin <= q <= qmax`` (no background).

    This is what a beamline ROI (e.g. the ID13 ``.dat`` maps) contains.
    """
    sel = (np.asarray(q) >= qmin) & (np.asarray(q) <= qmax)
    if not sel.any():
        raise ValueError(f"window {qmin}-{qmax} is outside the q range")
    return np.atleast_2d(intensity)[:, sel].sum(1)


def normalized_ratio(i_a: np.ndarray, i_b: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """Two-reflection phase ratio as used in Xu et al., ACS Energy Lett. 2025,
    10, 6347 (SI, eqns 1-3).

    Each ROI map is normalised to its own maximum, pixels where either
    normalised value is below ``threshold`` are masked (NaN), and the ratio
    ``a / (a + b)`` is returned. If the ROI sums still contain the diffuse
    background, the ratio is compressed towards 0.5; pass background-
    subtracted areas (e.g. from :func:`roi_area` on SNIP-corrected data) to
    avoid that. With background-free inputs, set ``threshold=0`` and mask by
    total signal instead: the per-channel cut would otherwise discard
    phase-pure pixels, whose other channel is ~0.
    """
    a = np.asarray(i_a, dtype=float) / np.nanmax(i_a)
    b = np.asarray(i_b, dtype=float) / np.nanmax(i_b)
    a = np.where(a < threshold, np.nan, a)
    b = np.where(b < threshold, np.nan, b)
    return a / (a + b)


def peak_position_map(intensity: np.ndarray, q: np.ndarray, qmin: float, qmax: float,
                      points: Optional[np.ndarray] = None, min_height: float = 0.0,
                      max_q_error: float = 0.002):
    """Fit a Gaussian + constant to one reflection in every pattern.

    Gives sub-bin peak positions, so small lattice-parameter changes (strain,
    Na content) can be mapped -- something clustering/NMF cannot resolve,
    because a continuous peak shift is not a mixture of fixed patterns.

    Returns a dict of per-point arrays (NaN where not fitted): ``q0``
    (centre), ``d`` (= 2*pi/q0), ``q0_err`` (1-sigma from the fit), ``height``
    and ``sigma`` (Gaussian width).
    """
    from scipy.optimize import curve_fit

    q = np.asarray(q)
    I = np.atleast_2d(intensity)
    win = (q >= qmin) & (q <= qmax)
    qw = q[win]
    n = I.shape[0]
    out = {k: np.full(n, np.nan) for k in ("q0", "d", "q0_err", "height", "sigma")}
    idx = np.arange(n) if points is None else np.flatnonzero(np.asarray(points, bool))
    step = float(np.median(np.diff(qw)))

    def gauss(x, a, c, w, b):
        return a * np.exp(-0.5 * ((x - c) / w) ** 2) + b

    for i in idx:
        y = I[i, win].astype(float)
        if y.max() - np.median(y) <= min_height:
            continue
        j = int(np.argmax(y))
        try:
            popt, pcov = curve_fit(gauss, qw, y, p0=[y.max() - y.min(), qw[j], 2 * step, y.min()],
                                   bounds=([0, qmin, step / 4, -np.inf], [np.inf, qmax, qmax - qmin, np.inf]),
                                   maxfev=4000)
        except (RuntimeError, ValueError):
            continue
        err = float(np.sqrt(pcov[1, 1])) if np.all(np.isfinite(pcov)) else np.inf
        if err > max_q_error:
            continue
        out["q0"][i], out["q0_err"][i], out["height"][i], out["sigma"][i] = popt[1], err, popt[0], popt[2]
    out["d"] = 2 * np.pi / out["q0"]
    return out
