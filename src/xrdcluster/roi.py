"""Classical region-of-interest (ROI) peak maps, used as a physics baseline."""
from __future__ import annotations

from typing import Dict, Iterable

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
