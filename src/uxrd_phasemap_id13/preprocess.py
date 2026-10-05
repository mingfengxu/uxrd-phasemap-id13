"""Pre-processing: background removal, foreground masking, feature building."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def snip_baseline(y: np.ndarray, iterations: int = 40, chunk: int = 4096) -> np.ndarray:
    """SNIP background estimate (Ryan et al. 1988; Morhac 2009), vectorised.

    Works on a single pattern (n_q,) or a stack (n_points, n_q). A
    log-log-sqrt (LLS) transform compresses the dynamic range so that sharp
    Bragg peaks are clipped while the slowly varying amorphous/air/holder
    scattering is retained. ``iterations`` is the half-width (in bins) of the
    widest feature treated as a peak.
    """
    y = np.asarray(y, dtype=float)
    single = y.ndim == 1
    y2 = np.atleast_2d(y)
    out = np.empty_like(y2)
    for start in range(0, y2.shape[0], chunk):
        block = np.clip(y2[start : start + chunk], 0, None)
        v = np.log(np.log(np.sqrt(block + 1.0) + 1.0) + 1.0)
        n = v.shape[1]
        for p in range(1, iterations + 1):
            if 2 * p >= n:
                break
            avg = 0.5 * (v[:, : n - 2 * p] + v[:, 2 * p :])
            v[:, p : n - p] = np.minimum(v[:, p : n - p], avg)
        out[start : start + chunk] = (np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0
    return out[0] if single else out


def subtract_background(intensity: np.ndarray, iterations: int = 40) -> np.ndarray:
    """Return the Bragg (peak-only) signal: ``max(I - SNIP(I), 0)``."""
    return np.clip(intensity - snip_baseline(intensity, iterations), 0, None)


def reference_background(intensity: np.ndarray, reference_points: np.ndarray,
                         fit_bins: Optional[np.ndarray] = None, n_iter: int = 6,
                         clip_sigma: float = 2.5):
    """Empty-region reference subtraction.

    Components that are (nearly) the same everywhere in the beam path -- cell
    window, electrolyte, separator, a uniform foil -- are removed by
    subtracting a *measured* reference pattern instead of a mathematical
    baseline::

        ref    = median of the patterns at ``reference_points`` (no sample)
        I_i   ~= s_i * ref + Bragg_i          (s_i: local thickness / flux)

    The scale ``s_i`` of every pattern is a least-squares fit over
    ``fit_bins`` in which bins with positive outliers (Bragg peaks of the
    sample add intensity, they never remove it) are iteratively excluded,
    using a robust (MAD) noise estimate.

    Returns ``(residual, scale, ref)``; the residual is ``I - s * ref`` and
    can be slightly negative where the sample absorbs or a reference
    component is locally weaker.
    """
    I = np.asarray(intensity, dtype=float)
    ref_pts = np.asarray(reference_points, dtype=bool)
    if ref_pts.sum() < 5:
        raise ValueError("need at least 5 reference (sample-free) points")
    ref = np.median(I[ref_pts], axis=0)
    fit = np.ones(I.shape[1], bool) if fit_bins is None else np.asarray(fit_bins, bool)
    fit &= ref > 0
    r = ref[fit]
    y = I[:, fit]
    s = np.median(y / r, axis=1)
    for _ in range(n_iter):
        res = y - s[:, None] * r
        med = np.median(res, axis=1, keepdims=True)
        sig = 1.4826 * np.median(np.abs(res - med), axis=1, keepdims=True) + 1e-12
        w = res < med + clip_sigma * sig
        s = (w * y * r).sum(1) / np.clip((w * r * r).sum(1), 1e-12, None)
    return I - s[:, None] * ref, s, ref


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold for a 1D sample (no scikit-image dependency)."""
    values = np.asarray(values, dtype=float)
    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = m0 / w0
        mu1 = (mt - m0) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = 0
    return float(centers[np.argmax(between)])


def foreground_mask(peaks: np.ndarray, threshold: Optional[float] = None,
                    baseline: Optional[np.ndarray] = None, nsigma: float = 3.0) -> Tuple[np.ndarray, float]:
    """Separate diffracting particles from empty/matrix scan points.

    The per-point signal is the integrated Bragg intensity. If ``baseline`` is
    given, only counts exceeding ``nsigma * sqrt(baseline)`` (Poisson noise of
    the background) are counted, so clipped noise in empty points does not
    add up to a spurious signal. The threshold is chosen on a log scale with
    Otsu's method unless given. Returns ``(mask, threshold_counts)``.
    """
    if baseline is not None:
        peaks = np.clip(peaks - nsigma * np.sqrt(np.clip(baseline, 0, None) + 1.0), 0, None)
    signal = np.log10(peaks.sum(axis=1) + 1.0)
    thr_log = otsu_threshold(signal) if threshold is None else np.log10(threshold + 1.0)
    return signal > thr_log, float(10**thr_log - 1.0)


def q_mask(q: np.ndarray, exclude: Iterable[Sequence[float]] = ()) -> np.ndarray:
    """Boolean mask over q that drops the given (qmin, qmax) windows."""
    keep = np.ones_like(q, dtype=bool)
    for lo, hi in exclude:
        keep &= ~((q >= lo) & (q <= hi))
    return keep


def build_features(peaks: np.ndarray, keep: Optional[np.ndarray] = None, transform: str = "sqrt") -> np.ndarray:
    """Turn peak-only patterns into clustering features.

    * ``sqrt`` (variance-stabilising for Poisson counts) or ``log1p`` or ``none``
    * each pattern is L2-normalised so that clustering reflects *which* peaks
      are present (phase), not how much material is in the beam (thickness).
    """
    X = peaks if keep is None else peaks[:, keep]
    X = np.clip(X, 0, None).astype(float)
    if transform == "sqrt":
        X = np.sqrt(X)
    elif transform == "log1p":
        X = np.log1p(X)
    elif transform != "none":
        raise ValueError(f"unknown transform {transform!r}")
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return X / norm
