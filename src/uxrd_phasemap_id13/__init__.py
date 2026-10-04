"""uxrd_phasemap_id13 – unsupervised phase mapping of scanning micro-XRD data."""
from .cluster import BACKGROUND, PhaseMapper, PhaseMapResult
from .io import XRDMap, load_ewoks_h5
from .preprocess import build_features, foreground_mask, snip_baseline, subtract_background
from .roi import normalized_ratio, phase_fraction, roi_area, window_sum
from .synthetic import make_synthetic_map

__version__ = "0.1.0"
__all__ = [
    "BACKGROUND", "PhaseMapper", "PhaseMapResult", "XRDMap", "load_ewoks_h5",
    "build_features", "foreground_mask", "snip_baseline", "subtract_background",
    "phase_fraction", "roi_area", "normalized_ratio", "window_sum", "make_synthetic_map",
]
