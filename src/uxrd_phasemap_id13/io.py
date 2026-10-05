"""Loading of azimuthally integrated scanning-XRD maps.

The main reader targets the HDF5 files written by the ESRF ``ewoks``
pipeline (``pyfai-integrate``), e.g. at ID13, where each scan point holds one
1D pattern ``I(q)``. Generic arrays can be wrapped with :class:`XRDMap`
directly.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class XRDMap:
    """A 2D map of 1D diffraction patterns.

    Attributes
    ----------
    intensity : (n_points, n_q) array
        One integrated pattern per scan point, in raster (row-major) order.
    q : (n_q,) array
        Scattering vector in inverse Angstrom.
    shape : (n_rows, n_cols)
        Map dimensions; ``n_rows * n_cols == n_points``.
    meta : dict
        Free-form metadata (source file, wavelength, integration config ...).
    """

    intensity: np.ndarray
    q: np.ndarray
    shape: Tuple[int, int]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.intensity = np.asarray(self.intensity)
        self.q = np.asarray(self.q, dtype=float)
        if self.intensity.ndim != 2:
            raise ValueError("intensity must be 2D (n_points, n_q)")
        if self.intensity.shape[1] != self.q.size:
            raise ValueError("intensity and q have inconsistent lengths")
        if self.shape[0] * self.shape[1] != self.intensity.shape[0]:
            raise ValueError(
                f"shape {self.shape} does not match {self.intensity.shape[0]} points"
            )

    @property
    def n_points(self) -> int:
        return self.intensity.shape[0]

    def to_image(self, values: np.ndarray) -> np.ndarray:
        """Reshape a per-point vector (n_points,) to a (rows, cols) image."""
        return np.asarray(values).reshape(self.shape)

    def crop_q(self, qmin: float = -np.inf, qmax: float = np.inf) -> "XRDMap":
        sel = (self.q >= qmin) & (self.q <= qmax)
        return XRDMap(self.intensity[:, sel], self.q[sel], self.shape, dict(self.meta))


def _guess_shape(n: int) -> Tuple[int, int]:
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(
            f"Cannot infer map shape for {n} points; pass shape=(rows, cols)."
        )
    return side, side


def load_ewoks_h5(
    path: str,
    shape: Optional[Tuple[int, int]] = None,
    scan: Optional[str] = None,
    dtype=np.float32,
    group: Optional[str] = None,
) -> XRDMap:
    """Read a pyFAI/ewoks integrated map.

    Looks for ``<scan>/*_integrate/integrated/{intensity,q}``. If ``scan`` is
    not given the first entry that contains such a group is used; ``group``
    selects a specific one (e.g. ``"eiger_sigmaclip"`` written by
    ``uxrd-phasemap-id13-integrate --sigma-clip``). The map shape is taken
    from ``shape``, else from a ``map_shape`` attribute, else assumed square.
    """
    import h5py

    with h5py.File(path, "r") as h5:
        entries = [scan] if scan else list(h5.keys())
        for entry in entries:
            grp = h5[entry]
            names = [group] if group else list(grp)
            for name in names:
                if name not in grp:
                    continue
                sub = grp[name]
                if isinstance(sub, h5py.Group) and "integrated" in sub:
                    integ = sub["integrated"]
                    intensity = integ["intensity"][()].astype(dtype)
                    q = integ["q"][()]
                    meta = {"source": str(path), "entry": entry, "group": name}
                    if "configuration" in sub and "data" in sub["configuration"]:
                        raw = sub["configuration"]["data"][()]
                        try:
                            cfg = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                            meta["integration"] = cfg
                            poni = cfg.get("poni", {})
                            if "wavelength" in poni:
                                meta["wavelength_A"] = poni["wavelength"] * 1e10
                        except (ValueError, AttributeError):
                            pass
                    if shape:
                        shp = tuple(shape)
                    elif "map_shape" in integ.attrs:
                        shp = tuple(int(v) for v in integ.attrs["map_shape"])
                    else:
                        shp = _guess_shape(intensity.shape[0])
                    return XRDMap(intensity, q, shp, meta)
    raise KeyError(f"No '*/integrated/intensity' dataset found in {path}")
