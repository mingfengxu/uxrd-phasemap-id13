"""Azimuthal integration of raw Eiger (LIMA) frames with pyFAI.

This reproduces the beamline Ewoks step ``IntegrateBlissScan`` on your own
computer: every 2D detector frame of a scan is integrated to a 1D pattern
I(q) and written to an HDF5 file with the same layout as the ESRF
PROCESSED_DATA files, so :func:`uxrd_phasemap_id13.load_ewoks_h5` (and
PyMca's "show as 1D stack") read it directly.

Frames are streamed from disk in small chunks, so a 40 GB scan needs only a
few hundred MB of memory.

Requires the optional dependencies: ``pip install -e ".[integrate]"``
(pyFAI, hdf5plugin for the Bitshuffle/LZ4-compressed Eiger files, fabio).
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

EIGER_DSETS = ("/entry_0000/instrument/eiger/data", "/entry_0000/measurement/data")


# --------------------------------------------------------------------- geometry
def find_integration_options(obj) -> Optional[dict]:
    """Return the first dict containing a ``"poni"`` entry, searching nested
    JSON (works for azimint.json, an Ewoks workflow json, or the
    ``configuration/data`` string stored in a PROCESSED_DATA file)."""
    if isinstance(obj, dict):
        if "poni" in obj and isinstance(obj["poni"], dict):
            return obj
        for v in obj.values():
            hit = find_integration_options(v)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = find_integration_options(v)
            if hit is not None:
                return hit
    return None


def load_geometry(path: str) -> dict:
    """Read integration options from a .poni file or an Ewoks/pyFAI json.

    Returns a dict with at least ``poni`` (dist, poni1, poni2, rot1-3,
    detector, detector_config, wavelength) and, when available,
    ``polarization_factor``, ``nbpt_rad``, ``unit``, ``mask_file``.
    """
    if path.endswith(".poni"):
        import pyFAI
        ai = pyFAI.load(path)
        return {"poni": {"dist": ai.dist, "poni1": ai.poni1, "poni2": ai.poni2,
                         "rot1": ai.rot1, "rot2": ai.rot2, "rot3": ai.rot3,
                         "wavelength": ai.wavelength, "detector": ai.detector.name,
                         "detector_config": ai.detector.get_config()},
                "_integrator": ai}
    with open(path) as fh:
        opts = find_integration_options(json.load(fh))
    if opts is None:
        raise ValueError(f"no 'poni' block found in {path}")
    out = {"poni": dict(opts["poni"])}
    for key in ("nbpt_rad", "unit", "mask_file"):
        if opts.get(key) is not None:
            out[key] = opts[key]
    pol = opts.get("polarization_description")
    if isinstance(pol, (list, tuple)) and pol:
        out["polarization_factor"] = float(pol[0])
    elif isinstance(pol, (int, float)):
        out["polarization_factor"] = float(pol)
    return out


def make_integrator(geometry: dict):
    """Build a pyFAI AzimuthalIntegrator from :func:`load_geometry` output."""
    if "_integrator" in geometry:
        return geometry["_integrator"]
    import pyFAI
    try:
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    except ImportError:  # pyFAI < 2024
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
    p = geometry["poni"]
    cfg = dict(p.get("detector_config") or {})
    cfg.pop("sensor", None)  # parallax correction is not applied by the beamline workflow
    detector = pyFAI.detector_factory(p.get("detector", "Eiger4M"), config=cfg or None)
    return AzimuthalIntegrator(dist=p["dist"], poni1=p["poni1"], poni2=p["poni2"],
                               rot1=p.get("rot1", 0.0), rot2=p.get("rot2", 0.0),
                               rot3=p.get("rot3", 0.0), wavelength=p["wavelength"],
                               detector=detector)


# --------------------------------------------------------------------- frames
@dataclass
class FrameSource:
    path: str
    dset: str
    n_frames: int


def sources_from_master(master: str, eiger_dir: Optional[str] = None,
                        entry: str = "1.1") -> Tuple[List[FrameSource], dict]:
    """Locate the Eiger files of a Bliss scan from its RAW_DATA master file.

    The master's ``instrument/eiger/image`` is a virtual dataset pointing at
    ``scan00001/eiger/*_data_XXXXXX.h5``. Those files are looked up first at
    the recorded relative path, then by file name inside ``eiger_dir``
    (default: the master's folder). Also returns scan metadata (map shape,
    monitor counters).
    """
    import h5py
    base = os.path.dirname(os.path.abspath(master))
    meta = {}
    with h5py.File(master, "r") as h5:
        g = h5[entry]
        img = g["instrument/eiger/image"]
        if not img.is_virtual:
            raise ValueError("instrument/eiger/image is not a virtual dataset")
        sources = []
        for vs in img.virtual_sources():
            n = vs.vspace.get_select_bounds()[1][0] - vs.vspace.get_select_bounds()[0][0] + 1
            candidates = [os.path.join(base, vs.file_name)]
            if eiger_dir:
                candidates.insert(0, os.path.join(eiger_dir, os.path.basename(vs.file_name)))
            candidates.append(os.path.join(base, os.path.basename(vs.file_name)))
            path = next((c for c in candidates if os.path.exists(c)), None)
            if path is None:
                raise FileNotFoundError(
                    f"cannot find {os.path.basename(vs.file_name)}; pass --eiger-dir")
            sources.append(FrameSource(path, vs.dset_name, int(n)))
        if "technique" in g:
            meta["shape"] = (int(g["technique/dim1"][()]), int(g["technique/dim0"][()]))
        meta["title"] = g["title"][()].decode() if "title" in g else ""
        meta["sample"] = g["sample/name"][()].decode() if "sample/name" in g else ""
        meta["monitors"] = {k: g[f"instrument/{k}/data"][()] for k in g["instrument"]
                            if k.startswith("ct") and f"instrument/{k}/data" in g}
    return sources, meta


def sources_from_files(paths: Sequence[str]) -> List[FrameSource]:
    """Use explicitly given Eiger files (sorted by name)."""
    import h5py
    out = []
    for p in sorted(paths):
        with h5py.File(p, "r") as h5:
            dset = next((d for d in EIGER_DSETS if d in h5), None)
            if dset is None:
                raise KeyError(f"no Eiger frame dataset in {p}")
            out.append(FrameSource(p, dset, h5[dset].shape[0]))
    return out


def iter_frames(sources: Sequence[FrameSource], indices: Optional[np.ndarray] = None,
                chunk: int = 32) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(global_index, frame)`` reading ``chunk`` frames at a time."""
    import h5py
    try:
        import hdf5plugin  # noqa: F401  (registers the Bitshuffle/LZ4 filter)
    except ImportError:
        pass
    offsets = np.cumsum([0] + [s.n_frames for s in sources])
    total = int(offsets[-1])
    wanted = np.arange(total) if indices is None else np.asarray(indices)
    for k, src in enumerate(sources):
        lo, hi = offsets[k], offsets[k + 1]
        local = wanted[(wanted >= lo) & (wanted < hi)] - lo
        if local.size == 0:
            continue
        with h5py.File(src.path, "r") as h5:
            d = h5[src.dset]
            # read contiguous runs in blocks for speed
            i = 0
            while i < local.size:
                j = i
                while j + 1 < local.size and local[j + 1] == local[j] + 1 and j - i + 1 < chunk:
                    j += 1
                block = d[local[i]:local[j] + 1]
                for n, fr in enumerate(block):
                    yield int(lo + local[i] + n), fr
                i = j + 1


def total_frames(sources: Sequence[FrameSource]) -> int:
    return int(sum(s.n_frames for s in sources))


# --------------------------------------------------------------------- mask
def build_mask(ai, probe_frames: np.ndarray, mask_file: Optional[str] = None,
               flag_level: Optional[int] = None) -> np.ndarray:
    """Pixels to ignore (True = masked).

    * module gaps of the detector model,
    * pixels flagged by the detector firmware: Eiger writes the maximum value
      of the counter depth (65535 or 2**32-1) into dead/gap pixels, so any
      pixel that stays at or above ``flag_level`` in *all* probe frames is
      masked (a real Bragg spot does not stay saturated in every frame),
    * an optional user mask (e.g. the beamline ``mask.edf``; non-zero = masked).
    """
    mask = np.zeros(ai.detector.shape, dtype=bool)
    if ai.detector.mask is not None:
        mask |= ai.detector.mask.astype(bool)
    probe = np.asarray(probe_frames)
    if flag_level is None:
        flag_level = 65535 if probe.dtype.itemsize >= 2 else np.iinfo(probe.dtype).max
    mask |= probe.min(axis=0) >= flag_level
    if mask_file:
        import fabio
        user = fabio.open(mask_file.replace("fabio://", "")).data.astype(bool)
        if user.shape != mask.shape:
            raise ValueError(f"mask shape {user.shape} != detector {mask.shape}")
        mask |= user
    return mask


# --------------------------------------------------------------------- integration
DEFAULT_METHOD = ("bbox", "csr", "cython")


def integrate_frames(sources: Sequence[FrameSource], ai, out_path: str, *,
                     npt: int = 1400, unit: str = "q_A^-1",
                     polarization_factor: Optional[float] = 0.99,
                     mask: Optional[np.ndarray] = None,
                     indices: Optional[np.ndarray] = None,
                     monitor: Optional[np.ndarray] = None,
                     sigma_clip: bool = False, sigma_thres: float = 3.0,
                     method=DEFAULT_METHOD, chunk: int = 32,
                     shape: Optional[Tuple[int, int]] = None,
                     meta: Optional[dict] = None, verbose: bool = True) -> str:
    """Integrate frames and write an ewoks-style HDF5 file.

    ``monitor`` (one value per frame, e.g. the ct34 I0 counter) normalises
    each frame to the median monitor value. With ``sigma_clip`` a second
    result is written in which, in every q bin, pixels deviating by more
    than ``sigma_thres`` standard deviations (bright single-crystal spots)
    are rejected before averaging (pyFAI ``sigma_clip_ng``).
    """
    import h5py
    n_total = total_frames(sources)
    idx = np.arange(n_total) if indices is None else np.asarray(indices)
    n = idx.size
    mon_ref = float(np.median(monitor)) if monitor is not None else None
    kw = dict(unit=unit, mask=mask, correctSolidAngle=True,
              polarization_factor=polarization_factor)

    t0 = time.time()
    with h5py.File(out_path, "w") as out:
        grp = out.create_group("1.1/eiger_integrate")
        grp["program"] = "pyFAI"
        import pyFAI
        grp["version"] = pyFAI.version
        cfg = {"application": "uxrd-phasemap-id13 integrate", "nbpt_rad": npt, "unit": unit,
               "polarization_factor": polarization_factor, "method": list(method),
               "correct_solid_angle": True, "monitor_normalised": monitor is not None,
               "sigma_clip": sigma_clip, "poni": {
                   "dist": ai.dist, "poni1": ai.poni1, "poni2": ai.poni2, "rot1": ai.rot1,
                   "rot2": ai.rot2, "rot3": ai.rot3, "wavelength": ai.wavelength,
                   "detector": ai.detector.name}}
        if meta:
            cfg.update({k: v for k, v in meta.items() if k in ("title", "sample", "shape")})
        grp["configuration/data"] = json.dumps(cfg)
        integ = grp.create_group("integrated")
        I_ds = integ.create_dataset("intensity", (n, npt), dtype="float32",
                                    chunks=(min(n, 64), npt), compression="gzip")
        integ["points"] = idx
        if shape is not None:
            integ.attrs["map_shape"] = shape
        if sigma_clip:
            sc = out.create_group("1.1/eiger_sigmaclip")
            sc["program"] = "pyFAI sigma_clip_ng"
            sc["configuration/data"] = json.dumps({**cfg, "sigma_thres": sigma_thres})
            sc_int = sc.create_group("integrated")
            S_ds = sc_int.create_dataset("intensity", (n, npt), dtype="float32",
                                         chunks=(min(n, 64), npt), compression="gzip")
            sc_int["points"] = idx
            if shape is not None:
                sc_int.attrs["map_shape"] = shape
        q_written = False
        for k, (gi, frame) in enumerate(iter_frames(sources, idx, chunk)):
            norm = float(monitor[gi]) / mon_ref if monitor is not None else 1.0
            res = ai.integrate1d(frame, npt, method=method, normalization_factor=norm, **kw)
            if not q_written:
                integ["q"] = res.radial
                if sigma_clip:
                    sc_int["q"] = res.radial
                q_written = True
            I_ds[k] = res.intensity
            if sigma_clip:
                r2 = ai.sigma_clip_ng(frame, npt, unit=unit, mask=mask,
                                      polarization_factor=polarization_factor,
                                      normalization_factor=norm, thres=sigma_thres,
                                      max_iter=5, error_model="azimuthal",
                                      method=("no", "csr", "cython"))
                S_ds[k] = r2.intensity
            if verbose and (k + 1) % 200 == 0 or (verbose and k + 1 == n):
                el = time.time() - t0
                rate = (k + 1) / el
                print(f"  {k + 1:6d}/{n} frames  {rate:5.1f} frames/s  "
                      f"elapsed {el / 60:5.1f} min  remaining ~{(n - k - 1) / rate / 60:5.1f} min",
                      flush=True)
    return out_path


# --------------------------------------------------------------------- preview
def preview_frame(ai, frame: np.ndarray, mask: np.ndarray, out_png: str, *,
                  npt: int = 1400, unit: str = "q_A^-1",
                  polarization_factor: Optional[float] = 0.99, title: str = ""):
    """One-frame sanity check: raw image with mask, cake (q-chi) and 1D
    patterns (plain average vs sigma-clipped)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    f = frame.astype(float)
    shown = np.where(mask, np.nan, f)
    r1 = ai.integrate1d(frame, npt, unit=unit, mask=mask, polarization_factor=polarization_factor,
                        method=DEFAULT_METHOD)
    rs = ai.sigma_clip_ng(frame, npt, unit=unit, mask=mask, polarization_factor=polarization_factor,
                          thres=3.0, max_iter=5, error_model="azimuthal", method=("no", "csr", "cython"))
    cake = ai.integrate2d(frame, 1000, 360, unit=unit, mask=mask,
                          polarization_factor=polarization_factor, method=("bbox", "csr", "cython"))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    vmax = np.nanpercentile(shown, 99.9)
    ax[0].imshow(np.clip(shown, 0.5, None), norm=LogNorm(vmin=0.5, vmax=max(vmax, 1)),
                 cmap="magma", origin="lower", interpolation="nearest")
    ax[0].imshow(np.where(mask, 1.0, np.nan), cmap="Blues", vmin=0, vmax=1.5, alpha=0.6,
                 origin="lower", interpolation="nearest")
    ax[0].set_title("raw frame (log), masked pixels in blue")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ci = cake.intensity
    pos = ci[ci > 0]
    ax[1].imshow(np.clip(ci, max(np.percentile(pos, 5), 1e-3) if pos.size else 1e-3, None),
                 aspect="auto", origin="lower", cmap="magma", norm=LogNorm(),
                 extent=[cake.radial[0], cake.radial[-1], cake.azimuthal[0], cake.azimuthal[-1]])
    ax[1].set_xlabel(r"q ($\AA^{-1}$)"); ax[1].set_ylabel(r"azimuth $\chi$ (deg)")
    ax[1].set_title("cake: rings become vertical lines, spots stay spots")
    ax[2].plot(r1.radial, r1.intensity, lw=0.8, label="azimuthal average")
    ax[2].plot(rs.radial, rs.intensity, lw=0.8, label="sigma-clipped (3σ)")
    ax[2].set_yscale("log"); ax[2].set_xlabel(r"q ($\AA^{-1}$)"); ax[2].legend(frameon=False)
    ax[2].set_title("1D pattern")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------- CLI
def _parse_range(text: str, n: int) -> np.ndarray:
    lo, _, hi = text.partition(":")
    return np.arange(int(lo or 0), int(hi) if hi else n)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog="uxrd-phasemap-id13-integrate",
        description="Integrate raw Eiger frames of an ID13 scan with pyFAI (2D -> 1D).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--master", help="RAW_DATA scan master .h5 (finds the Eiger files, map shape, monitors)")
    src.add_argument("--files", nargs="+", help="Eiger data files (*_data_000000.h5 ...)")
    p.add_argument("--eiger-dir", help="folder holding the Eiger files if not next to the master")
    p.add_argument("--geometry", required=True,
                   help=".poni file, azimint.json or an Ewoks workflow json containing a 'poni' block")
    p.add_argument("--mask", help="extra mask image (e.g. the beamline mask.edf)")
    p.add_argument("--npt", type=int, help="number of q bins (default: from geometry json, else 1400)")
    p.add_argument("--polarization", type=float,
                   help="polarization factor (default: from geometry json, else 0.99; use -1 for none)")
    p.add_argument("--rows", help="only these scan rows, e.g. 0:10 (needs --master or --shape)")
    p.add_argument("--frames", help="only these frame indices, e.g. 0:500")
    p.add_argument("--every", type=int, default=1, help="take every n-th frame (quick look)")
    p.add_argument("--shape", type=int, nargs=2, metavar=("ROWS", "COLS"))
    p.add_argument("--monitor", default=None,
                   help="normalise by this counter of the master, e.g. ct34 (beamline default)")
    p.add_argument("--sigma-clip", action="store_true", help="also write a sigma-clipped result")
    p.add_argument("--preview", type=int, metavar="FRAME",
                   help="only make a one-frame preview figure (image, cake, 1D) and stop")
    p.add_argument("--out", required=True, help="output .h5 (or .png with --preview)")
    a = p.parse_args(argv)

    geometry = load_geometry(a.geometry)
    ai = make_integrator(geometry)
    npt = a.npt or int(geometry.get("nbpt_rad", 1400))
    unit = geometry.get("unit", "q_A^-1")
    pol = a.polarization if a.polarization is not None else geometry.get("polarization_factor", 0.99)
    pol = None if pol is not None and pol < 0 else pol

    meta = {}
    if a.master:
        sources, meta = sources_from_master(a.master, a.eiger_dir)
    else:
        sources = sources_from_files(a.files)
    shape = tuple(a.shape) if a.shape else meta.get("shape")
    n = total_frames(sources)
    if shape and shape[0] * shape[1] != n:
        print(f"warning: map shape {shape} != {n} frames; ignoring shape")
        shape = None
    print(f"{len(sources)} Eiger file(s), {n} frames, map shape {shape}, detector {ai.detector.name}, "
          f"lambda {ai.wavelength * 1e10:.4f} A, dist {ai.dist * 1e3:.2f} mm, polarization {pol}")

    # mask from detector model + firmware-flagged pixels (+ optional file)
    probe = np.stack([f for _, f in iter_frames(sources, np.arange(min(8, n)))])
    mask = build_mask(ai, probe, a.mask or None)
    print(f"masked pixels: {int(mask.sum())} ({mask.mean() * 100:.1f} %)")

    if a.preview is not None:
        frame = next(iter_frames(sources, np.array([a.preview])))[1]
        out = a.out if a.out.endswith(".png") else os.path.splitext(a.out)[0] + ".png"
        preview_frame(ai, frame, mask, out, npt=npt, unit=unit, polarization_factor=pol,
                      title=f"{meta.get('sample', '')} frame {a.preview}")
        print(f"preview written to {out}")
        return

    idx = np.arange(n)
    if a.rows:
        if not shape:
            p.error("--rows needs the map shape (--master or --shape)")
        rows = _parse_range(a.rows, shape[0])
        idx = (rows[:, None] * shape[1] + np.arange(shape[1])[None, :]).ravel()
        shape = (rows.size, shape[1])
    if a.frames:
        idx = _parse_range(a.frames, n)
        shape = None
    if a.every > 1:
        idx = idx[::a.every]
        shape = None
    monitor = None
    if a.monitor:
        if a.monitor not in meta.get("monitors", {}):
            p.error(f"monitor {a.monitor} not found in master")
        monitor = meta["monitors"][a.monitor]
    print(f"integrating {idx.size} frames -> {a.out}")
    integrate_frames(sources, ai, a.out, npt=npt, unit=unit, polarization_factor=pol, mask=mask,
                     indices=idx, monitor=monitor, sigma_clip=a.sigma_clip, shape=shape, meta=meta)
    print("done")


if __name__ == "__main__":
    main()
