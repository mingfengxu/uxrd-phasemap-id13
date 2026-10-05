"""Tests for the raw-frame integration step (skipped if pyFAI is not installed)."""
import json
import os

import numpy as np
import pytest

pyFAI = pytest.importorskip("pyFAI")
h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

from uxrd_phasemap_id13 import load_ewoks_h5  # noqa: E402
from uxrd_phasemap_id13.integrate import (build_mask, integrate_frames, load_geometry,  # noqa: E402
                                          make_integrator, sources_from_files,
                                          sources_from_master)
from uxrd_phasemap_id13.synthetic import pseudo_voigt  # noqa: E402

FLAG = np.iinfo(np.uint32).max
GEOMETRY = {"poni": {"dist": 0.1, "poni1": 0.0073, "poni2": 0.0183, "rot1": 0.0, "rot2": 0.0,
                     "rot3": 0.0, "wavelength": 0.73e-10, "detector": "Pilatus100k",
                     "detector_config": {"orientation": 3}},
            "nbpt_rad": 300, "unit": "q_A^-1", "polarization_description": [0.99, 0.0]}
PEAKS = {"A": [0.6, 0.9], "B": [0.75]}


def _pattern(q, phase):
    return 5 + sum(400 * pseudo_voigt(q, c, 0.01) for c in PEAKS[phase])


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    """A 2x3 'scan' of synthetic frames written like a Bliss/LIMA dataset."""
    d = tmp_path_factory.mktemp("raw")
    geo_path = d / "workflow.json"
    geo_path.write_text(json.dumps({"graph": {"nodes": [{"default_inputs": [
        {"name": "integration_options", "value": GEOMETRY}]}]}}))
    ai = make_integrator(load_geometry(str(geo_path)))
    q = np.linspace(0.3, 1.2, 2000)
    frames, phases = [], ["A", "A", "B", "B", "A", "B"]
    rng = np.random.default_rng(0)
    for k, ph in enumerate(phases):
        img = ai.calcfrom1d(q, _pattern(q, ph), dim1_unit="q_A^-1", correctSolidAngle=True,
                            polarization_factor=0.99)
        img = rng.poisson(np.clip(img, 0, None) * (2.0 if k == 1 else 1.0)).astype(np.uint32)
        img[:, 100] = FLAG  # firmware-flagged dead column
        if k == 4:
            img[20:23, 300:303] = 60000  # one bright single-crystal spot
        frames.append(img)
    eiger = d / "scan00001" / "eiger"
    eiger.mkdir(parents=True)
    paths = []
    for f_idx, sl in enumerate([slice(0, 4), slice(4, 6)]):
        p = eiger / f"test_00001_data_{f_idx:06d}.h5"
        with h5py.File(p, "w") as h:
            h.create_dataset("/entry_0000/instrument/eiger/data", data=np.stack(frames[sl]),
                             chunks=(1,) + frames[0].shape, **hdf5plugin.Bitshuffle(cname="lz4"))
        paths.append(p)
    master = d / "test.h5"
    shp = frames[0].shape
    layout = h5py.VirtualLayout(shape=(6,) + shp, dtype=np.uint32)
    layout[0:4] = h5py.VirtualSource(f"scan00001/eiger/{paths[0].name}",
                                     "/entry_0000/instrument/eiger/data", shape=(4,) + shp)
    layout[4:6] = h5py.VirtualSource(f"scan00001/eiger/{paths[1].name}",
                                     "/entry_0000/instrument/eiger/data", shape=(2,) + shp)
    with h5py.File(master, "w") as h:
        h.create_virtual_dataset("1.1/instrument/eiger/image", layout)
        h["1.1/technique/dim0"] = 3  # fast axis (columns)
        h["1.1/technique/dim1"] = 2  # slow axis (rows)
        h["1.1/title"] = "akmap test"
        h["1.1/sample/name"] = "synthetic"
        h["1.1/instrument/ct34/data"] = np.array([1, 2, 1, 1, 1, 1], float)
    return dict(dir=d, master=str(master), files=[str(p) for p in paths], geo=str(geo_path),
                ai=ai, frames=frames, phases=phases)


def test_geometry_from_nested_workflow_json(scan):
    g = load_geometry(scan["geo"])
    assert g["poni"]["detector"] == "Pilatus100k"
    assert g["polarization_factor"] == pytest.approx(0.99)
    assert g["nbpt_rad"] == 300


def test_sources_from_master_and_files_agree(scan):
    s1, meta = sources_from_master(scan["master"])
    s2 = sources_from_files(scan["files"])
    assert [s.n_frames for s in s1] == [s.n_frames for s in s2] == [4, 2]
    assert meta["shape"] == (2, 3)
    assert "ct34" in meta["monitors"]


def test_mask_catches_flagged_column(scan):
    mask = build_mask(scan["ai"], np.stack(scan["frames"][:3]))
    assert mask[:, 100].all()
    assert not mask[:, 50].any()


def test_integration_recovers_peaks_and_reads_back(scan, tmp_path):
    sources, meta = sources_from_master(scan["master"])
    ai = scan["ai"]
    mask = build_mask(ai, np.stack(scan["frames"][:3]))
    out = integrate_frames(sources, ai, str(tmp_path / "int.h5"), npt=300, mask=mask,
                           monitor=meta["monitors"]["ct34"], sigma_clip=True,
                           shape=meta["shape"], meta=meta, verbose=False)
    xmap = load_ewoks_h5(out)
    assert xmap.shape == (2, 3)
    assert np.isfinite(xmap.intensity).all()
    for k, ph in enumerate(scan["phases"]):
        y = xmap.intensity[k]
        for c in PEAKS[ph]:  # each expected peak is a local maximum within one bin
            j = np.argmin(abs(xmap.q - c))
            assert y[j - 2:j + 3].max() > 3 * np.median(y)
    # frame 1 has twice the counts but monitor = 2 -> same normalised pattern as frame 0
    np.testing.assert_allclose(xmap.intensity[1], xmap.intensity[0], rtol=0.15, atol=2)
    # the bright spot in frame 4 inflates the plain average but not the sigma-clipped one
    clipped = load_ewoks_h5(out, group="eiger_sigmaclip")
    excess_plain = (xmap.intensity[4] - xmap.intensity[0]).max()
    excess_clip = (clipped.intensity[4] - clipped.intensity[0]).max()
    assert excess_plain > 3 * max(excess_clip, 1.0)


def test_subset_of_frames(scan, tmp_path):
    sources = sources_from_files(scan["files"])
    out = integrate_frames(sources, scan["ai"], str(tmp_path / "sub.h5"), npt=300,
                           indices=np.array([2, 3, 4]), verbose=False)
    with h5py.File(out, "r") as h:
        assert h["1.1/eiger_integrate/integrated/intensity"].shape == (3, 300)
        np.testing.assert_array_equal(h["1.1/eiger_integrate/integrated/points"][()], [2, 3, 4])
