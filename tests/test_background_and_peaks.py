import numpy as np
import pytest

from uxrd_phasemap_id13 import PhaseMapper, make_synthetic_map, peak_position_map, reference_background
from uxrd_phasemap_id13.synthetic import pseudo_voigt


def _cell_like_map(seed=0):
    """Synthetic map plus a narrow, spatially uniform 'cell component' (window/foil)."""
    xmap, truth = make_synthetic_map(shape=(40, 40), n_particles=20, impurity_particles=0, seed=seed)
    rng = np.random.default_rng(seed)
    cell = 60 * pseudo_voigt(xmap.q, 2.095, 0.012) + 30 * pseudo_voigt(xmap.q, 1.37, 0.04)
    scale = rng.uniform(0.95, 1.05, (xmap.n_points, 1))
    xmap.intensity = (xmap.intensity + rng.poisson(cell[None, :] * scale)).astype(np.float32)
    return xmap, truth, cell


def test_reference_background_removes_uniform_component():
    xmap, truth, cell = _cell_like_map()
    q = xmap.q
    empty = truth < 0
    resid, s, ref = reference_background(xmap.intensity, empty)
    w = np.abs(q - 2.095) < 0.01
    before = xmap.intensity[empty][:, w].mean()
    after = resid[empty][:, w].mean()
    assert abs(after) < 0.05 * before
    assert np.median(s) == pytest.approx(1.0, abs=0.05)


def test_reference_mode_keeps_cell_peak_out_of_features():
    xmap, truth, _ = _cell_like_map(seed=1)
    snip = PhaseMapper(n_clusters=2, snip_iterations=30).fit(xmap)
    ref = PhaseMapper(n_clusters=2, snip_iterations=30, background="reference").fit(xmap)
    w = np.abs(snip.q - 2.095) < 0.01
    # SNIP keeps the narrow uniform peak in every pattern; reference subtraction removes it
    assert snip.peaks[truth < 0][:, w].mean() > 10 * max(ref.peaks[truth < 0][:, w].mean(), 0.01)


def test_region_restriction():
    xmap, truth, _ = _cell_like_map(seed=2)
    region = np.zeros(xmap.n_points, bool)
    region[: xmap.n_points // 2] = True
    res = PhaseMapper(n_clusters=2, snip_iterations=30).fit(xmap, points=region)
    assert not res.foreground[~region].any()


def test_peak_position_map_recovers_shifts():
    q = np.linspace(1.0, 1.3, 120)
    centres = np.linspace(1.130, 1.142, 25)
    rng = np.random.default_rng(0)
    I = np.stack([rng.poisson(300 * pseudo_voigt(q, c, 0.006, eta=0) + 20) for c in centres]).astype(float)
    out = peak_position_map(I, q, 1.10, 1.17)
    np.testing.assert_allclose(out["q0"], centres, atol=4e-4)
    np.testing.assert_allclose(out["d"], 2 * np.pi / centres, rtol=5e-4)
    assert np.all(out["q0_err"] < 1e-3)
