import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from uxrd_phasemap_id13 import (PhaseMapper, XRDMap, load_ewoks_h5, make_synthetic_map,
                        normalized_ratio, phase_fraction, roi_area, snip_baseline, window_sum)
from uxrd_phasemap_id13.preprocess import build_features, otsu_threshold
from uxrd_phasemap_id13.synthetic import hexagonal_q, pseudo_voigt


# ---------------------------------------------------------------- preprocessing
def test_snip_recovers_smooth_background():
    q = np.linspace(0.5, 5, 1000)
    bg = 50 + 30 * np.exp(-((q - 2) / 1.0) ** 2)
    peaks = 400 * pseudo_voigt(q, 2.5, 0.02) + 200 * pseudo_voigt(q, 3.3, 0.02)
    # window must be wider than the peaks but narrower than the background hump
    est = snip_baseline(bg + peaks, iterations=20)
    far = (np.abs(q - 2.5) > 0.1) & (np.abs(q - 3.3) > 0.1)
    assert np.max(np.abs(est[far] - bg[far]) / bg[far]) < 0.03
    # peaks survive subtraction
    assert (bg + peaks - est)[np.argmin(abs(q - 2.5))] > 300


def test_snip_vectorised_matches_single():
    rng = np.random.default_rng(1)
    Y = rng.poisson(20, (5, 300)).astype(float)
    np.testing.assert_allclose(snip_baseline(Y, 20)[2], snip_baseline(Y[2], 20))


def test_otsu_splits_bimodal():
    rng = np.random.default_rng(0)
    v = np.r_[rng.normal(0, 1, 1000), rng.normal(10, 1, 1000)]
    assert 3 < otsu_threshold(v) < 7


def test_features_are_unit_norm_and_scale_invariant():
    rng = np.random.default_rng(0)
    P = rng.random((4, 50))
    F1, F2 = build_features(P), build_features(P * 37.0)
    np.testing.assert_allclose(np.linalg.norm(F1, axis=1), 1.0)
    np.testing.assert_allclose(F1, F2)


# ---------------------------------------------------------------- physics helpers
def test_hexagonal_q_basal_reflection():
    # 00l: d = c / l
    assert hexagonal_q(2.88, 11.1, [(0, 0, 2)])[0] == pytest.approx(2 * np.pi / 5.55)


def test_roi_area_linear_background():
    q = np.linspace(2, 3, 1001)
    y = 10 + 5 * q + 100 * pseudo_voigt(q, 2.5, 0.01, eta=0)
    area = roi_area(y, q, 2.5, half_width=0.03)
    expected = (100 * pseudo_voigt(q, 2.5, 0.01, eta=0))[np.abs(q - 2.5) <= 0.03].sum()
    assert area == pytest.approx(expected, rel=0.02)


def test_phase_fraction_sums_to_one():
    q = np.linspace(2, 3.5, 800)
    y = np.stack([30 * pseudo_voigt(q, 2.6, 0.01) + 10 * pseudo_voigt(q, 2.8, 0.01)])
    f = phase_fraction(y, q, {"A": [2.6], "B": [2.8]})
    assert f["A"][0] + f["B"][0] == pytest.approx(1)
    assert f["A"][0] == pytest.approx(0.75, abs=0.02)


# ---------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic_map(shape=(60, 60), n_particles=40, impurity_particles=3, seed=3)


def test_pipeline_recovers_synthetic_phases(synthetic):
    xmap, truth = synthetic
    res = PhaseMapper(exclude_q=[(1.0, 1.25)], n_clusters=3, snip_iterations=30).fit(xmap)
    fg = res.foreground & (truth >= 0)
    # points through the thick part of a particle are detected, empty points are not
    core = xmap.meta["thickness"] > 0.3
    assert res.foreground[core].mean() > 0.95
    assert (res.foreground & (truth < 0)).sum() / (truth < 0).sum() < 0.02
    assert adjusted_rand_score(truth[fg], res.labels[fg]) > 0.9


def test_automatic_k_selection(synthetic):
    xmap, _ = synthetic
    res = PhaseMapper(exclude_q=[(1.0, 1.25)], n_clusters=[2, 3, 4, 5], snip_iterations=30).fit(xmap)
    assert set(res.k_scan) == {2, 3, 4, 5}
    assert res.n_clusters == 3


def test_gmm_and_nmf_run(synthetic):
    xmap, truth = synthetic
    res = PhaseMapper(exclude_q=[(1.0, 1.25)], n_clusters=3, method="gmm", n_nmf=3,
                      snip_iterations=30).fit(xmap)
    assert res.probabilities.shape == (res.foreground.sum(), 3)
    np.testing.assert_allclose(res.probabilities.sum(1), 1, atol=1e-6)
    assert res.nmf_maps.shape == (xmap.n_points, 3)
    assert np.all(res.nmf_patterns >= 0)


def test_clusters_sorted_by_size(synthetic):
    xmap, _ = synthetic
    res = PhaseMapper(n_clusters=3, snip_iterations=30).fit(xmap)
    sizes = res.cluster_sizes()
    assert np.all(np.diff(sizes) <= 0)


# ---------------------------------------------------------------- I/O
def test_load_ewoks_h5_roundtrip(tmp_path):
    h5py = pytest.importorskip("h5py")
    I = np.random.default_rng(0).random((6, 20)).astype(np.float32)
    q = np.linspace(1, 5, 20)
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as h:
        g = h.create_group("1.1/eiger_integrate/integrated")
        g["intensity"], g["q"] = I, q
        h["1.1/eiger_integrate/configuration/data"] = '{"poni": {"wavelength": 9.5e-11}}'
    xmap = load_ewoks_h5(str(path), shape=(2, 3))
    np.testing.assert_allclose(xmap.intensity, I)
    assert xmap.shape == (2, 3)
    assert xmap.meta["wavelength_A"] == pytest.approx(0.95)


def test_xrdmap_validates_shape():
    with pytest.raises(ValueError):
        XRDMap(np.zeros((10, 5)), np.arange(5), (3, 3))


def test_normalized_ratio_matches_published_definition():
    a = np.array([10.0, 5.0, 1.0, 8.0])     # P2 102 ROI
    b = np.array([4.0, 8.0, 8.0, 0.5])      # P3 012 ROI
    r = normalized_ratio(a, b, threshold=0.15)
    # a/max = [1, .5, .1, .8]; b/max = [.5, 1, 1, .0625]
    np.testing.assert_allclose(r[:2], [1 / 1.5, 0.5 / 1.5])
    assert np.isnan(r[2]) and np.isnan(r[3])  # below 15 % in one channel


def test_window_sum():
    q = np.linspace(0, 1, 11)
    I = np.arange(22, dtype=float).reshape(2, 11)
    np.testing.assert_allclose(window_sum(I, q, 0.25, 0.45), [3 + 4, 14 + 15])
