"""Headless tests for the interactive viewer (matplotlib Agg backend)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from uxrd_phasemap_id13 import make_synthetic_map
from uxrd_phasemap_id13.viewer import MapViewer


class _Evt:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.fixture
def viewer():
    xmap, _ = make_synthetic_map(shape=(20, 30), n_particles=12, seed=2)
    return MapViewer(xmap, qwin=(1.0, 1.25), title="test")


def test_map_follows_q_window(viewer):
    m1 = viewer.roi_map().copy()
    viewer.set_q(2.6, 2.8)
    assert (viewer.qmin, viewer.qmax) == (2.6, 2.8)
    assert viewer._tb_min.text == "2.6000" and viewer._tb_max.text == "2.8000"
    assert not np.allclose(m1, viewer.roi_map())


def test_typed_q_window_is_kept(viewer):
    viewer._tb_max.set_val("1.40")
    viewer._set_text_q()
    assert viewer.qmax == pytest.approx(1.40) and viewer.qmin == pytest.approx(1.0)


def test_click_and_arrow_keys_select_pixels(viewer):
    viewer._on_click(_Evt(inaxes=viewer.ax_map, button=1, xdata=7.2, ydata=4.1))
    assert divmod(viewer.sel, 30) == (4, 7)
    viewer._on_key(_Evt(key="up"))
    viewer._on_key(_Evt(key="right"))
    assert divmod(viewer.sel, 30) == (5, 8)


def test_background_toggle_and_export(viewer, tmp_path):
    raw = viewer.roi_map().copy()
    viewer.toggle_bg()
    assert viewer.use_bg and (viewer.roi_map() <= raw + 1e-6).all()
    viewer.toggle_log()
    stem = viewer.export(str(tmp_path / "v"))
    assert np.loadtxt(stem + "_map.csv", delimiter=",").shape == (20, 30)
    assert (tmp_path / "v.png").exists() is False
    viewer.save(str(tmp_path / "v.png"))
    assert (tmp_path / "v.png").exists()


def test_raw_panel_with_frames(tmp_path):
    pyFAI = pytest.importorskip("pyFAI")
    from uxrd_phasemap_id13.integrate import make_integrator
    ai = make_integrator({"poni": {"dist": 0.1, "poni1": 0.0073, "poni2": 0.0183,
                                   "wavelength": 0.73e-10, "detector": "Pilatus100k"}})
    xmap, _ = make_synthetic_map(shape=(4, 5), n_particles=4, seed=1, q_range=(0.3, 1.2))
    frames = lambda i: ai.calcfrom1d(xmap.q, xmap.intensity[i], dim1_unit="q_A^-1")
    v = MapViewer(xmap, frames, ai, np.zeros(ai.detector.shape, bool), qwin=(0.6, 0.7))
    v.select(7)
    v.toggle_cake()
    v.save(str(tmp_path / "raw.png"))
    assert (tmp_path / "raw.png").exists()
