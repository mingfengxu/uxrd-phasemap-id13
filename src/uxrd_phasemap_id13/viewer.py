"""Interactive map viewer: imaging map <-> 1D pattern <-> raw 2D detector frame.

    uxrd-phasemap-id13-viewer PROCESSED.h5 [--master RAW.h5 --eiger-dir DIR
                                            --geometry m380.poni --mask mask.edf]

* Map (top left): intensity integrated over the current q window, one value
  per scan point. Click a pixel to select it; arrow keys move the selection.
* 1D pattern (bottom left): the selected pixel (and the map average, grey).
  Drag horizontally to choose a new q window; the map and the rings update.
* 2D detector (right, needs --master/--eiger-dir and --geometry): the raw Eiger
  frame of the selected pixel, masked pixels shaded, and the q window drawn as
  two rings (cyan = inner edge, yellow = outer edge). Toggle to a cake (q-chi)
  view to tell powder rings from single-crystal spots.

Keys: l = log scale, b = background subtraction (SNIP) for map and 1D,
c = raw image / cake, s = save figure, e = export map + pattern as CSV,
arrows = move selected pixel.

Only matplotlib is needed for the map and 1D panels; the 2D panel uses
pyFAI, h5py and hdf5plugin (``pip install -e ".[integrate]"``).
"""
from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from typing import Optional

import numpy as np

from .io import XRDMap, load_ewoks_h5
from .preprocess import snip_baseline


class FrameCache:
    """Reads single raw frames on demand and keeps the last few in memory."""

    def __init__(self, sources, size: int = 16):
        self.sources = sources
        self.size = size
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def __call__(self, index: int) -> np.ndarray:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        from .integrate import iter_frames
        frame = next(iter_frames(self.sources, np.array([index]), chunk=1))[1]
        self._cache[index] = frame
        if len(self._cache) > self.size:
            self._cache.popitem(last=False)
        return frame


def scan_extent(master: Optional[str], entry: str = "1.1"):
    """(x0, x1, y0, y1, xlabel, ylabel) in motor units from the Bliss kmap
    parameters, or None."""
    if not master:
        return None
    import h5py
    try:
        with h5py.File(master, "r") as h5:
            k = h5[f"{entry}/instrument/kmap_parameters"]
            x0, x1 = float(k["x_start"][()]), float(k["x_end"][()])
            y0, y1 = float(k["y_start"][()]), float(k["y_end"][()])
            title = h5[f"{entry}/title"][()].decode() if f"{entry}/title" in h5 else ""
    except (KeyError, OSError):
        return None
    unit = "µm" if abs(x1 - x0) < 1000 else ""
    return x0, x1, y0, y1, f"x ({unit})", f"y ({unit})"


class MapViewer:
    def __init__(self, xmap: XRDMap, frame_provider=None, integrator=None,
                 mask: Optional[np.ndarray] = None, extent=None,
                 qwin=None, title: str = "", snip_iterations: int = 40):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, SpanSelector, TextBox

        self.xmap, self.title = xmap, title
        self.q = xmap.q
        self.I = xmap.intensity
        self.rows, self.cols = xmap.shape
        self.frames, self.ai, self.mask = frame_provider, integrator, mask
        self.extent = extent
        self.snip_iterations = snip_iterations
        self._bg = None          # cached background-subtracted patterns
        self._q2d = None         # cached q (1/A) of every detector pixel
        self.log, self.use_bg, self.cake = False, False, False
        self._updating_boxes = False  # guards TextBox callbacks during programmatic updates
        self.avg = self.I.mean(0)
        if qwin is None:          # default: strongest peak of the average pattern
            base = snip_baseline(self.avg, snip_iterations)
            j = int(np.argmax(self.avg - base))
            dq = 6 * np.median(np.diff(self.q))
            qwin = (self.q[j] - dq, self.q[j] + dq)
        self.qmin, self.qmax = map(float, qwin)
        self.sel = int(np.argmax(self.roi_map()))

        has2d = frame_provider is not None and integrator is not None
        self.fig = plt.figure(figsize=(16, 9) if has2d else (11, 9))
        gs = self.fig.add_gridspec(2, 2 if has2d else 1, width_ratios=[1, 1.05] if has2d else None,
                                   height_ratios=[1.25, 1], left=0.06, right=0.97,
                                   bottom=0.17, top=0.93, wspace=0.18, hspace=0.30)
        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.ax_1d = self.fig.add_subplot(gs[1, 0])
        self.ax_2d = self.fig.add_subplot(gs[:, 1]) if has2d else None
        self.fig.suptitle(title)

        # widgets
        self._tb_min = TextBox(self.fig.add_axes([0.08, 0.025, 0.07, 0.035]), "q min ",
                               initial=f"{self.qmin:.4f}")
        self._tb_max = TextBox(self.fig.add_axes([0.21, 0.025, 0.07, 0.035]), "q max ",
                               initial=f"{self.qmax:.4f}")
        self._tb_min.on_submit(lambda t: self._set_text_q())
        self._tb_max.on_submit(lambda t: self._set_text_q())
        names = [("Log (l)", self.toggle_log), ("Bkg (b)", self.toggle_bg),
                 ("Raw/Cake (c)", self.toggle_cake), ("Save (s)", self.save),
                 ("Export (e)", self.export)]
        self._buttons = []
        for k, (name, fn) in enumerate(names):
            b = Button(self.fig.add_axes([0.32 + k * 0.09, 0.022, 0.08, 0.04]), name)
            b.on_clicked(lambda _evt, fn=fn: fn())
            self._buttons.append(b)
        self._status = self.fig.text(0.06, 0.08, "", fontsize=9, family="monospace")
        self._span = SpanSelector(self.ax_1d, self._on_span, "horizontal", useblit=False,
                                  props=dict(alpha=0.2, facecolor="tab:orange"), interactive=False)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._map_cbar = None
        self.redraw()

    # ------------------------------------------------------------ data
    def patterns(self) -> np.ndarray:
        if not self.use_bg:
            return self.I
        if self._bg is None:
            self._set_status("computing background (SNIP) for all patterns ...")
            self._bg = np.clip(self.I - snip_baseline(self.I, self.snip_iterations), 0, None)
        return self._bg

    def roi_map(self) -> np.ndarray:
        win = (self.q >= self.qmin) & (self.q <= self.qmax)
        if not win.any():
            return np.zeros(self.xmap.n_points)
        return self.patterns()[:, win].sum(1)

    def q_detector(self) -> np.ndarray:
        if self._q2d is None:
            self._q2d = self.ai.qArray() / 10.0   # pyFAI returns nm^-1
        return self._q2d

    def pixel_coords(self, idx: int):
        r, c = divmod(idx, self.cols)
        if self.extent:
            x0, x1, y0, y1 = self.extent[:4]
            return r, c, x0 + (c + 0.5) * (x1 - x0) / self.cols, y0 + (r + 0.5) * (y1 - y0) / self.rows
        return r, c, None, None

    # ------------------------------------------------------------ drawing
    def redraw(self):
        self.draw_map()
        self.draw_1d()
        if self.ax_2d is not None:
            self.draw_2d()
        self.update_status()
        self.fig.canvas.draw_idle()

    def draw_map(self):
        ax = self.ax_map
        ax.clear()
        img = self.xmap.to_image(self.roi_map())
        kw = dict(origin="lower", cmap="inferno", interpolation="nearest", aspect="equal")
        if self.extent:
            kw["extent"] = self.extent[:4]
            ax.set_xlabel(self.extent[4]); ax.set_ylabel(self.extent[5])
        else:
            ax.set_xlabel("column"); ax.set_ylabel("row")
        if self.log:
            from matplotlib.colors import LogNorm
            pos = img[img > 0]
            kw["norm"] = LogNorm(vmin=max(np.percentile(pos, 1), 1e-3) if pos.size else 1e-3,
                                 vmax=max(img.max(), 1e-2))
        else:
            kw["vmin"], kw["vmax"] = np.percentile(img, 0.5), np.percentile(img, 99.8)
        im = ax.imshow(img, **kw)
        if self._map_cbar is None:
            self._map_cbar = self.fig.colorbar(im, ax=ax, shrink=0.85, label="ROI intensity")
        else:
            self._map_cbar.update_normal(im)
        r, c, x, y = self.pixel_coords(self.sel)
        px, py = (x, y) if self.extent else (c, r)
        ax.plot(px, py, marker="s", mfc="none", mec="cyan", ms=9, mew=1.5)
        d1, d2 = 2 * np.pi / self.qmax, 2 * np.pi / max(self.qmin, 1e-6)
        ax.set_title(f"q {self.qmin:.3f}–{self.qmax:.3f} Å⁻¹  (d {d1:.3f}–{d2:.3f} Å)"
                     + ("  [bkg removed]" if self.use_bg else ""), fontsize=10)

    def draw_1d(self):
        ax = self.ax_1d
        ax.clear()
        P = self.patterns()
        avg = P.mean(0)
        sel = P[self.sel]
        if self.log:  # zeros (e.g. after background removal) must not dominate a log axis
            pos = np.concatenate([avg[avg > 0], sel[sel > 0]])
            floor = np.percentile(pos, 2) if pos.size else 1e-3
            avg, sel = np.clip(avg, floor, None), np.clip(sel, floor, None)
        ax.plot(self.q, avg, color="0.6", lw=0.8, label="map average")
        ax.plot(self.q, sel, color="tab:blue", lw=0.9, label="selected pixel")
        ax.axvspan(self.qmin, self.qmax, color="tab:orange", alpha=0.25)
        if self.log:
            ax.set_yscale("log")
        ax.set_xlabel(r"q ($\AA^{-1}$)"); ax.set_ylabel("intensity")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.set_title("drag to choose a q window", fontsize=9, color="0.4")

    def draw_2d(self):
        ax = self.ax_2d
        ax.clear()
        from matplotlib.colors import LogNorm
        try:
            frame = self.frames(self.sel).astype(float)
        except Exception as exc:  # missing file etc.
            ax.text(0.5, 0.5, f"cannot read raw frame:\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, wrap=True)
            return
        mask = self.mask if self.mask is not None else np.zeros(frame.shape, bool)
        if self.cake:
            res = self.ai.integrate2d(frame, 800, 360, unit="q_A^-1", mask=mask,
                                      polarization_factor=0.99, method=("bbox", "csr", "cython"))
            ci = np.where(res.intensity > 0, res.intensity, np.nan)
            ax.imshow(ci, origin="lower", aspect="auto", cmap="inferno",
                      norm=LogNorm(vmin=max(np.nanpercentile(ci, 2), 1e-2), vmax=np.nanpercentile(ci, 99.9)),
                      extent=[res.radial[0], res.radial[-1], res.azimuthal[0], res.azimuthal[-1]])
            for qv, col in ((self.qmin, "cyan"), (self.qmax, "yellow")):
                ax.axvline(qv, color=col, lw=0.8)
            ax.set_xlabel(r"q ($\AA^{-1}$)"); ax.set_ylabel(r"azimuth $\chi$ (°)")
            ax.set_title("cake: powder rings = vertical lines, crystallites = spots", fontsize=9)
        else:
            shown = np.where(mask, np.nan, np.clip(frame, 0.5, None))
            vmax = np.nanpercentile(shown, 99.95)
            ax.imshow(shown, origin="lower", cmap="inferno", interpolation="nearest",
                      norm=LogNorm(vmin=0.5, vmax=max(vmax, 1)))
            ax.imshow(np.where(mask, 1.0, np.nan), origin="lower", cmap="Wistia", vmin=0, vmax=1,
                      alpha=0.5, interpolation="nearest")
            qd = self.q_detector()
            ax.contourf(np.where(mask, np.nan, qd), levels=[self.qmin, self.qmax],
                        colors=["tab:blue"], alpha=0.25)
            ax.contour(qd, levels=[self.qmin], colors="cyan", linewidths=0.8)
            ax.contour(qd, levels=[self.qmax], colors="yellow", linewidths=0.8)
            ax.set_xlabel("detector x (px)"); ax.set_ylabel("detector y (px)")
            ax.set_title("raw frame (log) · cyan/yellow = q window · shaded = masked", fontsize=9)
        r, c, x, y = self.pixel_coords(self.sel)
        where = f"[{x:.2f}, {y:.2f}]" if x is not None else f"row {r}, col {c}"
        ax.text(0.01, 0.99, f"frame {self.sel}  {where}", transform=ax.transAxes, va="top",
                color="w", fontsize=9)

    def update_status(self):
        r, c, x, y = self.pixel_coords(self.sel)
        val = self.roi_map()[self.sel]
        pos = f"x={x:.3f} y={y:.3f}" if x is not None else ""
        self._set_status(f"pixel row {r} col {c} (frame {self.sel}) {pos}   ROI = {val:.4g}   "
                         f"keys: l log · b bkg · c cake · s save · e export · arrows move")

    def _set_status(self, text):
        self._status.set_text(text)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------ actions
    def select(self, idx: int):
        self.sel = int(np.clip(idx, 0, self.xmap.n_points - 1))
        self.redraw()

    def set_q(self, qmin: float, qmax: float):
        if qmax < qmin:
            qmin, qmax = qmax, qmin
        if qmax - qmin < np.median(np.diff(self.q)):
            return
        self.qmin, self.qmax = float(qmin), float(qmax)
        self._updating_boxes = True
        try:
            self._tb_min.set_val(f"{self.qmin:.4f}")
            self._tb_max.set_val(f"{self.qmax:.4f}")
        finally:
            self._updating_boxes = False
        self.redraw()

    def toggle_log(self):
        self.log = not self.log; self.redraw()

    def toggle_bg(self):
        self.use_bg = not self.use_bg; self.redraw()

    def toggle_cake(self):
        if self.ax_2d is not None:
            self.cake = not self.cake; self.draw_2d(); self.fig.canvas.draw_idle()

    def save(self, path: Optional[str] = None):
        path = path or f"viewer_frame{self.sel}_q{self.qmin:.3f}-{self.qmax:.3f}.png"
        self.fig.savefig(path, dpi=150)
        self._set_status(f"saved {os.path.abspath(path)}")
        return path

    def export(self, stem: Optional[str] = None):
        stem = stem or f"viewer_q{self.qmin:.3f}-{self.qmax:.3f}"
        np.savetxt(stem + "_map.csv", self.xmap.to_image(self.roi_map()), delimiter=",")
        np.savetxt(stem + f"_pattern_frame{self.sel}.csv",
                   np.column_stack([self.q, self.patterns()[self.sel], self.patterns().mean(0)]),
                   delimiter=",", header="q_A^-1,selected,map_average", comments="")
        self._set_status(f"exported {stem}_map.csv and pattern CSV")
        return stem

    # ------------------------------------------------------------ events
    def _set_text_q(self):
        if self._updating_boxes:
            return
        try:
            self.set_q(float(self._tb_min.text), float(self._tb_max.text))
        except ValueError:
            pass

    def _on_span(self, qmin, qmax):
        self.set_q(qmin, qmax)

    def _on_click(self, event):
        if event.inaxes is not self.ax_map or event.xdata is None or event.button != 1:
            return
        if self.extent:
            x0, x1, y0, y1 = self.extent[:4]
            c = int((event.xdata - x0) / (x1 - x0) * self.cols)
            r = int((event.ydata - y0) / (y1 - y0) * self.rows)
        else:
            c, r = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.select(r * self.cols + c)

    def _on_key(self, event):
        r, c = divmod(self.sel, self.cols)
        moves = {"left": (0, -1), "right": (0, 1), "up": (1, 0), "down": (-1, 0)}
        if event.key in moves:
            dr, dc = moves[event.key]
            r, c = np.clip(r + dr, 0, self.rows - 1), np.clip(c + dc, 0, self.cols - 1)
            self.select(int(r * self.cols + c))
        elif event.key == "l":
            self.toggle_log()
        elif event.key == "b":
            self.toggle_bg()
        elif event.key == "c":
            self.toggle_cake()
        elif event.key == "s":
            self.save()
        elif event.key == "e":
            self.export()


def build_viewer(processed: str, master: Optional[str] = None, eiger_dir: Optional[str] = None,
                 geometry: Optional[str] = None, mask_file: Optional[str] = None,
                 group: Optional[str] = None, qwin=None, shape=None) -> MapViewer:
    xmap = load_ewoks_h5(processed, shape=shape, group=group)
    frames = ai = mask = None
    if master and geometry:
        from .integrate import build_mask, load_geometry, make_integrator, sources_from_master
        ai = make_integrator(load_geometry(geometry))
        sources, _ = sources_from_master(master, eiger_dir)
        frames = FrameCache(sources)
        probe = np.stack([frames(i) for i in range(min(4, xmap.n_points))])
        mask = build_mask(ai, probe, mask_file)
    title = os.path.basename(processed)
    return MapViewer(xmap, frames, ai, mask, extent=scan_extent(master), qwin=qwin, title=title)


def main(argv=None):
    p = argparse.ArgumentParser(prog="uxrd-phasemap-id13-viewer",
                                description="Interactive map / 1D pattern / raw 2D frame viewer")
    p.add_argument("processed", help="integrated HDF5 (PROCESSED_DATA ..._0001.h5 or our integrate output)")
    p.add_argument("--master", help="RAW_DATA scan master .h5 (enables the raw 2D panel and µm axes)")
    p.add_argument("--eiger-dir", help="folder with the Eiger *_data_*.h5 files")
    p.add_argument("--geometry", help=".poni or azimint.json (needed for the 2D panel)")
    p.add_argument("--mask", help="mask.edf")
    p.add_argument("--group", help="HDF5 group to read, e.g. eiger_sigmaclip")
    p.add_argument("--shape", type=int, nargs=2, metavar=("ROWS", "COLS"))
    p.add_argument("--q", type=float, nargs=2, metavar=("QMIN", "QMAX"), help="initial q window (1/A)")
    a = p.parse_args(argv)
    import matplotlib.pyplot as plt
    build_viewer(a.processed, a.master, a.eiger_dir, a.geometry, a.mask, a.group,
                 qwin=a.q, shape=a.shape)
    plt.show()


if __name__ == "__main__":
    main()
