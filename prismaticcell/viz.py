"""Matplotlib visualization for :class:`~prismaticcell.coupling.Result` objects and
design sweeps (``api.Sweep.run``).

All plotting functions build and return a :class:`matplotlib.figure.Figure`; if a
``path`` is given the figure is also written there (PNG inferred from the extension).
No physics or geometry is hard-coded -- physical extents, cell sizes and the active-cell
mask are all read from ``result.geom``. The module is headless-safe: the ``Agg`` backend
is selected at import time, so it works with no display and never calls ``plt.show()``.

Robustness notes:
* Temperatures are stored in kelvin and converted to degrees Celsius for display.
* Inactive control volumes are masked to ``NaN`` so they render blank.
* Steady-mode results carry a single time point; the time-series plot degrades to
  labelled markers in that case.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend, selected before pyplot import

import matplotlib.pyplot as plt
import numpy as np

_KELVIN_OFFSET = 273.15
_CMAP_TEMP = "inferno"
_CMAP_CURRENT = "viridis"
_CMAP_SWEEP = "viridis"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _k_to_c(t_kelvin):
    """Kelvin -> Celsius (works on scalars and arrays)."""
    return np.asarray(t_kelvin, dtype=float) - _KELVIN_OFFSET


def _closure_rel(result) -> Optional[float]:
    """Best-effort extraction of the energy-balance relative closure."""
    eb = getattr(result, "energy_balance", None)
    if isinstance(eb, dict):
        val = eb.get("closure_rel")
        if val is not None:
            return float(val)
    return None


def _final_T_field(result) -> np.ndarray:
    """Return the final (nx,ny,nz) temperature field from a Result.

    ``T_field`` is stored as ``(nt, nx, ny, nz)``; we take the last snapshot. If a bare
    3-D field was supplied we use it as-is.
    """
    tf = np.asarray(result.T_field, dtype=float)
    if tf.ndim == 4:
        return tf[-1]
    if tf.ndim == 3:
        return tf
    raise ValueError(f"unexpected T_field ndim={tf.ndim}; expected 3 or 4")


def _extent_mm(geom):
    """Physical in-plane extent [xmin, xmax, ymin, ymax] in millimetres."""
    grid = geom.grid
    lx = grid.nx * grid.dx
    ly = grid.ny * grid.dy
    return [0.0, lx * 1e3, 0.0, ly * 1e3]


def _cell_center_mm(geom, i, j):
    """(x,y) cell-centre location in millimetres for in-plane index (i,j)."""
    grid = geom.grid
    return float(grid.xc[i] * 1e3), float(grid.yc[j] * 1e3)


def _save(fig, path):
    if path is not None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# time series
# --------------------------------------------------------------------------- #
def plot_time_series(result, path: Optional[str] = None):
    """Multi-panel time history: terminal voltage, current, mean SOC, and
    T_max / T_mean / T_min (degrees C) versus time.

    For a steady-mode result (a single time point) the traces degrade to markers so the
    single value is still visible. The suptitle reports the energy-balance ``closure_rel``.
    """
    t = np.asarray(result.t, dtype=float)
    single = t.size <= 1
    style = dict(marker="o", linestyle="none") if single else dict(marker="", linestyle="-")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    ax_v, ax_i, ax_soc, ax_t = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax_v.plot(t, np.asarray(result.v_terminal, dtype=float), color="C0", **style)
    ax_v.set_ylabel("Terminal voltage [V]")
    ax_v.set_title("Terminal voltage")
    ax_v.grid(True, alpha=0.3)

    ax_i.plot(t, np.asarray(result.i_terminal, dtype=float), color="C3", **style)
    ax_i.set_ylabel("Terminal current [A]")
    ax_i.set_title("Terminal current")
    ax_i.grid(True, alpha=0.3)

    ax_soc.plot(t, np.asarray(result.soc_mean, dtype=float), color="C2", **style)
    ax_soc.set_ylabel("Mean SOC [-]")
    ax_soc.set_xlabel("Time [s]")
    ax_soc.set_title("Mean state of charge")
    ax_soc.grid(True, alpha=0.3)

    ax_t.plot(t, _k_to_c(result.T_max), color="C1", label="T_max", **style)
    ax_t.plot(t, _k_to_c(result.T_mean), color="C4", label="T_mean", **style)
    ax_t.plot(t, _k_to_c(result.T_min), color="C0", label="T_min", **style)
    ax_t.set_ylabel("Temperature [°C]")
    ax_t.set_xlabel("Time [s]")
    ax_t.set_title("Temperature (max / mean / min)")
    ax_t.legend(loc="best", fontsize=8)
    ax_t.grid(True, alpha=0.3)

    closure = _closure_rel(result)
    suffix = f"  |  energy closure_rel = {closure:.2e}" if closure is not None else ""
    mode = "steady (single point)" if single else "transient"
    fig.suptitle(f"PrismaticCell time series [{mode}]{suffix}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# temperature slice
# --------------------------------------------------------------------------- #
def plot_temperature_slice(result, k_index: Optional[int] = None, path: Optional[str] = None):
    """In-plane (x,y) temperature heatmap of the final field at through-plane index
    ``k_index`` (default: mid-plane), in degrees C, with colorbar, physical mm axes, and
    a marker at the hotspot (hottest cell in the displayed slice).
    """
    geom = result.geom
    grid = geom.grid
    tfield = _final_T_field(result)  # (nx, ny, nz)
    nz = tfield.shape[2]
    if k_index is None:
        k_index = nz // 2
    k_index = int(np.clip(k_index, 0, nz - 1))

    slice_xy = _k_to_c(tfield[:, :, k_index])  # (nx, ny) in Celsius
    extent = _extent_mm(geom)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    # array is indexed [i(x), j(y)]; transpose so x is horizontal, y vertical.
    im = ax.imshow(
        slice_xy.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=_CMAP_TEMP,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature [°C]")

    # hotspot within the displayed slice
    ihot, jhot = np.unravel_index(np.argmax(slice_xy), slice_xy.shape)
    xhot, yhot = _cell_center_mm(geom, ihot, jhot)
    thot = float(slice_xy[ihot, jhot])
    ax.plot(xhot, yhot, marker="x", markersize=12, markeredgewidth=2.5,
            color="cyan", label=f"hotspot {thot:.2f} °C")
    ax.legend(loc="upper right", fontsize=8)

    z_mm = float(grid.zc[k_index] * 1e3)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Temperature slice at k={k_index} (z = {z_mm:.2f} mm)")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# current distribution
# --------------------------------------------------------------------------- #
def plot_current_distribution(result, path: Optional[str] = None):
    """In-plane heatmap of the final areal current density, averaged over the active
    z-cells of each column (inactive columns masked to NaN). Reveals current
    concentration near the tabs. Colorbar in A/m^2; tab node locations overlaid when
    reachable.
    """
    geom = result.geom
    j_field = np.asarray(result.j_field_final, dtype=float)  # (nx, ny, nz)

    mask = np.asarray(geom.active_mask, dtype=bool)
    j_masked = np.where(mask, j_field, np.nan)
    with np.errstate(invalid="ignore"):
        # mean over through-plane axis; all-NaN columns -> NaN (blanked)
        inplane = np.full(j_masked.shape[:2], np.nan)
        valid_cols = np.any(mask, axis=2)
        if valid_cols.any():
            inplane[valid_cols] = np.nanmean(j_masked, axis=2)[valid_cols]

    extent = _extent_mm(geom)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(
        inplane.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=_CMAP_CURRENT,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Areal current density [A/m$^2$]")

    # Overlay tab node locations if reachable (optional, guarded).
    try:
        pos_pts, neg_pts = [], []
        for roll in geom.rolls:
            for (i, j) in getattr(roll, "tab_pos_nodes", []):
                pos_pts.append(_cell_center_mm(geom, i, j))
            for (i, j) in getattr(roll, "tab_neg_nodes", []):
                neg_pts.append(_cell_center_mm(geom, i, j))
        if pos_pts:
            px, py = zip(*pos_pts)
            ax.scatter(px, py, marker="^", s=70, edgecolors="white",
                       facecolors="red", label="+ tab", zorder=5)
        if neg_pts:
            nx_, ny_ = zip(*neg_pts)
            ax.scatter(nx_, ny_, marker="v", s=70, edgecolors="white",
                       facecolors="black", label="- tab", zorder=5)
        if pos_pts or neg_pts:
            ax.legend(loc="upper right", fontsize=8)
    except Exception:
        pass  # tab overlay is best-effort only

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Areal current density (through-plane mean over active cells)")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# sweep heatmap
# --------------------------------------------------------------------------- #
def plot_sweep_heatmap(rows: Sequence[dict], x: str, y: str, z: str,
                       path: Optional[str] = None):
    """Design-sweep heatmap: pivot ``rows`` (list of dicts) on keys ``x`` and ``y`` and
    color by scalar key ``z``.

    A regular grid is rendered as an annotated heatmap with ticks at the unique x/y
    values. If (x, y) do not form a clean grid, it falls back to a colored scatter.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("plot_sweep_heatmap: `rows` is empty")

    xs = [float(r[x]) for r in rows]
    ys = [float(r[y]) for r in rows]
    zs = [float(r[z]) for r in rows]

    xu = sorted(set(xs))
    yu = sorted(set(ys))

    # Build a lookup; detect whether every (x,y) combination appears exactly once.
    grid = {}
    duplicate = False
    for xi, yi, zi in zip(xs, ys, zs):
        key = (xi, yi)
        if key in grid:
            duplicate = True
        grid[key] = zi
    is_full_grid = (not duplicate) and (len(grid) == len(xu) * len(yu))

    fig, ax = plt.subplots(figsize=(1.6 * max(len(xu), 4), 1.3 * max(len(yu), 4)))

    if is_full_grid:
        Z = np.full((len(yu), len(xu)), np.nan)  # rows=y, cols=x
        xi_of = {v: i for i, v in enumerate(xu)}
        yi_of = {v: j for j, v in enumerate(yu)}
        for (xi, yi), zi in grid.items():
            Z[yi_of[yi], xi_of[xi]] = zi

        im = ax.imshow(Z, origin="lower", aspect="auto", cmap=_CMAP_SWEEP)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(z)

        ax.set_xticks(range(len(xu)))
        ax.set_xticklabels([f"{v:g}" for v in xu])
        ax.set_yticks(range(len(yu)))
        ax.set_yticklabels([f"{v:g}" for v in yu])

        # annotate each cell with its z value
        zmin, zmax = np.nanmin(Z), np.nanmax(Z)
        mid = 0.5 * (zmin + zmax)
        for jj in range(len(yu)):
            for ii in range(len(xu)):
                val = Z[jj, ii]
                if np.isnan(val):
                    continue
                txt_color = "white" if val < mid else "black"
                ax.text(ii, jj, f"{val:.3g}", ha="center", va="center",
                        color=txt_color, fontsize=8)
        ax.set_title(f"Sweep: {z} over ({x}, {y})")
    else:
        sc = ax.scatter(xs, ys, c=zs, cmap=_CMAP_SWEEP, s=120,
                        edgecolors="black", linewidths=0.5)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(z)
        ax.set_title(f"Sweep (scatter, irregular grid): {z} over ({x}, {y})")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    return _save(fig, path)
