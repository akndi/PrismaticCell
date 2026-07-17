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


_AXNAME = {0: "x", 1: "y", 2: "z"}


def _axes(geom):
    """(stack, length, height) physical axis indices, defaulting to (z,x,y)."""
    return (getattr(geom, "stack_axis", 2), getattr(geom, "len_axis", 0), getattr(geom, "hgt_axis", 1))


def _extent_mm(geom):
    """Physical electrode-plane extent [len_min,len_max, hgt_min,hgt_max] in millimetres."""
    grid = geom.grid
    _, la, ha = _axes(geom)
    dn = ((grid.nx, grid.dx), (grid.ny, grid.dy), (grid.nz, grid.dz))
    return [0.0, dn[la][0] * dn[la][1] * 1e3, 0.0, dn[ha][0] * dn[ha][1] * 1e3]


def _cell_center_mm(geom, a, b):
    """(length, height) cell-centre location [mm] for electrode indices (a,b)."""
    grid = geom.grid
    coords = (grid.xc, grid.yc, grid.zc)
    _, la, ha = _axes(geom)
    return float(coords[la][a] * 1e3), float(coords[ha][b] * 1e3)


def _electrode_plane(geom, field3d, how="mid"):
    """Reduce a (nx,ny,nz) field to the electrode plane (length x height).

    how="mid": slice at the mid index along the stack axis; how="mean": average over the
    stack axis (NaN-aware). Returns (plane[a,b], info_string).
    """
    sa, la, ha = _axes(geom)
    if how == "mean":
        mask = np.asarray(geom.active_mask, dtype=bool)
        fm = np.where(mask, field3d, np.nan)
        with np.errstate(invalid="ignore"):
            plane = np.nanmean(fm, axis=sa)
        info = "mean over thickness"
    else:
        s = field3d.shape[sa] // 2
        plane = np.take(field3d, s, axis=sa)
        info = f"mid-thickness ({_AXNAME[sa]} index {s})"
    # remaining axes are (la, ha) in ascending order -> plane[a, b]
    return plane, info


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
    sa, la, ha = _axes(geom)
    tfield = _final_T_field(result)  # (nx, ny, nz)
    if k_index is not None:
        plane = np.take(tfield, int(np.clip(k_index, 0, tfield.shape[sa] - 1)), axis=sa)
        info = f"{_AXNAME[sa]} index {k_index}"
    else:
        plane, info = _electrode_plane(geom, tfield, how="mid")
    slice_ab = _k_to_c(plane)          # electrode plane [a(length), b(height)] in Celsius
    extent = _extent_mm(geom)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(slice_ab.T, origin="lower", extent=extent, aspect="auto", cmap=_CMAP_TEMP)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature [°C]")

    ahot, bhot = np.unravel_index(np.argmax(slice_ab), slice_ab.shape)
    xhot, yhot = _cell_center_mm(geom, ahot, bhot)
    thot = float(slice_ab[ahot, bhot])
    ax.plot(xhot, yhot, marker="x", markersize=12, markeredgewidth=2.5,
            color="cyan", label=f"hotspot {thot:.2f} °C")

    _overlay_tabs(geom, tfield, ax)
    ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel(f"length ({_AXNAME[la]}) [mm]")
    ax.set_ylabel(f"height ({_AXNAME[ha]}) [mm]")
    ax.set_title(f"Temperature on the electrode plane ({info})")
    fig.tight_layout()
    return _save(fig, path)


def _overlay_tabs(geom, tfield, ax):
    """Outline each tab's weld footprint on the electrode plane, labelled with its SOLVED root
    temperature (mean of the attachment CVs — includes the tab Joule backflow).

    The rectangle spans the actual attachment columns (the model's discrete footprint), extended
    by half a cell so it covers the full weld cells.
    """
    from .geometry import tab_attachment_cells
    from matplotlib.patches import Rectangle

    grid = geom.grid
    _, la, ha = _axes(geom)
    d = (grid.dx, grid.dy, grid.dz)
    hw, hh = d[la] * 1e3 / 2.0, d[ha] * 1e3 / 2.0
    style = {"pos": ("#ff5533", "+ tab"), "neg": ("#40c4ff", "− tab")}
    for pol, (color, name) in style.items():
        cells = tab_attachment_cells(geom, pol)
        if not cells:
            continue
        pts = sorted({(c[la], c[ha]) for c in cells})       # electrode-plane columns of the weld
        xy = [_cell_center_mm(geom, a, b) for (a, b) in pts]
        xs, ys = zip(*xy)
        rect = Rectangle((min(xs) - hw, min(ys) - hh),
                         (max(xs) - min(xs)) + 2 * hw, (max(ys) - min(ys)) + 2 * hh,
                         fill=False, edgecolor=color, linewidth=2.2, zorder=6)
        ax.add_patch(rect)
        t_root = _k_to_c(_tab_root_temp(geom, tfield, pol))
        # label below the footprint when it hugs the top edge of the plane, else above
        y_lo, y_hi = ax.get_ylim()
        near_top = (max(ys) + hh) > y_lo + 0.85 * (y_hi - y_lo)
        xy = (0.5 * (min(xs) + max(xs)), (min(ys) - hh) if near_top else (max(ys) + hh))
        ax.annotate(f"{name}  {t_root:.2f} °C", xy=xy,
                    xytext=(0, -6 if near_top else 6), textcoords="offset points",
                    ha="center", va="top" if near_top else "bottom",
                    fontsize=8, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, ec="none"),
                    zorder=7)


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
    sa, la, ha = _axes(geom)
    j_field = np.asarray(result.j_field_final, dtype=float)  # (nx, ny, nz)
    inplane, _ = _electrode_plane(geom, j_field, how="mean")  # electrode plane [a,b]

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

    ax.set_xlabel(f"length ({_AXNAME[la]}) [mm]")
    ax.set_ylabel(f"height ({_AXNAME[ha]}) [mm]")
    ax.set_title("Areal current density on the electrode plane (mean over thickness)")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# tab (1-D fin) temperature
# --------------------------------------------------------------------------- #
def _tab_sink_temp(cooling, geom, tfield) -> float:
    """Terminal/busbar sink temperature the tab far-end is heat-sunk to.

    Matches the thermal solver: the mean ``t_inf`` of the non-adiabatic faces. If ``cooling`` is
    not supplied, fall back to the coldest active cell (a reasonable coolant-side proxy).
    """
    if cooling is not None:
        faces = (cooling.top, cooling.bottom, cooling.x_min, cooling.x_max,
                 cooling.y_min, cooling.y_max)
        tinfs = [f.t_inf for f in faces if f.kind in ("convection", "dirichlet")]
        if tinfs:
            return float(np.mean(tinfs))
    mask = np.asarray(geom.active_mask, dtype=bool)
    return float(np.nanmin(np.where(mask, tfield, np.nan)))


def _tab_root_temp(geom, tfield, polarity) -> float:
    """Mean solved temperature of the control volumes a tab is welded to (its root).

    With two-way tab coupling the solved field already includes the tab's Joule backflow
    (half of I^2*R_tab injected at these cells), so no extra correction is applied here.
    """
    from .geometry import tab_attachment_cells
    vals = [float(tfield[i, j, k]) for (i, j, k) in tab_attachment_cells(geom, polarity)]
    return float(np.mean(vals)) if vals else float("nan")


def tab_thermal_profiles(result, cooling=None, n: int = 25) -> dict:
    """1-D fin temperature profile along each tab, derived from the solved field.

    The tab is a lumped conductor in the model (not a control volume). Its steady axial profile
    with uniform ohmic self-heating is fully set by the model's own lumped quantities:

        T(xi) = T_root + (T_sink - T_root) * xi  +  P / (2 G) * xi (1 - xi),   xi = x/L in [0, 1]

    where ``P = I^2 / g_tab`` is the tab ohmic dissipation, ``G = tab_heat_cond`` [W/K] is the tab's
    conduction-to-sink, ``T_root`` is the mean temperature of the attachment CVs (from the solved
    field) and ``T_sink`` the heat-sunk terminal temperature. Returns per polarity a dict with
    ``xi``, ``T`` [K], ``T_root``, ``T_sink``, ``T_peak`` [K], ``P`` [W] and ``R_tab`` [Ohm].
    ``T`` is None when the tab is not heat-sunk (``tab_heat_cond = 0``): with no sink the steady
    profile is undefined, so only ``T_root`` is meaningful.
    """
    geom = result.geom
    tfield = _final_T_field(result)
    i_term = float(np.asarray(result.i_terminal, dtype=float).ravel()[-1])
    t_sink = _tab_sink_temp(cooling, geom, tfield)
    p_exact = dict(getattr(result, "p_tab_final", {}) or {})
    xi = np.linspace(0.0, 1.0, int(max(n, 2)))
    out = {}
    for pol, gtab, gcond in (
        ("pos", float(getattr(geom, "g_tab_pos", 0.0)), float(getattr(geom, "tab_heat_cond_pos", 0.0))),
        ("neg", float(getattr(geom, "g_tab_neg", 0.0)), float(getattr(geom, "tab_heat_cond_neg", 0.0))),
    ):
        t_root = _tab_root_temp(geom, tfield, pol)
        r_tab = 1.0 / gtab if gtab > 0.0 else float("inf")
        # prefer the exact solved tab dissipation carried on the Result; lumped I^2*R fallback
        p_tab = float(p_exact.get(pol, 0.0)) or (i_term * i_term * r_tab if gtab > 0.0 else 0.0)
        if gcond > 0.0 and np.isfinite(p_tab):
            T = t_root + (t_sink - t_root) * xi + (p_tab / (2.0 * gcond)) * xi * (1.0 - xi)
            t_peak = float(np.max(T))
        else:
            T = None
            t_peak = t_root
        out[pol] = dict(xi=xi, T=T, T_root=t_root, T_sink=t_sink, T_peak=t_peak,
                        P=p_tab, R_tab=r_tab)
    return out


def plot_tab_temperature(result, cooling=None, path: Optional[str] = None):
    """Plot the 1-D temperature profile along each tab (root -> heat-sunk tip), including tab
    ohmic self-heating. Shows T_root (solved), the peak, and the terminal sink temperature.
    """
    prof = tab_thermal_profiles(result, cooling=cooling)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"pos": "#c0392b", "neg": "#2c3e50"}
    labels = {"pos": "+ tab", "neg": "− tab"}
    for pol, p in prof.items():
        c = colors[pol]
        if p["T"] is None:
            ax.axhline(_k_to_c(p["T_root"]), color=c, ls=":",
                       label=f"{labels[pol]} root {_k_to_c(p['T_root']):.2f} °C (not heat-sunk)")
            continue
        ax.plot(p["xi"], _k_to_c(p["T"]), color=c, lw=2,
                label=f"{labels[pol]}: root {_k_to_c(p['T_root']):.2f} → peak "
                      f"{_k_to_c(p['T_peak']):.2f} °C, I²R={p['P']:.2g} W")
        ipk = int(np.argmax(p["T"]))
        ax.plot(p["xi"][ipk], _k_to_c(p["T"][ipk]), marker="o", color=c, ms=7)
    t_sink = next(iter(prof.values()))["T_sink"]
    ax.axhline(_k_to_c(t_sink), color="gray", ls="--", lw=1,
               label=f"terminal sink {_k_to_c(t_sink):.2f} °C")
    ax.set_xlabel("position along tab  (0 = root / weld,  1 = heat-sunk tip)")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Tab temperature (1-D fin with I²R self-heating)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
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
