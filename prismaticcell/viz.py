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
    """In-plane temperature map of the final field at through-plane index ``k_index``
    (default: mid-plane), in degrees C.

    Drawn at the TRUE aspect ratio (the electrode really is a long strip), with a smooth
    (bilinear) rendering of the coarse control-volume field, the hotspot marked, and each tab
    drawn at the electrode edge: a flush weld band plus the physical tab protruding outside.
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
    L_mm, H_mm = extent[1], extent[3]

    fig_w = 13.0
    fig_h = max(fig_w * (H_mm / max(L_mm, 1e-9)) * 1.35 + 1.7, 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(slice_ab.T, origin="lower", extent=extent, aspect="equal",
                   cmap=_CMAP_TEMP, interpolation="bilinear")

    ahot, bhot = np.unravel_index(np.argmax(slice_ab), slice_ab.shape)
    xhot, yhot = _cell_center_mm(geom, ahot, bhot)
    thot = float(slice_ab[ahot, bhot])
    ax.plot(xhot, yhot, marker="o", ms=9, mfc="none", mec="#222222", mew=1.6)
    ax.annotate(f"hotspot {thot:.1f} °C", xy=(xhot, yhot), xytext=(10, -12),
                textcoords="offset points", fontsize=8.5, color="#222222",
                ha="left", va="top")

    _overlay_tabs(geom, tfield, ax, im, dict(getattr(result, "p_tab_final", {}) or {}))

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(labelsize=8.5, length=3)
    ax.set_xlabel(f"length ({_AXNAME[la]}) [mm]", fontsize=9.5)
    ax.set_ylabel(f"height ({_AXNAME[ha]}) [mm]", fontsize=9.5)
    ax.set_title(f"Temperature on the electrode plane  ·  {info}",
                 fontsize=11.5, loc="left", pad=10)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.055,
                        pad=0.16, shrink=0.5, anchor=(0.0, 1.0))
    cbar.set_label("temperature [°C]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


# tab identity colors (Okabe-Ito vermillion / sky blue — CVD-safe pair)
_TAB_STYLE = {"pos": ("#E69F00", "+ tab"), "neg": ("#56B4E9", "− tab")}


def _overlay_tabs(geom, tfield, ax, im=None, p_tab=None, polarities=("pos", "neg")):
    """Draw each tab AT the electrode edge: a flush weld band on the boundary (where the foils
    exit the electrode and bundle into the tab) and the physical tab protruding outside, at its
    real protrusion length. Labels report the SOLVED root temperature (mean of the attachment
    CVs — includes the tab Joule backflow) and sit outside the plane beside the stub.
    """
    from .geometry import tab_attachment_cells
    from matplotlib.patches import Rectangle

    grid = geom.grid
    _, la, ha = _axes(geom)
    d = (grid.dx, grid.dy, grid.dz)
    hw = d[la] * 1e3 / 2.0
    n_ha = (grid.nx, grid.ny, grid.nz)[ha]
    H_mm = n_ha * d[ha] * 1e3                              # electrode height extent [mm]
    p_tab = p_tab or {}
    y_lo, y_hi = ax.get_ylim()
    prot_max_mm = 0.0
    for pol in polarities:
        color, name = _TAB_STYLE[pol]
        cells = tab_attachment_cells(geom, pol)
        if not cells:
            continue
        pts = sorted({(c[la], c[ha]) for c in cells})       # electrode-plane columns of the weld
        xy = [_cell_center_mm(geom, a, b) for (a, b) in pts]
        xs, ys = zip(*xy)
        x0, x1 = min(xs) - hw, max(xs) + hw
        up = float(np.mean(ys)) > 0.5 * H_mm                # weld near the top edge?
        edge = H_mm if up else 0.0
        sgn = 1.0 if up else -1.0
        t_root = _tab_root_temp(geom, tfield, pol)
        prot_mm = float(getattr(geom, "tab_protrusion", {}).get(pol, 0.0)) * 1e3

        # weld band: flush ON the electrode edge (a slim band just inside the boundary)
        band = 0.03 * H_mm
        ax.add_patch(Rectangle((x0, edge - (band if up else 0.0)), x1 - x0, band,
                               facecolor=color, edgecolor="none", alpha=0.95, zorder=6))
        # physical tab: a clean stub outside the edge (neutral fill — its temperature lives in
        # the fin-profile plot; painting it with an off-scale colormap reads as black)
        if prot_mm > 0.0:
            ax.add_patch(Rectangle((x0, edge), x1 - x0, sgn * prot_mm,
                                   facecolor="#f0f0f0", edgecolor=color, linewidth=1.8,
                                   zorder=6, clip_on=False))
            prot_max_mm = max(prot_max_mm, prot_mm)
        # label beside the stub, outside the plane (neutral ink; the colored stub carries identity)
        ax.annotate(f"{name}   root {_k_to_c(t_root):.1f} °C",
                    xy=(x1 + 0.006 * max(x1, 1.0), edge + sgn * 0.5 * max(prot_mm, band)),
                    xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5, color="#333333",
                    annotation_clip=False, zorder=8)
    if prot_max_mm > 0.0:                                    # make room for the protruding tabs
        ax.set_ylim(min(y_lo, 0.0), max(y_hi, H_mm + 1.45 * prot_max_mm))


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

    Matches the thermal solver: an explicit ``geom.tab_sink_t`` (busbar temperature) wins;
    else the mean ``t_inf`` of the non-adiabatic faces; else the coldest active cell.
    """
    override = getattr(geom, "tab_sink_t", None)
    if override is not None:
        return float(override)
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


def plot_collector_planes(result, path: Optional[str] = None):
    """Per-current-collector view: one panel per foil, each with ONLY its own tab.

    Left  = cathode current collector (Al): temperature on the electrode plane with the + tab.
    Right = anode current collector (Cu): the same with the − tab.

    One tab per collector — the + tab exists only on the Al foil, the − tab only on the Cu foil.
    The thermal model homogenizes the sandwich, so both foils share the local temperature field;
    what distinguishes the panels is which tab (and its weld/protrusion) belongs to that foil.
    Each panel also overlays that foil's solved potential drop as contours (from ``phi_pos`` /
    ``phi_neg``, thickness-averaged), showing the in-plane current path toward its tab.
    """
    geom = result.geom
    _, la, ha = _axes(geom)
    tfield = _final_T_field(result)
    plane, info = _electrode_plane(geom, tfield, how="mid")
    slice_ab = _k_to_c(plane)
    extent = _extent_mm(geom)
    L_mm, H_mm = extent[1], extent[3]
    p_tab = dict(getattr(result, "p_tab_final", {}) or {})

    fig_w = 13.0
    strip_h = max(fig_w * (H_mm / max(L_mm, 1e-9)) * 1.35, 1.4)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 2 * strip_h + 2.1), sharex=True)
    fig.subplots_adjust(hspace=0.05)
    panels = (("pos", "Cathode current collector (Al)  ·  + tab", axes[0]),
              ("neg", "Anode current collector (Cu)  ·  − tab", axes[1]))
    im = None
    vmin, vmax = float(np.nanmin(slice_ab)), float(np.nanmax(slice_ab))
    for pol, title, ax in panels:
        im = ax.imshow(slice_ab.T, origin="lower", extent=extent, aspect="equal",
                       cmap=_CMAP_TEMP, interpolation="bilinear", vmin=vmin, vmax=vmax)
        # this foil's potential drop (thickness-averaged over rolls), as sparse subtle contours
        phi_rolls = getattr(result, "phi_final", None) or {}
        maps = [m for m in phi_rolls.get(pol, []) if m is not None]
        if maps:
            phi = np.nanmean(np.stack(maps), axis=0)
            if np.isfinite(phi).any():
                grid = geom.grid
                coords = (grid.xc, grid.yc, grid.zc)
                X, Y = np.meshgrid(coords[la] * 1e3, coords[ha] * 1e3, indexing="ij")
                dv = (phi - np.nanmin(phi)) * 1e3           # mV above the foil minimum
                cs = ax.contour(X, Y, dv, levels=4, colors="white",
                                linewidths=0.7, alpha=0.55)
                ax.clabel(cs, inline=True, fontsize=6.5, fmt="%.0f mV",
                          levels=cs.levels[::2])
        _overlay_tabs(geom, tfield, ax, im, p_tab, polarities=(pol,))
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(labelsize=8, length=3)
        ax.set_ylabel(f"height [mm]", fontsize=9)
        ax.set_title(title, fontsize=10.5, loc="left", pad=8)
    axes[1].set_xlabel(f"length ({_AXNAME[la]}) [mm]", fontsize=9.5)
    cbar = fig.colorbar(im, ax=list(axes), orientation="horizontal", fraction=0.05,
                        pad=0.14, shrink=0.45, anchor=(0.0, 1.0))
    cbar.set_label("temperature [°C]  ·  white contours = foil potential drop toward its tab",
                   fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle(f"Per-collector view ({info}) — one tab per foil", fontsize=12, x=0.02,
                 ha="left")
    return _save(fig, path)


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
