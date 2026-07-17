"""3-D field views of the cell for notebooks: exterior, cutaway with internals, jellyrolls.

Each view renders the cell (or its rolls) as boxes whose faces are colored by a solved field —
temperature, SOC, or areal current density — at a chosen time index ``ti``, and is designed to be
driven by a notebook time slider (call again with a new ``ti``) or saved as an animation with
:func:`save_animation`. Color limits are computed over the WHOLE history so animation colors are
stable in time.

Requires a transient :class:`~prismaticcell.coupling.Result`; SOC/current views need the run to
have recorded field histories (``solver.save_fields = true``). Nothing is hard-coded — geometry,
extents, tab positions and the fields all come from the Result.
"""
from __future__ import annotations

from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .viz import (_axes, field_frames, field_display, field_norm,
                  FIELD_CMAPS as _CMAPS, FIELD_LABELS as _LABEL)


def _frame(result, field: str, ti: int) -> np.ndarray:
    frames = field_display(field_frames(result, field), field)
    return frames[int(np.clip(ti, -frames.shape[0], frames.shape[0] - 1))]


# --------------------------------------------------------------------------- #
# drawing primitives
# --------------------------------------------------------------------------- #
def _edges_mm(geom):
    """Cell-EDGE coordinate vectors (mm) per physical axis."""
    g = geom.grid
    return (np.arange(g.nx + 1) * g.dx * 1e3,
            np.arange(g.ny + 1) * g.dy * 1e3,
            np.arange(g.nz + 1) * g.dz * 1e3)


def _face(ax, vals2d, u_edges, v_edges, const, axis, cmap, norm, scale=(1, 1, 1)):
    """Draw one axis-aligned face colored per cell from ``vals2d`` (nan -> gray)."""
    U, V = np.meshgrid(u_edges, v_edges, indexing="ij")
    W = np.full_like(U, const)
    xyz = {0: (W, U, V), 1: (U, W, V), 2: (U, V, W)}[axis]
    C = cmap(norm(vals2d))
    C[~np.isfinite(vals2d)] = (0.72, 0.72, 0.72, 1.0)
    ax.plot_surface(xyz[0] * scale[0], xyz[1] * scale[1], xyz[2] * scale[2],
                    facecolors=C, shade=False, rstride=1, cstride=1,
                    linewidth=0, antialiased=False)


def _box_faces(ax, vol, edges, lo, hi, cmap, norm, scale):
    """Draw the 6 outer faces of the sub-box vol[lo0:hi0, lo1:hi1, lo2:hi2]."""
    (l0, l1, l2), (h0, h1, h2) = lo, hi
    ex, ey, ez = edges
    sub = vol[l0:h0, l1:h1, l2:h2]
    exs, eys, ezs = ex[l0:h0 + 1], ey[l1:h1 + 1], ez[l2:h2 + 1]
    _face(ax, sub[0, :, :], eys, ezs, exs[0], 0, cmap, norm, scale)
    _face(ax, sub[-1, :, :], eys, ezs, exs[-1], 0, cmap, norm, scale)
    _face(ax, sub[:, 0, :], exs, ezs, eys[0], 1, cmap, norm, scale)
    _face(ax, sub[:, -1, :], exs, ezs, eys[-1], 1, cmap, norm, scale)
    _face(ax, sub[:, :, 0], exs, eys, ezs[0], 2, cmap, norm, scale)
    _face(ax, sub[:, :, -1], exs, eys, ezs[-1], 2, cmap, norm, scale)


def _wire_box(ax, p0, p1, scale, color="#555555", lw=0.9):
    x0, y0, z0 = [p * s for p, s in zip(p0, scale)]
    x1, y1, z1 = [p * s for p, s in zip(p1, scale)]
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    ax.add_collection3d(Line3DCollection([(c[a], c[b]) for a, b in e], colors=color,
                                         linewidths=lw))


def _setup_axes(ax, geom, scale, title):
    ex, ey, ez = _edges_mm(geom)
    dims = (ex[-1] * scale[0], ey[-1] * scale[1], ez[-1] * scale[2])
    ax.set_box_aspect(dims)
    ax.set_xlabel("x [mm]", fontsize=8)
    ax.set_ylabel("y [mm]", fontsize=8)
    ax.set_zlabel("z [mm]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)


def _colorbar(fig, ax, cmap, norm, field):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.10)
    cb.set_label(_LABEL[field], fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_visible(False)


def _prep(result, field, ti, vminmax):
    vol = _frame(result, field, ti)
    vmin, vmax = vminmax if vminmax is not None else field_norm(result, field)
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap(_CMAPS[field])
    t = float(np.asarray(result.t).ravel()[int(np.clip(ti, -len(result.t), len(result.t) - 1))])
    return vol, norm, cmap, t


# --------------------------------------------------------------------------- #
# the three views
# --------------------------------------------------------------------------- #
def plot_cell_3d(result, ti: int = 0, field: str = "T",
                 thickness_scale: float = 6.0, vminmax=None, path: Optional[str] = None):
    """(1) Full cell, EXTERIOR only: the outer surface colored by the field at time index ``ti``.

    The through-plane (thin) axis is drawn ``thickness_scale``x exaggerated for visibility; axis
    labels keep true mm.
    """
    geom = result.geom
    sa, _, _ = _axes(geom)
    scale = [1.0, 1.0, 1.0]
    scale[sa] = thickness_scale
    vol, norm, cmap, t = _prep(result, field, ti, vminmax)
    edges = _edges_mm(geom)
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111, projection="3d")
    nx, ny, nz = vol.shape
    _box_faces(ax, vol, edges, (0, 0, 0), (nx, ny, nz), cmap, norm, scale)
    _setup_axes(ax, geom, scale, f"Full cell exterior — {_LABEL[field]}  ·  t = {t:.0f} s")
    _colorbar(fig, ax, cmap, norm, field)
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def plot_cell_cutaway_3d(result, ti: int = 0, field: str = "T", cut_frac: float = 0.5,
                         thickness_scale: float = 6.0, vminmax=None,
                         path: Optional[str] = None):
    """(2) Full cell WITH internals: front part removed at ``cut_frac`` of the length, exposing
    the internal cross-section (through both jellyrolls); remaining exterior + cut face colored
    by the field. The can outline is drawn as a wireframe at the true outer dims.
    """
    geom = result.geom
    sa, la, ha = _axes(geom)
    scale = [1.0, 1.0, 1.0]
    scale[sa] = thickness_scale
    vol, norm, cmap, t = _prep(result, field, ti, vminmax)
    edges = _edges_mm(geom)
    nx, ny, nz = vol.shape
    n_ax = (nx, ny, nz)
    cut = int(np.clip(round(n_ax[la] * cut_frac), 1, n_ax[la]))
    lo, hi = [0, 0, 0], [nx, ny, nz]
    lo[la] = cut                                  # keep the far part [cut:] along the length
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111, projection="3d")
    _box_faces(ax, vol, edges, tuple(lo), tuple(hi), cmap, norm, scale)
    # can outline at the real outer dims (informational)
    od = getattr(geom, "outer_dims", None)
    if od is not None:
        _wire_box(ax, (0.0, 0.0, 0.0), tuple(d * 1e3 for d in od), scale)
    _setup_axes(ax, geom, scale,
                f"Cutaway at {cut_frac:.0%} length — {_LABEL[field]}  ·  t = {t:.0f} s")
    _colorbar(fig, ax, cmap, norm, field)
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def plot_jellyrolls_3d(result, ti: int = 0, field: str = "T",
                       thickness_scale: float = 6.0, gap_scale: float = 4.0,
                       vminmax=None, path: Optional[str] = None):
    """(3) The jellyrolls alone: each roll drawn as its own block (pulled slightly apart along
    the stacking axis for visibility), faces colored by the field at time index ``ti``.
    """
    geom = result.geom
    sa, _, _ = _axes(geom)
    scale = [1.0, 1.0, 1.0]
    scale[sa] = thickness_scale
    vol, norm, cmap, t = _prep(result, field, ti, vminmax)
    edges = list(_edges_mm(geom))
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111, projection="3d")
    cell_roll = np.asarray(geom.cell_roll)
    nrolls = int(cell_roll.max()) + 1
    d_sa = (geom.grid.dx, geom.grid.dy, geom.grid.dz)[sa] * 1e3
    shift = np.zeros(3)
    for r in range(nrolls):
        idxs = np.argwhere(cell_roll == r)
        lo = idxs.min(axis=0)
        hi = idxs.max(axis=0) + 1
        # separate the rolls visually along the stacking axis
        sep = np.zeros(3)
        sep[sa] = r * gap_scale * d_sa
        e = [edges[0].copy(), edges[1].copy(), edges[2].copy()]
        e[sa] = e[sa] + sep[sa]
        _box_faces(ax, vol, tuple(e), tuple(lo), tuple(hi), cmap, norm, scale)
        cen = [(edges[q][lo[q]] + edges[q][hi[q]]) / 2 * scale[q] + sep[q] * scale[q]
               for q in range(3)]
        top = edges[2][hi[2]] * scale[2] if sa != 2 else cen[2]
        ax.text(cen[0], cen[1], top * 1.06, f"jellyroll {r + 1}", fontsize=9,
                ha="center", color="#333333")
    _setup_axes(ax, geom, scale, f"Jellyrolls — {_LABEL[field]}  ·  t = {t:.0f} s")
    _colorbar(fig, ax, cmap, norm, field)
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# animation
# --------------------------------------------------------------------------- #
def save_animation(result, view, field: str = "T", path: str = "cell_anim.gif",
                   stride: int = 1, fps: int = 8, **view_kwargs):
    """Render ``view`` (one of the plot_* functions above) at every ``stride``-th time index and
    save an animated GIF. Color limits are held fixed over the whole run.
    """
    from matplotlib.animation import PillowWriter

    frames = range(0, len(result.t), max(int(stride), 1))
    vmm = field_norm(result, field)
    writer = PillowWriter(fps=fps)
    first = view(result, ti=0, field=field, vminmax=vmm, **view_kwargs)
    with writer.saving(first, path, dpi=100):
        writer.grab_frame()
        plt.close(first)
        for ti in list(frames)[1:]:
            fig = view(result, ti=ti, field=field, vminmax=vmm, **view_kwargs)
            writer.fig = fig
            writer.grab_frame()
            plt.close(fig)
    return path
