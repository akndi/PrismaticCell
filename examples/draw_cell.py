"""3-D view of the cell with the internal jellyrolls drawn as their individual stacks/layers.

A large 3-D cutaway renders EVERY stack (sandwich) as its five layers striped along the thickness
(Y): Cathode-CC | Cathode | Separator | Anode | Anode-CC, for both jellyrolls back-to-back, on the
bottom insulator, with the +/- tabs on the top edge. A side panel shows one sandwich to scale as
the colour/thickness legend.

Layer roles, materials, thicknesses, stack counts and the insulator are read from the config
(nothing hard-coded). The through-thickness (Y) is drawn exaggerated so the ~0.2 mm stacks are
visible; the sandwich detail (right) carries the true micron thicknesses.

Run:  python examples/draw_cell.py [config.yaml] [out.png]
"""
import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

RED, GREEN, GREY, MYLAR, BLUE = "#c0392b", "#5aa469", "#c9ccd1", "#8e44ad", "#3b7fd4"
ROLE_COL = {
    "pos_collector": "#a9adb5",     # Al cathode current collector (silver)
    "cathode_coating": "#7a9a4d",   # LFP cathode
    "separator": "#efdf8f",         # separator
    "anode_coating": "#4d4d4d",     # graphite anode
    "neg_collector": "#c07b3c",     # Cu anode current collector (copper)
}
ROLE_LABEL = {
    "pos_collector": "Cathode CC (Al)", "cathode_coating": "Cathode (LFP)",
    "separator": "Separator", "anode_coating": "Anode (graphite)",
    "neg_collector": "Anode CC (Cu)",
}
ROLE_ORDER = ["pos_collector", "cathode_coating", "separator", "anode_coating", "neg_collector"]


def _cfg_info(path):
    from prismaticcell.config import SimConfig
    from prismaticcell.materials import sandwich_thickness
    cfg = SimConfig.from_yaml(path)
    rolls = cfg.assembly.jellyrolls
    r0 = rolls[0]
    ins = cfg.enclosure.insulator
    info = dict(
        Lx=r0.stack.width, Lz=r0.stack.height,
        rolls=[(rr.n_stacks, [(l.role, l.material, l.thickness) for l in rr.stack.layers],
                sandwich_thickness(rr.stack.layers)) for rr in rolls],
        inter_gap=cfg.assembly.inter_gap,
        layers=[(l.role, l.material, l.thickness) for l in r0.stack.layers],
        sand_t=sandwich_thickness(r0.stack.layers),
        t_ins=(ins.thickness if ins is not None else 0.0),
        tabs=[(t.polarity, t.loc_length) for t in cfg.tabs],
    )
    info["Ly_true"] = sum(n * st for n, _, st in info["rolls"]) + info["inter_gap"] * (len(rolls) - 1)
    w = cfg.assembly.roll_wrap
    info["wrap"] = (w.thickness, w.coverage) if w is not None else None
    info["cavity_fill"] = cfg.assembly.cavity_fill
    od = cfg.enclosure.outer_dims
    info["outer_dims"] = list(od) if od is not None else None   # [x,y,z] can outer size, m
    info["wall"] = cfg.enclosure.wall_thickness
    if od is not None:
        wall = cfg.enclosure.wall_thickness
        info["headspace"] = max((od[2] - 2 * wall) - info["t_ins"] - info["Lz"], 0.0)  # Z leftover
        info["side_len"] = max(((od[0] - 2 * wall) - info["Lx"]) / 2.0, 0.0)            # X clearance
    else:
        info["headspace"] = 0.0
        info["side_len"] = 0.0
    return info


def _quad(ax, xs, ys, zs, color, alpha=1.0, lw=0.0, ec=None):
    ax.add_collection3d(Poly3DCollection(
        [list(zip(xs, ys, zs))], facecolor=color, edgecolor=ec, linewidths=lw, alpha=alpha))


def _box3d_internal(info, ax):
    Lx, Lz = info["Lx"], info["Lz"]
    Ly = 0.85 * Lx                       # drawn thickness (exaggerated so stacks are visible)
    scale = Ly / info["Ly_true"]
    t_ins = 0.06 * Lz                    # drawn insulator height (exaggerated)
    z0 = t_ins                           # stacks sit on the insulator
    z1 = z0 + Lz

    # Bottom insulator slab (green), full X-Y footprint.
    _quad(ax, [0, Lx, Lx, 0], [0, 0, Ly, Ly], [z0, z0, z0, z0], GREEN, 0.9)
    _quad(ax, [0, Lx, Lx, 0], [Ly, Ly, Ly, Ly], [0, 0, z0, z0], GREEN, 0.85, lw=0.4, ec="k")
    _quad(ax, [Lx, Lx, Lx, Lx], [0, Ly, Ly, 0], [0, 0, z0, z0], GREEN, 0.85, lw=0.4, ec="k")

    # Walk the layers along Y, drawing each layer's TOP (z=z1) and RIGHT (x=Lx) faces so the
    # Cathode-CC|Cathode|Sep|Anode|Anode-CC striping is visible along the thickness.
    y = 0.0
    nrolls = len(info["rolls"])
    for ri, (n, layers, st) in enumerate(info["rolls"]):
        for s in range(n):
            for role, _mat, t in layers:
                td = t * scale
                col = ROLE_COL[role]
                _quad(ax, [0, Lx, Lx, 0], [y, y, y + td, y + td], [z1, z1, z1, z1], col, 1.0)  # top
                _quad(ax, [Lx, Lx, Lx, Lx], [y, y + td, y + td, y], [z0, z0, z1, z1], col, 1.0)  # right
                y += td
        if ri < nrolls - 1:
            gd = info["inter_gap"] * scale
            _quad(ax, [0, Lx, Lx, 0], [y, y, y + gd, y + gd], [z1, z1, z1, z1], GREY, 1.0)
            _quad(ax, [Lx, Lx, Lx, Lx], [y, y + gd, y + gd, y], [z0, z0, z1, z1], GREY, 1.0)
            y += gd

    # Outer wireframe of the active block (edges).
    corners = np.array([[0, 0, z0], [Lx, 0, z0], [Lx, Ly, z0], [0, Ly, z0],
                        [0, 0, z1], [Lx, 0, z1], [Lx, Ly, z1], [0, Ly, z1]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    ax.add_collection3d(Line3DCollection([(corners[a], corners[b]) for a, b in edges],
                                         colors="k", linewidths=1.0))

    # Tabs on the top edge (front side, y=0 end where the first stack is), near the top.
    tw, td_ = 0.05 * Lx, 0.10 * Ly
    for pol, frac in info["tabs"]:
        cx = frac * Lx
        _quad(ax, [cx - tw / 2, cx + tw / 2, cx + tw / 2, cx - tw / 2],
              [0, 0, 0, 0], [z1, z1, z1 + 0.10 * Lz, z1 + 0.10 * Lz], RED, 1.0)
        ax.text(cx, 0, z1 + 0.16 * Lz, ("+" if pol == "pos" else "−") + " tab",
                color=RED, fontsize=11, fontweight="bold", ha="center")

    ax.set_xlabel(f"X = length {Lx*1e3:.0f} mm")
    ax.set_ylabel(f"Y = thickness {info['Ly_true']*1e3:.0f} mm  (exaggerated)")
    ax.set_zlabel(f"Z = height {Lz*1e3:.0f} mm")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect((Lx, Ly, Lz + z0))
    ax.view_init(elev=20, azim=-58)
    ntot = sum(n for n, _, _ in info["rolls"])
    rolls_txt = " + ".join(f"{n}" for n, _, _ in info["rolls"])
    ax.set_title(f"{ntot} stacks ({rolls_txt}) × 5 layers, "
                 f"inter-roll gap {info['inter_gap']*1e3:.1f} mm, insulator {info['t_ins']*1e3:.2f} mm",
                 fontsize=9, y=0.98)


def _zoom3d(info, ax, n_show=6):
    """3-D zoom of the first few stacks so the 5-layer pattern is clearly visible."""
    n_show = min(n_show, info["rolls"][0][0])
    layers = info["rolls"][0][1]
    Lx, Lz = 1.0, 1.0                      # local unit box (schematic proportions)
    # give each layer a drawn thickness proportional to its real thickness, per stack
    st = info["sand_t"]
    y = 0.0
    for s in range(n_show):
        for role, _mat, t in layers:
            td = (t / st) * 0.9            # one stack spans 0.9 units along Y
            col = ROLE_COL[role]
            _quad(ax, [0, Lx, Lx, 0], [y, y, y + td, y + td], [Lz, Lz, Lz, Lz], col, 1.0, lw=0.2, ec="#222")  # top
            _quad(ax, [Lx, Lx, Lx, Lx], [y, y + td, y + td, y], [0, 0, Lz, Lz], col, 1.0, lw=0.2, ec="#222")  # right
            _quad(ax, [0, Lx, Lx, 0], [y, y, y, y], [0, 0, Lz, Lz], col, 1.0, lw=0.2, ec="#222")             # front
            y += td
    ax.text2D(0.5, 0.98, f"zoom: first {n_show} stacks (1 stack = 5 layers)",
              transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("length", fontsize=8); ax.set_ylabel("thickness →", fontsize=8)
    ax.set_box_aspect((1.0, max(n_show * 0.9, 1.0), 1.0))
    ax.view_init(elev=16, azim=-62)


ELY = "#7fb2e5"    # electrolyte (light blue)
GAS = "#e5e7ea"    # gas headspace (light grey)


def _elevation_can(info, ax):
    """Side elevation (X-Z, not to scale): fixed can with insulator, roll, electrolyte side
    clearances (up to the roll top) and the gas headspace above."""
    if info["outer_dims"] is None:
        ax.axis("off"); ax.set_title("(auto-sized cavity: no fixed can)", fontsize=9); return
    # schematic band fractions (exaggerated so thin features are visible)
    wallf, insf, hsf = 0.05, 0.07, 0.22
    rollf = 1.0 - insf - hsf
    sidef = 0.10                                     # electrolyte side clearance (each side)
    x0, x1 = sidef, 1.0 - sidef
    # can wall outline
    ax.add_patch(Rectangle((-wallf, -wallf), 1 + 2 * wallf, 1 + 2 * wallf, facecolor="none",
                           ec="k", lw=2.5))
    ax.add_patch(Rectangle((0, insf + rollf), 1, hsf, facecolor=GAS, ec="k", lw=0.6))      # headspace
    ax.add_patch(Rectangle((x0, insf), x1 - x0, rollf, facecolor=BLUE, ec="k", lw=1.0, alpha=0.5))  # roll
    ax.add_patch(Rectangle((0, insf), sidef, rollf, facecolor=ELY, ec="k", lw=0.5))        # L clearance
    ax.add_patch(Rectangle((1 - sidef, insf), sidef, rollf, facecolor=ELY, ec="k", lw=0.5))  # R clearance
    ax.add_patch(Rectangle((0, 0), 1, insf, facecolor=GREEN, ec="k", lw=0.8))              # insulator
    ax.plot([0, 1], [insf + rollf, insf + rollf], color="#1f6fb2", lw=1.2, ls="--")        # elyte level
    ax.text(0.5, insf + rollf + hsf / 2, f"gas headspace\n{info['headspace']*1e3:.1f} mm",
            ha="center", va="center", fontsize=8, color="#444")
    ax.text(0.5, insf + rollf / 2, "jellyrolls", ha="center", va="center", fontsize=9,
            color="#12325a", fontweight="bold")
    ax.text(sidef / 2, insf + rollf * 0.5, "elyte", ha="center", va="center", fontsize=6,
            color="#1f6fb2", rotation=90)
    ax.text(0.5, insf / 2, "insulator", ha="center", va="center", fontsize=7, color="#0f4d22")
    ax.text(0.99, insf + rollf, "electrolyte level = roll top", ha="right", va="bottom",
            fontsize=7, color="#1f6fb2")
    ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.12, 1.12); ax.set_aspect("equal"); ax.axis("off")
    od = info["outer_dims"]
    ax.set_title(f"side elevation X–Z (not to scale)\nfixed can {od[0]*1e3:.0f}×{od[2]*1e3:.0f} mm, "
                 f"wall {info['wall']*1e3:.1f} mm", fontsize=9)


def _topview(info, ax):
    """Top view (X-Y, looking down Z): jellyrolls, mylar wrap border, and (fixed can) the
    electrolyte side clearance out to the can wall."""
    Lx, Ly, gap = info["Lx"], info["Ly_true"], info["inter_gap"]
    sx = info["side_len"] if info["outer_dims"] else 0.0     # X clearance to draw
    if info["outer_dims"]:                                   # electrolyte-filled cavity + can wall
        ax.add_patch(Rectangle((-sx, -sx), Lx + 2 * sx, Ly + 2 * sx, facecolor=ELY, ec="k", lw=2.0))
    y = 0.0
    for ri, (n, _l, st) in enumerate(info["rolls"]):
        rw = n * st
        ax.add_patch(Rectangle((0, y), Lx, rw, facecolor=BLUE, ec="none", alpha=0.35))
        ec, lw = (MYLAR, 3.0) if info["wrap"] is not None else ("k", 1.0)
        ax.add_patch(Rectangle((0, y), Lx, rw, facecolor="none", ec=ec, lw=lw))
        ax.text(Lx / 2, y + rw / 2, f"jellyroll {ri+1}", ha="center", va="center",
                fontsize=8, color="#12325a")
        y += rw
        if ri < len(info["rolls"]) - 1:
            ax.add_patch(Rectangle((0, y), Lx, gap, facecolor=ELY, ec="none"))
            y += gap
    m = max(sx, 0.06 * Ly)
    ax.set_xlim(-sx - 0.04 * Lx, Lx + sx + 0.04 * Lx); ax.set_ylim(-m, y + m)
    ax.set_aspect(Lx / max(y + 2 * m, 1e-9) * 0.6)
    fill = info["cavity_fill"] if info["outer_dims"] else "—"
    ax.set_title(f"top view X–Y: wrap (purple), cavity = {fill}", fontsize=9)
    ax.set_xlabel("X = length"); ax.set_ylabel("Y = thickness")
    ax.set_xticks([]); ax.set_yticks([])


def _sandwich(info, ax):
    layers = info["layers"]
    total = info["sand_t"]
    x = 0.0
    H = 1.0
    for role, mat, t in layers:
        ax.add_patch(Rectangle((x, 0), t, H, facecolor=ROLE_COL[role], ec="k", lw=0.8))
        x += t
    x = 0.0
    for i, (role, mat, t) in enumerate(layers):
        xc = x + t / 2
        yl = H + 0.12 + 0.42 * (i % 3)
        ax.annotate(f"{ROLE_LABEL[role]}\n{t*1e6:.0f} µm", xy=(xc, H), xytext=(xc, yl),
                    ha="center", va="bottom", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="#555", lw=0.6))
        x += t
    ax.set_xlim(-0.03 * total, 1.03 * total); ax.set_ylim(-0.15, H + 1.5)
    ax.set_xlabel(f"through-plane — one sandwich = {total*1e6:.0f} µm (to scale)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("one stack (sandwich): 5 layers", fontsize=10)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def draw(info, out="outputs/cell_internal_3d.png"):
    fig = plt.figure(figsize=(16, 8.5))
    axA = fig.add_axes([0.00, 0.05, 0.50, 0.88], projection="3d")     # main 3-D, all stacks
    axZ = fig.add_axes([0.52, 0.60, 0.22, 0.34], projection="3d")     # 3-D zoom, few stacks
    axD = fig.add_axes([0.52, 0.44, 0.34, 0.13])                      # sandwich detail (to scale)
    axE = fig.add_axes([0.52, 0.09, 0.20, 0.28])                      # side elevation (can/headspace)
    axT = fig.add_axes([0.76, 0.09, 0.22, 0.28])                      # top view (wrap/clearance)
    _box3d_internal(info, axA)
    _zoom3d(info, axZ)
    _sandwich(info, axD)
    _elevation_can(info, axE)
    _topview(info, axT)
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=ROLE_COL[r],
                      markeredgecolor="k", markersize=12, label=ROLE_LABEL[r]) for r in ROLE_ORDER]
    handles += [Line2D([0], [0], marker="s", color="w", markerfacecolor=GREEN,
                       markeredgecolor="k", markersize=12, label="Bottom insulator"),
                Line2D([0], [0], marker="s", color="w", markerfacecolor=RED,
                       markeredgecolor="k", markersize=12, label="Tab"),
                Line2D([0], [0], color=MYLAR, lw=3, label="Mylar wrap (roll sides)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.005),
               ncol=8, fontsize=8.5, frameon=True, title="layers")
    fig.suptitle("PrismaticCell — 3-D internal stack structure", fontsize=15, fontweight="bold")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/large_prismatic.yaml"
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/cell_internal_3d.png"
    draw(_cfg_info(cfg_path), out)
