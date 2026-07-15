"""3-D structured mesh, region map, and electro-network mapping (PHYSICS §0, §1, §5-mesh).

Builds the prismatic-cell geometry from a ``SimConfig``: a metal can enclosing one or more
jellyrolls, each a stack of Cathode-CC | Cathode | Sep | Anode | Anode-CC sandwiches. The cell is
discretized into a uniform ``nx x ny x nz`` control-volume grid; each cell is tagged with a region
(can wall / gap / active jellyroll) and given anisotropic thermal properties homogenized from the
sandwich layers. Active cells carry an ECM; the module also exposes the per-roll collector sheet
conductances, per-CV effective electrode area, per-CV capacity, and tab terminal nodes needed by
the distributed electro solver.

Implemented against docs/INTERFACES.md. No coefficients are hard-coded — all come from the config
and material database.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .config import SimConfig, Tab
from .materials import homogenize, isotropic_props, sandwich_thickness

# Region integer codes
REGION_CAN = 0
REGION_GAP = 1
REGION_ACTIVE = 2
REGION_TAB = 3


@dataclass
class Grid:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    xc: np.ndarray
    yc: np.ndarray
    zc: np.ndarray

    def volume(self) -> float:
        """Volume of a single control volume [m^3]."""
        return self.dx * self.dy * self.dz

    def flat(self, i: int, j: int, k: int) -> int:
        """Flat index for cell (i,j,k), C-order on (i,j,k)."""
        return k + self.nz * (j + self.ny * i)


@dataclass
class RollElectro:
    roll_index: int
    n_stacks: int
    columns: List[Tuple[int, int]]
    col_zcells: Dict[Tuple[int, int], List[int]]
    area_eff: np.ndarray            # (nx,ny,nz) effective electrode area per CV [m^2]
    sheet_cond_pos: float           # Al foil sheet conductance sigma*t*n_stacks [S]
    sheet_cond_neg: float           # Cu foil sheet conductance [S]
    tab_pos_nodes: List[Tuple[int, int]]
    tab_neg_nodes: List[Tuple[int, int]]


@dataclass
class Geometry:
    grid: Grid
    region: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    rho_cp: np.ndarray
    active_mask: np.ndarray
    cell_roll: np.ndarray
    cap_cv: np.ndarray
    rolls: List[RollElectro]
    outer_dims: Tuple[float, float, float]
    contact_conductance: float = 0.0   # W/m^2/K, stack<->can-wall interfacial conductance
    g_tab_pos: float = 0.0             # S, total physical conductance of the +tab(s): sigma*w*t/L
    g_tab_neg: float = 0.0             # S, total physical conductance of the -tab(s)
    tab_heat_cond_pos: float = 0.0     # W/K, +tab conduction to ambient (thermal loss path)
    tab_heat_cond_neg: float = 0.0     # W/K, -tab conduction to ambient

    @property
    def n_active(self) -> int:
        return int(self.active_mask.sum())


def _role_layer(layers, role):
    for lyr in layers:
        if lyr.role == role:
            return lyr
    raise ValueError(f"sandwich missing layer role '{role}'")


def build_geometry(cfg: SimConfig) -> Geometry:
    """Construct the full :class:`Geometry` from a validated ``SimConfig``."""
    wall = cfg.enclosure.wall_thickness
    clr = cfg.assembly.wall_clearance
    gap = cfg.assembly.inter_gap
    rolls_cfg = cfg.assembly.jellyrolls

    # Per-roll footprint and thickness
    roll_w = [r.stack.width for r in rolls_cfg]
    roll_h = [r.stack.height for r in rolls_cfg]
    roll_tz = [r.n_stacks * sandwich_thickness(r.stack.layers) for r in rolls_cfg]

    arrangement = cfg.assembly.arrangement
    nr = len(rolls_cfg)

    # Cavity dimensions from arrangement
    if arrangement == "side_by_side_x":
        cav_x = sum(roll_w) + gap * (nr - 1) + 2 * clr
        cav_y = max(roll_h) + 2 * clr
        cav_z = max(roll_tz) + 2 * clr
    elif arrangement == "side_by_side_y":
        cav_x = max(roll_w) + 2 * clr
        cav_y = sum(roll_h) + gap * (nr - 1) + 2 * clr
        cav_z = max(roll_tz) + 2 * clr
    elif arrangement == "stacked_z":
        cav_x = max(roll_w) + 2 * clr
        cav_y = max(roll_h) + 2 * clr
        cav_z = sum(roll_tz) + gap * (nr - 1) + 2 * clr
    else:  # pragma: no cover - guarded by config Literal
        raise ValueError(f"unknown arrangement '{arrangement}'")

    Lx = cav_x + 2 * wall
    Ly = cav_y + 2 * wall
    Lz = cav_z + 2 * wall

    # Roll bounding boxes in outer coordinates
    bboxes: List[Tuple[float, float, float, float, float, float]] = []
    cx = wall + clr
    cy = wall + clr
    cz = wall + clr
    for r in range(nr):
        if arrangement == "side_by_side_x":
            x0, x1 = cx, cx + roll_w[r]
            y0, y1 = wall + clr, wall + clr + roll_h[r]
            z0, z1 = wall + clr, wall + clr + roll_tz[r]
            cx = x1 + gap
        elif arrangement == "side_by_side_y":
            x0, x1 = wall + clr, wall + clr + roll_w[r]
            y0, y1 = cy, cy + roll_h[r]
            z0, z1 = wall + clr, wall + clr + roll_tz[r]
            cy = y1 + gap
        else:  # stacked_z
            x0, x1 = wall + clr, wall + clr + roll_w[r]
            y0, y1 = wall + clr, wall + clr + roll_h[r]
            z0, z1 = cz, cz + roll_tz[r]
            cz = z1 + gap
        bboxes.append((x0, x1, y0, y1, z0, z1))

    nx, ny, nz = cfg.mesh.nx, cfg.mesh.ny, cfg.mesh.nz
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    zc = (np.arange(nz) + 0.5) * dz
    grid = Grid(nx, ny, nz, dx, dy, dz, xc, yc, zc)

    region = np.full((nx, ny, nz), REGION_GAP, dtype=np.int32)
    cell_roll = np.full((nx, ny, nz), -1, dtype=np.int32)
    kx = np.zeros((nx, ny, nz))
    ky = np.zeros((nx, ny, nz))
    kz = np.zeros((nx, ny, nz))
    rho_cp = np.zeros((nx, ny, nz))

    can_mat = cfg.materials[cfg.enclosure.material]
    gap_mat = cfg.materials.get("gap_air")
    if gap_mat is None:
        raise ValueError(
            "No 'gap_air' material defined. The inter-roll/clearance gaps need an explicit "
            "filler material (define a low-conductivity 'gap_air' entry in the material DB) "
            "rather than silently inheriting the metal can's properties."
        )
    can_props = isotropic_props(can_mat, dz)
    gap_props = isotropic_props(gap_mat, dz)

    # Homogenized active props per roll (sandwich orientation: z through-plane)
    roll_props = []
    for r in rolls_cfg:
        hp = homogenize(r.stack.layers, cfg.materials)
        roll_props.append(hp)

    # Tag cells
    for i in range(nx):
        xi = xc[i]
        in_wall_x = xi < wall or xi > Lx - wall
        for j in range(ny):
            yj = yc[j]
            in_wall_y = yj < wall or yj > Ly - wall
            for k in range(nz):
                zk = zc[k]
                in_wall_z = zk < wall or zk > Lz - wall
                if in_wall_x or in_wall_y or in_wall_z:
                    region[i, j, k] = REGION_CAN
                    kx[i, j, k] = can_props.k_x
                    ky[i, j, k] = can_props.k_y
                    kz[i, j, k] = can_props.k_z
                    rho_cp[i, j, k] = can_props.rho_cp
                    continue
                # inside cavity: which roll?
                found = -1
                for r, (x0, x1, y0, y1, z0, z1) in enumerate(bboxes):
                    if x0 <= xi <= x1 and y0 <= yj <= y1 and z0 <= zk <= z1:
                        found = r
                        break
                if found >= 0:
                    hp = roll_props[found]
                    region[i, j, k] = REGION_ACTIVE
                    cell_roll[i, j, k] = found
                    kx[i, j, k] = hp.k_x
                    ky[i, j, k] = hp.k_y
                    kz[i, j, k] = hp.k_z
                    rho_cp[i, j, k] = hp.rho_cp
                else:
                    region[i, j, k] = REGION_GAP
                    kx[i, j, k] = gap_props.k_x
                    ky[i, j, k] = gap_props.k_y
                    kz[i, j, k] = gap_props.k_z
                    rho_cp[i, j, k] = gap_props.rho_cp

    active_mask = region == REGION_ACTIVE

    # Per-roll electro mapping
    rolls: List[RollElectro] = []
    area_eff_total = np.zeros((nx, ny, nz))
    for r, rc in enumerate(rolls_cfg):
        roll_cells = (cell_roll == r) & active_mask
        # active z-cells for this roll (assume uniform across columns)
        z_active = sorted({k for i in range(nx) for j in range(ny) for k in range(nz)
                           if roll_cells[i, j, k]})
        n_active_z = max(len(z_active), 1)
        n_sand_per_cell = rc.n_stacks / n_active_z

        columns: List[Tuple[int, int]] = []
        col_zcells: Dict[Tuple[int, int], List[int]] = {}
        area_eff = np.zeros((nx, ny, nz))
        for i in range(nx):
            for j in range(ny):
                ks = [k for k in range(nz) if roll_cells[i, j, k]]
                if ks:
                    columns.append((i, j))
                    col_zcells[(i, j)] = ks
                    for k in ks:
                        area_eff[i, j, k] = dx * dy * n_sand_per_cell
        area_eff_total += area_eff

        pos_layer = _role_layer(rc.stack.layers, "pos_collector")
        neg_layer = _role_layer(rc.stack.layers, "neg_collector")
        sigma_pos = cfg.materials[pos_layer.material].sigma_elec
        sigma_neg = cfg.materials[neg_layer.material].sigma_elec
        sheet_cond_pos = sigma_pos * pos_layer.thickness * rc.n_stacks
        sheet_cond_neg = sigma_neg * neg_layer.thickness * rc.n_stacks

        tab_pos_nodes: List[Tuple[int, int]] = []
        tab_neg_nodes: List[Tuple[int, int]] = []
        for tab in cfg.tabs:
            nodes = _tab_edge_nodes(tab, columns, grid, (Lx, Ly))
            if tab.polarity == "pos":
                tab_pos_nodes.extend(nodes)
            else:
                tab_neg_nodes.extend(nodes)

        rolls.append(RollElectro(
            roll_index=r,
            n_stacks=rc.n_stacks,
            columns=columns,
            col_zcells=col_zcells,
            area_eff=area_eff,
            sheet_cond_pos=sheet_cond_pos,
            sheet_cond_neg=sheet_cond_neg,
            tab_pos_nodes=tab_pos_nodes,
            tab_neg_nodes=tab_neg_nodes,
        ))

    # Tab electrical + thermal conductances from tab geometry/material (PHYSICS §3, §5):
    #   electrical G = sigma * (w*t) / L  [S];  thermal G = k * (w*t) / L  [W/K]
    g_tab_pos = g_tab_neg = 0.0
    tab_heat_pos = tab_heat_neg = 0.0
    for tab in cfg.tabs:
        mat = cfg.materials[tab.material]
        L = max(tab.length, 1e-9)
        xsec = tab.width * tab.thickness
        if tab.polarity == "pos":
            g_tab_pos += mat.sigma_elec * xsec / L
            tab_heat_pos += mat.k_in * xsec / L
        else:
            g_tab_neg += mat.sigma_elec * xsec / L
            tab_heat_neg += mat.k_in * xsec / L
    if not cfg.enclosure.tab_heat_sink:
        tab_heat_pos = tab_heat_neg = 0.0   # tabs thermally isolated (no far-end heat sink)

    # Capacity per CV proportional to effective electrode area; normalized to exact total
    cap_cv = np.zeros((nx, ny, nz))
    tot_area = area_eff_total.sum()
    if tot_area > 0:
        cap_cv = area_eff_total / tot_area * cfg.ecm.capacity_Ah

    return Geometry(
        grid=grid,
        region=region,
        kx=kx, ky=ky, kz=kz,
        rho_cp=rho_cp,
        active_mask=active_mask,
        cell_roll=cell_roll,
        cap_cv=cap_cv,
        rolls=rolls,
        outer_dims=(Lx, Ly, Lz),
        contact_conductance=cfg.enclosure.contact_conductance,
        g_tab_pos=g_tab_pos, g_tab_neg=g_tab_neg,
        tab_heat_cond_pos=tab_heat_pos, tab_heat_cond_neg=tab_heat_neg,
    )


def _tab_edge_nodes(tab: Tab, columns, grid: Grid, outer_xy) -> List[Tuple[int, int]]:
    """Return the roll's in-plane columns that lie under a tab footprint on its edge.

    Falls back to the single nearest edge column if the tab band overlaps no column, so
    every roll stays electrically terminated to the tab.
    """
    Lx, Ly = outer_xy
    if not columns:
        return []
    if tab.edge in ("y_min", "y_max"):
        # edge runs along x; select the extreme-j columns and band along x
        j_edge = min(j for _, j in columns) if tab.edge == "y_min" else max(j for _, j in columns)
        edge_cols = [(i, j) for (i, j) in columns if j == j_edge]
        center = tab.position * Lx
        lo, hi = center - tab.width / 2, center + tab.width / 2
        band = [(i, j) for (i, j) in edge_cols if lo <= grid.xc[i] <= hi]
        if not band:
            i_near = min(edge_cols, key=lambda c: abs(grid.xc[c[0]] - center))
            band = [i_near]
        return band
    else:
        # x_min / x_max: edge runs along y; band along y
        i_edge = min(i for i, _ in columns) if tab.edge == "x_min" else max(i for i, _ in columns)
        edge_cols = [(i, j) for (i, j) in columns if i == i_edge]
        center = tab.position * Ly
        lo, hi = center - tab.width / 2, center + tab.width / 2
        band = [(i, j) for (i, j) in edge_cols if lo <= grid.yc[j] <= hi]
        if not band:
            j_near = min(edge_cols, key=lambda c: abs(grid.yc[c[1]] - center))
            band = [j_near]
        return band
