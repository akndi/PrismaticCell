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
REGION_INSULATOR = 4   # bottom/top insulating slab (enclosure.insulator)

_AX = {"x": 0, "y": 1, "z": 2}   # physical grid-axis index for each named axis


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
    # Sub-grid wall shell (wall_model="shell"): the enclosure wall wraps the cavity boundary as a
    # conductive skin instead of being meshed. Zero when meshed as volume cells or no wall.
    wall_thickness: float = 0.0        # m, shell thickness
    wall_k: float = 0.0                # W/m/K, shell in-plane conductivity
    wall_rhocp: float = 0.0            # J/m^3/K, shell volumetric heat capacity
    # Axis roles (physical grid axis index 0/1/2 for x/y/z):
    stack_axis: int = 2                # through-plane / thickness / sandwich-stacking axis
    len_axis: int = 0                  # electrode length (first in-plane axis)
    hgt_axis: int = 1                  # electrode height (second in-plane axis)
    # Bottom/top insulating slab (enclosure.insulator). Being far thinner than one mesh cell it is
    # represented as a sub-grid conductive layer on the height face it occupies (same technique as
    # the wall shell): its series resistance-area is added to that face's BC and its areal heat
    # capacity to that face's cells. Empty/zero when no insulator is configured.
    insulator_face: str = ""           # physical face name it sits on ("bottom"/"top"/...), "" if none
    insulator_R_area: float = 0.0      # K*m^2/W, series resistance-area t_ins/k_ins on that face
    insulator_rhocp_t: float = 0.0     # J/m^2/K, areal heat capacity rho*cp*t_ins of the slab
    # Aggregate per-face sub-grid layers (insulator + jellyroll wrap), summed by physical face.
    # thermal.py adds face_R_area[f] in series on face f's BC and lumps face_rhocp_t[f] onto its
    # cells. Empty dicts when nothing is configured.
    face_R_area: Dict[str, float] = field(default_factory=dict)   # K*m^2/W per physical face
    face_rhocp_t: Dict[str, float] = field(default_factory=dict)  # J/m^2/K per physical face

    @property
    def n_active(self) -> int:
        return int(self.active_mask.sum())

    def phys_index(self, a: int, b: int, s: int) -> Tuple[int, int, int]:
        """Map electrode (length index a, height index b, stack index s) -> physical (i,j,k)."""
        t = [0, 0, 0]
        t[self.len_axis] = a
        t[self.hgt_axis] = b
        t[self.stack_axis] = s
        return t[0], t[1], t[2]


def _role_layer(layers, role):
    for lyr in layers:
        if lyr.role == role:
            return lyr
    raise ValueError(f"sandwich missing layer role '{role}'")


def build_geometry(cfg: SimConfig) -> Geometry:
    """Construct the full :class:`Geometry` from a validated ``SimConfig``."""
    wall_real = cfg.enclosure.wall_thickness
    # A fixed can size (enclosure.outer_dims) meshes only the roll bbox and represents the wall +
    # clearances + headspace as sub-grid layers, so it implies the shell wall model.
    explicit = cfg.enclosure.outer_dims is not None
    shell = (cfg.enclosure.wall_model == "shell") or explicit
    # In "shell" mode the wall is NOT meshed (it wraps the cavity as a sub-grid skin), so it
    # contributes no mesh thickness; in "mesh" mode it occupies volume cells at the box edge.
    wall = 0.0 if shell else wall_real
    clr = cfg.assembly.wall_clearance
    gap = cfg.assembly.inter_gap
    rolls_cfg = cfg.assembly.jellyrolls
    nr = len(rolls_cfg)

    # Axis roles: stack_axis = through-plane/thickness; the other two (in x<y<z order) are the
    # electrode LENGTH (la) and HEIGHT (ha) axes.
    sa = _AX[cfg.assembly.stack_axis]
    la, ha = [ax for ax in (0, 1, 2) if ax != sa]

    # Per-roll extent along each physical axis: length axis <- stack.width, height axis <-
    # stack.height, stack axis <- n_stacks * sandwich thickness.
    size = {la: [r.stack.width for r in rolls_cfg],
            ha: [r.stack.height for r in rolls_cfg],
            sa: [r.n_stacks * sandwich_thickness(r.stack.layers) for r in rolls_cfg]}

    arrangement = cfg.assembly.arrangement
    if arrangement == "stacked":
        arr_axis = sa                      # jellyrolls stacked back-to-back through the thickness
    elif arrangement == "side_by_side_length":
        arr_axis = la
    elif arrangement == "side_by_side_height":
        arr_axis = ha
    else:  # pragma: no cover - guarded by config Literal
        raise ValueError(f"unknown arrangement '{arrangement}'")

    # Tight roll-assembly extent per axis (rolls sum along the arrangement axis, else max).
    tight = [0.0, 0.0, 0.0]
    for ax in (0, 1, 2):
        tight[ax] = (sum(size[ax]) + gap * (nr - 1)) if ax == arr_axis else max(size[ax])

    # Insulator thickness (bottom slab) -- referenced here so the roll can sit on it.
    ins_cfg = cfg.enclosure.insulator
    t_ins_val = ins_cfg.thickness if ins_cfg is not None else 0.0

    outer = cfg.enclosure.outer_dims
    if explicit:
        # Fixed can: mesh only the roll-assembly bbox; wall / clearances / headspace / insulator are
        # all sub-grid layers. Inner cavity = outer - 2*wall. Roll is bottom-referenced along the
        # height axis (sits on the insulator) and centred in the two in-plane axes.
        inner = [outer[ax] - 2.0 * wall_real for ax in (0, 1, 2)]
        for ax in (0, 1, 2):
            slack = inner[ax] - tight[ax] - (t_ins_val if ax == ha else 0.0)
            if slack < -1e-12:
                raise ValueError(
                    f"enclosure.outer_dims too small along axis {ax}: inner cavity "
                    f"{inner[ax]*1e3:.2f} mm < roll {tight[ax]*1e3:.2f} mm"
                    + (f" + insulator {t_ins_val*1e3:.2f} mm" if ax == ha else "")
                    + ". Increase outer_dims or reduce the roll/insulator.")
        cav = list(tight)
        L = list(tight)                       # meshed domain = roll bbox (wall is sub-grid)
        roll_lo = [0.0, 0.0, 0.0]
    else:
        inner = None
        cav = [tight[ax] + 2.0 * clr for ax in (0, 1, 2)]
        L = [cav[ax] + 2.0 * wall for ax in (0, 1, 2)]
        roll_lo = [wall + clr, wall + clr, wall + clr]
    Lx, Ly, Lz = L

    # Roll bounding boxes (lo,hi) per physical axis
    bboxes: List[List[Tuple[float, float]]] = []
    cursor = roll_lo[arr_axis]
    for r in range(nr):
        box = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        for ax in (0, 1, 2):
            lo = cursor if ax == arr_axis else roll_lo[ax]
            box[ax] = (lo, lo + size[ax][r])
        cursor = box[arr_axis][1] + gap
        bboxes.append(box)

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
    fill_name = cfg.assembly.cavity_fill
    gap_mat = cfg.materials.get(fill_name)
    if gap_mat is None:
        raise ValueError(
            f"No '{fill_name}' material defined (assembly.cavity_fill). The cavity void "
            "(clearance / inter-roll / headspace) needs an explicit filler material in the "
            "material DB (e.g. 'gap_air' for a dry cell or 'electrolyte' for a flooded one)."
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
                cc = (xi, yj, zk)
                for r, box in enumerate(bboxes):
                    if all(box[ax][0] <= cc[ax] <= box[ax][1] for ax in (0, 1, 2)):
                        found = r
                        break
                if found >= 0:
                    hp = roll_props[found]
                    region[i, j, k] = REGION_ACTIVE
                    cell_roll[i, j, k] = found
                    # anisotropy follows the stack axis: k_through (hp.k_z) along the stack axis,
                    # k_in (hp.k_x) along the two in-plane axes.
                    kx[i, j, k] = hp.k_z if sa == 0 else hp.k_x
                    ky[i, j, k] = hp.k_z if sa == 1 else hp.k_x
                    kz[i, j, k] = hp.k_z if sa == 2 else hp.k_x
                    rho_cp[i, j, k] = hp.rho_cp
                else:
                    region[i, j, k] = REGION_GAP
                    kx[i, j, k] = gap_props.k_x
                    ky[i, j, k] = gap_props.k_y
                    kz[i, j, k] = gap_props.k_z
                    rho_cp[i, j, k] = gap_props.rho_cp

    active_mask = region == REGION_ACTIVE

    # Per-roll electro mapping in electrode coordinates: (a = length index, b = height index)
    # are the in-plane "columns"; col_zcells[(a,b)] are the through-thickness stack-cell indices.
    coords = [xc, yc, zc]
    d = [dx, dy, dz]
    d_la, d_ha = d[la], d[ha]
    rolls: List[RollElectro] = []
    area_eff_total = np.zeros((nx, ny, nz))
    for r, rc in enumerate(rolls_cfg):
        roll_cells = np.argwhere((cell_roll == r) & active_mask)   # physical (i,j,k) rows
        s_active = sorted({int(cell[sa]) for cell in roll_cells})
        n_active_s = max(len(s_active), 1)
        n_sand_per_cell = rc.n_stacks / n_active_s

        columns: List[Tuple[int, int]] = []
        col_zcells: Dict[Tuple[int, int], List[int]] = {}
        area_eff = np.zeros((nx, ny, nz))
        colmap: Dict[Tuple[int, int], List[int]] = {}
        for cell in roll_cells:
            i, j, k = int(cell[0]), int(cell[1]), int(cell[2])
            a, b, s = cell[la], cell[ha], cell[sa]
            colmap.setdefault((int(a), int(b)), []).append(int(s))
            area_eff[i, j, k] = d_la * d_ha * n_sand_per_cell
        for ab, ss in colmap.items():
            columns.append(ab)
            col_zcells[ab] = sorted(ss)
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
            nodes = _tab_footprint_nodes(tab, columns, coords, la, ha, (L[la], L[ha]))
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

    # Tab electrical + thermal conductances (footprint-based; PHYSICS §3, §5). The tab bundles
    # all n_stacks collector foils of both jellyrolls in parallel, each of cross-section
    # (size_length * thickness), conducting over the protrusion length:
    #   G_elec = sigma * n_foils * (size_length * thickness) / protrusion   [S]
    #   G_therm = k    * n_foils * (size_length * thickness) / protrusion   [W/K]
    # Tab thickness/material default to that polarity's current collector (Al pos / Cu neg).
    n_foils = sum(r.n_stacks for r in rolls_cfg)
    ref_layers = rolls_cfg[0].stack.layers
    coll = {"pos": _role_layer(ref_layers, "pos_collector"),
            "neg": _role_layer(ref_layers, "neg_collector")}
    g_tab_pos = g_tab_neg = 0.0
    tab_heat_pos = tab_heat_neg = 0.0
    for tab in cfg.tabs:
        clayer = coll[tab.polarity]
        thickness = tab.thickness if tab.thickness is not None else clayer.thickness
        mat = cfg.materials[tab.material if tab.material is not None else clayer.material]
        Lp = max(tab.protrusion, 1e-9)
        xsec = n_foils * tab.size_length * thickness
        if tab.polarity == "pos":
            g_tab_pos += mat.sigma_elec * xsec / Lp
            tab_heat_pos += mat.k_in * xsec / Lp
        else:
            g_tab_neg += mat.sigma_elec * xsec / Lp
            tab_heat_neg += mat.k_in * xsec / Lp
    if not cfg.enclosure.tab_heat_sink:
        tab_heat_pos = tab_heat_neg = 0.0   # tabs thermally isolated (no far-end heat sink)

    # Capacity per CV proportional to effective electrode area; normalized to exact total
    cap_cv = np.zeros((nx, ny, nz))
    tot_area = area_eff_total.sum()
    if tot_area > 0:
        cap_cv = area_eff_total / tot_area * cfg.ecm.capacity_Ah

    # Bottom/top insulating slab: a sub-grid conductive layer on the height-axis face it occupies
    # (bottom = -height end, opposite the tabs). Its series resistance t_ins/k_ins throttles heat
    # flow to that face and its (small) heat capacity is lumped onto the face cells.
    _face_names = {0: ("x_min", "x_max"), 1: ("y_min", "y_max"), 2: ("bottom", "top")}
    face_R_area: Dict[str, float] = {}
    face_rhocp_t: Dict[str, float] = {}

    def _add_face_layer(face: str, R_area: float, rhocp_t: float) -> None:
        face_R_area[face] = face_R_area.get(face, 0.0) + R_area
        face_rhocp_t[face] = face_rhocp_t.get(face, 0.0) + rhocp_t

    ins_cfg = cfg.enclosure.insulator
    ins_face = ""
    ins_R_area = 0.0
    ins_rhocp_t = 0.0
    if ins_cfg is not None:
        ins_mat = cfg.materials[ins_cfg.material]
        ins_R_area = ins_cfg.thickness / ins_mat.k_through   # through-plane resistance-area
        ins_rhocp_t = ins_mat.density * ins_mat.cp * ins_cfg.thickness
        ins_face = _face_names[ha][0 if ins_cfg.location == "bottom" else 1]
        _add_face_layer(ins_face, ins_R_area, ins_rhocp_t)

    # Jellyroll wrap (e.g. mylar): a film on each roll's side faces. Only the roll faces that border
    # the can/ambient (the external cell faces) carry the resistance; internal inter-roll faces are
    # neglected. Coverage is named by orientation: big faces = +/-stack axis, ends = +/-length axis,
    # height ends = +/-height axis.
    wrap_cfg = cfg.assembly.roll_wrap
    if wrap_cfg is not None:
        w_mat = cfg.materials[wrap_cfg.material]
        w_R = wrap_cfg.thickness / w_mat.k_through
        w_mt = w_mat.density * w_mat.cp * wrap_cfg.thickness
        axes = {"sides": (sa, la), "big_faces": (sa,), "ends": (la,), "all": (sa, la, ha)}[
            wrap_cfg.coverage]
        for ax in axes:
            for face in _face_names[ax]:
                _add_face_layer(face, w_R, w_mt)

    # Roll-to-can clearance filled with the cavity_fill material (e.g. electrolyte in a flooded
    # cell, air in a dry one). In shell mode the clearance is sub-grid (thinner than a cell), so add
    # its conduction resistance-area t/k in series on every external face and its areal heat capacity
    # to those cells -- this is what makes the fill choice (air k=0.03 vs electrolyte k=0.6) actually
    # change roll<->can heat transfer, independent of mesh. The internal inter-roll gap is left to
    # the meshed gap cells (resolved only on a fine enough grid).
    if explicit:
        # Fixed-can void as sub-grid layers: electrolyte side clearances (centred, spanning the
        # roll height) on the two in-plane faces of each in-plane axis, and a gas headspace on the
        # top face. The bottom insulator is already added above; the roll sits on it.
        for ax in (la, sa):
            side = max((inner[ax] - tight[ax]) / 2.0, 0.0)     # symmetric clearance, each side
            if side > 0.0 and gap_mat.k_through > 0.0:
                r_side = side / gap_mat.k_through
                m_side = gap_mat.density * gap_mat.cp * side
                for face in _face_names[ax]:
                    _add_face_layer(face, r_side, m_side)
        headspace = max(inner[ha] - t_ins_val - tight[ha], 0.0)   # all leftover height is on top
        if headspace > 0.0:
            hs_mat = cfg.materials[cfg.enclosure.headspace_fill]
            if hs_mat.k_through > 0.0:
                _add_face_layer(_face_names[ha][1], headspace / hs_mat.k_through,
                                hs_mat.density * hs_mat.cp * headspace)
    elif shell and clr > 0.0 and gap_mat.k_through > 0.0:
        r_clr = clr / gap_mat.k_through
        m_clr = gap_mat.density * gap_mat.cp * clr
        for ax in (0, 1, 2):
            for face in _face_names[ax]:
                _add_face_layer(face, r_clr, m_clr)

    return Geometry(
        grid=grid,
        region=region,
        kx=kx, ky=ky, kz=kz,
        rho_cp=rho_cp,
        active_mask=active_mask,
        cell_roll=cell_roll,
        cap_cv=cap_cv,
        rolls=rolls,
        outer_dims=(tuple(outer) if explicit else (Lx, Ly, Lz)),
        contact_conductance=cfg.enclosure.contact_conductance,
        g_tab_pos=g_tab_pos, g_tab_neg=g_tab_neg,
        tab_heat_cond_pos=tab_heat_pos, tab_heat_cond_neg=tab_heat_neg,
        wall_thickness=(wall_real if shell else 0.0),
        wall_k=(can_mat.k_in if shell else 0.0),
        wall_rhocp=(can_mat.density * can_mat.cp if shell else 0.0),
        stack_axis=sa, len_axis=la, hgt_axis=ha,
        insulator_face=ins_face, insulator_R_area=ins_R_area, insulator_rhocp_t=ins_rhocp_t,
        face_R_area=face_R_area, face_rhocp_t=face_rhocp_t,
    )


def _tab_footprint_nodes(tab: Tab, columns, coords, la: int, ha: int,
                         L_lh) -> List[Tuple[int, int]]:
    """Return the electrode columns (a=length index, b=height index) under a tab footprint.

    The footprint is centered at fractional (loc_length, loc_height) of the electrode extent,
    with size (size_length x size_height). Falls back to the single nearest column if the
    footprint overlaps none, so every roll stays terminated to the tab.
    """
    if not columns:
        return []
    L_la, L_ha = L_lh
    cl = tab.loc_length * L_la
    ch = tab.loc_height * L_ha
    lo_l, hi_l = cl - tab.size_length / 2.0, cl + tab.size_length / 2.0
    lo_h, hi_h = ch - tab.size_height / 2.0, ch + tab.size_height / 2.0
    la_c, ha_c = coords[la], coords[ha]
    band = [(a, b) for (a, b) in columns
            if lo_l <= la_c[a] <= hi_l and lo_h <= ha_c[b] <= hi_h]
    if not band:
        near = min(columns, key=lambda c: (la_c[c[0]] - cl) ** 2 + (ha_c[c[1]] - ch) ** 2)
        band = [near]
    return band
