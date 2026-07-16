"""Configuration schema for PrismaticCell.

These dataclasses are the shared contract between every module (geometry, echem,
thermal, distributed, coupling, api). They hold *only* data — no physics — and are
populated from YAML config files. No physical constant or coefficient is baked into
solver code; everything a solver needs is reachable from a ``SimConfig``.

Loading: ``SimConfig.from_yaml(path)`` parses a YAML file (with optional ``include``
of a shared materials file) into a fully typed tree and validates it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional, List, Dict, Any, Literal, get_args, get_origin, get_type_hints
import os
import yaml


# --------------------------------------------------------------------------- #
# Physical constants (universal — not tunable model parameters)
# --------------------------------------------------------------------------- #
R_GAS = 8.314462618          # J/mol/K, universal gas constant
SIGMA_SB = 5.670374419e-8    # W/m^2/K^4, Stefan-Boltzmann constant
T_ABS_ZERO = 0.0             # K
KELVIN_0C = 273.15           # K


# --------------------------------------------------------------------------- #
# Materials
# --------------------------------------------------------------------------- #
@dataclass
class Material:
    """Thermo-electric properties of a material used in the stack/enclosure.

    Anisotropy is supported directly: ``k_through`` (z) and ``k_in`` (x,y) may
    differ. For isotropic materials set them equal (loader does this if only ``k``
    is given). ``sigma_elec`` is the electronic conductivity used for current
    collectors (0 for non-conductors).
    """
    name: str
    density: float              # kg/m^3
    cp: float                   # J/kg/K
    k_through: float            # W/m/K, through-plane (z)
    k_in: float                 # W/m/K, in-plane (x,y)
    sigma_elec: float = 0.0     # S/m, electronic conductivity


# --------------------------------------------------------------------------- #
# Stack geometry
# --------------------------------------------------------------------------- #
LayerRole = Literal[
    "pos_collector", "cathode_coating", "separator", "anode_coating", "neg_collector"
]


@dataclass
class Layer:
    """One physical layer within a single repeat unit of the stack."""
    role: LayerRole
    material: str               # key into the material database
    thickness: float            # m


@dataclass
class Stack:
    """One electrode sandwich (repeat unit): CathodeCC | Cathode | Sep | Anode | AnodeCC.

    ``width`` x ``height`` is the in-plane electrode footprint of the sandwich: ``width`` maps to
    the electrode LENGTH axis and ``height`` to the electrode HEIGHT axis (the two axes that are
    not ``assembly.stack_axis``).
    """
    layers: List[Layer]         # ordered layers of ONE sandwich
    width: float                # m, electrode LENGTH extent (first in-plane axis)
    height: float               # m, electrode HEIGHT extent (second in-plane axis)


@dataclass
class Jellyroll:
    """A jellyroll = ``n_stacks`` electrode sandwiches stacked along ``assembly.stack_axis``.

    ``origin`` is the lower corner of the roll's bounding box inside the can cavity
    (m, [x,y,z]); if ``None`` the geometry builder auto-arranges the rolls per
    ``Assembly.arrangement``. All rolls are wired in parallel to the cell tabs.
    """
    stack: Stack                # the repeated sandwich
    n_stacks: int               # number of sandwiches stacked along the through-plane axis
    origin: Optional[List[float]] = None


@dataclass
class Wrap:
    """A thin film wrapped around each jellyroll's side faces (e.g. mylar/PET tape).

    Holds the winding/stack together and electrically isolates the roll from the can. Thermally it
    is a sub-grid conductive layer that adds series resistance ``t/k`` on the roll-to-can heat path
    over the wrapped faces. ``coverage`` selects which roll faces it covers, named by orientation
    (independent of ``stack_axis``):
      - ``"sides"``     (default) = the big front/back faces (normal to the thickness/stack axis)
                        AND the small end faces (normal to the length axis); top/bottom open.
      - ``"big_faces"`` = only the big front/back faces (normal to the stack axis).
      - ``"ends"``      = only the small end faces (normal to the length axis).
      - ``"all"``       = all six faces (adds the height ends too).
    Only the roll faces that border the can/ambient (the external cell faces) carry the resistance;
    the internal inter-roll faces are neglected (their wrap resistance is tiny next to the air gap).
    """
    material: str               # key into material database
    thickness: float            # m, film thickness
    coverage: Literal["sides", "big_faces", "ends", "all"] = "sides"


@dataclass
class Assembly:
    """The internals of the can: one or more jellyrolls and how they are arranged.

    ``stack_axis`` is the physical axis along which the sandwiches (and the jellyrolls, when
    ``arrangement="stacked"``) build up — i.e. the through-plane / thickness direction. The other
    two axes are the in-plane electrode plane: the first (in x<y<z order) is the electrode LENGTH,
    the second is the electrode HEIGHT. Example for a long flat prismatic cell: ``stack_axis="y"``
    → length = X, height = Z, thickness = Y (two jellyrolls back-to-back along Y).
    """
    jellyrolls: List[Jellyroll]
    stack_axis: Literal["x", "y", "z"] = "z"
    arrangement: Literal["stacked", "side_by_side_length", "side_by_side_height"] = "stacked"
    inter_gap: float = 1e-3     # m, gap between adjacent jellyrolls
    wall_clearance: float = 5e-4  # m, gap between roll bounding box and can wall
    roll_wrap: Optional["Wrap"] = None  # film wrapped around each jellyroll's side faces; None=absent
    cavity_fill: str = "gap_air"  # material filling the void (clearance/inter-roll/headspace):
                                  # "gap_air" (dry) or e.g. "electrolyte" (flooded cell)


@dataclass
class Tab:
    """A current tab: a rectangular contact footprint on the electrode (in-plane) plane.

    The footprint is where the collector foils bundle and weld to the terminal; it sets where
    current enters/leaves the collector (the tab-placement design lever). Location is fractional
    (0..1) along the electrode LENGTH and HEIGHT axes; ``size_length``/``size_height`` are the
    footprint extents [m]. ``thickness`` defaults to this polarity's current-collector thickness
    (e.g. Al 13 µm / Cu 6 µm) and ``material`` to its collector material — i.e. "a tab the size of
    the current-collector thickness". All ``n_stacks`` collector foils (both jellyrolls) bus to it
    in parallel. ``protrusion`` is the out-of-cell tab length used for the series R_tab and the
    tab heat-loss path; the current-carrying width is taken as ``size_length`` (so
    R_tab = protrusion / (sigma * n_foils * size_length * thickness)), while ``size_height`` only
    positions the footprint.
    """
    polarity: Literal["pos", "neg"]
    loc_length: float = 0.5     # fractional center along the electrode length axis (0..1)
    loc_height: float = 0.5     # fractional center along the electrode height axis (0..1)
    size_length: float = 0.02   # m, footprint extent along the length axis
    size_height: float = 0.02   # m, footprint extent along the height axis
    protrusion: float = 0.01    # m, tab protrusion out of the cell (R_tab & heat path)
    thickness: Optional[float] = None   # m; default = this polarity's collector thickness
    material: Optional[str] = None      # default = this polarity's collector material


@dataclass
class Insulator:
    """A solid insulating slab inside the can, spanning the full cell cross-section.

    Models the bottom-insulation film of a prismatic cell: a thin polymer slab sitting on the
    can floor (or under the lid) between the jellyrolls and the can, over the whole in-plane
    footprint (electrode length x thickness). It electrically isolates the roll from the can and
    adds thermal resistance on that heat path — so it matters when the corresponding face is
    cooled. ``location`` selects which end of the HEIGHT axis it occupies: ``"bottom"`` = the
    -height end (opposite the tabs, the usual place), ``"top"`` = the +height end. ``thickness``
    is its extent along the height axis [m]; the rolls are shifted to rest against it. Purely a
    thermal element here (the can carries no ECM current in this model)."""
    material: str               # key into material database
    thickness: float            # m, slab extent along the height axis
    location: Literal["bottom", "top"] = "bottom"


@dataclass
class Enclosure:
    """Cell enclosure: pouch (laminate) or prismatic (metal can)."""
    kind: Literal["pouch", "prismatic"]
    material: str               # key into material database
    wall_thickness: float       # m
    contact_conductance: float  # W/m^2/K, stack<->wall interfacial conductance
    tab_heat_sink: bool = True  # if True, tabs conduct heat to ambient at their far end
                                # (busbar/terminal heat-sunk near the coolant temperature)
    insulator: Optional["Insulator"] = None  # bottom (or top) insulating slab; None = absent
    # How the enclosure wall is represented thermally:
    #  - "shell" (default): a sub-grid conductive shell wraps the cavity mesh, giving the wall
    #    in-plane spreading + through-wall BC + thermal mass WITHOUT resolving its (thin) thickness
    #    on the grid. Correct for thin walls in large faces (no >100-cell requirement).
    #  - "mesh": the wall is meshed as volume cells (only resolved where dx/dy/dz < wall_thickness).
    wall_model: Literal["shell", "mesh"] = "shell"
    # Fixed can OUTER dimensions [Lx, Ly, Lz] in metres. When given, the can size is fixed (not
    # auto-sized to hug the roll): inner cavity = outer - 2*wall_thickness, the roll sits on the
    # bottom insulator (bottom-referenced in height) and is centred in the two in-plane axes, the
    # side clearances are filled with `assembly.cavity_fill` up to the roll top, and the leftover
    # space above the roll is the gas headspace. When None, the cavity is auto-sized (roll + uniform
    # `assembly.wall_clearance`).
    outer_dims: Optional[List[float]] = None
    headspace_fill: str = "gap_air"   # gas filling the headspace above the roll (electrolyte level)


# --------------------------------------------------------------------------- #
# Electrochemistry (per-CV ECM parameters; tables live in data/ files)
# --------------------------------------------------------------------------- #
@dataclass
class RCPair:
    """A single Thevenin RC pair with Arrhenius temperature dependence.

    ``r_table`` / ``c_table`` are CSV paths (columns: soc, value) giving the
    reference (T = ``t_ref``) SOC dependence. ``ea_r`` / ``ea_c`` are activation
    energies (J/mol) for Arrhenius scaling of R and C respectively.
    """
    r_table: str
    c_table: str
    ea_r: float = 0.0
    ea_c: float = 0.0


@dataclass
class ECM:
    """Equivalent-circuit-model parameter set (shared by every CV; state is per-CV).

    All SOC/temperature dependence is data-driven via CSV tables referenced here.
    ``capacity_Ah`` is the *total* nominal cell capacity; per-CV capacity is derived
    by the geometry/coupling layer from the mesh division.
    """
    capacity_Ah: float
    ocv_table: str              # CSV: soc, ocv[V]
    entropy_table: str          # CSV: soc, dUdT[V/K]
    r0_table: str               # CSV: soc, r0[Ohm*m^2]  (areal, at t_ref)
    ea_r0: float                # J/mol, Arrhenius activation energy for R0
    rc_pairs: List[RCPair] = field(default_factory=list)
    t_ref: float = 298.15       # K, reference temperature for tables
    # Optional capacity temperature dependence CSV: temp[K], capacity_scale
    capacity_temp_table: Optional[str] = None


# --------------------------------------------------------------------------- #
# Mesh, cooling, load, solver
# --------------------------------------------------------------------------- #
@dataclass
class Mesh:
    """Structured grid resolution along the physical x/y/z axes. The two axes that are not
    ``assembly.stack_axis`` set the in-plane ECM-network resolution; the stack axis resolves the
    through-thickness stack layering."""
    nx: int                     # cells along physical x
    ny: int                     # cells along physical y
    nz: int                     # cells along physical z


FaceName = Literal["top", "bottom", "x_min", "x_max", "y_min", "y_max"]


@dataclass
class FaceBC:
    """Boundary condition on one external face of the cell.

    Face names are physical: ``top``/``bottom`` are the +z/-z faces; the four sides are x/y
    min/max. Which of these are the large flat faces depends on ``assembly.stack_axis`` (e.g. for
    ``stack_axis='y'`` the large faces are ``y_min``/``y_max``).
    """
    kind: Literal["convection", "dirichlet", "neumann", "adiabatic"]
    h: float = 0.0              # W/m^2/K, convective coefficient (convection)
    t_inf: float = 298.15       # K, ambient/coolant/wall temperature
    flux: float = 0.0           # W/m^2, prescribed flux (neumann; +out)
    emissivity: float = 0.0     # 0..1 surface emissivity; >0 adds linearized radiation to t_inf
                                # on any non-dirichlet face (thermal-radiation heat path)


@dataclass
class Cooling:
    """Per-face boundary conditions. Any omitted face defaults to adiabatic.

    Faces may be given either by **physical** name (``top``/``bottom`` = +z/-z, ``x_min``/``x_max``,
    ``y_min``/``y_max``) or by **role** name that follows the cell orientation
    (``top_face``/``bottom_face``/``side_face_1..4``, see :func:`face_role_map`). Role names are
    translated to physical faces at load time based on ``assembly.stack_axis``.
    """
    top: FaceBC = field(default_factory=lambda: FaceBC("adiabatic"))
    bottom: FaceBC = field(default_factory=lambda: FaceBC("adiabatic"))
    x_min: FaceBC = field(default_factory=lambda: FaceBC("adiabatic"))
    x_max: FaceBC = field(default_factory=lambda: FaceBC("adiabatic"))
    y_min: FaceBC = field(default_factory=lambda: FaceBC("adiabatic"))
    y_max: FaceBC = field(default_factory=lambda: FaceBC("adiabatic"))


# Role-based face naming. Physical faces per axis as (negative_face, positive_face):
_AXIS_FACES = {0: ("x_min", "x_max"), 1: ("y_min", "y_max"), 2: ("bottom", "top")}
FACE_ROLES = ("top_face", "bottom_face", "front_face", "back_face", "left_face", "right_face",
              "side_face_1", "side_face_2", "side_face_3", "side_face_4")


def face_role_map(stack_axis: str) -> Dict[str, str]:
    """Map role face names to physical faces for a given ``stack_axis`` (length = first non-stack
    axis, height = second).

    Intuitive names (for a cell with length=X, height=Z, thickness/stack=Y, i.e. stack_axis='y'):
      - ``top_face`` / ``bottom_face``  = the +/- HEIGHT faces  (XY plane, normal to height Z)
      - ``front_face`` / ``back_face``  = the two LARGE flat faces (XZ plane, normal to the stack
        axis Y) -- the big front/back faces
      - ``left_face`` / ``right_face``  = the small end faces  (YZ plane, normal to length X)
    ``side_face_1/2`` are aliases for front/back (large faces); ``side_face_3/4`` for left/right.
    """
    sa = {"x": 0, "y": 1, "z": 2}[stack_axis]
    la, ha = [a for a in (0, 1, 2) if a != sa]
    return {
        "top_face": _AXIS_FACES[ha][1], "bottom_face": _AXIS_FACES[ha][0],
        # large flat faces: normal to the stack axis (thickness) -> the biggest faces
        "back_face": _AXIS_FACES[sa][0], "front_face": _AXIS_FACES[sa][1],
        "side_face_1": _AXIS_FACES[sa][0], "side_face_2": _AXIS_FACES[sa][1],
        # small end faces: normal to the length axis
        "left_face": _AXIS_FACES[la][0], "right_face": _AXIS_FACES[la][1],
        "side_face_3": _AXIS_FACES[la][0], "side_face_4": _AXIS_FACES[la][1],
    }


def _cooling_roles_to_physical(cooling_raw: Dict[str, Any], stack_axis: str) -> Dict[str, Any]:
    """Translate a role-keyed cooling dict to physical face keys (from YAML preprocessing)."""
    mapping = face_role_map(stack_axis)
    out: Dict[str, Any] = {}
    for role, bc in cooling_raw.items():
        if role not in mapping:
            raise ValueError(
                f"unknown cooling face role '{role}'; use one of {FACE_ROLES} "
                f"(or physical names top/bottom/x_min/x_max/y_min/y_max)")
        if mapping[role] in out:
            raise ValueError(f"cooling role '{role}' maps to physical face '{mapping[role]}' "
                             "which is already assigned")
        out[mapping[role]] = bc
    return out


@dataclass
class Load:
    """Applied electrical load / current profile.

    - ``constant_current``: ``value`` in A (positive = discharge)
    - ``constant_crate``:   ``value`` in C (multiplied by capacity)
    - ``constant_voltage``: ``value`` in V (potentiostatic)
    - ``profile``:          ``profile_csv`` with columns time[s], value; ``kind_of``
                            selects whether values are A, C, or V.
    """
    kind: Literal["constant_current", "constant_crate", "constant_voltage", "profile"]
    value: float = 0.0
    profile_csv: Optional[str] = None
    profile_units: Literal["A", "C", "V"] = "A"


@dataclass
class Solver:
    """Time-integration / solve controls."""
    mode: Literal["transient", "steady"] = "transient"
    dt: float = 1.0             # s, timestep (transient)
    t_end: float = 3600.0       # s, end time (transient)
    max_subiter: int = 20       # electro<->thermal sub-iterations per step
    coupling_tol: float = 1e-6  # convergence tol on coupling sub-iteration
    newton_tol: float = 1e-9    # convergence tol on the electro network solve
    newton_max: int = 50        # max Newton iterations for the network
    linear_solver: Literal["direct", "cg"] = "direct"
    steady_relax: float = 0.5   # under-relaxation factor for the steady fixed-point (0<w<=1)
    steady_max: int = 300       # max iterations for the steady coupled fixed-point
    # Collector network fidelity: "planar" = one shared 2-D foil potential per jellyroll
    # (2.5-D; fast); "layered" = a separate 2-D foil potential per through-thickness stack
    # layer, all in parallel at the tabs (full 3-D collector; resolves through-thickness
    # potential/current gradients driven by the 3-D temperature field).
    collector_model: Literal["planar", "layered"] = "planar"


# --------------------------------------------------------------------------- #
# Top-level configuration
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    """Complete specification of one simulation run."""
    name: str
    materials: Dict[str, Material]
    assembly: Assembly
    tabs: List[Tab]
    enclosure: Enclosure
    ecm: ECM
    mesh: Mesh
    cooling: Cooling
    load: Load
    solver: Solver
    t_init: float = 298.15      # K, initial uniform temperature
    soc_init: float = 1.0       # initial state of charge (uniform), 0..1
    data_root: str = "."        # base dir for resolving relative CSV table paths

    # ---- loading / validation ------------------------------------------- #
    @classmethod
    def from_yaml(cls, path: str) -> "SimConfig":
        """Parse and validate a YAML config into a typed ``SimConfig`` tree.

        Supports a top-level ``include: [file, ...]`` list (merged first, shallow)
        so materials can live in a shared file. Relative CSV table paths are
        resolved against ``data_root`` (defaults to the config file's directory).
        """
        with open(path, "r") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}
        base_dir = os.path.dirname(os.path.abspath(path))
        for inc in raw.pop("include", []) or []:
            inc_path = inc if os.path.isabs(inc) else os.path.join(base_dir, inc)
            with open(inc_path, "r") as fh:
                inc_raw = yaml.safe_load(fh) or {}
            for k, v in inc_raw.items():
                if k not in raw:
                    raw[k] = v
                elif isinstance(raw[k], dict) and isinstance(v, dict):
                    merged = dict(v)
                    merged.update(raw[k])
                    raw[k] = merged
        raw.setdefault("data_root", base_dir)
        # Translate role-based cooling face names (top_face/bottom_face/side_face_1..4) to
        # physical faces, based on the assembly's stack axis.
        cooling_raw = raw.get("cooling")
        if isinstance(cooling_raw, dict) and any(k in FACE_ROLES for k in cooling_raw):
            phys = {k for k in cooling_raw if k not in FACE_ROLES}
            if phys:
                raise ValueError(f"cooling mixes role and physical face names: {sorted(phys)}. "
                                 "Use one naming scheme.")
            stack_axis = (raw.get("assembly") or {}).get("stack_axis", "z")
            raw["cooling"] = _cooling_roles_to_physical(cooling_raw, stack_axis)
        cfg = _build(cls, raw)
        cfg.validate()
        return cfg

    def resolve(self, table_path: Optional[str]) -> Optional[str]:
        """Resolve a (possibly relative) CSV table path against ``data_root``."""
        if table_path is None:
            return None
        if os.path.isabs(table_path):
            return table_path
        return os.path.join(self.data_root, table_path)

    def validate(self) -> None:
        """Raise ``ValueError`` on structurally invalid configs."""
        errors: List[str] = []
        # material references resolve
        known = set(self.materials)
        if not self.assembly.jellyrolls:
            errors.append("assembly must contain at least one jellyroll")
        for ridx, roll in enumerate(self.assembly.jellyrolls):
            if roll.n_stacks < 1:
                errors.append(f"jellyroll[{ridx}].n_stacks must be >= 1")
            roles = {lyr.role for lyr in roll.stack.layers}
            for req in ("pos_collector", "cathode_coating", "separator",
                        "anode_coating", "neg_collector"):
                if req not in roles:
                    errors.append(f"jellyroll[{ridx}] sandwich missing required layer role '{req}'")
            for lyr in roll.stack.layers:
                if lyr.material not in known:
                    errors.append(f"jellyroll[{ridx}] layer role={lyr.role} references unknown material '{lyr.material}'")
                if lyr.thickness <= 0:
                    errors.append(f"jellyroll[{ridx}] layer role={lyr.role} thickness must be > 0")
        for t in self.tabs:
            if t.material is not None and t.material not in known:
                errors.append(f"tab {t.polarity} references unknown material '{t.material}'")
            if not (0.0 <= t.loc_length <= 1.0) or not (0.0 <= t.loc_height <= 1.0):
                errors.append(f"tab {t.polarity} loc_length/loc_height must be in [0,1]")
            if t.size_length <= 0 or t.size_height <= 0:
                errors.append(f"tab {t.polarity} size_length/size_height must be > 0")
        if self.enclosure.material not in known:
            errors.append(f"enclosure references unknown material '{self.enclosure.material}'")
        od = self.enclosure.outer_dims
        if od is not None:
            if len(od) != 3 or any(v <= 0 for v in od):
                errors.append("enclosure.outer_dims must be 3 positive values [Lx, Ly, Lz]")
            if self.enclosure.headspace_fill not in known:
                errors.append(f"enclosure.headspace_fill references unknown material "
                              f"'{self.enclosure.headspace_fill}'")
        ins = self.enclosure.insulator
        if ins is not None:
            if ins.material not in known:
                errors.append(f"enclosure.insulator references unknown material '{ins.material}'")
            if ins.thickness <= 0:
                errors.append("enclosure.insulator.thickness must be > 0")
        if self.assembly.cavity_fill not in known:
            errors.append(f"assembly.cavity_fill references unknown material "
                          f"'{self.assembly.cavity_fill}'")
        wrap = self.assembly.roll_wrap
        if wrap is not None:
            if wrap.material not in known:
                errors.append(f"assembly.roll_wrap references unknown material '{wrap.material}'")
            if wrap.thickness <= 0:
                errors.append("assembly.roll_wrap.thickness must be > 0")
        # need at least one pos and one neg tab
        pol = {t.polarity for t in self.tabs}
        if "pos" not in pol or "neg" not in pol:
            errors.append("need at least one 'pos' and one 'neg' tab")
        # mesh sanity
        if min(self.mesh.nx, self.mesh.ny, self.mesh.nz) < 1:
            errors.append("mesh dimensions must be >= 1")
        if self.ecm.capacity_Ah <= 0:
            errors.append("ecm.capacity_Ah must be > 0")
        if errors:
            raise ValueError("Invalid SimConfig:\n  - " + "\n  - ".join(errors))


# --------------------------------------------------------------------------- #
# Generic typed-dataclass builder (keeps loader DRY, no per-field boilerplate)
# --------------------------------------------------------------------------- #
def _build(tp: Any, value: Any) -> Any:
    """Recursively coerce ``value`` into type ``tp`` (dataclass/list/dict/scalar)."""
    if is_dataclass(tp) and isinstance(value, dict):
        kwargs: Dict[str, Any] = {}
        # Resolve string annotations (PEP 563 / `from __future__ import annotations`)
        # to real types so origin/dataclass introspection below works.
        type_hints = get_type_hints(tp)
        for f in fields(tp):
            if f.name in value:
                kwargs[f.name] = _build(type_hints[f.name], value[f.name])
        return tp(**kwargs)
    origin = get_origin(tp)
    if origin in (list, List):
        (elem_tp,) = get_args(tp) or (Any,)
        return [_build(elem_tp, v) for v in value]
    if origin in (dict, Dict):
        args = get_args(tp)
        if len(args) == 2 and value is not None:
            _, vt = args
            out = {}
            vt_is_named = is_dataclass(vt) and any(f.name == "name" for f in fields(vt))
            for k, v in value.items():
                if vt_is_named and isinstance(v, dict) and "name" not in v:
                    v = {"name": k, **v}
                out[k] = _build(vt, v)
            return out
        return value
    # Optional[X] / Union
    if origin is not None and args_has_none(tp):
        for a in get_args(tp):
            if a is type(None):
                continue
            if value is None:
                return None
            return _build(a, value)
    # Scalar coercion. YAML 1.1 (PyYAML) parses unsigned-exponent floats like "3.5e7" as
    # strings; coerce numeric-typed fields so such values still load correctly.
    if tp is float and isinstance(value, (str, int)):
        return float(value)
    if tp is int and isinstance(value, str):
        return int(value)
    return value


def args_has_none(tp: Any) -> bool:
    return type(None) in get_args(tp)
