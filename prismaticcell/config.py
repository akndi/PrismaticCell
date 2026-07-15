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

    ``width`` x ``height`` is the in-plane electrode footprint of the sandwich.
    """
    layers: List[Layer]         # ordered layers of ONE sandwich
    width: float                # m, in-plane x extent (electrode)
    height: float               # m, in-plane y extent (electrode)


@dataclass
class Jellyroll:
    """A jellyroll = ``n_stacks`` electrode sandwiches through the thickness (z).

    ``origin`` is the lower corner of the roll's bounding box inside the can cavity
    (m, [x,y,z]); if ``None`` the geometry builder auto-arranges the rolls per
    ``Assembly.arrangement``. Both rolls are wired in parallel to the cell tabs.
    """
    stack: Stack                # the repeated sandwich
    n_stacks: int               # number of sandwiches stacked through z
    origin: Optional[List[float]] = None


@dataclass
class Assembly:
    """The internals of the can: one or more jellyrolls and how they are arranged."""
    jellyrolls: List[Jellyroll]
    arrangement: Literal["side_by_side_x", "side_by_side_y", "stacked_z"] = "side_by_side_x"
    inter_gap: float = 1e-3     # m, gap between adjacent jellyrolls
    wall_clearance: float = 5e-4  # m, gap between roll bounding box and can wall


@dataclass
class Tab:
    """A current tab welded to a collector at a parametric edge position."""
    polarity: Literal["pos", "neg"]
    edge: Literal["x_min", "x_max", "y_min", "y_max"]
    position: float             # fractional location along the edge, 0..1
    width: float                # m, tab width along the edge
    thickness: float            # m
    length: float               # m, protrusion beyond the enclosure
    material: str               # key into material database


@dataclass
class Enclosure:
    """Cell enclosure: pouch (laminate) or prismatic (metal can)."""
    kind: Literal["pouch", "prismatic"]
    material: str               # key into material database
    wall_thickness: float       # m
    contact_conductance: float  # W/m^2/K, stack<->wall interfacial conductance


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
    """Structured grid resolution. ``nx``,``ny`` also set the ECM network size."""
    nx: int                     # in-plane x cells (electrode + tab columns)
    ny: int                     # in-plane y cells
    nz: int                     # through-plane thermal cells (stack layering)


FaceName = Literal["top", "bottom", "x_min", "x_max", "y_min", "y_max"]


@dataclass
class FaceBC:
    """Boundary condition on one external face of the cell.

    ``top``/``bottom`` are the +z/-z (large) faces; the four sides are x/y min/max.
    """
    kind: Literal["convection", "dirichlet", "neumann", "adiabatic"]
    h: float = 0.0              # W/m^2/K, convective coefficient (convection)
    t_inf: float = 298.15       # K, ambient/coolant/wall temperature
    flux: float = 0.0           # W/m^2, prescribed flux (neumann; +out)


@dataclass
class Cooling:
    """Per-face boundary conditions. Any omitted face defaults to adiabatic."""
    top: FaceBC = field(default_factory=lambda: FaceBC("adiabatic"))
    bottom: FaceBC = field(default_factory=lambda: FaceBC("adiabatic"))
    x_min: FaceBC = field(default_factory=lambda: FaceBC("adiabatic"))
    x_max: FaceBC = field(default_factory=lambda: FaceBC("adiabatic"))
    y_min: FaceBC = field(default_factory=lambda: FaceBC("adiabatic"))
    y_max: FaceBC = field(default_factory=lambda: FaceBC("adiabatic"))


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
            if t.material not in known:
                errors.append(f"tab {t.polarity} references unknown material '{t.material}'")
            if not (0.0 <= t.position <= 1.0):
                errors.append(f"tab {t.polarity} position must be in [0,1]")
        if self.enclosure.material not in known:
            errors.append(f"enclosure references unknown material '{self.enclosure.material}'")
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
    # Material dict special-case handled via Dict[str, Material] above; scalars pass through
    return value


def args_has_none(tp: Any) -> bool:
    return type(None) in get_args(tp)
