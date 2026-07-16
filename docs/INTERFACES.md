# Module interface contract

Every module implements exactly these public objects/signatures so the pieces compose without
drift. Physics equations are in `PHYSICS.md` (referenced by §). Config types are in
`prismaticcell/config.py`; homogenization in `prismaticcell/materials.py` (both already written).
All arrays are NumPy `float64`; SI units throughout. **No coefficient may be hard-coded** — read
everything from `SimConfig` / material DB / CSV tables.

Modeling convention (agreed): **2.5-D electro-thermal.** Thermal is full 3-D (`nx×ny×nz`).
Collector potentials `φ⁺,φ⁻` are 2-D per jellyroll (the many sandwiches share tabs). **Every
active 3-D control volume still carries its own ECM state** (`soc`, RC overpotentials, local `T`);
within an in-plane column the active z-cells share the local `Δφ = φ⁺−φ⁻` and each contributes its
own current — so 3-D SOC/temperature non-uniformity is preserved (PHYSICS §2–§3).

---

## `geometry.py`  (implements PHYSICS §0, §1, §5-mesh)

```python
@dataclass
class Grid:
    nx: int; ny: int; nz: int
    dx: float; dy: float; dz: float          # uniform cell size [m]
    xc: np.ndarray; yc: np.ndarray; zc: np.ndarray   # 1-D cell-center coords
    def volume(self) -> float                # dx*dy*dz
    def flat(self, i, j, k) -> int           # k + nz*(j + ny*i)  (C-order on (i,j,k))

# Region integer codes
REGION_CAN = 0; REGION_GAP = 1; REGION_ACTIVE = 2; REGION_TAB = 3

@dataclass
class RollElectro:
    roll_index: int
    n_stacks: int
    columns: list[tuple[int,int]]            # active in-plane (i,j) columns of this roll
    col_zcells: dict[tuple[int,int], list[int]]   # (i,j) -> active z indices k
    area_eff: np.ndarray                     # (nx,ny,nz) effective electrode area per CV [m^2]
                                             #   = dx*dy * (n_stacks / n_active_z_in_roll); 0 if inactive
    sheet_cond_pos: float                    # Al foil in-plane sheet conductance σ*t*n_stacks [S]
    sheet_cond_neg: float                    # Cu foil                              [S]
    tab_pos_nodes: list[tuple[int,int]]      # (i,j) columns where + tab injects current
    tab_neg_nodes: list[tuple[int,int]]      # (i,j) columns where - tab extracts current

@dataclass
class Geometry:
    grid: Grid
    region: np.ndarray                       # (nx,ny,nz) int region code
    kx: np.ndarray; ky: np.ndarray; kz: np.ndarray   # (nx,ny,nz) conductivity [W/m/K]
    rho_cp: np.ndarray                       # (nx,ny,nz) volumetric heat capacity [J/m^3/K]
    active_mask: np.ndarray                  # (nx,ny,nz) bool: has an ECM
    cell_roll: np.ndarray                    # (nx,ny,nz) int roll idx, -1 if none
    cap_cv: np.ndarray                       # (nx,ny,nz) capacity [Ah] per CV (Σ == cfg.ecm.capacity_Ah)
    rolls: list[RollElectro]
    outer_dims: tuple[float,float,float]     # (Lx,Ly,Lz) [m]

def build_geometry(cfg: SimConfig) -> Geometry: ...
```

Construction rules: outer box = can wall (`enclosure.wall_thickness`) around a cavity holding the
jellyrolls (arranged per `assembly.arrangement`, separated by `inter_gap`, `wall_clearance` to the
wall). Uniform grid. Tag each cell by center location: can-wall shell → REGION_CAN
(`isotropic_props(can material)`); clearance/inter-roll bands → REGION_GAP (filled with
`assembly.cavity_fill`, e.g. `gap_air` or `electrolyte`); inside a
roll bbox → REGION_ACTIVE with `materials.homogenize(sandwich_layers)` mapped so **k_z = through-
plane, k_x=k_y = in-plane**. `cap_cv` normalized so it sums exactly to `cfg.ecm.capacity_Ah`.
Foil sheet conductance uses the collector layer's `sigma_elec` × its thickness × `n_stacks`. Tab
nodes = the electrode columns under each tab's rectangular footprint (fractional `loc_length`,
`loc_height` center; `size_length` × `size_height` extents), for every roll (both rolls parallel
to the same terminal). Anisotropy and the electrode plane follow `assembly.stack_axis` (the
through-plane axis); the two in-plane axes are the electrode length and height.

Optional sub-grid layers — `enclosure.insulator` (bottom/top film) and `assembly.roll_wrap` (film
on each roll's side faces) — are aggregated into `geom.face_R_area` and `geom.face_rhocp_t`, dicts
keyed by physical face name (`t/k` [K·m²/W] and `ρc·t` [J/m²/K], summed where layers coincide).
`thermal.py` adds `face_R_area[f]` in series on face `f`'s BC and lumps `face_rhocp_t[f]` onto its
cells (neither is meshed as a region). The insulator's single-face values are also kept as
`insulator_face` / `insulator_R_area` / `insulator_rhocp_t` for reference. In shell mode the
roll-to-can `wall_clearance` (filled with `cavity_fill`) also contributes `wall_clearance/k_fill`
to every external face via the same dicts. With a fixed can (`enclosure.outer_dims`) the mesh is the
roll bbox and the void is per-face instead: electrolyte side clearances on the in-plane faces, a gas
`headspace_fill` layer on the top face, and the insulator on the bottom (roll bottom-referenced,
centred in-plane).

---

## `echem.py`  (implements PHYSICS §2, §4)

```python
class Table:                                  # CSV lookup with linear interp + flat extrapolation
    @classmethod
    def from_csv(cls, path: str) -> "Table": ...
    def __call__(self, x: np.ndarray | float) -> np.ndarray | float: ...

@dataclass
class ECMModel:                               # stateless parameter provider (state lives in ECMState)
    ocv: Table; entropy: Table; r0: Table
    rc_r: list[Table]; rc_c: list[Table]
    ea_r0: float; ea_r: list[float]; ea_c: list[float]; t_ref: float
    capacity_temp: Table | None
    @classmethod
    def from_config(cls, cfg: SimConfig) -> "ECMModel": ...
    def ocv_v(self, soc) -> np.ndarray                 # U_ocv(soc)
    def dudt(self, soc) -> np.ndarray                  # entropic dU/dT(soc)
    def r0_area(self, soc, T) -> np.ndarray            # areal R0 [Ω·m²], Arrhenius (PHYSICS §2.2)
    def rc_area(self, p, soc, T) -> tuple[np.ndarray, np.ndarray]  # (R_p[Ω·m²], C_p[F/m²]) Arrhenius

@dataclass
class ECMState:                               # per active CV, flat arrays over active CVs
    soc: np.ndarray                           # (n_active,)
    rc_u: np.ndarray                          # (n_rc, n_active) overpotentials [V]
    def advance_rc(self, model, j_area, T, dt): ...    # du/dt = -u/(R C) + j/C  (PHYSICS §2.1)
    def advance_soc(self, model, i_cv, cap_cv, T, dt): ...  # dz/dt = -i/(3600 Q(T)) (PHYSICS §2.3)

# Local through-CV areal current density from shared column Δφ and this CV's state:
#   v = ocv(soc,T) - j_area*R0 - Σ_p u_p ;  and v == Δφ  ->  j_area = (ocv - Σu - Δφ)/R0
# Heat per CV (PHYSICS §4): q_ecm = i*(ocv - v); q_rev = -i*T*dudt(soc) [Bernardi minus sign];
# returned by coupling.
```

`Arrhenius`: `X(T) = X_ref * exp[(Ea/R_GAS)(1/T - 1/t_ref)]` using `config.R_GAS`.

---

## `thermal.py`  (implements PHYSICS §5)

```python
@dataclass
class ThermalOperator:
    A: scipy.sparse.csr_matrix                # (N,N) conductance + BC, N = nx*ny*nz
    b_bc: np.ndarray                          # (N,) BC source (convection/dirichlet/neumann)
    M: np.ndarray                             # (N,) lumped capacitance rho_cp*V per cell
    @classmethod
    def assemble(cls, geom: Geometry, cooling: Cooling) -> "ThermalOperator": ...
        # face conductance = harmonic mean of the two cells' directional k (series); external
        # faces per `cooling` (convection/dirichlet/neumann/adiabatic), independently per face.

def solve_steady(op: ThermalOperator, q_vol: np.ndarray) -> np.ndarray: ...   # A T = b_bc + q_vol*V
def step_transient(op, T_prev, q_vol, dt, linear_solver="direct") -> np.ndarray: ...
        # (M/dt + A) T = (M/dt) T_prev + b_bc + q_vol*V     (backward Euler, PHYSICS §5.3)

def boundary_heat_removed(op, T) -> float: ...   # net W leaving through all external faces
```
`q_vol` is volumetric heat density [W/m³] per cell (N,). `V = grid.volume()`.

---

## `distributed.py`  (implements PHYSICS §3)

```python
@dataclass
class NetworkSolution:
    j_area: np.ndarray            # (nx,ny,nz) local areal current density [A/m²], 0 if inactive
    i_cv: np.ndarray              # (nx,ny,nz) current per CV [A] (= j_area*area_eff)
    phi_pos: dict[int, np.ndarray]  # roll_index -> (nx,ny) + foil potential
    phi_neg: dict[int, np.ndarray]
    v_terminal: float             # cell terminal voltage [V]
    i_terminal: float             # total cell current [A] (== I_app for galvanostatic)
    q_ohm_vol: np.ndarray         # (nx,ny,nz) foil ohmic heat density [W/m³]

def solve_network(geom, model, state, T_field, applied, *, mode, tol, maxiter) -> NetworkSolution:
    # `applied` is current [A] (galvanostatic, mode="current") or voltage [V] (mode="voltage").
    # Per roll assemble 2-D collector Laplacians (sheet_cond * neighbor coupling) with tab terminals;
    # per in-plane column the active z-cells share Δφ and each gives j_area=(ocv-Σu-Δφ)/R0 (PHYSICS §3.1);
    # column source into foils = Σ_z i_cv. Terminal constraint Σ i_cv = I_app (or fixed ΔV_terminal).
    # Solve the coupled linear system for (φ⁺,φ⁻, terminal); Newton only if you choose to iterate R(state).
```
`T_field` is the (nx,ny,nz) temperature; active CVs read their local T.

---

## `coupling.py`  (implements PHYSICS §6)

```python
@dataclass
class Result:
    t: np.ndarray                 # (nt,) time [s]
    v_terminal: np.ndarray        # (nt,)
    i_terminal: np.ndarray        # (nt,)
    soc_mean: np.ndarray          # (nt,)
    T_mean: np.ndarray; T_max: np.ndarray; T_min: np.ndarray  # (nt,)
    T_field: np.ndarray           # (nt, nx, ny, nz) or final field + snapshots
    q_total: np.ndarray           # (nt,) total heat generation [W]
    energy_balance: dict          # gen, stored, removed [J] for the conservation check
    geom: Geometry

def run(cfg: SimConfig) -> Result:
    # transient: for each dt -> solve_network -> heat map (q_ecm+q_rev+q_ohm) -> step_transient
    #            -> advance soc/rc -> re-evaluate T-dependent params. Sub-iterate to coupling_tol.
    # steady: fixed operating current; iterate network<->solve_steady to self-consistency.

def current_at(cfg, t) -> float: ...   # resolves Load (constant_current/crate/voltage/profile)
```

---

## `api.py`, `viz.py`, `cli.py`

```python
# api.py
class Simulation:
    def __init__(self, cfg: SimConfig | str): ...   # accept SimConfig or path
    def run(self) -> Result: ...
class Sweep:
    def __init__(self, base_cfg, param_grid: dict[str, list]): ...  # dotted paths, e.g. "cooling.bottom.h"
    def run(self) -> "pandas-free list[dict]": ...   # rows: params + scalar outputs (Tmax, dV, ...)

# viz.py  (matplotlib; save or return figure)
def plot_time_series(result: Result, path=None): ...       # V, I, SOC, Tmax/Tmean vs t
def plot_temperature_slice(result, k_index=None, path=None): ...  # in-plane T map + hotspot
def plot_current_distribution(result, path=None): ...      # j_area map (tab effect)
def plot_sweep_heatmap(rows, x, y, z, path=None): ...       # design-sweep heatmap
def tab_thermal_profiles(result, cooling=None, n=25): ...   # 1-D tab fin T(x): root->sink + I²R
def plot_tab_temperature(result, cooling=None, path=None): ...  # plot the tab fin profiles

# cli.py
def main(argv=None) -> int: ...    # `prismaticcell run CONFIG [--steady] [--out DIR]`
```
