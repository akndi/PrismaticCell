"""Two-way electro-thermal co-simulation driver (PHYSICS §6).

Orchestrates the closed loop: the distributed ECM network produces a per-control-volume heat
map from the local electrochemical state and temperature; the 3-D thermal solver advances the
temperature field; the new temperature feeds back into the (Arrhenius) ECM parameters for the
next step. Genuinely bidirectional — verified by the global energy balance recorded in
``Result.energy_balance``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import numpy as np

from .config import SimConfig
from .geometry import build_geometry, Geometry
from .echem import ECMModel, ECMState
from .thermal import ThermalOperator, solve_steady, step_transient, boundary_heat_removed
from .distributed import solve_network


@dataclass
class Result:
    t: np.ndarray
    v_terminal: np.ndarray
    i_terminal: np.ndarray
    soc_mean: np.ndarray
    T_mean: np.ndarray
    T_max: np.ndarray
    T_min: np.ndarray
    T_field: np.ndarray            # (nt, nx, ny, nz)
    q_total: np.ndarray            # (nt,) total heat generation [W]
    j_field_final: np.ndarray      # (nx,ny,nz) final areal current density [A/m^2]
    soc_field_final: np.ndarray    # (nx,ny,nz) final SOC (NaN where inactive)
    energy_balance: dict
    geom: Geometry
    snapshots: dict = field(default_factory=dict)   # optional labelled T snapshots


def applied_at(cfg: SimConfig, t: float) -> Tuple[str, float]:
    """Resolve the load at time ``t`` into (mode, value).

    mode is "current" (value in A, positive = discharge) or "voltage" (value in V).
    """
    load = cfg.load
    if load.kind == "constant_current":
        return "current", load.value
    if load.kind == "constant_crate":
        return "current", load.value * cfg.ecm.capacity_Ah
    if load.kind == "constant_voltage":
        return "voltage", load.value
    if load.kind == "profile":
        path = cfg.resolve(load.profile_csv)
        data = _load_profile(path)
        val = float(np.interp(t, data[:, 0], data[:, 1]))
        if load.profile_units == "V":
            return "voltage", val
        if load.profile_units == "C":
            return "current", val * cfg.ecm.capacity_Ah
        return "current", val
    raise ValueError(f"unknown load kind '{load.kind}'")


_PROFILE_CACHE: Dict[str, np.ndarray] = {}


def _load_profile(path: str) -> np.ndarray:
    if path not in _PROFILE_CACHE:
        rows = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                try:
                    rows.append([float(parts[0]), float(parts[1])])
                except (ValueError, IndexError):
                    continue  # header
        _PROFILE_CACHE[path] = np.array(rows, dtype=float)
    return _PROFILE_CACHE[path]


def _heat_map(geom: Geometry, model: ECMModel, state: ECMState, sol,
              T_field: np.ndarray, active_ijk: np.ndarray) -> np.ndarray:
    """Volumetric heat density [W/m^3] per cell = ECM (irrev + reversible) + foil ohmic (PHYSICS §4)."""
    grid = geom.grid
    V = grid.volume()
    q_vol = np.array(sol.q_ohm_vol, dtype=float)  # foil ohmic already W/m^3
    ii, jj, kk = active_ijk[:, 0], active_ijk[:, 1], active_ijk[:, 2]
    T_act = T_field[ii, jj, kk]
    soc = state.soc
    ocv = model.ocv_v(soc)
    # local cell voltage = Δφ across the stack at this column (vectorized per roll)
    dphi = np.empty_like(soc)
    roll_of = geom.cell_roll[ii, jj, kk]
    for r, pp in sol.phi_pos.items():
        m = roll_of == r
        if np.any(m):
            dphi[m] = pp[ii[m], jj[m]] - sol.phi_neg[r][ii[m], jj[m]]
    i_cv = sol.i_cv[ii, jj, kk]
    # Bernardi decomposition (PHYSICS §4): Q = I(U_ocv - V) - I*T*(dU/dT).
    # Irreversible overpotential heat (dominant; = I^2*R over a cycle):
    q_irrev = i_cv * (ocv - dphi)
    # Reversible/entropic heat GENERATED = -I*T*(dU/dT). dU/dT comes from the entropy
    # table (small, sign-changing for LFP); the leading minus is the Bernardi sign.
    q_rev = -i_cv * T_act * model.dudt(soc)
    q_ecm = (q_irrev + q_rev) / V
    q_vol[ii, jj, kk] += q_ecm
    return q_vol


def run(cfg: SimConfig) -> Result:
    """Run the coupled simulation described by ``cfg`` (transient or steady)."""
    geom = build_geometry(cfg)
    model = ECMModel.from_config(cfg)
    op = ThermalOperator.assemble(geom, cfg.cooling)
    grid = geom.grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    V = grid.volume()

    active_ijk = np.argwhere(geom.active_mask)          # (n_active, 3), row-major (i,j,k)
    n_active = len(active_ijk)
    ii, jj, kk = active_ijk[:, 0], active_ijk[:, 1], active_ijk[:, 2]
    n_rc = len(cfg.ecm.rc_pairs)

    state = ECMState(soc=np.full(n_active, cfg.soc_init),
                     rc_u=np.zeros((n_rc, n_active)))
    cap_act = geom.cap_cv[ii, jj, kk]
    T_field = np.full((nx, ny, nz), cfg.t_init)

    if cfg.solver.mode == "steady":
        return _run_steady(cfg, geom, model, op, state, T_field, active_ijk, cap_act)

    dt = cfg.solver.dt
    t_end = cfg.solver.t_end
    nsteps = int(round(t_end / dt))
    times, vt, it_, socm = [], [], [], []
    Tm, Tmx, Tmn, qtot = [], [], [], []
    T_hist: List[np.ndarray] = []

    gen_energy = 0.0
    removed_energy = 0.0
    M = op.M
    U0 = float((M * T_field.ravel(order="C")).sum())

    last_sol = None
    for s in range(nsteps):
        t = s * dt
        mode, applied = applied_at(cfg, t)
        # ---- coupling sub-iteration: network<->thermal at fixed z,u ----
        T_iter = T_field
        for _ in range(cfg.solver.max_subiter):
            sol = solve_network(geom, model, state, T_iter, applied,
                                mode=mode, tol=cfg.solver.newton_tol,
                                maxiter=cfg.solver.newton_max)
            q_vol = _heat_map(geom, model, state, sol, T_iter, active_ijk)
            T_new = step_transient(op, T_field, q_vol, dt,
                                   linear_solver=cfg.solver.linear_solver
                                   ).reshape(nx, ny, nz)
            if np.max(np.abs(T_new - T_iter)) < cfg.solver.coupling_tol:
                T_iter = T_new
                break
            T_iter = T_new
        last_sol = sol
        # ---- advance electrochemical state with the converged current FIRST, so the
        #      recorded SOC is the end-of-step value consistent with time t+dt and T_new ----
        T_act = T_new[ii, jj, kk]
        j_area_act = sol.j_area[ii, jj, kk]
        i_cv_act = sol.i_cv[ii, jj, kk]
        state.advance_rc(model, j_area_act, T_act, dt)
        state.advance_soc(model, i_cv_act, cap_act, T_act, dt)
        T_field = T_new
        # ---- record end-of-step state ----
        q_step = float((q_vol * V).sum())
        gen_energy += q_step * dt
        removed_energy += boundary_heat_removed(op, T_new.ravel(order="C")) * dt
        times.append(t + dt)
        vt.append(sol.v_terminal)
        it_.append(sol.i_terminal)
        socm.append(float(state.soc.mean()))
        Tm.append(float(T_new.mean()))
        Tmx.append(float(T_new.max()))
        Tmn.append(float(T_new.min()))
        qtot.append(q_step)
        T_hist.append(T_new.copy())
        # ---- stop once the cell is fully discharged/charged (avoid unphysical over-run) ----
        soc_mean = float(state.soc.mean())
        if soc_mean <= 1e-6 or soc_mean >= 1.0 - 1e-6:
            break

    U1 = float((M * T_field.ravel(order="C")).sum())
    stored = U1 - U0
    energy_balance = {
        "generated_J": gen_energy,
        "stored_J": stored,
        "removed_J": removed_energy,
        "residual_J": gen_energy - stored - removed_energy,
        "closure_rel": (gen_energy - stored - removed_energy) / (abs(gen_energy) + 1e-30),
    }

    soc_field = np.full((nx, ny, nz), np.nan)
    soc_field[ii, jj, kk] = state.soc
    return Result(
        t=np.array(times), v_terminal=np.array(vt), i_terminal=np.array(it_),
        soc_mean=np.array(socm), T_mean=np.array(Tm), T_max=np.array(Tmx),
        T_min=np.array(Tmn), T_field=np.array(T_hist), q_total=np.array(qtot),
        j_field_final=(last_sol.j_area if last_sol is not None else np.zeros((nx, ny, nz))),
        soc_field_final=soc_field, energy_balance=energy_balance, geom=geom,
    )


def _run_steady(cfg, geom, model, op, state, T_field, active_ijk, cap_act) -> Result:
    """Steady operating point: iterate network<->steady-thermal to self-consistency."""
    grid = geom.grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    V = grid.volume()
    ii, jj, kk = active_ijk[:, 0], active_ijk[:, 1], active_ijk[:, 2]
    mode, applied = applied_at(cfg, 0.0)
    T_iter = T_field
    sol = None
    for _ in range(max(cfg.solver.max_subiter, 50)):
        sol = solve_network(geom, model, state, T_iter, applied,
                            mode=mode, tol=cfg.solver.newton_tol, maxiter=cfg.solver.newton_max)
        q_vol = _heat_map(geom, model, state, sol, T_iter, active_ijk)
        T_new = solve_steady(op, q_vol).reshape(nx, ny, nz)
        if np.max(np.abs(T_new - T_iter)) < cfg.solver.coupling_tol:
            T_iter = T_new
            break
        T_iter = T_new
    q_step = float((q_vol * V).sum())
    removed = boundary_heat_removed(op, T_iter.ravel(order="C"))
    soc_field = np.full((nx, ny, nz), np.nan)
    soc_field[ii, jj, kk] = state.soc
    energy_balance = {"generated_W": q_step, "removed_W": removed,
                      "residual_W": q_step - removed,
                      "closure_rel": (q_step - removed) / (abs(q_step) + 1e-30)}
    return Result(
        t=np.array([0.0]), v_terminal=np.array([sol.v_terminal]),
        i_terminal=np.array([sol.i_terminal]), soc_mean=np.array([float(state.soc.mean())]),
        T_mean=np.array([float(T_iter.mean())]), T_max=np.array([float(T_iter.max())]),
        T_min=np.array([float(T_iter.min())]), T_field=T_iter[None, ...],
        q_total=np.array([q_step]), j_field_final=sol.j_area, soc_field_final=soc_field,
        energy_balance=energy_balance, geom=geom,
    )
