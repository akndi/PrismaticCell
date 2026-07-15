"""Current-collector potential network — distributes current across each jellyroll.

Implements PHYSICS.md §3 (and feeds §4 foil ohmic heat). This is the 2.5-D electro
core: per jellyroll there are two in-plane potential fields, ``phi_pos`` on the Al
foil and ``phi_neg`` on the Cu foil. The many through-thickness sandwiches of a roll
share the two collector fields; each active 3-D control volume keeps its own ECM
state (soc, RC overpotentials, local T) and contributes its own current from the
shared column ``Δφ = φ⁺ − φ⁻``. Tab placement and electrode/foil thickness therefore
shape the current and heat maps — the whole point of the module.

Model (see PHYSICS §3, INTERFACES ``distributed.py``)
----------------------------------------------------
For each roll, discretize the 2-D sheet Laplacian ``∇·(G ∇φ)`` on the roll's active
in-plane columns with 4-neighbour coupling. The in-plane foil conductance between two
adjacent active columns is::

    G_edge = sheet_cond * (transverse_length / spacing)
           = sheet_cond * (dy/dx)   for x-neighbours
           = sheet_cond * (dx/dy)   for y-neighbours

(``sheet_cond`` = ``RollElectro.sheet_cond_pos`` / ``sheet_cond_neg`` [S]; only columns
active in *both* endpoints are connected.)

Within an in-plane column ``(i,j)`` the active z-cells share ``Δφ_ij = φ⁺_ij − φ⁻_ij``.
Each active CV ``k`` gives an areal current density (echem local closure)::

    j_k = (ocv_v(soc_k) − Σ_p u_{k,p} − Δφ_ij) / R0_area(soc_k, T_k)

and CV current ``i_k = j_k * area_eff[i,j,k]``. Because ``R0`` depends only on state
(fixed within a step) and *not* on ``Δφ``, ``j_k`` is affine in ``Δφ_ij``, so the
column behaves like a local conductance and current source::

    g_ij   = Σ_k area_eff_k / R0_k
    Isrc_ij= Σ_k area_eff_k * (ocv_k − Σu_k) / R0_k
    I_col  = Σ_k i_k = Isrc_ij − g_ij * Δφ_ij

``I_col`` is injected into the + foil node and extracted from the − foil node (source
into ``φ⁺``, sink out of ``φ⁻``), coupling the two foil Laplacians.

Because the closure is affine in the unknown potentials, the whole coupled system
(two foil Laplacians + column source/sink coupling + tab terminals + terminal
constraint) is **linear** — ONE sparse solve is exact. No Newton iteration on
``R(state)`` is needed within a step; the ``tol``/``maxiter`` arguments are honoured
for API compatibility (the single solve trivially satisfies ``tol``).

Terminals / tabs
----------------
Both jellyrolls connect to the SAME two terminals (in parallel). The − terminal is the
potential reference (grounded, ``φ⁻_term = 0``). Each tab is modelled as a *strong
conductance* ``G_tab`` from its foil column node(s) to the terminal node — chosen as a
large multiple (``TAB_CONDUCTANCE_FACTOR``) of the roll's foil sheet-conductance scale
so the tab footprint is effectively equipotential with the terminal, while keeping the
matrix non-singular and well-conditioned (unlike a hard Dirichlet pin, this also lets
several tab columns share a terminal cleanly and yields the terminal current directly).

- ``mode="current"`` (galvanostatic): the + terminal potential ``v_terminal`` is an
  unknown; extra row enforces ``Σ_(all + tab branches) G_tab (Vp − φ⁺) = applied [A]``.
  Summing the foil balances shows this equals ``Σ_k i_k`` — i.e. total tab current =
  total stack current = ``applied`` (charge conservation, asserted in the self-test).
- ``mode="voltage"`` (potentiostatic): ``Vp − Vn = applied`` with ``Vn = 0`` so
  ``Vp = applied`` is fixed; total current is an output.

Ohmic-heat mapping (feeds PHYSICS §4)
-------------------------------------
The in-plane foil Joule heat of an edge is ``P_edge = G_edge (φ_a − φ_b)²``. Each edge's
power is split equally to its two endpoint columns and summed over both foils, giving a
per-column foil heat ``P_col``. ``P_col`` is distributed to the column's active z-cells in
proportion to their ``area_eff`` (uniform per column here), then divided by the CV thermal
volume ``grid.volume()`` to yield ``q_ohm_vol`` [W/m³]. (Tab/contact Joule heating is a
separate region contribution per PHYSICS §4 and is not included here.)

State-array ordering
--------------------
``ECMState`` flat arrays (``soc``, ``rc_u``) are indexed by the C-order enumeration of
``geom.active_mask`` — equivalently ``geom.grid.flat(i,j,k)`` restricted to active cells.
This is the natural/only consistent mapping given the documented interfaces; see
``_active_index_field``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .geometry import Geometry, RollElectro

# Fallback tab conductance = TAB_CONDUCTANCE_FACTOR * (roll foil sheet-conductance scale [S]),
# used only if no physical tab conductance is defined (sigma or geometry missing). Otherwise the
# physical tab conductance G = sigma*w*t/L (geom.g_tab_pos/neg) is distributed over the tab nodes,
# so tab material/size/length produce a real terminal IR drop (tab-placement design lever).
TAB_CONDUCTANCE_FACTOR = 1.0e4


def _tab_conductances(geom, n_pos_nodes, n_neg_nodes):
    """Per-tab-node conductance to the terminal for + and - polarity.

    Distributes the physical tab conductance (geom.g_tab_pos/neg [S]) over its nodes; falls
    back to a strong near-equipotential conductance if no physical value is available.
    """
    sheet_scale = max((max(r.sheet_cond_pos, r.sheet_cond_neg) for r in geom.rolls), default=1.0)
    fallback = TAB_CONDUCTANCE_FACTOR * sheet_scale
    g_pos = float(getattr(geom, "g_tab_pos", 0.0) or 0.0)
    g_neg = float(getattr(geom, "g_tab_neg", 0.0) or 0.0)
    G_pos = (g_pos / n_pos_nodes) if (g_pos > 0.0 and n_pos_nodes > 0) else fallback
    G_neg = (g_neg / n_neg_nodes) if (g_neg > 0.0 and n_neg_nodes > 0) else fallback
    return G_pos, G_neg


@dataclass
class NetworkSolution:
    j_area: np.ndarray                # (nx,ny,nz) local areal current density [A/m^2], 0 if inactive
    i_cv: np.ndarray                  # (nx,ny,nz) current per CV [A] (= j_area*area_eff)
    phi_pos: Dict[int, np.ndarray]    # roll_index -> (nx,ny) + foil potential (NaN at inactive columns)
    phi_neg: Dict[int, np.ndarray]    # roll_index -> (nx,ny) - foil potential
    v_terminal: float                 # cell terminal voltage [V] (Vp - Vn, Vn grounded to 0)
    i_terminal: float                 # total cell current [A] (== applied for galvanostatic)
    q_ohm_vol: np.ndarray             # (nx,ny,nz) foil ohmic heat density [W/m^3]
    dphi_field: np.ndarray = None     # (nx,ny,nz) local stack Δφ = φ⁺-φ⁻ per active CV [V]
                                      # (per-column in "planar", per-CV in "layered")


def _active_index_field(geom: Geometry) -> np.ndarray:
    """(nx,ny,nz) int field: flat active-CV index for active cells, -1 elsewhere.

    Active index = C-order enumeration of ``active_mask`` (matches ``grid.flat`` order
    restricted to active cells), i.e. the assumed ordering of ``ECMState`` arrays.
    """
    nx, ny, nz = geom.grid.nx, geom.grid.ny, geom.grid.nz
    idx = np.full(nx * ny * nz, -1, dtype=np.int64)
    active_flat = np.flatnonzero(geom.active_mask.ravel(order="C"))
    idx[active_flat] = np.arange(active_flat.size, dtype=np.int64)
    return idx.reshape(nx, ny, nz)


def solve_network(
    geom: Geometry,
    model,
    state,
    T_field: np.ndarray,
    applied: float,
    *,
    mode: str = "current",
    tol: float = 1e-9,
    maxiter: int = 50,
    collector_model: str = "planar",
) -> NetworkSolution:
    """Solve the current-collector potential network (PHYSICS §3).

    Parameters
    ----------
    geom : Geometry
    model : ECMModel      (needs ``ocv_v(soc)`` and ``r0_area(soc, T)``)
    state : ECMState      (``soc`` (n_active,), ``rc_u`` (n_rc, n_active))
    T_field : (nx,ny,nz) temperature field [K]; active CVs read their local T.
    applied : current [A] if ``mode == "current"`` else terminal voltage [V].
    mode : "current" (galvanostatic) or "voltage" (potentiostatic).
    collector_model : "planar" (one shared 2-D foil potential per jellyroll; 2.5-D) or
        "layered" (a separate 2-D foil potential per through-thickness stack layer, all in
        parallel at the tabs -> full 3-D collector resolving through-thickness gradients).
    tol, maxiter : API compatibility; the affine system is solved exactly in one solve.
    """
    if mode not in ("current", "voltage"):
        raise ValueError(f"mode must be 'current' or 'voltage', got {mode!r}")
    if collector_model not in ("planar", "layered"):
        raise ValueError(f"collector_model must be 'planar' or 'layered', got {collector_model!r}")

    grid = geom.grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    T_field = np.asarray(T_field, dtype=np.float64)

    # ---- degeneracy guards (prevent silently-singular systems) -----------------
    if not any(roll.columns for roll in geom.rolls):
        raise ValueError(
            "No active control volumes in the mesh: the collector network is empty. "
            "Refine the mesh so at least one jellyroll region is resolved (the documented "
            "1x1 lumped limit needs a single-roll config or a mesh that resolves a roll)."
        )
    n_pos_nodes = sum(len(roll.tab_pos_nodes) for roll in geom.rolls)
    n_neg_nodes = sum(len(roll.tab_neg_nodes) for roll in geom.rolls)
    if n_pos_nodes == 0 or n_neg_nodes == 0:
        raise ValueError(
            "Network is not terminated: need at least one positive and one negative tab "
            f"node (got {n_pos_nodes} pos, {n_neg_nodes} neg). Check tab edges/positions."
        )

    if collector_model == "layered":
        return _solve_layered(geom, model, state, T_field, applied, mode=mode)

    # ---- per-active-CV state gathered onto (nx,ny,nz) fields --------------------
    aidx = _active_index_field(geom)                 # (nx,ny,nz) active index / -1
    soc = np.asarray(state.soc, dtype=np.float64)
    n_active = soc.size
    rc_u = np.asarray(state.rc_u, dtype=np.float64)
    usum = rc_u.sum(axis=0) if rc_u.size else np.zeros(n_active)

    T_active = T_field.ravel(order="C")[np.flatnonzero(geom.active_mask.ravel(order="C"))]
    ocv_active = np.asarray(model.ocv_v(soc), dtype=np.float64).reshape(-1)
    r0_active = np.asarray(model.r0_area(soc, T_active), dtype=np.float64).reshape(-1)
    if np.any(r0_active <= 0.0):
        raise ValueError("r0_area returned a non-positive resistance; check the R0 table/state")

    # per-CV coefficients of the affine closure:  i_k = a_k - b_k * Δφ  (b_k = area/R0)
    # source-part a_k = area * (ocv - Σu) / R0 ; conductance-part b_k = area / R0.
    # (area_eff differs per roll; fetched below when we walk each roll's columns.)

    # ---- global DOF layout -----------------------------------------------------
    # unknowns: phi_pos per (roll, column), phi_neg per (roll, column), [+ Vp if current]
    pos_dof: Dict[Tuple[int, Tuple[int, int]], int] = {}
    neg_dof: Dict[Tuple[int, Tuple[int, int]], int] = {}
    ndof = 0
    for roll in geom.rolls:
        for c in roll.columns:
            pos_dof[(roll.roll_index, c)] = ndof
            ndof += 1
    for roll in geom.rolls:
        for c in roll.columns:
            neg_dof[(roll.roll_index, c)] = ndof
            ndof += 1

    current_mode = mode == "current"
    if current_mode:
        vp_dof = ndof
        ndof += 1
        Vp_fixed = None
    else:
        vp_dof = None
        Vp_fixed = float(applied)   # Vp - Vn = applied, Vn = 0
    Vn = 0.0  # negative terminal is the grounded reference

    # physical tab conductance distributed over the tab nodes (real terminal IR drop)
    n_pos_nodes = sum(len(r.tab_pos_nodes) for r in geom.rolls)
    n_neg_nodes = sum(len(r.tab_neg_nodes) for r in geom.rolls)
    G_pos, G_neg = _tab_conductances(geom, n_pos_nodes, n_neg_nodes)

    rows: list = []
    cols: list = []
    vals: list = []
    rhs = np.zeros(ndof)

    def add(r_, c_, v_):
        rows.append(r_)
        cols.append(c_)
        vals.append(v_)

    # per-column stored quantities for later ohmic-heat / current reconstruction
    col_g: Dict[Tuple[int, Tuple[int, int]], float] = {}
    col_isrc: Dict[Tuple[int, Tuple[int, int]], float] = {}

    # ---- assemble foil Laplacians + column source/sink coupling ----------------
    for roll in geom.rolls:
        r = roll.roll_index
        colset = set(roll.columns)
        sc_p = roll.sheet_cond_pos
        sc_n = roll.sheet_cond_neg
        Gx_p, Gy_p = sc_p * (dy / dx), sc_p * (dx / dy)
        Gx_n, Gy_n = sc_n * (dy / dx), sc_n * (dx / dy)

        # in-plane 4-neighbour coupling (add each undirected edge once via +x,+y)
        for (i, j) in roll.columns:
            p = pos_dof[(r, (i, j))]
            q = neg_dof[(r, (i, j))]
            for (ni, nj), Gp, Gn in (
                ((i + 1, j), Gx_p, Gx_n),
                ((i, j + 1), Gy_p, Gy_n),
            ):
                if (ni, nj) in colset:
                    p2 = pos_dof[(r, (ni, nj))]
                    q2 = neg_dof[(r, (ni, nj))]
                    # positive foil edge
                    add(p, p, Gp); add(p2, p2, Gp); add(p, p2, -Gp); add(p2, p, -Gp)
                    # negative foil edge
                    add(q, q, Gn); add(q2, q2, Gn); add(q, q2, -Gn); add(q2, q, -Gn)

        # column source/sink coupling the two foils
        for (i, j) in roll.columns:
            p = pos_dof[(r, (i, j))]
            q = neg_dof[(r, (i, j))]
            g_ij = 0.0
            isrc_ij = 0.0
            for k in roll.col_zcells[(i, j)]:
                a = aidx[i, j, k]
                area = roll.area_eff[i, j, k]
                b = area / r0_active[a]                      # conductance part
                g_ij += b
                isrc_ij += b * (ocv_active[a] - usum[a])     # source part
            col_g[(r, (i, j))] = g_ij
            col_isrc[(r, (i, j))] = isrc_ij
            # eq (A) phi_pos node:  ... + g(φ⁺ − φ⁻) = Isrc
            add(p, p, g_ij); add(p, q, -g_ij); rhs[p] += isrc_ij
            # eq (B) phi_neg node:  ... − g(φ⁺ − φ⁻) = −Isrc
            add(q, q, g_ij); add(q, p, -g_ij); rhs[q] += -isrc_ij

        # tabs -> terminal via the physical tab conductance (per node)
        for c in roll.tab_pos_nodes:
            p = pos_dof[(r, c)]
            add(p, p, G_pos)
            if current_mode:
                add(p, vp_dof, -G_pos)
                add(vp_dof, p, -G_pos)
                add(vp_dof, vp_dof, G_pos)
            else:
                rhs[p] += G_pos * Vp_fixed          # Vp known
        for c in roll.tab_neg_nodes:
            q = neg_dof[(r, c)]
            add(q, q, G_neg)
            rhs[q] += G_neg * Vn                     # Vn = 0 (grounded)

    # terminal constraint row (galvanostatic): the applied current is EXTRACTED at the +
    # terminal on discharge, i.e. Σ_(+ tabs) G_tab (φ⁺ − Vp) = applied. With the assembled
    # (symmetric) coupling Σ G_tab (Vp − φ⁺) on the LHS, this is rhs = −applied. Summing the
    # φ⁺ balances then gives Σ_k i_cv = applied (charge conservation, positive => discharge).
    if current_mode:
        rhs[vp_dof] += -float(applied)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))
    x = spsolve(A.tocsc(), rhs)
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))

    if current_mode:
        Vp = float(x[vp_dof])
    else:
        Vp = Vp_fixed
    v_terminal = Vp - Vn

    # ---- reconstruct potentials, currents, and ohmic heat ----------------------
    phi_pos: Dict[int, np.ndarray] = {}
    phi_neg: Dict[int, np.ndarray] = {}
    j_area = np.zeros((nx, ny, nz))
    i_cv = np.zeros((nx, ny, nz))
    q_ohm_vol = np.zeros((nx, ny, nz))
    dphi_field = np.zeros((nx, ny, nz))
    vol = grid.volume()

    for roll in geom.rolls:
        r = roll.roll_index
        pp = np.full((nx, ny), np.nan)
        pn = np.full((nx, ny), np.nan)
        for c in roll.columns:
            pp[c] = x[pos_dof[(r, c)]]
            pn[c] = x[neg_dof[(r, c)]]
        phi_pos[r] = pp
        phi_neg[r] = pn

        # local current densities from shared column Δφ
        for (i, j) in roll.columns:
            dphi = pp[i, j] - pn[i, j]
            for k in roll.col_zcells[(i, j)]:
                a = aidx[i, j, k]
                j_local = (ocv_active[a] - usum[a] - dphi) / r0_active[a]
                j_area[i, j, k] = j_local
                i_cv[i, j, k] = j_local * roll.area_eff[i, j, k]
                dphi_field[i, j, k] = dphi

        # foil ohmic heat: split each edge's power to its two endpoint columns,
        # accumulate per column over both foils.
        p_col = np.zeros((nx, ny))
        colset = set(roll.columns)
        sc_p, sc_n = roll.sheet_cond_pos, roll.sheet_cond_neg
        Gx_p, Gy_p = sc_p * (dy / dx), sc_p * (dx / dy)
        Gx_n, Gy_n = sc_n * (dy / dx), sc_n * (dx / dy)
        for (i, j) in roll.columns:
            for (ni, nj), Gp, Gn in (
                ((i + 1, j), Gx_p, Gx_n),
                ((i, j + 1), Gy_p, Gy_n),
            ):
                if (ni, nj) in colset:
                    dpp = pp[i, j] - pp[ni, nj]
                    dpn = pn[i, j] - pn[ni, nj]
                    p_edge = Gp * dpp * dpp + Gn * dpn * dpn
                    p_col[i, j] += 0.5 * p_edge
                    p_col[ni, nj] += 0.5 * p_edge
        # distribute each column's foil heat to its active z-cells (area-weighted),
        # then convert to volumetric density.
        for (i, j) in roll.columns:
            if p_col[i, j] == 0.0:
                continue
            ks = roll.col_zcells[(i, j)]
            area_tot = sum(roll.area_eff[i, j, k] for k in ks)
            for k in ks:
                w = (roll.area_eff[i, j, k] / area_tot) if area_tot > 0 else (1.0 / len(ks))
                q_ohm_vol[i, j, k] += p_col[i, j] * w / vol

    i_terminal = float(i_cv.sum())

    return NetworkSolution(
        j_area=j_area,
        i_cv=i_cv,
        phi_pos=phi_pos,
        phi_neg=phi_neg,
        v_terminal=float(v_terminal),
        i_terminal=i_terminal,
        q_ohm_vol=q_ohm_vol,
        dphi_field=dphi_field,
    )


def _solve_layered(geom, model, state, T_field, applied, *, mode="current"):
    """Full 3-D collector solve (PHYSICS §3, layered variant).

    Instead of one shared 2-D foil potential per jellyroll, each through-thickness stack layer
    ``k`` gets its OWN pair of 2-D foil potentials ``φ⁺[r,k]``, ``φ⁻[r,k]``. Layers carry a
    fraction ``1/n_layers`` of the roll's foil sheet conductance and are electrically independent
    except that every layer's tab nodes connect in PARALLEL to the shared cell terminals. This
    resolves through-thickness potential/current gradients: a hotter (lower-R0) inner layer draws
    more current and develops its own foil IR profile. Reduces to ``planar`` when ``nz`` places a
    single active layer in the roll.
    """
    grid = geom.grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    vol = grid.volume()
    T_field = np.asarray(T_field, dtype=np.float64)

    aidx = _active_index_field(geom)
    soc = np.asarray(state.soc, dtype=np.float64)
    n_active = soc.size
    rc_u = np.asarray(state.rc_u, dtype=np.float64)
    usum = rc_u.sum(axis=0) if rc_u.size else np.zeros(n_active)
    T_active = T_field.ravel(order="C")[np.flatnonzero(geom.active_mask.ravel(order="C"))]
    ocv_active = np.asarray(model.ocv_v(soc), dtype=np.float64).reshape(-1)
    r0_active = np.asarray(model.r0_area(soc, T_active), dtype=np.float64).reshape(-1)
    if np.any(r0_active <= 0.0):
        raise ValueError("r0_area returned a non-positive resistance; check the R0 table/state")

    current_mode = mode == "current"
    Vn = 0.0

    # active z-layers per roll (roll bbox is uniform in z, so shared across columns)
    roll_layers = {roll.roll_index: sorted({k for ks in roll.col_zcells.values() for k in ks})
                   for roll in geom.rolls}

    def is_active(roll, k, c):
        return (c in roll.col_zcells) and (k in roll.col_zcells[c])

    # ---- DOF layout: φ⁺ then φ⁻ per (roll, layer, column), then Vp -------------
    pos_dof: Dict[Tuple[int, int, Tuple[int, int]], int] = {}
    neg_dof: Dict[Tuple[int, int, Tuple[int, int]], int] = {}
    ndof = 0
    for roll in geom.rolls:
        r = roll.roll_index
        for k in roll_layers[r]:
            for c in roll.columns:
                if is_active(roll, k, c):
                    pos_dof[(r, k, c)] = ndof
                    ndof += 1
    for roll in geom.rolls:
        r = roll.roll_index
        for k in roll_layers[r]:
            for c in roll.columns:
                if is_active(roll, k, c):
                    neg_dof[(r, k, c)] = ndof
                    ndof += 1
    if current_mode:
        vp_dof = ndof
        ndof += 1
        Vp_fixed = None
    else:
        vp_dof = None
        Vp_fixed = float(applied)

    # physical tab conductance distributed over ALL (layer, node) attachments so the total
    # terminal tab conductance equals the planar case (no n_layers double-counting)
    n_pos_nodes = sum(len(roll.tab_pos_nodes) * len(roll_layers[roll.roll_index])
                      for roll in geom.rolls)
    n_neg_nodes = sum(len(roll.tab_neg_nodes) * len(roll_layers[roll.roll_index])
                      for roll in geom.rolls)
    G_pos, G_neg = _tab_conductances(geom, n_pos_nodes, n_neg_nodes)

    rows: list = []
    cols: list = []
    vals: list = []
    rhs = np.zeros(ndof)

    def add(a, b, v):
        rows.append(a); cols.append(b); vals.append(v)

    # ---- assemble per-layer foil Laplacians + column coupling + tabs -----------
    for roll in geom.rolls:
        r = roll.roll_index
        nlz = max(len(roll_layers[r]), 1)
        sc_p = roll.sheet_cond_pos / nlz          # per-layer foil sheet conductance
        sc_n = roll.sheet_cond_neg / nlz
        Gx_p, Gy_p = sc_p * (dy / dx), sc_p * (dx / dy)
        Gx_n, Gy_n = sc_n * (dy / dx), sc_n * (dx / dy)
        for k in roll_layers[r]:
            # in-plane 4-neighbour foil coupling within this layer
            for (i, j) in roll.columns:
                if not is_active(roll, k, (i, j)):
                    continue
                p = pos_dof[(r, k, (i, j))]
                q = neg_dof[(r, k, (i, j))]
                for (ni, nj), Gp, Gn in (((i + 1, j), Gx_p, Gx_n), ((i, j + 1), Gy_p, Gy_n)):
                    if is_active(roll, k, (ni, nj)):
                        p2 = pos_dof[(r, k, (ni, nj))]
                        q2 = neg_dof[(r, k, (ni, nj))]
                        add(p, p, Gp); add(p2, p2, Gp); add(p, p2, -Gp); add(p2, p, -Gp)
                        add(q, q, Gn); add(q2, q2, Gn); add(q, q2, -Gn); add(q2, q, -Gn)
                # column source/sink: a single CV per (layer, column)
                a = aidx[i, j, k]
                b = roll.area_eff[i, j, k] / r0_active[a]
                isrc = b * (ocv_active[a] - usum[a])
                add(p, p, b); add(p, q, -b); rhs[p] += isrc
                add(q, q, b); add(q, p, -b); rhs[q] += -isrc
            # tabs: this layer's tab nodes join the SHARED terminals (parallel across layers)
            for c in roll.tab_pos_nodes:
                if not is_active(roll, k, c):
                    continue
                p = pos_dof[(r, k, c)]
                add(p, p, G_pos)
                if current_mode:
                    add(p, vp_dof, -G_pos); add(vp_dof, p, -G_pos); add(vp_dof, vp_dof, G_pos)
                else:
                    rhs[p] += G_pos * Vp_fixed
            for c in roll.tab_neg_nodes:
                if not is_active(roll, k, c):
                    continue
                q = neg_dof[(r, k, c)]
                add(q, q, G_neg)
                rhs[q] += G_neg * Vn

    if current_mode:
        rhs[vp_dof] += -float(applied)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))
    x = np.atleast_1d(np.asarray(spsolve(A.tocsc(), rhs), dtype=np.float64))
    Vp = float(x[vp_dof]) if current_mode else Vp_fixed
    v_terminal = Vp - Vn

    # ---- reconstruct fields ----------------------------------------------------
    phi_pos: Dict[int, np.ndarray] = {}
    phi_neg: Dict[int, np.ndarray] = {}
    j_area = np.zeros((nx, ny, nz))
    i_cv = np.zeros((nx, ny, nz))
    q_ohm_vol = np.zeros((nx, ny, nz))
    dphi_field = np.zeros((nx, ny, nz))

    for roll in geom.rolls:
        r = roll.roll_index
        nlz = max(len(roll_layers[r]), 1)
        sc_p = roll.sheet_cond_pos / nlz
        sc_n = roll.sheet_cond_neg / nlz
        Gx_p, Gy_p = sc_p * (dy / dx), sc_p * (dx / dy)
        Gx_n, Gy_n = sc_n * (dy / dx), sc_n * (dx / dy)
        pp_sum = np.zeros((nx, ny)); pn_sum = np.zeros((nx, ny)); cnt = np.zeros((nx, ny))
        for k in roll_layers[r]:
            for (i, j) in roll.columns:
                if not is_active(roll, k, (i, j)):
                    continue
                pv = x[pos_dof[(r, k, (i, j))]]
                nv = x[neg_dof[(r, k, (i, j))]]
                dphi = pv - nv
                a = aidx[i, j, k]
                jl = (ocv_active[a] - usum[a] - dphi) / r0_active[a]
                j_area[i, j, k] = jl
                i_cv[i, j, k] = jl * roll.area_eff[i, j, k]
                dphi_field[i, j, k] = dphi
                pp_sum[i, j] += pv; pn_sum[i, j] += nv; cnt[i, j] += 1.0
                # per-layer foil ohmic heat, split to endpoint CVs of this layer
                for (ni, nj), Gp, Gn in (((i + 1, j), Gx_p, Gx_n), ((i, j + 1), Gy_p, Gy_n)):
                    if is_active(roll, k, (ni, nj)):
                        dpp = pv - x[pos_dof[(r, k, (ni, nj))]]
                        dpn = nv - x[neg_dof[(r, k, (ni, nj))]]
                        pe = Gp * dpp * dpp + Gn * dpn * dpn
                        q_ohm_vol[i, j, k] += 0.5 * pe / vol
                        q_ohm_vol[ni, nj, k] += 0.5 * pe / vol
        pp = np.full((nx, ny), np.nan); pn = np.full((nx, ny), np.nan)
        m = cnt > 0
        pp[m] = pp_sum[m] / cnt[m]                  # thickness-averaged (for plotting)
        pn[m] = pn_sum[m] / cnt[m]
        phi_pos[r] = pp
        phi_neg[r] = pn

    return NetworkSolution(
        j_area=j_area, i_cv=i_cv, phi_pos=phi_pos, phi_neg=phi_neg,
        v_terminal=float(v_terminal), i_terminal=float(i_cv.sum()),
        q_ohm_vol=q_ohm_vol, dphi_field=dphi_field,
    )
