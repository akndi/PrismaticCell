"""3-D anisotropic finite-volume thermal solver (PHYSICS §5).

Discretizes the energy equation

    (rho cp) dT/dt = div(K grad T) + q'''            K = diag(k_x, k_y, k_z)

on the structured ``nx x ny x nz`` control-volume grid built by :mod:`geometry`.
Interior face conductances use the harmonic mean of the two neighbour cells'
directional conductivities (series thermal resistance). External faces carry
independent per-face boundary conditions (convection / dirichlet / neumann /
adiabatic) read from a :class:`~prismaticcell.config.Cooling`.

Cell ordering matches ``Grid.flat(i,j,k) = k + nz*(j + ny*i)`` (C-order on the
(nx,ny,nz) fields). No physical coefficient is hard-coded: everything comes from
``Geometry`` and ``Cooling``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .config import Cooling, FaceBC, SIGMA_SB
from .geometry import Geometry, REGION_ACTIVE, REGION_CAN


@dataclass
class ThermalOperator:
    """Assembled thermal system operator.

    ``A`` is the (N,N) conductance-plus-boundary matrix, ``b_bc`` the (N,) boundary
    source vector, and ``M`` the (N,) lumped capacitance ``rho_cp * V`` per cell.
    The remaining fields are boundary bookkeeping used by
    :func:`boundary_heat_removed` and are not part of the physics contract.
    """

    A: sp.csr_matrix
    b_bc: np.ndarray
    M: np.ndarray
    V: float = 0.0
    # Boundary bookkeeping (per cell) for the energy accounting:
    #   g_amb : sum of convection/dirichlet conductances to ambient  [W/K]
    #   gt_amb: sum of (conductance * t_inf)                          [W]
    #   q_flux_out: prescribed (neumann) outward power per cell       [W]
    g_amb: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gt_amb: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q_flux_out: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def assemble(cls, geom: Geometry, cooling: Cooling,
                 t_surf_field: "np.ndarray | None" = None) -> "ThermalOperator":
        """Assemble the sparse thermal operator.

        ``t_surf_field`` (nx,ny,nz), if given, is the current temperature estimate used to
        linearize radiation about the mean film temperature Tm=(Ts+T_inf)/2 (h_rad=4εσTm³),
        far more accurate than linearizing about T_inf at large ΔT (<0.6% error to ΔT~50 K,
        <1.4% to 80 K). When omitted, radiation linearizes about T_inf (exact at ΔT=0,
        conservative/over-predicts T).
        """
        grid = geom.grid
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        t_surf_flat = (np.ascontiguousarray(t_surf_field, dtype=np.float64).ravel()
                       if t_surf_field is not None else None)
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        N = nx * ny * nz

        kx = np.ascontiguousarray(geom.kx, dtype=np.float64)
        ky = np.ascontiguousarray(geom.ky, dtype=np.float64)
        kz = np.ascontiguousarray(geom.kz, dtype=np.float64)

        # Flat index of every cell, same C-order as Grid.flat.
        idx = np.arange(N, dtype=np.int64).reshape(nx, ny, nz)

        # Face areas normal to each direction and center-to-center distances.
        area_x, area_y, area_z = dy * dz, dx * dz, dx * dy

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        def _harmonic(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
            """Harmonic mean of two conductivity fields (0 where either is 0)."""
            s = k1 + k2
            out = np.zeros_like(s)
            m = s > 0.0
            out[m] = 2.0 * k1[m] * k2[m] / s[m]
            return out

        region = np.ascontiguousarray(geom.region)
        gc = float(getattr(geom, "contact_conductance", 0.0))
        # Sub-grid wall shell: the enclosure wall wraps the cavity boundary as a conductive skin.
        wall_t = float(getattr(geom, "wall_thickness", 0.0))
        wall_k = float(getattr(geom, "wall_k", 0.0))
        wall_rhocp = float(getattr(geom, "wall_rhocp", 0.0))
        shell = wall_t > 0.0 and wall_k > 0.0
        # Extra series resistance-area [K*m^2/W] from stack->wall contact and through-wall
        # conduction, added to every external-face BC when the shell is active.
        shell_R_area = 0.0
        if shell:
            shell_R_area = wall_t / wall_k + (1.0 / gc if gc > 0.0 else 0.0)

        def _add_interior(k, lo_idx, hi_idx, area, dist, reg_lo, reg_hi):
            kh = _harmonic(k[0], k[1])
            g = (kh * area / dist).ravel()
            # Series interfacial contact resistance at the can-wall interface (PHYSICS §5):
            #   1/G_face = 1/G_cond + 1/(contact_conductance * area)
            # Applied on every face separating the interior (stack or gap) from the can wall.
            if gc > 0.0:
                iface = ((reg_lo == REGION_CAN) ^ (reg_hi == REGION_CAN)).ravel()
                if np.any(iface):
                    gcond = g[iface]
                    gcontact = gc * area
                    g = g.copy()
                    g[iface] = 1.0 / (1.0 / np.maximum(gcond, 1e-300) + 1.0 / gcontact)
            p = lo_idx.ravel()
            q = hi_idx.ravel()
            rows.append(p); cols.append(p); vals.append(g)
            rows.append(q); cols.append(q); vals.append(g)
            rows.append(p); cols.append(q); vals.append(-g)
            rows.append(q); cols.append(p); vals.append(-g)

        # Interior faces in each direction (adjacent cells share a face).
        if nx > 1:
            _add_interior((kx[:-1, :, :], kx[1:, :, :]),
                          idx[:-1, :, :], idx[1:, :, :], area_x, dx,
                          region[:-1, :, :], region[1:, :, :])
        if ny > 1:
            _add_interior((ky[:, :-1, :], ky[:, 1:, :]),
                          idx[:, :-1, :], idx[:, 1:, :], area_y, dy,
                          region[:, :-1, :], region[:, 1:, :])
        if nz > 1:
            _add_interior((kz[:, :, :-1], kz[:, :, 1:]),
                          idx[:, :, :-1], idx[:, :, 1:], area_z, dz,
                          region[:, :, :-1], region[:, :, 1:])

        # --- External faces / boundary conditions ------------------------------ #
        b_bc = np.zeros(N)
        g_amb = np.zeros(N)
        gt_amb = np.zeros(N)
        q_flux_out = np.zeros(N)
        diag_bc = np.zeros(N)

        def _apply_face(bc: FaceBC, cell_idx: np.ndarray, k_norm: np.ndarray,
                        area: float, half_dist: float, extra_R_area: float = 0.0):
            """Apply one external face's BC to the given boundary cells.

            cell_idx     : (M,) flat indices of the boundary cells on this face
            k_norm       : (M,) directional conductivity normal to the face
            area         : per-cell face area [m^2]
            half_dist    : cell-center-to-face distance [m]
            extra_R_area : extra series resistance-area [K*m^2/W] on this face (e.g. a bottom/top
                           insulating slab), in series with the half-cell conduction + wall shell.
            """
            kind = bc.kind
            # Linearized radiative surface coefficient: q_rad = eps*sigma*(Ts^4 - Tinf^4)
            #   ~= h_rad*(Ts - Tinf). Linearize about the mean film temperature Tm=(Ts+Tinf)/2
            #   when a surface-T estimate is available (error <0.6% to ΔT~80 K), else about Tinf.
            if bc.emissivity > 0.0:
                if t_surf_flat is not None:
                    t_m = 0.5 * (t_surf_flat[cell_idx] + bc.t_inf)
                else:
                    t_m = bc.t_inf
                h_rad = 4.0 * bc.emissivity * SIGMA_SB * t_m**3
            else:
                h_rad = 0.0
            g_rad = h_rad * area

            # Through-wall + contact series resistance (per cell) when the shell is active, plus
            # any face-specific insulating layer (its resistance-area over the face area).
            r_series = ((half_dist / np.maximum(k_norm, 1e-300)) / area
                        + shell_R_area / area + extra_R_area / area)

            if kind == "dirichlet":
                # fixed temperature at the OUTER wall surface, through the shell + half-cell
                g_bc = np.asarray(1.0 / r_series, dtype=np.float64)
                diag_bc[cell_idx] += g_bc
                b_bc[cell_idx] += g_bc * bc.t_inf
                g_amb[cell_idx] += g_bc
                gt_amb[cell_idx] += g_bc * bc.t_inf
                return

            if kind == "neumann":
                # -k dT/dn = flux  (flux>0 leaves the cell). Heat INTO cell = -flux*area.
                q_out = bc.flux * area
                b_bc[cell_idx] += -q_out
                q_flux_out[cell_idx] += q_out

            # Surface-to-ambient conductance: convective film and radiation act in PARALLEL,
            # in SERIES with the half-cell conduction from the cell center to the surface.
            g_film = bc.h * area if (kind == "convection" and bc.h > 0.0) else 0.0
            g_surface = g_film + g_rad
            if np.isscalar(g_surface) and g_surface <= 0.0:
                return                                            # adiabatic / pure-flux, no ambient path
            # series: half-cell conduction (+ shell contact/through-wall) then surface exchange
            g_bc = 1.0 / (r_series + 1.0 / g_surface)
            g_bc = np.asarray(g_bc, dtype=np.float64)
            diag_bc[cell_idx] += g_bc
            b_bc[cell_idx] += g_bc * bc.t_inf
            g_amb[cell_idx] += g_bc
            gt_amb[cell_idx] += g_bc * bc.t_inf

        # Per-face extra series resistance from sub-grid layers (bottom/top insulator + roll wrap).
        six_faces = ("bottom", "top", "x_min", "x_max", "y_min", "y_max")
        face_R = dict(getattr(geom, "face_R_area", {}) or {})
        if not face_R:   # backward-compat: fall back to the insulator-only scalar
            _if = str(getattr(geom, "insulator_face", ""))
            if _if:
                face_R[_if] = float(getattr(geom, "insulator_R_area", 0.0))
        xR = {f: float(face_R.get(f, 0.0)) for f in six_faces}

        # top = +z max, bottom = -z min; sides map to x/y min/max.
        # z faces use kz, x faces use kx, y faces use ky.
        _apply_face(cooling.bottom, idx[:, :, 0].ravel(), kz[:, :, 0].ravel(),
                    area_z, dz / 2.0, xR["bottom"])
        _apply_face(cooling.top, idx[:, :, -1].ravel(), kz[:, :, -1].ravel(),
                    area_z, dz / 2.0, xR["top"])
        _apply_face(cooling.x_min, idx[0, :, :].ravel(), kx[0, :, :].ravel(),
                    area_x, dx / 2.0, xR["x_min"])
        _apply_face(cooling.x_max, idx[-1, :, :].ravel(), kx[-1, :, :].ravel(),
                    area_x, dx / 2.0, xR["x_max"])
        _apply_face(cooling.y_min, idx[:, 0, :].ravel(), ky[:, 0, :].ravel(),
                    area_y, dy / 2.0, xR["y_min"])
        _apply_face(cooling.y_max, idx[:, -1, :].ravel(), ky[:, -1, :].ravel(),
                    area_y, dy / 2.0, xR["y_max"])

        # --- Sub-grid wall shell: in-plane wall conduction (spreading + a metal path to the
        #     cooled faces) + wall thermal mass, on the cavity-boundary cells (PHYSICS §5). ---
        M_wall = np.zeros(N)
        if shell:
            gwt = wall_k * wall_t   # wall sheet conductance base [W/K] (× transverse/spacing)

            def _add_pair(p, q, g):
                rows.append(p); cols.append(p); vals.append(np.full(p.size, g))
                rows.append(q); cols.append(q); vals.append(np.full(q.size, g))
                rows.append(p); cols.append(q); vals.append(np.full(p.size, -g))
                rows.append(q); cols.append(p); vals.append(np.full(q.size, -g))

            def _shell_face(cells2d, g0, g1):
                if cells2d.shape[0] > 1:
                    _add_pair(cells2d[:-1, :].ravel(), cells2d[1:, :].ravel(), g0)
                if cells2d.shape[1] > 1:
                    _add_pair(cells2d[:, :-1].ravel(), cells2d[:, 1:].ravel(), g1)

            _shell_face(idx[:, :, 0],  gwt * dy / dx, gwt * dx / dy)   # bottom (z=min)
            _shell_face(idx[:, :, -1], gwt * dy / dx, gwt * dx / dy)   # top (z=max)
            _shell_face(idx[0, :, :],  gwt * dz / dy, gwt * dy / dz)   # x_min
            _shell_face(idx[-1, :, :], gwt * dz / dy, gwt * dy / dz)   # x_max
            _shell_face(idx[:, 0, :],  gwt * dz / dx, gwt * dx / dz)   # y_min
            _shell_face(idx[:, -1, :], gwt * dz / dx, gwt * dx / dz)   # y_max

            mw = wall_rhocp * wall_t
            Mw3 = M_wall.reshape(nx, ny, nz)
            Mw3[:, :, 0] += mw * area_z; Mw3[:, :, -1] += mw * area_z
            Mw3[0, :, :] += mw * area_x; Mw3[-1, :, :] += mw * area_x
            Mw3[:, 0, :] += mw * area_y; Mw3[:, -1, :] += mw * area_y

        # Sub-grid layer thermal mass (insulator + roll wrap): lump each layer's areal heat
        # capacity onto its face cells.
        face_M = dict(getattr(geom, "face_rhocp_t", {}) or {})
        if not face_M:   # backward-compat: insulator-only scalar
            _if = str(getattr(geom, "insulator_face", ""))
            if _if:
                face_M[_if] = float(getattr(geom, "insulator_rhocp_t", 0.0))
        if face_M:
            Mi3 = M_wall.reshape(nx, ny, nz)
            _face_slice = {
                "bottom": (np.s_[:, :, 0], area_z), "top": (np.s_[:, :, -1], area_z),
                "x_min": (np.s_[0, :, :], area_x), "x_max": (np.s_[-1, :, :], area_x),
                "y_min": (np.s_[:, 0, :], area_y), "y_max": (np.s_[:, -1, :], area_y),
            }
            for f, mt in face_M.items():
                if f in _face_slice and mt > 0.0:
                    sl, af = _face_slice[f]
                    Mi3[sl] += mt * af

        # Tab conduction-to-ambient heat loss (PHYSICS §5): the tab's far end is heat-sunk near
        # the coolant temperature. Distribute each polarity's tab thermal conductance over its
        # attachment control volumes (the tab-node columns, across their active z-cells). Tab
        # sink temperature = mean of the non-adiabatic face temperatures (config-driven).
        tinfs = [f.t_inf for f in (cooling.top, cooling.bottom, cooling.x_min,
                                   cooling.x_max, cooling.y_min, cooling.y_max)
                 if f.kind in ("convection", "dirichlet")]
        t_tab = float(np.mean(tinfs)) if tinfs else float(cooling.top.t_inf)

        def _apply_tab_thermal(node_lists, g_total):
            if g_total <= 0.0:
                return
            cells = []
            for roll, nodes in node_lists:
                for (a, b) in nodes:                       # electrode (length,height) columns
                    for s in roll.col_zcells.get((a, b), []):
                        cells.append(idx[geom.phys_index(a, b, s)])
            if not cells:
                return
            g_each = g_total / len(cells)
            for f in cells:
                diag_bc[f] += g_each
                b_bc[f] += g_each * t_tab
                g_amb[f] += g_each
                gt_amb[f] += g_each * t_tab

        _apply_tab_thermal([(r, r.tab_pos_nodes) for r in geom.rolls],
                           float(getattr(geom, "tab_heat_cond_pos", 0.0)))
        _apply_tab_thermal([(r, r.tab_neg_nodes) for r in geom.rolls],
                           float(getattr(geom, "tab_heat_cond_neg", 0.0)))

        # Boundary conductances add to the diagonal.
        if np.any(diag_bc):
            nz_bc = np.nonzero(diag_bc)[0]
            rows.append(nz_bc); cols.append(nz_bc); vals.append(diag_bc[nz_bc])

        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(N, N),
        ).tocsr()

        V = grid.volume()
        M = np.ascontiguousarray(geom.rho_cp, dtype=np.float64).ravel() * V + M_wall

        return cls(A=A, b_bc=b_bc, M=M, V=V,
                   g_amb=g_amb, gt_amb=gt_amb, q_flux_out=q_flux_out)


def _q_source(op: ThermalOperator, q_vol: np.ndarray) -> np.ndarray:
    """Return the (N,) generation source ``q_vol * V``, accepting 3-D or flat input."""
    q = np.asarray(q_vol, dtype=np.float64)
    if q.ndim > 1:
        q = q.ravel(order="C")
    N = op.b_bc.shape[0]
    if q.shape[0] != N:
        raise ValueError(f"q_vol has {q.shape[0]} entries, expected {N}")
    return q * op.V


def solve_steady(op: ThermalOperator, q_vol: np.ndarray) -> np.ndarray:
    """Solve the steady conduction system ``A T = b_bc + q_vol * V``."""
    rhs = op.b_bc + _q_source(op, q_vol)
    T = spla.spsolve(op.A.tocsc(), rhs)
    return np.asarray(T, dtype=np.float64)


def step_transient(op: ThermalOperator, T_prev: np.ndarray, q_vol: np.ndarray,
                   dt: float, linear_solver: str = "direct") -> np.ndarray:
    """Advance one backward-Euler step (PHYSICS §5.3).

    Solves ``(M/dt + A) T = (M/dt) T_prev + b_bc + q_vol*V``.
    ``linear_solver`` is ``"direct"`` (spsolve) or ``"cg"`` (Jacobi-preconditioned CG).
    """
    T_prev = np.asarray(T_prev, dtype=np.float64).ravel()
    mdt = op.M / dt
    Asys = (op.A + sp.diags(mdt)).tocsr()
    rhs = mdt * T_prev + op.b_bc + _q_source(op, q_vol)

    if linear_solver == "direct":
        T = spla.spsolve(Asys.tocsc(), rhs)
    elif linear_solver == "cg":
        diag = Asys.diagonal()
        diag = np.where(diag != 0.0, diag, 1.0)
        precond = sp.diags(1.0 / diag)            # Jacobi / diagonal preconditioner
        cg_kwargs = dict(M=precond, x0=T_prev, maxiter=10 * Asys.shape[0])
        try:
            T, info = spla.cg(Asys, rhs, rtol=1e-12, atol=0.0, **cg_kwargs)
        except TypeError:  # older scipy uses `tol` instead of `rtol`
            T, info = spla.cg(Asys, rhs, tol=1e-12, atol=0.0, **cg_kwargs)
        if info != 0:
            # Fall back to a robust direct solve rather than return a bad answer.
            T = spla.spsolve(Asys.tocsc(), rhs)
    else:
        raise ValueError(f"unknown linear_solver '{linear_solver}'")
    return np.asarray(T, dtype=np.float64)


def boundary_heat_removed(op: ThermalOperator, T: np.ndarray) -> float:
    """Net power [W] leaving through all external faces (>0 = heat leaving cell).

    Convection/dirichlet faces remove ``g_bc*(T_cell - t_inf)``; neumann faces
    remove the prescribed ``flux*area``. Summed over every external face.
    """
    T = np.asarray(T, dtype=np.float64).ravel()
    conv = float(op.g_amb @ T - op.gt_amb.sum())
    flux = float(op.q_flux_out.sum())
    return conv + flux
