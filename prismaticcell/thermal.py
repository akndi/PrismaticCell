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
from .geometry import Geometry


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
    def assemble(cls, geom: Geometry, cooling: Cooling) -> "ThermalOperator":
        grid = geom.grid
        nx, ny, nz = grid.nx, grid.ny, grid.nz
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

        def _add_interior(k, lo_idx, hi_idx, area, dist):
            kh = _harmonic(k[0], k[1])
            g = (kh * area / dist).ravel()
            p = lo_idx.ravel()
            q = hi_idx.ravel()
            rows.append(p); cols.append(p); vals.append(g)
            rows.append(q); cols.append(q); vals.append(g)
            rows.append(p); cols.append(q); vals.append(-g)
            rows.append(q); cols.append(p); vals.append(-g)

        # Interior faces in each direction (adjacent cells share a face).
        if nx > 1:
            _add_interior((kx[:-1, :, :], kx[1:, :, :]),
                          idx[:-1, :, :], idx[1:, :, :], area_x, dx)
        if ny > 1:
            _add_interior((ky[:, :-1, :], ky[:, 1:, :]),
                          idx[:, :-1, :], idx[:, 1:, :], area_y, dy)
        if nz > 1:
            _add_interior((kz[:, :, :-1], kz[:, :, 1:]),
                          idx[:, :, :-1], idx[:, :, 1:], area_z, dz)

        # --- External faces / boundary conditions ------------------------------ #
        b_bc = np.zeros(N)
        g_amb = np.zeros(N)
        gt_amb = np.zeros(N)
        q_flux_out = np.zeros(N)
        diag_bc = np.zeros(N)

        def _apply_face(bc: FaceBC, cell_idx: np.ndarray, k_norm: np.ndarray,
                        area: float, half_dist: float):
            """Apply one external face's BC to the given boundary cells.

            cell_idx : (M,) flat indices of the boundary cells on this face
            k_norm   : (M,) directional conductivity normal to the face
            area     : per-cell face area [m^2]
            half_dist: cell-center-to-face distance [m]
            """
            kind = bc.kind
            # Linearized radiative surface coefficient (about the sink temperature t_inf):
            #   q_rad = eps*sigma*(Ts^4 - Tinf^4) ~= h_rad*(Ts - Tinf),  h_rad = 4*eps*sigma*Tinf^3
            # Applied on any non-dirichlet face with emissivity>0 (a real thermal-radiation path).
            h_rad = (4.0 * bc.emissivity * SIGMA_SB * bc.t_inf**3) if bc.emissivity > 0.0 else 0.0
            g_rad = h_rad * area

            if kind == "dirichlet":
                g_bc = np.asarray(k_norm * area / half_dist, dtype=np.float64)
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
            r_cond = half_dist / np.maximum(k_norm, 1e-300)       # half-cell conduction resistance
            g_bc = 1.0 / (r_cond / area + 1.0 / g_surface)
            g_bc = np.asarray(g_bc, dtype=np.float64)
            diag_bc[cell_idx] += g_bc
            b_bc[cell_idx] += g_bc * bc.t_inf
            g_amb[cell_idx] += g_bc
            gt_amb[cell_idx] += g_bc * bc.t_inf

        # top = +z max, bottom = -z min; sides map to x/y min/max.
        # z faces use kz, x faces use kx, y faces use ky.
        _apply_face(cooling.bottom, idx[:, :, 0].ravel(), kz[:, :, 0].ravel(),
                    area_z, dz / 2.0)
        _apply_face(cooling.top, idx[:, :, -1].ravel(), kz[:, :, -1].ravel(),
                    area_z, dz / 2.0)
        _apply_face(cooling.x_min, idx[0, :, :].ravel(), kx[0, :, :].ravel(),
                    area_x, dx / 2.0)
        _apply_face(cooling.x_max, idx[-1, :, :].ravel(), kx[-1, :, :].ravel(),
                    area_x, dx / 2.0)
        _apply_face(cooling.y_min, idx[:, 0, :].ravel(), ky[:, 0, :].ravel(),
                    area_y, dy / 2.0)
        _apply_face(cooling.y_max, idx[:, -1, :].ravel(), ky[:, -1, :].ravel(),
                    area_y, dy / 2.0)

        # Boundary conductances add to the diagonal.
        if np.any(diag_bc):
            nz_bc = np.nonzero(diag_bc)[0]
            rows.append(nz_bc); cols.append(nz_bc); vals.append(diag_bc[nz_bc])

        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(N, N),
        ).tocsr()

        V = grid.volume()
        M = np.ascontiguousarray(geom.rho_cp, dtype=np.float64).ravel() * V

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
