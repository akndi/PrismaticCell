"""Per-control-volume equivalent-circuit model (implements PHYSICS.md §2, §4).

Each electrochemically active control volume (CV) carries its own Thevenin ECM
(``U_ocv``, series ``R0`` and ``m`` RC pairs), all of whose parameters are read from
data-driven CSV tables and Arrhenius-scaled with temperature — *nothing* is hard-coded.
State (SOC and RC overpotentials) lives in :class:`ECMState`; the parameter provider
:class:`ECMModel` is stateless. All arrays are NumPy ``float64`` and every operation is
vectorized over the flat set of active CVs (no Python loops over CVs).

Sign convention: positive current = discharge -> SOC decreases; irreversible
overpotential heat is >= 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from .config import ECM, R_GAS, SimConfig

ArrayLike = Union[np.ndarray, float]


# --------------------------------------------------------------------------- #
# CSV lookup table
# --------------------------------------------------------------------------- #
class Table:
    """1-D CSV lookup with linear interpolation and flat extrapolation.

    The CSV is a two-column table ``x, value``. Lines starting with ``#`` are
    comments and the first non-comment line is a header (both skipped). Values are
    linearly interpolated; queries beyond the tabulated range are clamped to the end
    values (flat extrapolation). Evaluation is vectorized over NumPy arrays.
    """

    __slots__ = ("x", "y")

    def __init__(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
            raise ValueError("Table requires two equal-length 1-D columns")
        if x.size == 0:
            raise ValueError("Table requires at least one row")
        # np.interp requires monotonically increasing x; sort defensively.
        order = np.argsort(x, kind="stable")
        self.x = x[order]
        self.y = y[order]

    @classmethod
    def from_csv(cls, path: str) -> "Table":
        """Load a two-column table from ``path`` (skips ``#`` comments + header)."""
        xs: List[float] = []
        ys: List[float] = []
        header_skipped = False
        with open(path, "r") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if not header_skipped:
                    # First non-comment line is the header row.
                    header_skipped = True
                    continue
                parts = s.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
        return cls(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Interpolate value(s) at ``x`` (scalar or ndarray), flat outside range."""
        xq = np.asarray(x, dtype=np.float64)
        # np.interp clamps to y[0]/y[-1] outside [x[0], x[-1]] -> flat extrapolation.
        out = np.interp(xq, self.x, self.y)
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
            return float(out)
        return out


# --------------------------------------------------------------------------- #
# Arrhenius scaling
# --------------------------------------------------------------------------- #
def _arrhenius(x_ref: ArrayLike, ea: float, T: ArrayLike, t_ref: float) -> np.ndarray:
    """Arrhenius temperature scaling (PHYSICS §2.2).

    ``X(T) = X_ref * exp[ (Ea / R_GAS) * (1/T - 1/t_ref) ]``  using ``config.R_GAS``.

    For a positive activation energy ``Ea`` the factor *decreases* as ``T`` rises
    above ``t_ref`` (1/T shrinks), so a resistance falls with increasing temperature.
    """
    x_ref = np.asarray(x_ref, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    return x_ref * np.exp((ea / R_GAS) * (1.0 / T - 1.0 / t_ref))


# --------------------------------------------------------------------------- #
# Stateless ECM parameter provider
# --------------------------------------------------------------------------- #
@dataclass
class ECMModel:
    """Stateless provider of ECM parameters (SOC- and T-dependent).

    Tables give the reference (``t_ref``) SOC dependence; temperature dependence is
    applied on top via Arrhenius scaling with per-quantity activation energies. The
    dynamic state (SOC, RC overpotentials) lives in :class:`ECMState`.
    """

    ocv: Table                      # U_ocv(soc) [V]
    entropy: Table                  # entropic dU/dT(soc) [V/K]
    r0: Table                       # areal R0(soc) [Ohm*m^2] at t_ref
    rc_r: List[Table]               # per-RC areal R_p(soc) [Ohm*m^2] at t_ref
    rc_c: List[Table]               # per-RC areal C_p(soc) [F/m^2] at t_ref
    ea_r0: float                    # J/mol, activation energy for R0
    ea_r: List[float]               # J/mol, activation energy per RC resistance
    ea_c: List[float]               # J/mol, activation energy per RC capacitance
    t_ref: float                    # K, reference temperature of the tables
    capacity_temp: Optional[Table]  # capacity_scale(T) [-], or None

    @classmethod
    def from_config(cls, cfg: SimConfig) -> "ECMModel":
        """Build all tables from ``cfg.ecm``, resolving CSV paths via ``cfg.resolve``."""
        ecm: ECM = cfg.ecm
        ocv = Table.from_csv(cfg.resolve(ecm.ocv_table))
        entropy = Table.from_csv(cfg.resolve(ecm.entropy_table))
        r0 = Table.from_csv(cfg.resolve(ecm.r0_table))

        rc_r: List[Table] = []
        rc_c: List[Table] = []
        ea_r: List[float] = []
        ea_c: List[float] = []
        for pair in ecm.rc_pairs:
            rc_r.append(Table.from_csv(cfg.resolve(pair.r_table)))
            rc_c.append(Table.from_csv(cfg.resolve(pair.c_table)))
            ea_r.append(float(pair.ea_r))
            ea_c.append(float(pair.ea_c))

        capacity_temp: Optional[Table] = None
        if ecm.capacity_temp_table is not None:
            capacity_temp = Table.from_csv(cfg.resolve(ecm.capacity_temp_table))

        return cls(
            ocv=ocv,
            entropy=entropy,
            r0=r0,
            rc_r=rc_r,
            rc_c=rc_c,
            ea_r0=float(ecm.ea_r0),
            ea_r=ea_r,
            ea_c=ea_c,
            t_ref=float(ecm.t_ref),
            capacity_temp=capacity_temp,
        )

    @property
    def n_rc(self) -> int:
        """Number of RC pairs."""
        return len(self.rc_r)

    # ---- direct table lookups ------------------------------------------- #
    def ocv_v(self, soc: ArrayLike) -> np.ndarray:
        """Open-circuit voltage ``U_ocv(soc)`` [V]."""
        return self.ocv(soc)

    def dudt(self, soc: ArrayLike) -> np.ndarray:
        """Entropic coefficient ``dU/dT(soc)`` [V/K] (from the table; sign varies)."""
        return self.entropy(soc)

    # ---- Arrhenius-scaled parameters ------------------------------------ #
    def r0_area(self, soc: ArrayLike, T: ArrayLike) -> np.ndarray:
        """Areal series resistance ``R0(soc, T)`` [Ohm*m^2] (Arrhenius, PHYSICS §2.2)."""
        return _arrhenius(self.r0(soc), self.ea_r0, T, self.t_ref)

    def rc_area(self, p: int, soc: ArrayLike, T: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Areal RC-pair ``p`` parameters ``(R_p [Ohm*m^2], C_p [F/m^2])`` at ``(soc, T)``.

        ``R_p`` scales with ``ea_r[p]`` and ``C_p`` with ``ea_c[p]``.
        """
        r = _arrhenius(self.rc_r[p](soc), self.ea_r[p], T, self.t_ref)
        c = _arrhenius(self.rc_c[p](soc), self.ea_c[p], T, self.t_ref)
        return r, c


# --------------------------------------------------------------------------- #
# Per-CV dynamic state
# --------------------------------------------------------------------------- #
@dataclass
class ECMState:
    """Dynamic ECM state over the flat set of active CVs.

    ``soc``   : shape ``(n_active,)`` state of charge in [0, 1].
    ``rc_u``  : shape ``(n_rc, n_active)`` RC-pair overpotentials [V].
    """

    soc: np.ndarray          # (n_active,)
    rc_u: np.ndarray         # (n_rc, n_active)

    @classmethod
    def initial(cls, model: ECMModel, n_active: int, soc0: ArrayLike = 1.0) -> "ECMState":
        """Convenience constructor: uniform (or per-CV) initial SOC, zero RC overpotentials."""
        soc = np.full(n_active, float(soc0), dtype=np.float64) if np.isscalar(soc0) \
            else np.asarray(soc0, dtype=np.float64).copy()
        rc_u = np.zeros((model.n_rc, n_active), dtype=np.float64)
        return cls(soc=soc, rc_u=rc_u)

    def rc_u_sum(self) -> np.ndarray:
        """Sum of RC overpotentials over pairs, shape ``(n_active,)``."""
        if self.rc_u.shape[0] == 0:
            return np.zeros_like(self.soc)
        return self.rc_u.sum(axis=0)

    # ---- RC dynamics ---------------------------------------------------- #
    def advance_rc(self, model: ECMModel, j_area: ArrayLike, T: ArrayLike, dt: float) -> None:
        """Advance each RC overpotential by ``dt`` (PHYSICS §2.1).

        Solves ``du/dt = -u/(R C) + j_area/C`` per pair with the *exact* exponential
        (analytic) update, unconditionally stable for any ``dt``::

            u_new = u * exp(-dt/tau) + j_area * R * (1 - exp(-dt/tau)),   tau = R C

        ``R``/``C`` are evaluated (Arrhenius) at the current ``soc`` and ``T``.
        ``j_area`` is the local areal current density [A/m^2].
        """
        j = np.asarray(j_area, dtype=np.float64)
        for p in range(model.n_rc):
            r, c = model.rc_area(p, self.soc, T)
            tau = r * c
            # exp(-dt/tau); tau>0 for physical R,C. Guard tau==0 -> instantaneous relaxation.
            with np.errstate(divide="ignore", over="ignore"):
                decay = np.where(tau > 0.0, np.exp(-dt / tau), 0.0)
            self.rc_u[p] = self.rc_u[p] * decay + j * r * (1.0 - decay)

    # ---- SOC dynamics --------------------------------------------------- #
    def advance_soc(self, model: ECMModel, i_cv: ArrayLike, cap_cv: ArrayLike,
                    T: ArrayLike, dt: float) -> None:
        """Advance SOC by ``dt`` (PHYSICS §2.3): ``dz/dt = -i_cv / (3600 Q(T))``.

        ``i_cv`` is the current per CV [A] (positive = discharge -> SOC drops),
        ``cap_cv`` the per-CV capacity [Ah]. Temperature-dependent capacity is
        ``Q(T) = cap_cv * capacity_temp(T)`` when a ``capacity_temp`` table exists,
        else ``Q = cap_cv``. SOC is clipped to [0, 1].
        """
        i = np.asarray(i_cv, dtype=np.float64)
        cap = np.asarray(cap_cv, dtype=np.float64)
        q = cap * model.capacity_temp(T) if model.capacity_temp is not None else cap
        dz = -i * dt / (3600.0 * q)
        self.soc = np.clip(self.soc + dz, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Coupling helper: local through-CV areal current density
# --------------------------------------------------------------------------- #
def local_current_density(model: ECMModel, soc: ArrayLike, T: ArrayLike,
                          rc_u_sum: ArrayLike, dphi: ArrayLike) -> np.ndarray:
    """Local areal current density from the shared column ``Delta phi`` (PHYSICS §3.1).

    The local stack voltage equals the collector potential difference,
    ``v = U_ocv(soc) - j_area*R0(soc,T) - sum_p u_p`` and ``v == dphi``, so::

        j_area = (U_ocv(soc) - sum_p u_p - dphi) / R0_area(soc, T)     [A/m^2]

    This is the key coupling equation between the collector potential field and each
    CV's ECM; the distributed/coupling layers call it per active CV. ``rc_u_sum`` is
    the summed RC overpotential per CV (``ECMState.rc_u_sum()``); ``dphi`` is the local
    ``phi_pos - phi_neg``. Positive ``j_area`` = discharge current out of the + foil.
    """
    ocv = np.asarray(model.ocv_v(soc), dtype=np.float64)
    usum = np.asarray(rc_u_sum, dtype=np.float64)
    dphi = np.asarray(dphi, dtype=np.float64)
    r0 = model.r0_area(soc, T)
    return (ocv - usum - dphi) / r0
