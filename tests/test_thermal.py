"""Analytical validations of the 3-D anisotropic thermal solver (PHYSICS §5, §7)."""
import numpy as np

from prismaticcell.config import Cooling, FaceBC
from prismaticcell.thermal import (
    ThermalOperator, solve_steady, step_transient, boundary_heat_removed,
)
from conftest import make_uniform_geometry


def test_1d_steady_conduction_linear_profile():
    """Dirichlet on x_min/x_max, adiabatic elsewhere -> linear profile & correct flux."""
    nx, ny, nz, dx, k = 20, 3, 3, 1e-3, 5.0
    geom = make_uniform_geometry(nx, ny, nz, dx, k, rho_cp=2e6)
    T_hot, T_cold = 350.0, 300.0
    cooling = Cooling(
        x_min=FaceBC("dirichlet", t_inf=T_hot),
        x_max=FaceBC("dirichlet", t_inf=T_cold),
    )
    op = ThermalOperator.assemble(geom, cooling)
    T = solve_steady(op, np.zeros(nx * ny * nz)).reshape(nx, ny, nz)
    # profile along x at mid-plane is linear
    line = T[:, ny // 2, nz // 2]
    L = nx * dx
    x = (np.arange(nx) + 0.5) * dx
    expected = T_hot + (T_cold - T_hot) * x / L
    assert np.allclose(line, expected, rtol=1e-3)
    # conductive flux through the block ~ k*A*dT/L
    A = (ny * dx) * (nz * dx)
    q_analytic = k * A * (T_hot - T_cold) / L
    q_removed = boundary_heat_removed(op, T)  # net leaving ~0 at steady state overall
    assert abs(q_removed) < 1e-6 * max(1.0, q_analytic)


def test_adiabatic_energy_conservation_one_step():
    """All faces adiabatic, uniform generation -> ΔT = q*V*dt/(Σ m cp)."""
    nx, ny, nz, dx = 6, 5, 4, 2e-3
    rho_cp = 2.2e6
    geom = make_uniform_geometry(nx, ny, nz, dx, k=2.0, rho_cp=rho_cp)
    cooling = Cooling()  # all adiabatic by default
    op = ThermalOperator.assemble(geom, cooling)
    q = 5000.0  # W/m^3
    dt = 20.0
    T0 = np.full(nx * ny * nz, 298.15)
    T1 = step_transient(op, T0, np.full((nx, ny, nz), q), dt)
    dT_expected = q * dt / rho_cp  # V and mass cancel for uniform props
    assert np.allclose(T1 - 298.15, dT_expected, rtol=1e-8)


def test_convection_steady_energy_balance():
    """Single cooled face at steady state: generated == removed."""
    nx, ny, nz, dx = 5, 5, 5, 2e-3
    geom = make_uniform_geometry(nx, ny, nz, dx, k=3.0, rho_cp=2e6)
    cooling = Cooling(bottom=FaceBC("convection", h=50.0, t_inf=298.15))
    op = ThermalOperator.assemble(geom, cooling)
    q = 8000.0
    T = solve_steady(op, np.full((nx, ny, nz), q)).reshape(nx, ny, nz)
    V_total = nx * ny * nz * dx**3
    gen = q * V_total
    removed = boundary_heat_removed(op, T)
    assert np.isclose(gen, removed, rtol=1e-6)
    assert T.min() > 298.15  # warmer than coolant


def test_symmetric_cooling_gives_symmetric_field():
    """Equal convection on x_min and x_max -> field symmetric about mid-x."""
    nx, ny, nz, dx = 9, 4, 4, 1e-3
    geom = make_uniform_geometry(nx, ny, nz, dx, k=2.0, rho_cp=2e6)
    bc = FaceBC("convection", h=30.0, t_inf=298.15)
    cooling = Cooling(x_min=bc, x_max=FaceBC("convection", h=30.0, t_inf=298.15))
    op = ThermalOperator.assemble(geom, cooling)
    T = solve_steady(op, np.full((nx, ny, nz), 6000.0)).reshape(nx, ny, nz)
    assert np.allclose(T, T[::-1, :, :], atol=1e-9)


def test_transient_relaxes_to_steady():
    """Transient with fixed BC/gen converges to the steady solution."""
    nx, ny, nz, dx = 5, 5, 4, 2e-3
    geom = make_uniform_geometry(nx, ny, nz, dx, k=2.0, rho_cp=1e6)
    cooling = Cooling(top=FaceBC("convection", h=40.0, t_inf=298.15))
    op = ThermalOperator.assemble(geom, cooling)
    q = np.full((nx, ny, nz), 5000.0)
    T_steady = solve_steady(op, q)
    T = np.full(nx * ny * nz, 298.15)
    for _ in range(4000):
        T = step_transient(op, T, q, dt=1.0)
    assert np.allclose(T, T_steady, rtol=1e-4)


def test_radiation_only_face_energy_balance():
    """An adiabatic-but-emissive face still rejects heat by radiation; gen == removed."""
    from prismaticcell.config import SIGMA_SB
    nx, ny, nz, dx = 5, 5, 5, 2e-3
    geom = make_uniform_geometry(nx, ny, nz, dx, k=3.0, rho_cp=2e6)
    cooling = Cooling(top=FaceBC("adiabatic", t_inf=298.15, emissivity=0.85))
    op = ThermalOperator.assemble(geom, cooling)
    q = 4000.0
    T = solve_steady(op, np.full((nx, ny, nz), q)).reshape(nx, ny, nz)
    V_total = nx * ny * nz * dx**3
    assert np.isclose(q * V_total, boundary_heat_removed(op, T), rtol=1e-6)
    assert T.min() > 298.15


def test_emissivity_augments_convection():
    """Adding emissivity to a convective face increases heat removal (lower steady T)."""
    nx, ny, nz, dx = 5, 5, 5, 2e-3
    geom = make_uniform_geometry(nx, ny, nz, dx, k=3.0, rho_cp=2e6)
    op_conv = ThermalOperator.assemble(geom, Cooling(top=FaceBC("convection", h=10.0, t_inf=298.15)))
    op_both = ThermalOperator.assemble(geom, Cooling(top=FaceBC("convection", h=10.0, t_inf=298.15, emissivity=0.9)))
    q = np.full((nx, ny, nz), 5000.0)
    T_conv = solve_steady(op_conv, q)
    T_both = solve_steady(op_both, q)
    assert T_both.max() < T_conv.max()
