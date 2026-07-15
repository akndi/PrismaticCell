"""Shared pytest fixtures and helpers for the PrismaticCell validation suite."""
import os

import numpy as np
import pytest

from prismaticcell.config import SimConfig
from prismaticcell.geometry import Geometry, Grid, REGION_ACTIVE

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "baseline_prismatic.yaml")


@pytest.fixture
def baseline_cfg():
    return SimConfig.from_yaml(CONFIG)


@pytest.fixture
def small_cfg():
    """Baseline physics on a small mesh + short horizon for fast tests."""
    cfg = SimConfig.from_yaml(CONFIG)
    cfg.mesh.nx, cfg.mesh.ny, cfg.mesh.nz = 5, 6, 4
    cfg.solver.dt = 30.0
    cfg.solver.t_end = 300.0
    return cfg


def make_uniform_geometry(nx, ny, nz, dx, k, rho_cp):
    """A homogeneous isotropic block Geometry for analytical thermal tests.

    Electro fields are left empty (these tests exercise only the thermal operator).
    """
    grid = Grid(nx, ny, nz, dx, dx, dx,
                (np.arange(nx) + 0.5) * dx,
                (np.arange(ny) + 0.5) * dx,
                (np.arange(nz) + 0.5) * dx)
    shape = (nx, ny, nz)
    return Geometry(
        grid=grid,
        region=np.full(shape, REGION_ACTIVE, dtype=np.int32),
        kx=np.full(shape, k), ky=np.full(shape, k), kz=np.full(shape, k),
        rho_cp=np.full(shape, rho_cp),
        active_mask=np.zeros(shape, dtype=bool),
        cell_roll=np.full(shape, -1, dtype=np.int32),
        cap_cv=np.zeros(shape),
        rolls=[],
        outer_dims=(nx * dx, ny * dx, nz * dx),
    )
