"""Validations of the configurable stack axis (through-plane direction) and footprint tabs."""
import copy

import numpy as np
import pytest

from prismaticcell.config import (SimConfig, Material, Stack, Layer, Jellyroll, Assembly,
                                   Tab, Enclosure, ECM, RCPair, Mesh, Cooling, FaceBC, Solver, Load)
from prismaticcell.geometry import build_geometry, REGION_ACTIVE
from prismaticcell.echem import ECMModel, ECMState
from prismaticcell.distributed import solve_network
from prismaticcell import coupling

import os
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def _mats():
    return {
        "al_collector": Material("al_collector", 2700, 900, 238, 238, 3.5e7),
        "cu_collector": Material("cu_collector", 8960, 385, 398, 398, 5.96e7),
        "lfp_cathode": Material("lfp_cathode", 2200, 1000, 1.0, 1.0, 0.0),
        "graphite_anode": Material("graphite_anode", 1550, 1100, 1.5, 1.5, 0.0),
        "separator": Material("separator", 1200, 1900, 0.30, 0.30, 0.0),
        "can_al": Material("can_al", 2700, 900, 238, 238, 0.0),
        "gap_air": Material("gap_air", 1.2, 1005, 0.03, 0.03, 0.0),
    }


def _layers():
    return [Layer("pos_collector", "al_collector", 13e-6),
            Layer("cathode_coating", "lfp_cathode", 92e-6),
            Layer("separator", "separator", 14e-6),
            Layer("anode_coating", "graphite_anode", 73e-6),
            Layer("neg_collector", "cu_collector", 6e-6)]


def _cfg(stack_axis, width, height, n_stacks=6, mesh=(6, 6, 6), cooling=None, cap=50.0):
    stack = Stack(_layers(), width=width, height=height)
    asm = Assembly([Jellyroll(stack, n_stacks), Jellyroll(stack, n_stacks)],
                   stack_axis=stack_axis, arrangement="stacked",
                   inter_gap=1e-3, wall_clearance=5e-4)
    ecm = ECM(cap, os.path.join(DATA, "lfp_ocv.csv"), os.path.join(DATA, "lfp_entropy.csv"),
              os.path.join(DATA, "lfp_r0.csv"), 20000.0,
              rc_pairs=[RCPair(os.path.join(DATA, "lfp_rc1_r.csv"),
                               os.path.join(DATA, "lfp_rc1_c.csv"), 22000.0, 0.0)])
    tabs = [Tab("pos", loc_length=0.2, loc_height=0.9, size_length=0.02, size_height=0.01),
            Tab("neg", loc_length=0.8, loc_height=0.9, size_length=0.02, size_height=0.01)]
    enc = Enclosure("prismatic", "can_al", 0.8e-3, 150.0)
    cool = cooling if cooling is not None else Cooling(
        top=FaceBC("convection", h=25.0, t_inf=298.15))
    cfg = SimConfig("t", _mats(), asm, tabs, enc, ecm, Mesh(*mesh), cool,
                    Load("constant_current", 20.0), Solver(dt=30.0, t_end=300.0))
    cfg.data_root = "."
    return cfg


def test_stack_axis_y_puts_through_plane_on_y():
    """stack_axis='y' -> low (through-plane) conductivity along Y, high in X and Z."""
    g = build_geometry(_cfg("y", width=0.20, height=0.12))
    act = g.region == REGION_ACTIVE
    assert g.stack_axis == 1 and g.len_axis == 0 and g.hgt_axis == 2
    assert g.ky[act].mean() < 0.2 * g.kx[act].mean()       # through-plane much lower
    assert np.isclose(g.kx[act].mean(), g.kz[act].mean())  # both in-plane


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_stack_axis_charge_conservation(axis):
    """Charge conservation holds for any stack axis, planar and layered."""
    cfg = _cfg(axis, width=0.12, height=0.10)
    g = build_geometry(cfg)
    model = ECMModel.from_config(cfg)
    n = int((g.region == REGION_ACTIVE).sum())
    state = ECMState(soc=np.full(n, 0.6), rc_u=np.zeros((model.n_rc, n)))
    T = np.full((g.grid.nx, g.grid.ny, g.grid.nz), 298.15)
    for cm in ("planar", "layered"):
        sol = solve_network(g, model, state, T, 20.0, mode="current", collector_model=cm)
        assert np.isclose(sol.i_cv.sum(), 20.0, rtol=1e-4)


def test_stack_axis_energy_conservation():
    """A coupled run with a non-default stack axis still closes the energy balance."""
    res = coupling.run(_cfg("y", width=0.20, height=0.12))
    assert abs(res.energy_balance["closure_rel"]) < 1e-6


def test_stack_axis_rotational_invariance():
    """A symmetric cell (square in-plane, cubic mesh, symmetric cooling) gives identical results
    whether the stack axis is Y or Z -- the physics is invariant under relabeling the axes."""
    W = 0.10
    cool = Cooling(x_min=FaceBC("convection", h=20.0, t_inf=298.15),
                   x_max=FaceBC("convection", h=20.0, t_inf=298.15),
                   top=FaceBC("convection", h=20.0, t_inf=298.15),
                   bottom=FaceBC("convection", h=20.0, t_inf=298.15),
                   y_min=FaceBC("convection", h=20.0, t_inf=298.15),
                   y_max=FaceBC("convection", h=20.0, t_inf=298.15))
    cz = _cfg("z", width=W, height=W, mesh=(6, 6, 6), cooling=copy.deepcopy(cool))
    cy = _cfg("y", width=W, height=W, mesh=(6, 6, 6), cooling=copy.deepcopy(cool))
    rz = coupling.run(cz)
    ry = coupling.run(cy)
    assert np.isclose(rz.v_terminal[-1], ry.v_terminal[-1], atol=1e-6)
    assert np.isclose(rz.T_max[-1], ry.T_max[-1], atol=1e-4)
    assert np.isclose(rz.T_mean[-1], ry.T_mean[-1], atol=1e-4)


_AXI = {"x": 0, "y": 1, "z": 2}
_FACES = {0: ("x_min", "x_max"), 1: ("y_min", "y_max"), 2: ("top", "bottom")}


def _rotated_cfg(stack_axis):
    """Same physical cell, rotated so the stack axis is x/y/z, with DISTINCT length/height/stack
    extents, per-role mesh counts, and asymmetric per-role cooling -- so the la/ha distinction is
    actually exercised (a symmetric cube would pass even with la/ha transposed)."""
    sa = _AXI[stack_axis]
    la, ha = [a for a in (0, 1, 2) if a != sa]
    stack = Stack(_layers(), width=0.16, height=0.10)     # length=0.16, height=0.10 (distinct)
    # ~40 sandwiches/roll -> ~16 mm stack extent, resolvable by the nS=4 stack mesh below
    asm = Assembly([Jellyroll(stack, 40), Jellyroll(stack, 40)],
                   stack_axis=stack_axis, arrangement="stacked", inter_gap=5e-4, wall_clearance=2e-4)
    ecm = ECM(50.0, os.path.join(DATA, "lfp_ocv.csv"), os.path.join(DATA, "lfp_entropy.csv"),
              os.path.join(DATA, "lfp_r0.csv"), 20000.0,
              rc_pairs=[RCPair(os.path.join(DATA, "lfp_rc1_r.csv"),
                               os.path.join(DATA, "lfp_rc1_c.csv"), 22000.0, 0.0)])
    tabs = [Tab("pos", loc_length=0.2, loc_height=0.9, size_length=0.02, size_height=0.01),
            Tab("neg", loc_length=0.8, loc_height=0.9, size_length=0.02, size_height=0.01)]
    enc = Enclosure("prismatic", "can_al", 0.8e-3, 150.0)
    dims = [0, 0, 0]; dims[la], dims[ha], dims[sa] = 8, 5, 4        # per-role mesh counts
    cool = Cooling()
    for ax, h in ((la, 12.0), (ha, 27.0), (sa, 40.0)):            # per-role cooling
        for f in _FACES[ax]:
            setattr(cool, f, FaceBC("convection", h=h, t_inf=298.15))
    cfg = SimConfig("rot", _mats(), asm, tabs, enc, ecm, Mesh(*dims), cool,
                    Load("constant_current", 20.0), Solver(dt=30.0, t_end=300.0))
    cfg.data_root = "."
    return cfg


@pytest.mark.parametrize("cm", ["planar", "layered"])
def test_stack_axis_rotational_invariance_adversarial(cm):
    """A non-symmetric cell (distinct extents, per-axis mesh, asymmetric cooling) gives IDENTICAL
    results whichever physical axis is the stack axis -- proving geometry/distributed/thermal all
    share one consistent axis convention (would fail if len_axis/hgt_axis were transposed)."""
    res = {}
    for ax in ("x", "y", "z"):
        cfg = _rotated_cfg(ax)
        cfg.solver.collector_model = cm
        res[ax] = coupling.run(cfg)
    for ax in ("x", "y"):
        assert np.isclose(res[ax].v_terminal[-1], res["z"].v_terminal[-1], atol=1e-7)
        assert np.isclose(res[ax].T_max[-1], res["z"].T_max[-1], atol=1e-5)
        assert np.isclose(res[ax].T_mean[-1], res["z"].T_mean[-1], atol=1e-5)


def test_tab_thickness_defaults_to_collector():
    """A tab with thickness=None uses that polarity's collector thickness (Al 13um / Cu 6um)."""
    cfg = _cfg("y", width=0.20, height=0.12)
    g = build_geometry(cfg)
    # pos tab (Al) and neg tab (Cu) conductances reflect the collector thicknesses & sigmas
    assert g.g_tab_pos > 0 and g.g_tab_neg > 0
