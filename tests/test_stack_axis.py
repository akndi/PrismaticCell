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
        "pp_insulator": Material("pp_insulator", 905, 1900, 0.20, 0.20, 0.0),
        "mylar_film": Material("mylar_film", 1390, 1170, 0.15, 0.15, 0.0),
        "electrolyte": Material("electrolyte", 1200, 2000, 0.60, 0.60, 0.0),
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


def test_face_role_map_matches_orientation():
    """Role face names map to the correct physical faces / planes for the stack axis.

    For length=X, height=Z, thickness=Y (stack_axis='y'):
      top/bottom -> XY plane (normal Z); front/back -> XZ plane (normal Y, big faces);
      left/right -> YZ plane (normal X, small faces).
    """
    from prismaticcell.config import face_role_map
    my = face_role_map("y")
    assert my["front_face"] == "y_max" and my["back_face"] == "y_min"      # big faces (normal Y)
    assert my["top_face"] == "top" and my["bottom_face"] == "bottom"       # height ends (normal Z)
    assert my["left_face"] == "x_min" and my["right_face"] == "x_max"      # length ends (normal X)
    # numbered aliases: 1/2 = big front/back, 3/4 = small ends
    assert my["side_face_1"] == "y_min" and my["side_face_2"] == "y_max"
    assert my["side_face_3"] == "x_min" and my["side_face_4"] == "x_max"
    mz = face_role_map("z")            # default: big faces normal to Z = top/bottom
    assert {mz["front_face"], mz["back_face"]} == {"top", "bottom"}


def test_insulator_geometry_face_and_params():
    """A bottom insulator becomes a sub-grid layer on the -height face with R_area = t/k."""
    from prismaticcell.config import Insulator
    cfg = _cfg("y", width=0.20, height=0.12)
    cfg.enclosure.insulator = Insulator("pp_insulator", thickness=5e-4, location="bottom")
    g = build_geometry(cfg)
    # stack_axis='y' -> height = Z, so the -height end is the physical "bottom" face
    assert g.insulator_face == "bottom"
    assert np.isclose(g.insulator_R_area, 5e-4 / 0.20)          # t / k_through
    assert np.isclose(g.insulator_rhocp_t, 905 * 1900 * 5e-4)   # rho*cp*t
    # location='top' moves it to the +height face
    cfg.enclosure.insulator = Insulator("pp_insulator", thickness=5e-4, location="top")
    assert build_geometry(cfg).insulator_face == "top"


def test_insulator_throttles_bottom_cooling():
    """With only the bottom face cooled, a bottom insulator raises the steady temperature by
    adding series resistance on that (the only) heat-exit path -- the feedback the slab exists for."""
    from prismaticcell.config import Insulator

    def mk(with_ins):
        cool = Cooling(bottom=FaceBC("convection", h=200.0, t_inf=298.15))
        cfg = _cfg("y", width=0.20, height=0.12, cooling=cool)
        cfg.solver.mode = "steady"      # isolate the resistance effect (mass irrelevant at steady)
        if with_ins:
            cfg.enclosure.insulator = Insulator("pp_insulator", thickness=2e-3, location="bottom")
        return cfg

    r_no = coupling.run(mk(False))
    r_ins = coupling.run(mk(True))
    assert r_ins.T_max[-1] > r_no.T_max[-1] + 1e-3      # insulator -> less heat out -> hotter


def test_roll_wrap_covers_side_faces():
    """A 'sides' mylar wrap adds series resistance on the big (stack-axis) AND end (length-axis)
    external faces, but not the height ends; each adds t/k on top of the clearance baseline."""
    from prismaticcell.config import Wrap
    base = build_geometry(_cfg("y", width=0.20, height=0.12)).face_R_area   # clearance-only baseline
    cfg = _cfg("y", width=0.20, height=0.12)   # stack=Y, length=X, height=Z
    cfg.assembly.roll_wrap = Wrap("mylar_film", thickness=5e-5, coverage="sides")
    g = build_geometry(cfg)
    R = 5e-5 / 0.15
    # big faces (normal to stack axis Y) = y_min/y_max; end faces (normal to length X) = x_min/x_max
    for f in ("y_min", "y_max", "x_min", "x_max"):
        assert np.isclose(g.face_R_area[f] - base.get(f, 0.0), R)
    # height ends (top/bottom) are open -> wrap adds nothing there
    for f in ("top", "bottom"):
        assert np.isclose(g.face_R_area[f] - base.get(f, 0.0), 0.0)


def test_roll_wrap_and_insulator_compose_on_bottom():
    """Insulator (bottom) and an 'all'-coverage wrap stack in series on the same bottom face."""
    from prismaticcell.config import Wrap, Insulator
    base = build_geometry(_cfg("y", width=0.20, height=0.12)).face_R_area.get("bottom", 0.0)
    cfg = _cfg("y", width=0.20, height=0.12)
    cfg.enclosure.insulator = Insulator("pp_insulator", thickness=5e-4, location="bottom")
    cfg.assembly.roll_wrap = Wrap("mylar_film", thickness=5e-5, coverage="all")
    g = build_geometry(cfg)
    # bottom now carries clearance (base) + insulator + wrap, all in series
    assert np.isclose(g.face_R_area["bottom"] - base, 5e-4 / 0.20 + 5e-5 / 0.15)


def test_roll_wrap_throttles_side_cooling():
    """A side-face wrap raises steady temperature when that side is the cooled face."""
    from prismaticcell.config import Wrap

    def mk(with_wrap):
        cool = Cooling(y_min=FaceBC("convection", h=200.0, t_inf=298.15),
                       y_max=FaceBC("convection", h=200.0, t_inf=298.15))
        cfg = _cfg("y", width=0.20, height=0.12, cooling=cool)
        cfg.solver.mode = "steady"
        if with_wrap:
            cfg.assembly.roll_wrap = Wrap("mylar_film", thickness=5e-4, coverage="big_faces")
        return cfg

    r_no = coupling.run(mk(False))
    r_wrap = coupling.run(mk(True))
    assert r_wrap.T_max[-1] > r_no.T_max[-1] + 1e-3


def test_cavity_fill_material_sets_clearance_resistance():
    """The cavity_fill (air vs electrolyte) sets the roll-to-can clearance conduction on every
    external face: R_area = wall_clearance / k_fill. Electrolyte (k=0.6) >> air (k=0.03) coupling."""
    def clearance_R(fill):
        cfg = _cfg("y", width=0.20, height=0.12)
        cfg.assembly.cavity_fill = fill
        return build_geometry(cfg).face_R_area

    clr = 5e-4   # _cfg default wall_clearance
    air = clearance_R("gap_air")
    ely = clearance_R("electrolyte")
    # every external face carries the clearance layer
    for f in ("x_min", "x_max", "y_min", "y_max", "top", "bottom"):
        assert np.isclose(air[f], clr / 0.03)
    # electrolyte-filled clearance is ~20x less resistive than air
    assert np.isclose(ely["top"], clr / 0.60)
    assert ely["top"] < 0.1 * air["top"]


def test_explicit_can_placement_and_void():
    """A fixed can (outer_dims) meshes the roll bbox; roll is bottom-referenced on the insulator
    (headspace all on top) and centred in-plane (electrolyte side clearances)."""
    from prismaticcell.config import Insulator
    wall = 0.8e-3     # _cfg enclosure wall thickness
    cfg = _cfg("y", width=0.20, height=0.12, n_stacks=6)
    cfg.assembly.cavity_fill = "electrolyte"
    cfg.enclosure.insulator = Insulator("pp_insulator", thickness=5e-4, location="bottom")
    cfg.enclosure.headspace_fill = "gap_air"
    cfg.enclosure.outer_dims = [0.2096, 0.005976, 0.1281]   # [X len, Y thick, Z height] outer
    g = build_geometry(cfg)
    assert tuple(round(v, 6) for v in g.outer_dims) == (0.2096, 0.005976, 0.1281)  # reports the can
    # bottom = insulator only: the roll sits on it, so there is NO electrolyte clearance below
    assert np.isclose(g.face_R_area["bottom"], 5e-4 / 0.20)
    # top = gas headspace = all the leftover height (bottom-referenced placement)
    hs = (0.1281 - 2 * wall) - 5e-4 - 0.12
    assert hs > 0 and np.isclose(g.face_R_area["top"], hs / 0.03)     # gap_air k=0.03
    # length ends (X) = centred electrolyte side clearance (k=0.6), equal on both faces
    side_x = ((0.2096 - 2 * wall) - 0.20) / 2.0
    assert np.isclose(g.face_R_area["x_min"], side_x / 0.60)
    assert np.isclose(g.face_R_area["x_max"], side_x / 0.60)


def test_explicit_can_too_small_raises():
    """A can smaller than the roll assembly is rejected with a clear error."""
    cfg = _cfg("y", width=0.20, height=0.12, n_stacks=6)
    cfg.enclosure.outer_dims = [0.10, 0.005, 0.05]     # far smaller than the 0.20 m roll
    with pytest.raises(ValueError):
        build_geometry(cfg)


def test_tab_joule_backflow_injected():
    """_heat_map deposits exactly frac * I^2/g_tab per polarity into the attachment CVs:
    frac = 1/2 when the tab is heat-sunk (fin split), 1 when its tip is adiabatic."""
    from prismaticcell.echem import ECMModel, ECMState
    from prismaticcell.distributed import solve_network
    from prismaticcell.geometry import tab_attachment_cells

    def injected(tab_heat_sink):
        cfg = _cfg("y", width=0.20, height=0.12)
        cfg.enclosure.tab_heat_sink = tab_heat_sink
        g = build_geometry(cfg)
        model = ECMModel.from_config(cfg)
        n = int(g.active_mask.sum())
        state = ECMState(soc=np.full(n, 0.6), rc_u=np.zeros((model.n_rc, n)))
        T = np.full((g.grid.nx, g.grid.ny, g.grid.nz), 298.15)
        I = 80.0
        sol = solve_network(g, model, state, T, I, mode="current")
        active_ijk = np.argwhere(g.active_mask)
        q = coupling._heat_map(g, model, state, sol, T, active_ijk)
        V = g.grid.volume()
        # subtract the ECM + foil-ohmic parts to isolate the tab injection
        base = np.array(sol.q_ohm_vol, dtype=float)
        ii, jj, kk = active_ijk[:, 0], active_ijk[:, 1], active_ijk[:, 2]
        dphi = sol.dphi_field[ii, jj, kk]
        i_cv = sol.i_cv[ii, jj, kk]
        q_ecm = (i_cv * (model.ocv_v(state.soc) - dphi)
                 - i_cv * T[ii, jj, kk] * model.dudt(state.soc)) / V
        base[ii, jj, kk] += q_ecm
        extra_W = float(((q - base) * V).sum())
        p_full = I * I / g.g_tab_pos + I * I / g.g_tab_neg
        return extra_W, p_full, g

    extra, p_full, g = injected(True)                     # heat-sunk: half of each tab's P
    assert np.isclose(extra, 0.5 * p_full, rtol=1e-9)
    extra0, p_full0, _ = injected(False)                  # adiabatic tip: all of P
    assert np.isclose(extra0, 1.0 * p_full0, rtol=1e-9)
    # injection lands only on attachment CVs
    assert len(tab_attachment_cells(g, "pos")) > 0


def test_tab_joule_energy_closure_high_current():
    """Energy balance still closes with the tab Joule backflow active at high current."""
    cool = Cooling(y_min=FaceBC("convection", h=100.0, t_inf=298.15),
                   y_max=FaceBC("convection", h=100.0, t_inf=298.15))
    cfg = _cfg("y", width=0.20, height=0.12, cooling=cool)
    cfg.load = Load("constant_current", 150.0)
    res = coupling.run(cfg)
    assert abs(res.energy_balance["closure_rel"]) < 1e-6


def test_tab_thermal_fin_profile():
    """The 1-D tab fin profile matches its closed form, the bump is non-negative, the zero-current
    limit is linear, and a non-heat-sunk tab returns no profile."""
    from prismaticcell import viz
    cool = Cooling(y_min=FaceBC("convection", h=100.0, t_inf=298.15),
                   y_max=FaceBC("convection", h=100.0, t_inf=298.15))
    cfg = _cfg("y", width=0.20, height=0.12, cooling=cool)
    cfg.solver.mode = "steady"
    cfg.load = Load("constant_current", 60.0)
    r = coupling.run(cfg)
    prof = viz.tab_thermal_profiles(r, cool)
    for pol, gcond in (("pos", r.geom.tab_heat_cond_pos), ("neg", r.geom.tab_heat_cond_neg)):
        p = prof[pol]
        xi = p["xi"]
        # closed form: T = T_root + (T_sink-T_root) xi + P/(2G) xi(1-xi)
        expect = p["T_root"] + (p["T_sink"] - p["T_root"]) * xi + p["P"] / (2 * gcond) * xi * (1 - xi)
        assert np.allclose(p["T"], expect, rtol=1e-9, atol=1e-9)
        assert p["P"] > 0.0 and np.isclose(p["R_tab"], 1.0 / (r.geom.g_tab_pos if pol == "pos"
                                                              else r.geom.g_tab_neg))
        lin = p["T_root"] + (p["T_sink"] - p["T_root"]) * xi          # self-heating bump >= 0
        assert np.all(p["T"] >= lin - 1e-9)

    # non-heat-sunk tab -> no fin profile (only the root is meaningful)
    cfg.enclosure.tab_heat_sink = False
    p2 = viz.tab_thermal_profiles(coupling.run(cfg), cool)["pos"]
    assert p2["T"] is None and np.isfinite(p2["T_root"])


def test_inter_roll_gap_is_electrolyte_layer():
    """The inter-roll gap is a sub-grid electrolyte layer: R = inter_gap / k_fill; 0 if touching."""
    cfg = _cfg("y", width=0.20, height=0.12)
    cfg.assembly.cavity_fill = "electrolyte"
    cfg.assembly.inter_gap = 1e-3
    g = build_geometry(cfg)
    assert np.isclose(g.inter_roll_R_area, 1e-3 / 0.60)
    assert np.isclose(g.inter_roll_rhocp_t, 1200 * 2000 * 1e-3)
    cfg.assembly.inter_gap = 0.0                       # back-to-back
    assert build_geometry(cfg).inter_roll_R_area == 0.0


def test_inter_roll_gap_decouples_rolls():
    """With one big face cooled, an electrolyte inter-roll gap makes the far roll run hotter than
    a back-to-back (touching) pair -- the gap impedes heat crossing between the rolls."""
    def mk(gap):
        cool = Cooling(y_min=FaceBC("convection", h=200.0, t_inf=298.15))
        cfg = _cfg("y", width=0.20, height=0.12, cooling=cool)
        cfg.assembly.cavity_fill = "electrolyte"
        cfg.assembly.inter_gap = gap
        cfg.solver.mode = "steady"
        return cfg
    r_touch = coupling.run(mk(0.0))
    r_gap = coupling.run(mk(3e-3))
    assert r_gap.T_max[-1] > r_touch.T_max[-1] + 1e-4


def test_explicit_can_rotated_axis_maps_faces():
    """The fixed-can void follows the axis roles: for stack_axis='z' (height=Y), the insulator is
    on y_min, the headspace on y_max, and side clearances on the X (length) and Z (stack) faces."""
    from prismaticcell.config import Insulator
    wall = 0.8e-3
    cfg = _cfg("z", width=0.20, height=0.12, n_stacks=6)   # stack=Z, length=X, height=Y
    cfg.assembly.cavity_fill = "electrolyte"
    cfg.enclosure.insulator = Insulator("pp_insulator", thickness=5e-4, location="bottom")
    cfg.enclosure.headspace_fill = "gap_air"
    cfg.enclosure.outer_dims = [0.2096, 0.1281, 0.005976]   # [X, Y=height, Z=stack] outer
    g = build_geometry(cfg)
    assert g.hgt_axis == 1 and g.insulator_face == "y_min"          # height = Y here
    ins = 5e-4 / 0.20
    hs = (0.1281 - 2 * wall) - 5e-4 - 0.12
    assert np.isclose(g.face_R_area["y_min"], ins)                  # insulator on -height (y_min)
    assert hs > 0 and np.isclose(g.face_R_area["y_max"], hs / 0.03)  # headspace on +height (y_max)
    side_x = ((0.2096 - 2 * wall) - 0.20) / 2.0                     # length clearance on x faces
    assert np.isclose(g.face_R_area["x_min"], side_x / 0.60)


def test_fixed_can_credits_cooling_over_can_surface():
    """In fixed-can mode the ambient exchange is scaled to the real can face (mesh is the roll bbox);
    auto-size mode leaves it at 1 (the cavity boundary already is the external surface)."""
    cfg = SimConfig.from_yaml(os.path.join(os.path.dirname(__file__), "..", "configs",
                                           "large_prismatic.yaml"))
    g = build_geometry(cfg)
    Lx, Ly, Lz = cfg.enclosure.outer_dims       # [X, Y(thick), Z(height)]
    # big front/back faces are normal to Y (stack axis): they span X*Z
    assert np.isclose(g.face_area_scale["y_min"], (Lx * Lz) / (0.720 * 0.120))
    assert all(v > 1.0 for v in g.face_area_scale.values())
    assert build_geometry(_cfg("y", width=0.20, height=0.12)).face_area_scale == {}   # auto: unscaled


def test_new_enclosure_assembly_field_validation():
    """Typos in insulator.location / roll_wrap.coverage, and mesh + outer_dims, are rejected."""
    from prismaticcell.config import Insulator, Wrap
    cfg = _cfg("y", width=0.20, height=0.12)
    cfg.enclosure.insulator = Insulator("pp_insulator", 5e-4, location="botom")   # typo
    with pytest.raises(ValueError):
        cfg.validate()
    cfg.enclosure.insulator = None
    cfg.assembly.roll_wrap = Wrap("mylar_film", 5e-5, coverage="sidez")           # typo
    with pytest.raises(ValueError):
        cfg.validate()
    cfg.assembly.roll_wrap = None
    cfg.enclosure.outer_dims = [0.30, 0.30, 0.30]
    cfg.enclosure.wall_model = "mesh"                                             # contradiction
    with pytest.raises(ValueError):
        cfg.validate()


def test_cavity_fill_unknown_material_rejected():
    cfg = _cfg("y", width=0.20, height=0.12)
    cfg.assembly.cavity_fill = "nonexistent_fluid"
    with pytest.raises(ValueError):
        cfg.validate()


def test_roll_wrap_unknown_material_rejected():
    from prismaticcell.config import Wrap
    cfg = _cfg("y", width=0.20, height=0.12)
    cfg.assembly.roll_wrap = Wrap("nope", thickness=5e-5)
    with pytest.raises(ValueError):
        cfg.validate()


def test_insulator_unknown_material_rejected():
    """enclosure.insulator with an unknown material / non-positive thickness fails validation."""
    from prismaticcell.config import Insulator
    cfg = _cfg("y", width=0.20, height=0.12)
    cfg.enclosure.insulator = Insulator("does_not_exist", thickness=5e-4)
    with pytest.raises(ValueError):
        cfg.validate()
    cfg.enclosure.insulator = Insulator("pp_insulator", thickness=-1.0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_role_based_cooling_from_yaml():
    """A YAML config using role face names cools the intended physical faces."""
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "large_prismatic.yaml")
    cfg = SimConfig.from_yaml(path)
    # front/back (large flat faces) -> y_min/y_max got the convection BC
    assert cfg.cooling.y_min.kind == "convection" and cfg.cooling.y_max.kind == "convection"
    assert cfg.cooling.top.kind == "adiabatic" and cfg.cooling.x_min.kind == "adiabatic"
