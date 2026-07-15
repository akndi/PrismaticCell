"""Validations of the distributed collector network (PHYSICS §3, §7)."""
import numpy as np

from prismaticcell.geometry import build_geometry
from prismaticcell.echem import ECMModel, ECMState
from prismaticcell.distributed import solve_network


def _setup(cfg):
    geom = build_geometry(cfg)
    model = ECMModel.from_config(cfg)
    n = int(geom.active_mask.sum())
    state = ECMState(soc=np.full(n, 1.0), rc_u=np.zeros((model.n_rc, n)))
    T = np.full((geom.grid.nx, geom.grid.ny, geom.grid.nz), cfg.t_init)
    return geom, model, state, T


def test_charge_conservation(small_cfg):
    """Σ i_cv over all CVs == i_terminal == applied current."""
    geom, model, state, T = _setup(small_cfg)
    I_app = 15.0
    sol = solve_network(geom, model, state, T, I_app, mode="current", tol=1e-9, maxiter=50)
    assert np.isclose(sol.i_cv.sum(), I_app, rtol=1e-4)
    assert np.isclose(sol.i_terminal, I_app, rtol=1e-4)


def test_discharge_below_ocv(small_cfg):
    """Positive applied current (discharge) => terminal voltage below OCV."""
    geom, model, state, T = _setup(small_cfg)
    ocv = model.ocv_v(1.0)
    sol = solve_network(geom, model, state, T, 20.0, mode="current", tol=1e-9, maxiter=50)
    assert sol.v_terminal < ocv
    # and charging (negative current) sits above OCV
    sol_c = solve_network(geom, model, state, T, -20.0, mode="current", tol=1e-9, maxiter=50)
    assert sol_c.v_terminal > ocv


def test_single_roll_reduces_to_lumped(baseline_cfg):
    """A single-roll uniform cell approaches the lumped ECM: V ≈ OCV − (I/A)·R0.

    The tiny remaining gap is the in-plane foil spreading drop (highly conductive foils),
    which we allow for with a loose tolerance.
    """
    cfg = baseline_cfg
    cfg.assembly.jellyrolls = cfg.assembly.jellyrolls[:1]   # one jellyroll
    cfg.mesh.nx, cfg.mesh.ny, cfg.mesh.nz = 4, 4, 2
    geom = build_geometry(cfg)
    model = ECMModel.from_config(cfg)
    n = int(geom.active_mask.sum())
    assert n >= 1
    state = ECMState(soc=np.full(n, 1.0), rc_u=np.zeros((model.n_rc, n)))
    T = np.full((geom.grid.nx, geom.grid.ny, geom.grid.nz), cfg.t_init)
    I = 5.0
    sol = solve_network(geom, model, state, T, I, mode="current", tol=1e-12, maxiter=50)
    total_area = float(geom.rolls[0].area_eff[geom.active_mask].sum())
    r0 = float(model.r0_area(1.0, cfg.t_init))
    v_lumped = float(model.ocv_v(1.0)) - (I / total_area) * r0
    assert abs(sol.v_terminal - v_lumped) < 0.02       # within 20 mV of the lumped estimate
    assert np.isclose(sol.i_cv.sum(), I, rtol=1e-4)


def test_empty_mesh_raises(baseline_cfg):
    """Degenerate mesh with no resolved roll must raise, not silently return garbage."""
    import pytest
    cfg = baseline_cfg
    cfg.mesh.nx = cfg.mesh.ny = cfg.mesh.nz = 1
    geom = build_geometry(cfg)
    model = ECMModel.from_config(cfg)
    n = int(geom.active_mask.sum())
    state = ECMState(soc=np.full(max(n, 1), 1.0), rc_u=np.zeros((model.n_rc, max(n, 1))))
    T = np.full((1, 1, 1), cfg.t_init)
    if n == 0:
        with pytest.raises(ValueError):
            solve_network(geom, model, state, T, 5.0, mode="current", tol=1e-9, maxiter=50)


def test_current_concentrates_near_tabs(baseline_cfg):
    """Current density should be higher near the tab columns than far from them."""
    cfg = baseline_cfg
    cfg.mesh.nx, cfg.mesh.ny, cfg.mesh.nz = 10, 14, 4
    geom, model = build_geometry(cfg), ECMModel.from_config(cfg)
    n = int(geom.active_mask.sum())
    state = ECMState(soc=np.full(n, 0.5), rc_u=np.zeros((model.n_rc, n)))
    T = np.full((geom.grid.nx, geom.grid.ny, geom.grid.nz), cfg.t_init)
    sol = solve_network(geom, model, state, T, 40.0, mode="current", tol=1e-9, maxiter=50)
    roll = geom.rolls[0]
    tab_js = [c[1] for c in roll.tab_pos_nodes + roll.tab_neg_nodes]
    if not tab_js:
        return
    j_edge = np.nanmean([sol.j_area[i, j, :][geom.active_mask[i, j, :]].mean()
                         for (i, j) in roll.columns if j >= max(tab_js)])
    j_far = np.nanmean([sol.j_area[i, j, :][geom.active_mask[i, j, :]].mean()
                        for (i, j) in roll.columns if j <= min(tab_js)])
    assert j_edge >= j_far          # near-tab current >= far-from-tab


def _thermal_gradient_field(geom, base=298.15, amp=25.0):
    import numpy as np
    T = np.full((geom.grid.nx, geom.grid.ny, geom.grid.nz), base)
    kc = (geom.grid.nz - 1) / 2.0
    for k in range(geom.grid.nz):
        T[:, :, k] += amp * np.exp(-((k - kc) / 2.0) ** 2)   # hotter inner layers
    return T


def test_layered_charge_conservation(baseline_cfg):
    cfg = baseline_cfg
    cfg.mesh.nx, cfg.mesh.ny, cfg.mesh.nz = 8, 10, 6
    geom, model = build_geometry(cfg), ECMModel.from_config(cfg)
    n = int(geom.active_mask.sum())
    state = ECMState(soc=np.full(n, 0.6), rc_u=np.zeros((model.n_rc, n)))
    T = _thermal_gradient_field(geom)
    sol = solve_network(geom, model, state, T, 40.0, mode="current", tol=1e-9, maxiter=50,
                        collector_model="layered")
    assert np.isclose(sol.i_cv.sum(), 40.0, rtol=1e-4)
    assert np.isclose(sol.i_terminal, 40.0, rtol=1e-4)


def test_layered_agrees_with_planar_for_conductive_foils(baseline_cfg):
    """With the (very conductive) real Al/Cu foils, layered ~ planar terminal voltage."""
    cfg = baseline_cfg
    cfg.mesh.nx, cfg.mesh.ny, cfg.mesh.nz = 8, 10, 6
    geom, model = build_geometry(cfg), ECMModel.from_config(cfg)
    n = int(geom.active_mask.sum())
    state = ECMState(soc=np.full(n, 0.6), rc_u=np.zeros((model.n_rc, n)))
    T = _thermal_gradient_field(geom)
    sp_ = solve_network(geom, model, state, T, 40.0, mode="current", collector_model="planar")
    sl_ = solve_network(geom, model, state, T, 40.0, mode="current", collector_model="layered")
    assert abs(sp_.v_terminal - sl_.v_terminal) < 5e-3


def test_layered_resolves_through_thickness_potential(baseline_cfg):
    """Planar shares Δφ across z within a column; layered can differ layer-to-layer.

    With reduced foil conductance + a z thermal gradient, the layered model develops a
    genuine through-thickness Δφ variation that the planar model cannot represent.
    """
    cfg = baseline_cfg
    cfg.assembly.jellyrolls = cfg.assembly.jellyrolls[:1]
    cfg.mesh.nx, cfg.mesh.ny, cfg.mesh.nz = 6, 8, 6
    # thin/resistive foils so the foil IR drop matters
    cfg.materials["al_collector"].sigma_elec = 3.5e4
    cfg.materials["cu_collector"].sigma_elec = 5.96e4
    geom, model = build_geometry(cfg), ECMModel.from_config(cfg)
    n = int(geom.active_mask.sum())
    state = ECMState(soc=np.full(n, 0.6), rc_u=np.zeros((model.n_rc, n)))
    T = _thermal_gradient_field(geom, amp=30.0)
    sp_ = solve_network(geom, model, state, T, 60.0, mode="current", collector_model="planar")
    sl_ = solve_network(geom, model, state, T, 60.0, mode="current", collector_model="layered")

    roll = geom.rolls[0]
    # pick a column with several active z-cells
    col = max(roll.columns, key=lambda c: len(roll.col_zcells[c]))
    ks = roll.col_zcells[col]
    dphi_planar = np.array([sp_.dphi_field[col[0], col[1], k] for k in ks])
    dphi_layered = np.array([sl_.dphi_field[col[0], col[1], k] for k in ks])
    assert dphi_planar.max() - dphi_planar.min() < 1e-9        # planar: identical across z
    assert dphi_layered.max() - dphi_layered.min() > 1e-4      # layered: genuinely varies in z
