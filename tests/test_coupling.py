"""End-to-end two-way coupling validations (PHYSICS §6, §7)."""
import copy

import numpy as np

from prismaticcell.coupling import run


def test_energy_conservation(small_cfg):
    """Generated heat == stored + removed (closure to solver tolerance)."""
    res = run(small_cfg)
    eb = res.energy_balance
    assert abs(eb["closure_rel"]) < 1e-6


def test_full_discharge_soc_and_voltage(baseline_cfg):
    """A 1C discharge empties the cell in ~1 h with a falling terminal voltage."""
    cfg = baseline_cfg
    cfg.mesh.nx, cfg.mesh.ny, cfg.mesh.nz = 5, 6, 4
    cfg.solver.dt = 30.0
    cfg.solver.t_end = 3600.0
    res = run(cfg)
    assert res.soc_mean[0] > 0.99
    assert res.soc_mean[-1] < 0.05                 # nearly empty after 1C for 1 h
    assert res.v_terminal[-1] < res.v_terminal[0]  # voltage falls over discharge
    assert np.all(res.q_total >= -1e-9)            # net generation non-negative


def test_temperature_rises_under_load(small_cfg):
    """Ohmic + entropic heating warms the cell above ambient."""
    res = run(small_cfg)
    assert res.T_max[-1] > small_cfg.t_init
    assert res.T_max[-1] >= res.T_mean[-1] >= res.T_min[-1]


def test_more_cooling_lowers_temperature(small_cfg):
    """Two-way coupling closes: stronger convection => lower steady-ish temperature."""
    cfg_lo = copy.deepcopy(small_cfg)
    cfg_hi = copy.deepcopy(small_cfg)
    cfg_lo.cooling.bottom.h = 10.0
    cfg_hi.cooling.bottom.h = 200.0
    res_lo = run(cfg_lo)
    res_hi = run(cfg_hi)
    assert res_hi.T_max[-1] < res_lo.T_max[-1]


def test_steady_mode_energy_balance(small_cfg):
    """Steady operating point: generation balances boundary removal."""
    cfg = copy.deepcopy(small_cfg)
    cfg.solver.mode = "steady"
    cfg.load.kind = "constant_current"
    cfg.load.value = 10.0
    res = run(cfg)
    eb = res.energy_balance
    assert abs(eb["closure_rel"]) < 1e-3


def test_layered_collector_energy_closure(small_cfg):
    """The full 3-D (layered) collector model also closes the energy balance."""
    cfg = small_cfg
    cfg.solver.collector_model = "layered"
    res = run(cfg)
    assert abs(res.energy_balance["closure_rel"]) < 1e-6
    assert res.v_terminal[-1] < res.v_terminal[0] or res.soc_mean[-1] < res.soc_mean[0]


def test_steady_converges_and_is_monotonic(small_cfg):
    """Steady fixed point converges (flag set) and stronger cooling lowers temperature."""
    import copy
    from prismaticcell.config import Cooling, FaceBC

    def steady_at(h):
        cfg = copy.deepcopy(small_cfg)
        cfg.solver.mode = "steady"
        cfg.load.kind = "constant_current"
        cfg.load.value = 20.0
        bc = lambda: FaceBC("convection", h=h, t_inf=298.15)
        cfg.cooling = Cooling(top=bc(), bottom=bc(), x_min=bc(), x_max=bc(), y_min=bc(), y_max=bc())
        return run(cfg)

    lo, hi = steady_at(5.0), steady_at(100.0)
    assert lo.energy_balance["converged"] and hi.energy_balance["converged"]
    assert lo.T_max[-1] > hi.T_max[-1]                 # less cooling => hotter
    assert abs(lo.energy_balance["closure_rel"]) < 1e-3


def test_contact_conductance_raises_stack_temperature(small_cfg):
    """A poorer stack<->wall contact traps heat -> higher peak temperature.

    Needs nz fine enough to resolve the can wall (dz < wall_thickness) so REGION_CAN cells
    exist for the contact interface.
    """
    import copy
    base = copy.deepcopy(small_cfg)
    base.mesh.nx, base.mesh.ny, base.mesh.nz = 6, 6, 12   # resolve the top/bottom wall in z
    cfg_lo = copy.deepcopy(base); cfg_lo.enclosure.contact_conductance = 20.0
    cfg_hi = copy.deepcopy(base); cfg_hi.enclosure.contact_conductance = 1e6
    assert run(cfg_lo).T_max[-1] > run(cfg_hi).T_max[-1]


def test_tab_heat_sink_lowers_temperature(small_cfg):
    """Enabling the tab conduction-to-ambient path removes heat -> lower peak temperature."""
    import copy
    cfg_off = copy.deepcopy(small_cfg); cfg_off.enclosure.tab_heat_sink = False
    cfg_on = copy.deepcopy(small_cfg); cfg_on.enclosure.tab_heat_sink = True
    assert run(cfg_on).T_max[-1] < run(cfg_off).T_max[-1]
