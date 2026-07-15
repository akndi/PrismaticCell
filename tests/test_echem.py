"""Validations of the per-CV equivalent-circuit model (PHYSICS §2, §7)."""
import numpy as np

from prismaticcell.echem import ECMModel, ECMState


def test_ocv_endpoints_and_monotone(baseline_cfg):
    m = ECMModel.from_config(baseline_cfg)
    assert m.ocv_v(0.0) < m.ocv_v(1.0)          # rises from empty to full
    assert 2.0 < m.ocv_v(0.0) < 3.6
    assert 3.2 < m.ocv_v(1.0) < 3.7


def test_arrhenius_resistance_decreases_with_temperature(baseline_cfg):
    m = ECMModel.from_config(baseline_cfg)
    soc = 0.5
    r_cold = m.r0_area(soc, 288.15)
    r_ref = m.r0_area(soc, m.t_ref)
    r_hot = m.r0_area(soc, 318.15)
    assert r_cold > r_ref > r_hot               # positive Ea => R falls as T rises


def test_soc_coulomb_balance(baseline_cfg):
    """∫ I dt / 3600 == ΔSOC * Q (charge conservation of the integrator)."""
    m = ECMModel.from_config(baseline_cfg)
    n = 4
    cap = np.full(n, 5.0)                        # Ah per CV
    state = ECMState(soc=np.full(n, 1.0), rc_u=np.zeros((m.n_rc, n)))
    i_cv = np.full(n, 2.0)                       # A, discharge
    T = np.full(n, 298.15)
    dt = 10.0
    steps = 500
    for _ in range(steps):
        state.advance_soc(m, i_cv, cap, T, dt)
    charge_out = i_cv * dt * steps / 3600.0      # Ah
    dsoc = 1.0 - state.soc
    assert np.allclose(dsoc * cap, charge_out, rtol=1e-6)


def test_rc_relaxes_to_jR(baseline_cfg):
    """Under constant current the RC overpotential relaxes to j*R."""
    if baseline_cfg.ecm.rc_pairs == []:
        return
    m = ECMModel.from_config(baseline_cfg)
    n = 3
    soc = np.full(n, 0.5)
    T = np.full(n, 298.15)
    j = np.full(n, 100.0)                        # A/m^2
    state = ECMState(soc=soc.copy(), rc_u=np.zeros((m.n_rc, n)))
    for _ in range(2000):
        state.advance_rc(m, j, T, dt=1.0)
    R0_area = m.rc_area(0, 0.5, 298.15)[0]
    assert np.allclose(state.rc_u[0], j * R0_area, rtol=1e-3)


def test_entropy_table_sign_changes(baseline_cfg):
    """LFP entropic coefficient is small and changes sign across SOC."""
    m = ECMModel.from_config(baseline_cfg)
    vals = np.array([m.dudt(s) for s in np.linspace(0, 1, 11)])
    assert np.any(vals > 0) and np.any(vals < 0)
    assert np.all(np.abs(vals) < 5e-3)          # small (V/K)
