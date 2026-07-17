"""Tests for the cycling driver, field-history recording, and the notebook-facing viz helpers."""
import os

import numpy as np
import pytest

from prismaticcell.config import Cooling, FaceBC
from prismaticcell.cycling import build_cycle_profile, configure_cycling
from prismaticcell import coupling, viz, viz3d

from tests.test_stack_axis import _cfg

SCRATCH = os.environ.get("TMPDIR", "/tmp")


def _cycled(tmp_path, n_cycles=1, dt=120.0, soc=(0.3, 0.8)):
    cool = Cooling(y_min=FaceBC("convection", h=50.0, t_inf=298.15),
                   y_max=FaceBC("convection", h=50.0, t_inf=298.15))
    cfg = _cfg("y", width=0.20, height=0.12, cooling=cool, cap=50.0)
    configure_cycling(cfg, c_rate=1.0, n_cycles=n_cycles, soc_min=soc[0], soc_max=soc[1],
                      dt=dt, profile_path=str(tmp_path / "prof.csv"))
    return cfg, coupling.run(cfg)


def test_cycle_profile_is_square_wave():
    """The profile alternates +/- C*capacity with half-cycle length (dSOC*3600/C)."""
    prof = build_cycle_profile(50.0, c_rate=2.0, n_cycles=2, soc_min=0.2, soc_max=0.7)
    seg = 0.5 * 3600.0 / 2.0
    assert np.isclose(prof[-1, 0], 4 * seg)                 # total duration
    assert np.isclose(prof[0, 1], +100.0)                   # discharge first at 2C * 50Ah
    # value just before the first switch is still +100; just after is -100
    assert np.isclose(np.interp(seg - 1e-3, prof[:, 0], prof[:, 1]), +100.0, atol=0.2)
    assert np.isclose(np.interp(seg + 1e-3, prof[:, 0], prof[:, 1]), -100.0, atol=0.2)
    with pytest.raises(ValueError):
        build_cycle_profile(50.0, 1.0, 1, 0.8, 0.2)         # inverted window


def test_cycling_run_tracks_soc_window(tmp_path):
    """Mean SOC traverses exactly [soc_min, soc_max]; current is the +/-1C square wave; the
    recorded field histories align with the time axis; energy closes."""
    cfg, res = _cycled(tmp_path, n_cycles=2)
    assert np.isclose(res.soc_mean.min(), 0.3, atol=5e-3)
    assert np.isclose(res.soc_mean.max(), 0.8, atol=5e-3)
    assert np.isclose(res.i_terminal.max(), +50.0, rtol=1e-6)
    assert np.isclose(res.i_terminal.min(), -50.0, rtol=1e-6)
    nt = len(res.t)
    assert res.soc_hist.shape[0] == nt and res.j_hist.shape[0] == nt
    # the SOC field history at the last step matches the final field
    assert np.allclose(np.nan_to_num(res.soc_hist[-1]), np.nan_to_num(res.soc_field_final))
    assert abs(res.energy_balance["closure_rel"]) < 1e-6


def test_field_helpers_and_plane_plot(tmp_path):
    cfg, res = _cycled(tmp_path)
    for f in ("T", "soc", "current"):
        frames = viz.field_frames(res, f)
        assert frames.shape[0] == len(res.t)
        lo, hi = viz.field_norm(res, f)
        assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi
        fig = viz.plot_plane_field(res, field=f, ti=len(res.t) // 2)
        assert fig is not None
    # without recording, soc/current access raises a clear error
    cfg2 = _cfg("y", width=0.20, height=0.12)
    res2 = coupling.run(cfg2)
    with pytest.raises(ValueError):
        viz.field_frames(res2, "soc")


def test_sandwich_mapping_and_layer_maps(tmp_path):
    cfg, res = _cycled(tmp_path)
    g = res.geom
    total = sum(r.n_stacks for r in g.rolls)
    r0, s0, per = viz.sandwich_to_slice(g, 1)
    rN, sN, _ = viz.sandwich_to_slice(g, total)
    assert r0 == 0 and rN == len(g.rolls) - 1               # first/last land in first/last roll
    assert per > 0
    with pytest.raises(ValueError):
        viz.sandwich_to_slice(g, 0)
    with pytest.raises(ValueError):
        viz.sandwich_to_slice(g, total + 1)
    for layer in ("neg_collector", "cathode_coating"):
        fig = viz.plot_stack_layer_maps(res, sandwich=3, layer=layer, ti=2)
        assert fig is not None
    with pytest.raises(ValueError):
        viz.plot_stack_layer_maps(res, sandwich=3, layer="not_a_layer")


def test_viz3d_views_render(tmp_path):
    cfg, res = _cycled(tmp_path)
    for view in (viz3d.plot_cell_3d, viz3d.plot_cell_cutaway_3d, viz3d.plot_jellyrolls_3d):
        for f in ("T", "soc"):
            fig = view(res, ti=1, field=f)
            assert fig is not None
