"""Baseline 1C discharge of the LFP prismatic cell, with plots.

Runs the full two-way coupled model on configs/baseline_prismatic.yaml and writes:
  - time_series.png          (V, I, SOC, temperature vs time)
  - temperature_slice.png    (in-plane T field at mid-plane, hotspot marked)
  - current_distribution.png (areal current density map -- tab effect)

Usage:  python examples/discharge_baseline.py [--out DIR] [--fine]
"""
import argparse
import os

from prismaticcell.api import Simulation
from prismaticcell import viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/baseline")
    ap.add_argument("--fine", action="store_true", help="use the full config mesh (slower)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, "..", "configs", "baseline_prismatic.yaml")
    sim = Simulation(cfg_path)
    if not args.fine:
        sim.cfg.mesh.nx, sim.cfg.mesh.ny, sim.cfg.mesh.nz = 12, 16, 6
        sim.cfg.solver.dt = 15.0

    print(f"Running baseline discharge ({sim.cfg.load.value}C) ...")
    res = sim.run()
    print(f"  final V={res.v_terminal[-1]:.3f} V, SOC={res.soc_mean[-1]:.3f}, "
          f"Tmax={res.T_max[-1]-273.15:.2f} C, "
          f"peak ΔT={float((res.T_max-res.T_min).max()):.2f} K")
    print(f"  energy closure = {res.energy_balance['closure_rel']:.2e}")

    viz.plot_time_series(res, os.path.join(args.out, "time_series.png"))
    viz.plot_temperature_slice(res, path=os.path.join(args.out, "temperature_slice.png"))
    viz.plot_current_distribution(res, path=os.path.join(args.out, "current_distribution.png"))
    print(f"Plots written to {args.out}/")


if __name__ == "__main__":
    main()
