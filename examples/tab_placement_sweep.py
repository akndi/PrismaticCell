"""Design study: how tab placement affects hotspot temperature and current uniformity.

Sweeps the positive and negative tab positions along their edge and records, for each layout,
the peak temperature, the in-plane temperature spread, and the current-density non-uniformity.
Demonstrates the intended use of the tool for tab-placement design.

Usage:  python examples/tab_placement_sweep.py [--out DIR]
"""
import argparse
import os

import numpy as np

from prismaticcell.api import Simulation, Sweep
from prismaticcell import viz


def uniformity_reducer(result):
    """Custom scalar outputs for the sweep row."""
    j = result.j_field_final[result.geom.active_mask]
    return {
        "T_max_C": float(result.T_max[-1] - 273.15),
        "dT_K": float((result.T_max[-1] - result.T_min[-1])),
        "j_spread_pct": float(100.0 * (np.nanmax(j) - np.nanmin(j)) / np.nanmean(j)),
        "v_final": float(result.v_terminal[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tab_sweep")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, "..", "configs", "baseline_prismatic.yaml")
    base = Simulation(cfg_path).cfg
    # keep the sweep affordable
    base.mesh.nx, base.mesh.ny, base.mesh.nz = 12, 16, 4
    base.solver.dt = 30.0
    base.solver.t_end = 1800.0
    base.load.value = 2.0   # 2C to accentuate ohmic/tab effects

    # sweep the two tab positions along their (y_max) edge
    grid = {
        "tabs.0.position": [0.15, 0.5, 0.85],
        "tabs.1.position": [0.15, 0.5, 0.85],
    }
    print("Running tab-placement sweep (9 layouts) ...")
    rows = Sweep(base, grid).run(reducer=uniformity_reducer)
    for r in rows:
        print(f"  pos+={r['tabs.0.position']:.2f} pos-={r['tabs.1.position']:.2f} "
              f"-> Tmax={r['T_max_C']:.2f}C dT={r['dT_K']:.2f}K "
              f"j_spread={r['j_spread_pct']:.1f}%")

    viz.plot_sweep_heatmap(rows, x="tabs.0.position", y="tabs.1.position", z="T_max_C",
                           path=os.path.join(args.out, "tab_sweep_Tmax.png"))
    viz.plot_sweep_heatmap(rows, x="tabs.0.position", y="tabs.1.position", z="j_spread_pct",
                           path=os.path.join(args.out, "tab_sweep_jspread.png"))
    print(f"Sweep heatmaps written to {args.out}/")


if __name__ == "__main__":
    main()
