"""Design study: which cooled face (top / bottom / side) controls temperature best.

For a fixed discharge, applies convective cooling to one face at a time (others adiabatic)
and compares peak temperature and in-plane gradient. Directly answers the "cool the top face
vs bottom face vs side face" design question.

Usage:  python examples/cooling_comparison.py [--out DIR]
"""
import argparse
import copy
import os

from prismaticcell.api import Simulation
from prismaticcell.config import FaceBC, Cooling
from prismaticcell import viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/cooling")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, "..", "configs", "baseline_prismatic.yaml")
    base = Simulation(cfg_path).cfg
    base.mesh.nx, base.mesh.ny, base.mesh.nz = 12, 16, 6
    base.solver.dt = 30.0
    base.solver.t_end = 1800.0
    base.load.value = 2.0
    h = 40.0

    faces = ["top", "bottom", "x_min", "y_max"]
    print("Comparing single-face cooling options (2C, h=40 W/m^2/K) ...")
    results = {}
    for face in faces:
        cfg = copy.deepcopy(base)
        cfg.cooling = Cooling()  # all adiabatic
        setattr(cfg.cooling, face, FaceBC("convection", h=h, t_inf=298.15))
        res = Simulation(cfg).run()
        results[face] = res
        print(f"  cool {face:6s}: Tmax={res.T_max[-1]-273.15:6.2f}C  "
              f"Tmean={res.T_mean[-1]-273.15:6.2f}C  peakDT={float((res.T_max-res.T_min).max()):.2f}K")

    # plot the temperature field for the best (lowest Tmax) option
    best = min(results, key=lambda f: results[f].T_max[-1])
    print(f"Lowest peak temperature: cooling the '{best}' face.")
    viz.plot_temperature_slice(results[best],
                               path=os.path.join(args.out, f"best_{best}_Tfield.png"))
    viz.plot_time_series(results[best], os.path.join(args.out, f"best_{best}_timeseries.png"))
    print(f"Plots written to {args.out}/")


if __name__ == "__main__":
    main()
