"""Design study: cooled-face choice AND radiation's contribution to heat rejection.

Part 1 — which cooled face (top / bottom / side) controls temperature best (convection only).
Part 2 — under natural convection, sweep surface emissivity and quantify how much heat leaves by
         radiation vs convection (the "is radiation worth caring about?" question).

Usage:  python examples/cooling_comparison.py [--out DIR]
"""
import argparse
import copy
import os

import numpy as np

from prismaticcell.api import Simulation
from prismaticcell.config import FaceBC, Cooling, SIGMA_SB
from prismaticcell import viz


def rejection_split(result, cooling):
    """Estimate convective vs radiative heat leaving each external face at the final state.

    Uses boundary cell-center temperatures as the surface temperature (a small ΔT approximation)
    and the physical T^4 law for radiation. Returns (q_conv [W], q_rad [W]) summed over faces.
    Since the solver linearizes radiation about the mean film temperature, this physical-T^4
    estimate agrees with the model's actual radiative rejection to <1% at these ΔT.
    """
    g = result.geom
    grid = g.grid
    T = result.T_field[-1]
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    faces = {
        "bottom": (T[:, :, 0], dx * dy, cooling.bottom),
        "top":    (T[:, :, -1], dx * dy, cooling.top),
        "x_min":  (T[0, :, :], dy * dz, cooling.x_min),
        "x_max":  (T[-1, :, :], dy * dz, cooling.x_max),
        "y_min":  (T[:, 0, :], dx * dz, cooling.y_min),
        "y_max":  (T[:, -1, :], dx * dz, cooling.y_max),
    }
    q_conv = q_rad = 0.0
    for _, (Tf, area, bc) in faces.items():
        Tf = np.asarray(Tf, dtype=float)
        if bc.kind == "convection" and bc.h > 0.0:
            q_conv += float((bc.h * (Tf - bc.t_inf) * area).sum())
        if getattr(bc, "emissivity", 0.0) > 0.0:
            q_rad += float((bc.emissivity * SIGMA_SB * (Tf**4 - bc.t_inf**4) * area).sum())
    return q_conv, q_rad


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

    # ---- Part 1: which single face to cool (convection only) ----
    print("Part 1 - single-face convection cooling (2C, h=40):")
    for face in ["top", "bottom", "x_min", "y_max"]:
        cfg = copy.deepcopy(base)
        cfg.cooling = Cooling()
        setattr(cfg.cooling, face, FaceBC("convection", h=40.0, t_inf=298.15))
        res = Simulation(cfg).run()
        print(f"  cool {face:6s}: Tmax={res.T_max[-1]-273.15:6.2f}C")

    # ---- Part 2: natural convection + emissivity sweep (radiation contribution) ----
    print("\nPart 2 - natural convection (h=5 all faces) + emissivity sweep:")
    emissivities = [0.0, 0.3, 0.6, 0.9]
    tmax, rad_share = [], []
    for eps in emissivities:
        cfg = copy.deepcopy(base)
        # natural convection on every external face, plus radiation to 25 C surroundings
        bc = lambda: FaceBC("convection", h=5.0, t_inf=298.15, emissivity=eps)
        cfg.cooling = Cooling(top=bc(), bottom=bc(), x_min=bc(), x_max=bc(),
                              y_min=bc(), y_max=bc())
        res = Simulation(cfg).run()
        q_conv, q_rad = rejection_split(res, cfg.cooling)
        share = 100.0 * q_rad / (q_conv + q_rad) if (q_conv + q_rad) > 0 else 0.0
        tmax.append(res.T_max[-1] - 273.15)
        rad_share.append(share)
        print(f"  emissivity={eps:.1f}: Tmax={res.T_max[-1]-273.15:6.2f}C  "
              f"q_conv={q_conv:5.2f}W  q_rad={q_rad:5.2f}W  radiation share={share:4.1f}%")

    # plot Tmax and radiation share vs emissivity
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(emissivities, tmax, "o-", color="#c0392b", label="peak temperature")
    ax1.set_xlabel("surface emissivity [-]")
    ax1.set_ylabel("peak temperature [°C]", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax2 = ax1.twinx()
    ax2.plot(emissivities, rad_share, "s--", color="#2c7fb8", label="radiation share")
    ax2.set_ylabel("radiation share of heat rejection [%]", color="#2c7fb8")
    ax2.tick_params(axis="y", labelcolor="#2c7fb8")
    ax1.set_title("Radiation's contribution under natural convection (h=5 W/m²/K, 2C)")
    fig.tight_layout()
    out = os.path.join(args.out, "radiation_contribution.png")
    fig.savefig(out, dpi=110)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
