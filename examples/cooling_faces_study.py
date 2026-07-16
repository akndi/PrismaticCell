"""Cool each of the 6 cell faces in turn and report the steady temperatures.

Demonstrates that boundary conditions are per-face configurable data (nothing is hard-coded):
the same cell is run once per face with that face set to convection and the others adiabatic,
plus an "all six faces" case. Prints a table of hotspot / mean / gradient so the effect of the
cooling choice is visible.

Run:  python examples/cooling_faces_study.py
"""
import os
import copy

from prismaticcell.config import SimConfig, FaceBC, Cooling
from prismaticcell import coupling

HERE = os.path.dirname(__file__)
CFG = os.path.join(HERE, "..", "configs", "large_prismatic.yaml")

# The 6 physical faces and a human label (for this stack_axis='y' cell).
FACES = [
    ("y_max", "front (big XZ)"),
    ("y_min", "back  (big XZ)"),
    ("x_min", "left  (end YZ)"),
    ("x_max", "right (end YZ)"),
    ("top",   "top   (XY)"),
    ("bottom", "bottom(XY)"),
]
H = 25.0
T_INF = 298.15


def run_case(cool: Cooling) -> dict:
    cfg = SimConfig.from_yaml(CFG)
    cfg.solver.mode = "steady"
    cfg.cooling = cool
    res = coupling.run(cfg)
    return dict(tmax=res.T_max[-1], tmean=res.T_mean[-1], tmin=res.T_min[-1])


def one_face(face: str) -> Cooling:
    c = Cooling()                                   # every face defaults to adiabatic...
    setattr(c, face, FaceBC("convection", h=H, t_inf=T_INF))   # ...set just this one to convection
    return c


def all_faces() -> Cooling:
    c = Cooling()
    for f, _ in FACES:
        setattr(c, f, FaceBC("convection", h=H, t_inf=T_INF))
    return c


def main():
    print(f"Cell: configs/large_prismatic.yaml  ·  convection h={H} W/m^2K, T_inf={T_INF} K\n")
    print(f"{'cooled face':22s} {'T_max[C]':>9s} {'T_mean[C]':>10s} {'ΔT (max-min)[K]':>16s}")
    print("-" * 60)
    for face, label in FACES:
        r = run_case(one_face(face))
        print(f"{label:22s} {r['tmax']-273.15:9.2f} {r['tmean']-273.15:10.2f} "
              f"{r['tmax']-r['tmin']:16.2f}")
    r = run_case(all_faces())
    print("-" * 60)
    print(f"{'ALL six faces':22s} {r['tmax']-273.15:9.2f} {r['tmean']-273.15:10.2f} "
          f"{r['tmax']-r['tmin']:16.2f}")


if __name__ == "__main__":
    main()
