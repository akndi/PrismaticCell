"""Cycling protocol builder: constant-C discharge/charge over an SOC window for N cycles.

At constant current the per-CV coulomb counting makes the mean SOC exactly linear in time, so a
CC/CC cycling protocol reduces to a piecewise-constant current profile:

    segment time = (soc_max - soc_min) * 3600 / C_rate   [s]

``configure_cycling(cfg, ...)`` builds that profile (a CSV the existing ``Load(kind="profile")``
machinery consumes), sets ``soc_init = soc_max``, ``solver.t_end`` to cover all cycles, and turns
on field recording for animation. Nothing is hard-coded: the current is derived from the config's
own capacity and the requested C-rate.

Note the SOC window is tracked open-loop (time-based). Non-uniform SOC across the cell makes
individual CVs deviate slightly from the window edges (a real, physical effect the plots show);
the MEAN SOC hits the window exactly.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .config import SimConfig, Load


def build_cycle_profile(capacity_ah: float, c_rate: float, n_cycles: int,
                        soc_min: float, soc_max: float,
                        start: str = "discharge") -> np.ndarray:
    """Return the (time [s], current [A]) breakpoints of a CC/CC cycling profile.

    Positive current = discharge (model convention). The profile is piecewise constant;
    each half-cycle spans the SOC window at the given C-rate.
    """
    if not (0.0 <= soc_min < soc_max <= 1.0):
        raise ValueError(f"need 0 <= soc_min < soc_max <= 1, got [{soc_min}, {soc_max}]")
    if c_rate <= 0 or n_cycles < 1:
        raise ValueError("c_rate must be > 0 and n_cycles >= 1")
    amps = c_rate * capacity_ah
    seg = (soc_max - soc_min) * 3600.0 / c_rate          # s per half-cycle
    sign0 = 1.0 if start == "discharge" else -1.0
    # STEP-shaped breakpoints: the consumer interpolates linearly (np.interp), so each segment
    # gets a breakpoint at its start AND just before its end — the current switches sign in an
    # instant instead of ramping through zero across the whole half-cycle.
    eps = 1e-6                                            # s, switch sharpness
    rows = []
    t = 0.0
    for _cyc in range(n_cycles):
        for half in range(2):
            sign = sign0 * (1.0 if half == 0 else -1.0)
            rows.append([t, sign * amps])
            rows.append([t + seg - eps, sign * amps])
            t += seg
    rows.append([t, 0.0])                                 # end marker
    return np.array(rows)


def configure_cycling(cfg: SimConfig, c_rate: float = 1.0, n_cycles: int = 1,
                      soc_min: float = 0.1, soc_max: float = 0.9,
                      start: str = "discharge", dt: Optional[float] = None,
                      profile_path: Optional[str] = None,
                      save_fields: bool = True) -> SimConfig:
    """Configure ``cfg`` in place for a CC/CC cycling run; returns it for chaining.

    Writes the profile CSV next to ``cfg.data_root`` (or to ``profile_path``), points the load
    at it, starts the cell at ``soc_max`` (or ``soc_min`` when charging first), sets ``t_end``
    to the protocol length, and (by default) enables field-history recording for animations.
    """
    prof = build_cycle_profile(cfg.ecm.capacity_Ah, c_rate, n_cycles, soc_min, soc_max, start)
    path = profile_path or os.path.join(cfg.data_root, "_cycling_profile.csv")
    with open(path, "w") as fh:
        fh.write("time_s,current_A\n")
        for t, a in prof:
            fh.write(f"{t:.6f},{a:.6f}\n")
    cfg.load = Load(kind="profile", profile_csv=path, profile_units="A")
    cfg.soc_init = soc_max if start == "discharge" else soc_min
    cfg.solver.mode = "transient"
    cfg.solver.t_end = float(prof[-1, 0])
    if dt is not None:
        cfg.solver.dt = float(dt)
    cfg.solver.save_fields = save_fields
    return cfg
