"""Command-line interface for PrismaticCell.

Entry point ``main`` (wired to the ``prismaticcell`` console script in
``pyproject.toml``). Currently one subcommand::

    prismaticcell run CONFIG [--steady] [--out DIR]

Loads a config, runs the coupled simulation (steady or transient), prints a concise
summary, and — when ``--out`` is given — writes the time series (CSV, always) plus
time-series and temperature-slice plots (via :mod:`prismaticcell.viz`, imported
lazily; if unavailable, the CSV alone is written).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prismaticcell",
        description="Distributed thermo-electrochemical simulator for prismatic cells.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a simulation from a config file")
    run_p.add_argument("config", help="path to a YAML SimConfig file")
    run_p.add_argument(
        "--steady", action="store_true",
        help="run a steady-state operating point instead of transient",
    )
    run_p.add_argument(
        "--out", metavar="DIR", default=None,
        help="directory to write time-series CSV and plots into",
    )
    run_p.set_defaults(func=_cmd_run)
    return parser


def _print_summary(result, steady: bool) -> None:
    eb = result.energy_balance
    closure = eb.get("closure_rel", float("nan"))
    v_final = float(result.v_terminal[-1])
    t_mean = float(result.T_mean[-1])
    t_max = float(max(result.T_max))
    dT = float(result.T_max[-1] - result.T_min[-1])
    mode = "steady" if steady else "transient"
    print(f"PrismaticCell run summary ({mode})")
    print(f"  terminal voltage (final): {v_final:.4f} V")
    print(f"  temperature  mean/max:    {t_mean - 273.15:.2f} / "
          f"{t_max - 273.15:.2f} degC  ({t_mean:.2f} / {t_max:.2f} K)")
    print(f"  spatial dT (final):       {dT:.3f} K")
    print(f"  final mean SOC:           {float(result.soc_mean[-1]):.4f}")
    print(f"  energy-balance closure:   {closure:.3e} (relative residual)")


def _write_csv(result, path: str) -> None:
    import csv
    cols = [
        ("t", result.t), ("v_terminal", result.v_terminal),
        ("i_terminal", result.i_terminal), ("soc_mean", result.soc_mean),
        ("T_mean", result.T_mean), ("T_max", result.T_max),
        ("T_min", result.T_min), ("q_total", result.q_total),
    ]
    names = [name for name, _ in cols]
    nrows = len(result.t)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for i in range(nrows):
            writer.writerow([f"{float(arr[i]):.8g}" for _, arr in cols])


def _save_outputs(result, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "time_series.csv")
    _write_csv(result, csv_path)
    print(f"  wrote time series: {csv_path}")
    try:
        from . import viz  # lazy: matplotlib not required unless plotting
    except Exception as exc:  # noqa: BLE001 - viz optional (missing module or matplotlib)
        print(f"  (plots skipped: prismaticcell.viz unavailable: {exc})")
        return
    ts_path = os.path.join(out_dir, "time_series.png")
    slice_path = os.path.join(out_dir, "temperature_slice.png")
    viz.plot_time_series(result, path=ts_path)
    viz.plot_temperature_slice(result, path=slice_path)
    print(f"  wrote plots: {ts_path}, {slice_path}")


def _cmd_run(args: argparse.Namespace) -> int:
    from .api import Simulation  # lazy

    sim = Simulation(args.config)
    result = sim.steady() if args.steady else sim.run()
    _print_summary(result, steady=args.steady)
    if args.out:
        _save_outputs(result, args.out)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns 0 on success, non-zero on error."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse exits on --help (code 0) and on parse errors (code 2);
        # surface the code as a return value instead of propagating.
        return int(exc.code or 0)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - top-level guard: report and fail cleanly
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
