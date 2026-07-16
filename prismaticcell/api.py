"""High-level programmatic API for PrismaticCell.

Thin orchestration over :mod:`prismaticcell.coupling` — no physics lives here. Two
objects are exposed:

* :class:`Simulation` — wrap a single :class:`~prismaticcell.config.SimConfig` (or a
  path to a YAML config) and run it (transient or steady).
* :class:`Sweep` — run the cartesian product of a parameter grid, addressing config
  fields by dotted paths (e.g. ``"cooling.bottom.h"``), collecting scalar outputs.

Heavy solver modules (and matplotlib) are imported lazily so that ``import
prismaticcell.api`` stays cheap and does not hard-require the full solver stack.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import is_dataclass, fields as dc_fields
from typing import Any, Callable, Dict, List, Optional, Union

from .config import SimConfig


# --------------------------------------------------------------------------- #
# Dotted-path access into the SimConfig tree
# --------------------------------------------------------------------------- #
def _descend(obj: Any, key: str) -> Any:
    """Return the child of ``obj`` addressed by a single path segment ``key``.

    Lists/tuples are indexed with ``int(key)``; dicts by string key; everything
    else (dataclass instances / plain objects) by attribute name.
    """
    if isinstance(obj, (list, tuple)):
        return obj[int(key)]
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def _assign(obj: Any, key: str, value: Any) -> None:
    """Assign ``value`` onto ``obj`` at the final path segment ``key``.

    For dataclass targets an unknown field raises (rather than silently attaching an ignored
    attribute) so a mistyped sweep/override path fails loudly instead of being a silent no-op.
    """
    if isinstance(obj, list):
        obj[int(key)] = value
    elif isinstance(obj, dict):
        obj[key] = value
    else:
        if is_dataclass(obj) and key not in {f.name for f in dc_fields(obj)}:
            raise AttributeError(
                f"{type(obj).__name__} has no field '{key}' (dotted-path typo?)")
        setattr(obj, key, value)


def set_dotted(root: Any, path: str, value: Any) -> None:
    """Set ``root`` at a dotted ``path`` to ``value``, in place.

    Supports dataclass attributes, ``dict`` keys, and integer ``list`` indices,
    mixed freely, e.g. ``"assembly.jellyrolls.0.stack.layers.1.thickness"`` or
    ``"cooling.bottom.h"``.
    """
    parts = path.split(".")
    node = root
    for seg in parts[:-1]:
        node = _descend(node, seg)
    _assign(node, parts[-1], value)


def get_dotted(root: Any, path: str) -> Any:
    """Read the value of ``root`` at a dotted ``path`` (mirror of :func:`set_dotted`)."""
    node = root
    for seg in path.split("."):
        node = _descend(node, seg)
    return node


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
class Simulation:
    """A single simulation: a config plus the ability to run it.

    ``cfg`` may be a :class:`~prismaticcell.config.SimConfig` or a path string to a
    YAML config (loaded via :meth:`SimConfig.from_yaml`).
    """

    def __init__(self, cfg: Union[SimConfig, str]):
        if isinstance(cfg, SimConfig):
            self.cfg = cfg
        elif isinstance(cfg, str):
            self.cfg = SimConfig.from_yaml(cfg)
        else:
            raise TypeError(
                f"Simulation expects a SimConfig or path str, got {type(cfg).__name__}"
            )

    def run(self):
        """Run the coupled simulation and return a ``coupling.Result``."""
        from . import coupling  # lazy: keeps solver stack off the import path
        return coupling.run(self.cfg)

    def steady(self):
        """Run with ``solver.mode`` forced to ``"steady"``.

        A deep copy of the config is used so the caller's ``cfg`` is left untouched.
        """
        from . import coupling
        cfg = copy.deepcopy(self.cfg)
        cfg.solver.mode = "steady"
        return coupling.run(cfg)


# --------------------------------------------------------------------------- #
# Default scalar reducer
# --------------------------------------------------------------------------- #
def _last(arr) -> float:
    return float(arr[-1])


def default_scalars(result) -> Dict[str, float]:
    """Extract a flat dict of scalar summary metrics from a ``coupling.Result``.

    Works for both transient (multi-step arrays) and steady (single-element arrays)
    results. ``energy_balance`` always carries a ``closure_rel`` key.
    """
    eb = result.energy_balance
    return {
        "T_max": float(max(result.T_max)),
        "T_mean_final": _last(result.T_mean),
        "dT_max": float(result.T_max[-1] - result.T_min[-1]),
        "v_final": _last(result.v_terminal),
        "soc_final": _last(result.soc_mean),
        "q_total_mean": float(sum(result.q_total) / len(result.q_total)),
        "energy_closure_rel": float(eb.get("closure_rel", float("nan"))),
    }


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
class Sweep:
    """Run the cartesian product of a parameter grid over a base config.

    ``param_grid`` maps dotted config paths to lists of values, e.g.::

        Sweep(cfg, {"cooling.bottom.h": [10, 30], "load.value": [1.0, 2.0]})

    Each combination gets its own deep copy of the base config (runs are fully
    isolated), the dotted paths are set, the sim is run, and a row combining the
    swept parameters with the run's scalar outputs is collected.
    """

    def __init__(self, base_cfg: Union[SimConfig, str], param_grid: Dict[str, List[Any]]):
        if isinstance(base_cfg, str):
            base_cfg = SimConfig.from_yaml(base_cfg)
        elif not isinstance(base_cfg, SimConfig):
            raise TypeError(
                f"Sweep expects a SimConfig or path str, got {type(base_cfg).__name__}"
            )
        self.base_cfg = base_cfg
        self.param_grid = dict(param_grid)

    def combinations(self) -> List[Dict[str, Any]]:
        """Return the list of {path: value} dicts for the cartesian product."""
        if not self.param_grid:
            return [{}]
        keys = list(self.param_grid.keys())
        return [
            dict(zip(keys, combo))
            for combo in itertools.product(*(self.param_grid[k] for k in keys))
        ]

    def run(self, reducer: Optional[Callable[[Any], Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Execute every grid point and return a list of result rows.

        Each row is ``{**swept_params, **scalar_outputs}``. ``reducer`` may be given
        to customize the scalar outputs; it receives the ``coupling.Result`` and must
        return a dict. Defaults to :func:`default_scalars`.
        """
        from . import coupling  # lazy
        reduce_fn = reducer or default_scalars
        rows: List[Dict[str, Any]] = []
        for params in self.combinations():
            cfg = copy.deepcopy(self.base_cfg)
            for path, value in params.items():
                set_dotted(cfg, path, value)
            result = coupling.run(cfg)
            row: Dict[str, Any] = dict(params)
            row.update(reduce_fn(result))
            rows.append(row)
        return rows
