"""PrismaticCell -- physics-based thermo-electrochemical model for LFP prismatic cells.

A distributed electro-thermal simulator: a 3-D control-volume grid over a prismatic
cell (can + two jellyrolls of stacked Cathode-CC | Cathode | Sep | Anode | Anode-CC
sandwiches). Every active control volume carries its own equivalent-circuit model,
networked through the current-collector potential fields and two-way coupled to a
3-D anisotropic thermal solver (transient & steady state).

Public API is assembled in :mod:`prismaticcell.api`; the configuration schema lives
in :mod:`prismaticcell.config`.
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import config as config  # noqa: F401
from .config import SimConfig  # noqa: F401

__all__ = ["config", "SimConfig", "__version__"]
