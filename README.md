# PrismaticCell

A physics-based **distributed thermo-electrochemical model** for LFP (LiFePO₄ / artificial-
graphite) prismatic cells, built for **design exploration**: sweep stack dimensions, enclosure
type/thickness, electrode thicknesses, and tab placement, and read out heat generation, the 3-D
temperature field, thermal gradients/hotspots, and electrochemical performance.

## What it models

```
Prismatic cell (metal can)
 └─ 2 jellyrolls  (parallel to the tabs, thermally coupled through the can)
     └─ each jellyroll = many stacked sandwiches
         └─ sandwich = Cathode-CC | Cathode | Separator | Anode | Anode-CC
Control volumes = full 3-D structured grid → one ECM per active control volume
```

- **Distributed ECM network** — every active control volume runs its own equivalent-circuit
  model (OCV, R0, RC pairs, entropy) at its *local* SOC and temperature. CVs are networked
  through the Al/Cu **current-collector potential fields**, so **tab placement, electrode
  thickness and stack size actually redistribute current, SOC and heat**.
- **3-D anisotropic thermal solver** — finite-volume, transient (implicit) and steady-state,
  with independent per-face boundary conditions (convection / fixed-temperature / fixed-flux /
  adiabatic) on the top, bottom and four side faces.
- **Two-way coupling** — Bernardi heat generation (irreversible + reversible + ohmic) feeds the
  thermal solver; the temperature field feeds back into the (Arrhenius) ECM parameters. A real
  closed loop, verified by an energy-conservation test.

Governing equations: [`docs/PHYSICS.md`](docs/PHYSICS.md).

## Install & run

```bash
pip install -e .            # numpy, scipy, matplotlib, pyyaml
prismaticcell run configs/baseline_prismatic.yaml          # run a simulation
python -m pytest                                           # validation suite
python examples/discharge_baseline.py                      # example + plots
python examples/tab_placement_sweep.py                     # design sweep + plots
```

## Layout

| Path | Purpose |
|------|---------|
| `prismaticcell/config.py`     | typed configuration schema (the shared contract) |
| `prismaticcell/materials.py`  | material DB + thermal homogenization |
| `prismaticcell/geometry.py`   | 3-D mesh, region map, tabs, enclosure |
| `prismaticcell/echem.py`      | per-CV ECM (OCV, R0, RC, entropy, SOC, Bernardi heat) |
| `prismaticcell/distributed.py`| collector-potential network + current distribution solve |
| `prismaticcell/thermal.py`    | 3-D anisotropic FVM (steady + transient) |
| `prismaticcell/coupling.py`   | two-way co-simulation driver + energy accounting |
| `prismaticcell/api.py`        | `Simulation` / `Sweep` high-level objects |
| `prismaticcell/viz.py`        | field slices, hotspot maps, time series, sweep heatmaps |
| `configs/`, `data/`           | example configs and (replaceable) LFP data tables |
| `tests/`, `examples/`         | analytical validations and runnable examples |

## Example results

Baseline 1C discharge (`examples/discharge_baseline.py`) — LFP voltage plateau with end knee,
exact 1C SOC ramp, and the characteristic double-hump temperature (hot at the SOC extremes where
resistance is high, cool on the flat plateau); global energy closes to ~1e-12:

![time series](docs/figures/baseline_time_series.png)

In-plane temperature field with the hotspot marked, and the areal current-density map showing
current concentrating toward the tabs:

![temperature field](docs/figures/baseline_temperature_slice.png)
![current distribution](docs/figures/baseline_current_distribution.png)

Design studies find real trade-offs: cooling a large face (top/bottom) holds ~45 °C vs ~80 °C for
a side face at 2C (`examples/cooling_comparison.py`); and placing the two tabs at opposite ends of
the edge minimizes current-density non-uniformity (`examples/tab_placement_sweep.py`).

![tab sweep](docs/figures/tab_sweep_current_spread.png)

## Validation

`python -m pytest` runs 20 checks (PHYSICS §7): 1-D steady conduction vs closed form, lumped-
capacitance transient, adiabatic-pulse and steady energy conservation, symmetric-cooling symmetry,
SOC Coulomb balance, Arrhenius direction, entropy sign, charge conservation, discharge-below-OCV,
the single-roll lumped limit, and end-to-end two-way-coupling energy closure.

Parameters and data tables are **representative** starting points — replace `data/*.csv` and
`configs/*.yaml` with your cell's characterization for quantitative design work.
