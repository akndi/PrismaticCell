# Data tables

All electrochemical dependencies are **data-driven** — no coefficients are baked into solver
code. Each CSV is a lookup table interpolated by the ECM module. Lines starting with `#` are
comments; the first non-comment line is a header.

| File | Columns | Meaning |
|------|---------|---------|
| `lfp_ocv.csv`     | `soc, ocv`   | open-circuit voltage [V] vs SOC [-] |
| `lfp_entropy.csv` | `soc, dUdT`  | entropic coefficient [V/K] vs SOC |
| `lfp_r0.csv`      | `soc, r0`    | areal ohmic resistance [Ω·m²] vs SOC at `t_ref` |
| `lfp_rc1_r.csv`   | `soc, r1`    | areal RC-pair resistance [Ω·m²] vs SOC at `t_ref` |
| `lfp_rc1_c.csv`   | `soc, c1`    | areal RC-pair capacitance [F/m²] vs SOC at `t_ref` |

**Areal convention:** resistances are area-specific (Ω·m²); multiply by local current density
(A/m²) to get a voltage. This lets one table serve every control volume regardless of mesh.

Temperature dependence is applied on top of these `t_ref` tables via Arrhenius scaling using the
activation energies in the config (`ea_r0`, `ea_r`, …). Values here are **representative**
published-order LFP/graphite numbers for wiring up and testing the model — replace them with your
own cell's characterization data.
