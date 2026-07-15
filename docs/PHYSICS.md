# PrismaticCell — Governing Physics & Numerics

This document is the **equation contract**. Every solver module implements these equations and
nothing may be hard-coded: all coefficients come from the material database / config / data
tables. All quantities are SI unless noted.

---

## 0. Geometry hierarchy

```
Prismatic cell = metal can (enclosure)
  └─ 2 jellyrolls (electrically in parallel to the cell tabs, thermally coupled via can + gap)
       └─ each jellyroll = n_stacks electrode sandwiches through the thickness
            └─ each sandwich = CathodeCC | Cathode | Separator | Anode | AnodeCC
```

The cell volume is discretized into a **3-D structured control-volume grid** (`nx × ny × nz`).
`x,y` are in-plane; `z` runs through the sandwich-stacking direction. Each CV is tagged with a
**region** (jellyroll-active / inter-roll gap / can wall / tab / headspace) and inherits that
region's material properties. Every *electrochemically active* CV carries its own ECM (§2). The
two jellyrolls are separate active regions summed in parallel at the terminal constraint (§3.2).

## 1. Thermal homogenization (per active CV)

One sandwich (from config) is an ordered list of layers, each with a `role`, `material`, and
`thickness` tᵢ:

```
pos_collector (Al) | cathode_coating (LFP) | separator | anode_coating (graphite) | neg_collector (Cu)
```

For a homogenized thermal control volume spanning the stack, effective properties are computed
from the layer set — **never assigned literals**:

- Through-plane (z, series / harmonic):  `k_z = (Σ tᵢ) / Σ (tᵢ / kᵢ)`
- In-plane (x,y, parallel / thickness-weighted): `k_xy = Σ (tᵢ kᵢ) / Σ tᵢ`
- Volumetric heat capacity: `(ρ cp)_eff = Σ (tᵢ ρᵢ cpᵢ) / Σ tᵢ`
- Density: `ρ_eff = Σ (tᵢ ρᵢ) / Σ tᵢ`

Current collectors dominate `k_xy` (metal); contact/coating layers dominate the low `k_z`.

Active electrode area `A = width × height`. Number of repeat units `N`. Total stack thickness
`H = N · Σ tᵢ`. Tabs and enclosure are separate conductive regions with their own material
properties and contact conductances.

---

## 2. Distributed equivalent-circuit network (one ECM per control volume)

The active jellyroll volume is discretized into 3-D control volumes (CVs). **Each active CV k
has its own ECM** at local state of charge `zₖ` and local temperature `Tₖ`. Index `k` ranges
over all active CVs in *both* jellyrolls; `z` (through-thickness) indexes the sandwich stacks,
so utilization varies through the roll as well as in-plane.

### 2.1 Single-CV ECM (Thévenin, m RC pairs)

Local through-stack cell voltage across CV k:

```
v_k = U_ocv(z_k, T_k) − i_k R0(z_k, T_k) − Σ_p u_{k,p}
```

with each RC-pair overpotential state evolving as

```
du_{k,p}/dt = − u_{k,p} / (R_p C_p) + i_k / C_p
```

`i_k` is the through-stack current of CV k (A), positive on discharge. `A_cv = A / (nx·ny)`.

### 2.2 Temperature dependence (Arrhenius — coefficients from config)

```
R(z,T) = R_ref(z) · exp[ (E_a/R_gas) · (1/T − 1/T_ref) ]
```

applied to R0 and each Rp (own `E_a`). Capacity `Q(T)` and RC time constants likewise from
config-supplied tables/coefficients. `U_ocv` from the OCV table; entropic term below.

### 2.3 State of charge (per CV)

```
dz_k/dt = − i_k / (3600 · Q_cv(T_k))            (i_k in A, Q_cv in Ah)
```

`Q_cv` is the capacity of one CV (Ah), derived from total capacity / (nx·ny). SOC diverges
CV-to-CV → non-uniform utilization (near-tab vs far-tab) is a first-class output.

---

## 3. Current-collector coupling (what makes tab placement matter)

Per jellyroll, two in-plane potential fields, one per foil: `φ⁺` on the positive (Al) collector,
`φ⁻` on the negative (Cu) collector. Because the many parallel sandwiches of a roll share common
tabs, the collector fields are solved in-plane (x,y) per roll and the through-thickness sandwiches
draw from them in parallel; each 3-D CV keeps its own SOC/thermal/RC state. Sheet-charge
conservation on each foil (2D Poisson with the local through-stack current as source/sink):

```
∇·(σ⁺ t⁺ ∇φ⁺) = + j_k          (current leaves the + foil into the stack)
∇·(σ⁻ t⁻ ∇φ⁻) = − j_k          (current returns to the − foil)
```

where `j_k = i_k / A_cv` is the local areal current density (A/m²), `σ` foil electronic
conductivity, `t` foil thickness. Tabs are **boundary current terminals** at their parametric
edge positions.

### 3.1 Local closure (couples foils ↔ ECM)

The stack voltage seen locally equals the foil potential difference:

```
φ⁺_k − φ⁻_k = v_k = U_ocv(z_k,T_k) − i_k R0(z_k,T_k) − Σ_p u_{k,p}
```

### 3.2 Terminal constraint

- **Galvanostatic:** `Σ_k i_k = I_app(t)` with a common tab potential (unknown terminal voltage).
- **Potentiostatic:** tab potential difference fixed = `V_app(t)`; total current is an output.

Assembled each timestep as a coupled sparse system (two foil Laplacians + per-CV I–V relations +
terminal constraint), solved by Newton / fixed-point because `U_ocv`, `R` depend on state.
Mesh = 1×1 reduces exactly to a single lumped ECM (validation limit).

**Collector fidelity (`solver.collector_model`):**
- `planar` (default, 2.5-D): one shared `φ⁺,φ⁻` per jellyroll; all through-thickness stack layers
  in a column share `Δφ(x,y)` but each 3-D CV still carries its own SOC/T/RC state and current
  `j_k=(U_ocv−Σu−Δφ)/R0(T_k)`, so through-thickness current variation from `R0(T)` is captured.
- `layered` (full 3-D collector): each stack layer `k` gets its **own** `φ⁺[k],φ⁻[k]` (2-D each),
  carrying `1/n_layers` of the foil sheet conductance, all joined **in parallel at the tabs**. This
  additionally resolves the through-thickness *potential* gradient (foil IR differing layer-to-
  layer). Reduces to `planar` in the limit of highly conductive foils. Charge is conserved in both.

---

## 4. Heat generation (per CV → thermal source map)

Bernardi decomposition, per CV, plus foil ohmic and contact heating:

```
q_ecm,k  = i_k (U_ocv,k − v_k)              [irreversible overpotential heat; = i²R over a cycle]
q_rev,k  = − i_k T_k (dU/dT)(z_k)           [reversible/entropic; leading minus is the Bernardi sign]
q_ohm,k  = σ⁺ t⁺ |∇φ⁺_k|² + σ⁻ t⁻ |∇φ⁻_k|²  [foil Joule heat, ≥ 0]
```

Full Bernardi form: `Q = I(U_ocv − V) − I·T·(dU/dT)`, with `I` discharge-positive. The entropic
coefficient `dU/dT` is stored directly in the entropy table, so the minus sign lives in the
equation. (For LFP mid-SOC, `dU/dT < 0`, so discharge is mildly exothermic reversibly.)

Total CV heat `Q_k = q_ecm,k + q_rev,k + q_ohm,k` (W), converted to a volumetric density
`q'''_k = Q_k / V_cv` and injected into the thermal solver. Tab and contact-resistance Joule
heating are added at their regions.

`dU/dT` is the entropic coefficient from the entropy table. For LFP it is small and sign-changing
across SOC — captured by the table, not assumed.

---

## 5. Thermal solver (3D anisotropic finite volume)

Energy conservation on the structured `nx × ny × nz` grid:

```
(ρ cp)  ∂T/∂t = ∇·( K ∇T ) + q'''
```

with anisotropic conductivity tensor `K = diag(k_x, k_y, k_z)` per cell (from §1 and the region
map: stack / collector / tab / enclosure). Face conductances between neighbor cells use the
harmonic mean of the two cells' directional conductivities (series resistance).

### 5.1 Boundary conditions (independent per face: top, bottom, 4 sides)

- **Convection:** `−k ∂T/∂n = h (T_s − T_∞)`
- **Dirichlet:** `T_s = T_wall` (constant-temperature cooling)
- **Neumann:** `−k ∂T/∂n = q″` (fixed flux; `q″=0` → adiabatic)
- **Radiation** (any non-Dirichlet face with `emissivity > 0`): `q_rad = εσ(T_s⁴ − T_∞⁴)`,
  linearized about the sink as `h_rad = 4εσT_∞³` and added **in parallel** with the convective
  film (both in series with the half-cell conduction). Valid for moderate ΔT; for large ΔT the
  coefficient can be iterated on the surface temperature.

Heat-transfer modes covered: **conduction** (3-D anisotropic, everywhere), **convection**
(external Newton cooling; internal gaps are effective-conduction media, not resolved fluid flow),
and **radiation** (external faces, linearized Stefan–Boltzmann).

### 5.2 Steady state

Assemble sparse `A T = b` (conductances + BC), solve directly/iteratively.

### 5.3 Transient

Backward-Euler (unconditionally stable): `(M/Δt + A) Tⁿ⁺¹ = (M/Δt) Tⁿ + bⁿ⁺¹`, `M = (ρ cp) V`.
Energy is conserved to solver tolerance (checked in tests).

---

## 6. Two-way coupling (co-simulation loop)

Per timestep, staggered with optional sub-iteration to convergence:

```
1. Given Tⁿ, zⁿ, uⁿ  →  solve §3 network for i_k, φ, terminal V   (electro)
2. Form heat map §4  →  q'''ⁿ
3. Advance thermal §5.3 with q'''ⁿ  →  Tⁿ⁺¹
4. Advance z, u (§2.1, §2.3) with i_k  →  zⁿ⁺¹, uⁿ⁺¹
5. Re-evaluate T-dependent params (Arrhenius) at Tⁿ⁺¹ for next step
```

The loop is genuinely bidirectional: heat comes from the electro state; the electro parameters
(R, Q, dynamics) depend on the thermal field. Global energy balance
`∫ Σ Q_k dt = ΔU_thermal + Q_removed_at_boundaries` is asserted in tests.

---

## 7. Validation targets

| Test | Expectation |
|------|-------------|
| 1-D steady conduction, fixed T both ends | linear profile, flux = kΑΔT/L |
| Lumped capacitance transient (uniform, convective) | `T(t)=T∞+(T0−T∞)e^{−hA t/(mcp)}` |
| Adiabatic pulse | `ΔT = Σ Q Δt /(m cp)` |
| Symmetric cooling | symmetric T field |
| Single-CV network | reduces to lumped ECM V(t) |
| SOC balance | `∫ I dt / 3600 = ΔSOC · Q` |
| Energy conservation | gen = stored + removed, within tol |
| Cooling ↑ (h↑) | steady T ↓ and ECM params shift correctly (coupling closes) |
