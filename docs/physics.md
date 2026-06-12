# Physics of the POP Ocean Emulator

This document states the POP2 primitive-equation set that the emulator targets,
and describes precisely how the learned update is constrained to respect the
ocean model's physics and conservation laws. The aim is *physical
faithfulness*: the emulator is a learned discrete time-integrator that lives on
the same `gx1v6` grid and obeys the same budgets as POP, not a generic regressor.

All quantities use POP's native **CGS** units (cm, g, s, °C, ergs) so that the
physical constants stored in the dataset can be used directly. Unit conversions
are confined to `constants.py` and the diagnostics.

---

## 1. The POP primitive equations

POP solves the hydrostatic, Boussinesq primitive equations. With horizontal
velocity **u** = (u, v), vertical velocity w, potential temperature T,
salinity S, in-situ density ρ, pressure p, and Coriolis parameter f:

**Momentum** (horizontal)
```
∂u/∂t + (u·∇)u + w ∂u/∂z + f k×u = -(1/ρ0) ∇_h p + ∇·(A_h ∇u) + ∂/∂z (A_v ∂u/∂z)
```

**Hydrostatic balance**
```
∂p/∂z = -ρ g
```

**Continuity (Boussinesq incompressibility)**
```
∇_h · u + ∂w/∂z = 0
```

**Tracer transport** (for C ∈ {T, S})
```
∂C/∂t + ∇_h·(u C) + ∂(w C)/∂z = ∇·(κ_h ∇C) + ∂/∂z(κ_v ∂C/∂z) + Q_C
```

**Equation of state** (McDougall, Wright, Jackett & Feistel 2003; "MWJF")
```
ρ = ρ(T, S, p)
```

**Surface boundary conditions** (the model forcing)
```
A_v ∂u/∂z |_{z=0} = τ / ρ0           (wind stress τ = (TAUX, TAUY))
κ_v ∂T/∂z |_{z=0} = Q_heat / (ρ0 c_p) (surface heat flux SHF, incl. SHF_QSW)
κ_v ∂S/∂z |_{z=0} = F_salt           (virtual salt flux from SFWF)
```

These boundary fluxes are precisely the fields supplied to the emulator as
forcing inputs, and they are what closes the conservation budgets below.

---

## 2. The emulator as a discrete integrator

Let `x = (T, S, u, v, η)` be the prognostic state stacked over the 60 vertical
levels (η is 2-D). The network predicts a **tendency / residual**, not the next
state directly:

```
x(t+Δt) = x(t) + Δt · F_theta( x(t), b(t), g )
```

where `b(t)` is the surface forcing `(SHF, SHF_QSW, SFWF, TAUX, TAUY)` and `g`
are static grid fields (Coriolis f, depth/`KMT` mask, layer thickness `dz`,
cell area `TAREA`, land mask, latitude/longitude encodings).

Predicting the tendency mirrors POP's own time-stepping (a forward update of a
prognostic field) and gives the network an exact identity at Δt→0, which makes
multi-step rollout far more stable than direct state-to-state prediction.

The vertical velocity is **never learned**. After each update it is *diagnosed*
from the continuity equation (Section 4), so incompressibility holds by
construction rather than being hoped for.

---

## 3. Grid, staggering, and masking

- **Horizontal:** B-grid. Tracers (T, S) at T-cell centers; velocities (u, v)
  at U-cell corners (NE corner of the T-cell); w at layer top faces.
- **Vertical:** 60 z-levels with thicknesses `dz(k)`; partial bottom topography
  encoded by `KMT(i,j)` = index of the deepest active cell in each column.
- **Masks:** the 3-D ocean mask is `mask3d(k,i,j) = (k < KMT(i,j))`. Land cells
  carry POP's fill value in the raw data; the loader replaces them with zeros
  and every loss term multiplies by `mask3d` so land never contributes.
- **Boundaries:** longitude is periodic → **circular** padding in x. The
  northern boundary is a tripole fold; we approximate it with **replicate**
  padding in y. This is the one place the emulator is not bit-faithful to POP's
  grid topology, and it only affects a few rows adjacent to the Arctic fold.
- **Coriolis:** `f = 2 Ω sin(φ)` with Ω = `omega` from the dataset and φ =
  `TLAT`. Supplied as a static input channel so the network can represent
  geostrophic/Coriolis dynamics.

---

## 4. Incompressibility: diagnosing w from continuity

Integrating `∇_h·u + ∂w/∂z = 0` upward from the (rigid) bottom with `w=0` at
the floor gives, for each column,

```
w(k) = w(k+1) - dz(k) · (∇_h · u)_k
```

`physics.diagnose_wvel` implements this with the B-grid divergence operator
(`grid.divergence_ub`), using the proper cell-edge lengths (`HTN`, `HTE`) and
T-cell area (`TAREA`). Because w is reconstructed from the predicted (u, v),
the discrete continuity equation is satisfied to machine precision regardless
of network error — the emulator cannot produce a spuriously divergent flow.

A **vertical-divergence penalty** additionally discourages the predicted
horizontal flow from implying a non-zero vertical velocity through the (rigid)
sea surface, which is the discrete analogue of the rigid-lid constraint:

```
L_cont = || w_diagnosed(k=0) ||²_ocean        (target: ≈ 0 at the surface)
```

---

## 5. Conservation budgets

For an extensive quantity with density ψ, the global ocean content is
`G[ψ] = Σ_ocean ψ · dV`, with cell volume `dV = TAREA · dz`. POP conserves
these up to the surface fluxes. The emulator's loss ties the predicted *change*
in each budget to the forcing integrated over the step, rather than imposing
naive strict conservation (which would be wrong whenever the surface flux is
non-zero).

**Heat content** `H = Σ ρ0 c_p T dV` (ρ0 = `rho_sw`, c_p = `cp_sw`):
```
ΔH_pred  should equal  Δt · Σ_surface SHF · TAREA
L_heat = ( ΔH_pred - ΔH_flux )²  /  scale_H²
```

**Salt content** `Sc = Σ ρ0 S dV`:
```
ΔSc_pred  should equal  Δt · Σ_surface F_salt · TAREA      (from SFWF)
L_salt = ( ΔSc_pred - ΔSc_flux )² / scale_S²
```

**Volume / barotropic mass (Boussinesq):** with a rigid lid the ocean volume is
fixed; the vertically integrated divergence must vanish:
```
L_baro = || Σ_k dz(k) (∇_h·u)_k ||²_ocean
```
(When SSH is treated as a free surface, this term is relaxed and the η tendency
absorbs the column divergence; controlled by `physics.barotropic` in the config.)

All budgets are computed in double precision and masked to ocean cells. Their
*absolute* values are also logged each epoch as physical diagnostics (PW for
heat, Sv·psu for salt) so conservation drift is monitorable, not just penalized.

---

## 6. Equation of state and static stability

`physics.mwjf_density` implements the POP MWJF (2003) 25-term rational
polynomial for in-situ density ρ(T, S, p), with pressure approximated
hydrostatically from depth (`p ≈ ρ0 g z`, POP's `pressure()` routine). This is
the *same* EOS POP uses, so densities derived from the emulator's (T, S) are
consistent with the model's.

Density buys two physical constraints:

1. **Hydrostatic consistency** — ρ from the predicted (T, S) is used in the
   hydrostatic relation to ensure the temperature/salinity update does not
   produce physically inconsistent pressure structure.
2. **Static stability** — POP's convective adjustment keeps the water column
   gravitationally stable (∂ρ_θ/∂z ≥ 0 for potential density referenced to the
   level). The loss penalizes *newly created* instability relative to the
   target field:
   ```
   L_stab = mean( relu( N²_unstable_pred ) - relu( N²_unstable_true ) )₊
   ```
   so the emulator is not pushed to be more stable than POP, only discouraged
   from inventing unstable stratification.

---

## 7. Composite loss

```
L = L_data
  + λ_cont  L_cont          (rigid-lid / surface continuity)
  + λ_baro  L_baro          (barotropic / volume)
  + λ_heat  L_heat          (heat-content budget)
  + λ_salt  L_salt          (salt-content budget)
  + λ_stab  L_stab          (static stability)
  + λ_spec  L_spectral      (optional: KE spectrum slope, anti-blurring)
```

`L_data` is a per-variable, per-level-weighted MSE in normalized space (deeper
levels and velocity components are reweighted so the optimizer is not dominated
by the high-variance surface). The physics weights `λ` are set in
`configs/default.yaml` and are annealed in: the model first learns the data
map, then the conservation constraints are ramped up. This curriculum avoids
the well-known failure mode where hard physics penalties stall early training.

---

## 8. What "faithful to POP" does and does not mean here

**Enforced by construction**
- Land/topography mask and partial bottom cells (`KMT`).
- Incompressibility: w diagnosed from continuity (machine precision).
- Same EOS (MWJF 2003) and same physical constants as POP.
- Tendency-form update mirroring POP's prognostic time step.

**Enforced softly (penalized, monitored)**
- Heat-, salt-, and volume-content budgets tied to surface fluxes.
- Static stability (no spurious convective instability).
- Rigid-lid surface kinematic condition.

**Deliberate approximations (documented)**
- Tripole northern fold approximated by replicate padding.
- Monthly-mean state (the emulator integrates monthly, not at POP's baroclinic
  time step); sub-monthly transport correlations are represented statistically.
- Sub-grid mixing (KPP, GM/Redi) is learned implicitly from data rather than
  reproduced term-by-term.

These choices are revisited in `docs/data.md` and are all switchable in config.
