# POP Ocean Emulator

A physics-informed deep-learning emulator that integrates a three-dimensional
ocean state (potential temperature, salinity, and the horizontal velocity
field) forward in time. The emulator is trained on the **CESM Large Ensemble**
(CESM-CAM5-BGC-LE) ocean output, which is produced by the **POP2** ocean model
on the `gx1v6` displaced-pole grid.

The design goal is to stay as faithful as possible to the POP primitive
equations and to the conservation principles of the underlying ocean model
(volume / Boussinesq mass, heat content, salt content, incompressibility), so
that the learned map behaves like a discrete time-integrator of the ocean
rather than a generic image-to-image regressor.

```
x(t+1) = x(t) + F_theta( x(t), forcing(t), grid )      # prognostic update
w(t+1) = -∫ ∇_h · u  dz                                 # diagnosed, not learned
```

See **[docs/physics.md](docs/physics.md)** for the equation set and how each
conservation law is enforced, and **[docs/data.md](docs/data.md)** for the
dataset and the rationale behind the variable selection.

---

## What it predicts

| Symbol | Variable | POP name | Grid location | Role |
|--------|----------|----------|---------------|------|
| T | Potential temperature | `TEMP` | T-cell (3D) | prognostic |
| S | Salinity | `SALT` | T-cell (3D) | prognostic |
| u | Grid-x velocity | `UVEL` | U-cell (3D) | prognostic |
| v | Grid-y velocity | `VVEL` | U-cell (3D) | prognostic |
| η | Sea-surface height | `SSH` | T-cell (2D) | prognostic |
| w | Vertical velocity | `WVEL` | W-face (3D) | **diagnosed** from continuity |

The emulator is *forced* at the surface by the same boundary fields that drive
POP: net surface heat flux (`SHF`, with the shortwave part `SHF_QSW`),
surface freshwater/virtual-salt flux (`SFWF`), and the wind-stress components
(`TAUX`, `TAUY`). These enter the network as input channels and also close the
conservation budgets (a non-zero heat flux *should* change the global heat
content; the loss accounts for that rather than demanding strict conservation).

## Grid

- POP `gx1v6`: 384 (`nlat`) × 320 (`nlon`) horizontal, 60 vertical levels
  (`z_t`, 5 m → 5375 m, partial bottom cells via `KMT`).
- B-grid staggering: tracers on T-cells, velocities on U-cells (NE corner),
  vertical velocity on layer top faces.
- Longitude is periodic (circular padding); the latitude direction terminates
  in a tripole fold at the North Pole (handled with replicate padding — an
  approximation documented in `docs/physics.md`).

## Repository layout

```
configs/            YAML experiment configuration
docs/               physics.md, data.md
src/pop_emulator/
  constants.py      POP physical constants (CGS, read from the data)
  grid.py           grid geometry, masks, Coriolis, differential operators
  data.py           CESM-LENS dataset / dataloader (reads the raw netCDF)
  normalization.py  per-level, per-variable standardization
  physics.py        MWJF equation of state, continuity, conservation losses
  losses.py         composite data + physics loss
  model.py          physics-informed spherical U-Net (residual/tendency form)
  rollout.py        autoregressive multi-step integration
  utils.py          checkpointing, logging, seeding
scripts/
  build_grid.py     extract static grid + constants to grid_gx1v6.nc
  compute_stats.py  normalization statistics over the training members
  train.py          training entry point
  evaluate.py       rollout skill metrics + conservation diagnostics
  smoke_test.py     end-to-end check on synthetic data (no GPU/data needed)
jobs/               PBS submission scripts for Casper (GPU)
tests/              unit tests for the physics operators
```

## Quick start (NCAR Casper)

```bash
# environment with torch 2.10 + xarray + einops + timm
CONDA=/glade/work/acsubram/conda-envs/credit/bin

# 0. self-contained smoke test (CPU, synthetic data, ~seconds)
$CONDA/python scripts/smoke_test.py

# 1. build the static grid file (geometry, masks, Coriolis, constants)
$CONDA/python scripts/build_grid.py --out data/grid_gx1v6.nc

# 2. compute normalization statistics over the training members
$CONDA/python scripts/compute_stats.py --config configs/default.yaml

# 3. train (submit to a GPU node)
qsub jobs/train.pbs
```

The data root and ensemble-member split are set in `configs/default.yaml`.

## Status

This repository is a complete, runnable scaffold: the physics operators, data
pipeline, model, training loop, and evaluation are implemented and covered by
a synthetic smoke test and unit tests. Large-scale training on the full
ensemble is launched through the PBS scripts in `jobs/`.
