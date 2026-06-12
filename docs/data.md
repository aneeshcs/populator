# Data: CESM Large Ensemble ocean output

## Source

```
/glade/campaign/collections/gdex/data/d651027/cesmLE/CESM-CAM5-BGC-LE/ocn/proc/tseries/
```

The CESM-CAM5-BGC Large Ensemble (CESM-LENS) ocean component is **POP2** on the
`gx1v6` displaced-pole grid. Output is provided as single-variable time series
at `annual`, `monthly`, and `daily` frequency. **Monthly** is used for the
emulator because it is the only frequency with the full set of 3-D prognostic
fields (the `daily` stream contains only surface / 2-D diagnostics such as
`SST`, `SSH_2`, `HMXL_2`, `WVEL_50m`).

### Experiments and ensemble members

| Tag | Experiment | Period | Members |
|-----|-----------|--------|---------|
| `B20TRC5CNBDRD` | Historical (20th century) | 1850/1920–2005 | 52 |
| `BRCP85C5CNBDRD` | RCP8.5 future | 2006–2080/2100 | 73 |
| `B1850C5CN` | Pre-industrial control | multi-century | (long control runs) |

The historical + RCP8.5 members give ~125 independent trajectories sharing the
same model physics but differing in initial condition / forcing — an ideal
corpus for learning an initial-condition-to-forecast map. Member `001` of the
historical run is held out for validation by default; the split is configurable
(`configs/default.yaml: data.{train,val,test}_members`).

### Grid dimensions

```
nlat = 384      nlon = 320      z_t = 60  (5 m surface → 5375 m bottom)
```

A single 3-D monthly variable for one member over 1850–2005 is ~23 GB
(1872 months × 60 × 384 × 320, float32), so the pipeline streams lazily with
xarray/dask and never materializes a whole member in memory.

## Variable selection

The task — integrate a 3-D ocean state forward — fixes the **prognostic** set to
the variables POP itself time-steps:

| Variable | POP name | dims | units (raw) | why |
|----------|----------|------|-------------|-----|
| Potential temperature | `TEMP` | (t, z_t, nlat, nlon) | °C | prognostic tracer |
| Salinity | `SALT` | (t, z_t, nlat, nlon) | g/kg | prognostic tracer |
| Grid-x velocity | `UVEL` | (t, z_t, nlat, nlon) | cm/s | prognostic momentum |
| Grid-y velocity | `VVEL` | (t, z_t, nlat, nlon) | cm/s | prognostic momentum |
| Sea-surface height | `SSH` | (t, nlat, nlon) | cm | barotropic / free surface |

**Diagnosed, not predicted**

| `WVEL` | vertical velocity | reconstructed from continuity (see physics.md §4); including it as a target would let the network violate incompressibility. |
| `RHO` / `PD` | density | computed from (T, S) via the MWJF EOS so it is exactly consistent with the predicted tracers. |

**Surface forcing (network inputs, close the budgets)**

| Variable | POP name | role |
|----------|----------|------|
| Net surface heat flux | `SHF` | heat-content budget, T surface BC |
| Shortwave heat flux | `SHF_QSW` | penetrative solar (separated from SHF) |
| Surface freshwater / virtual-salt flux | `SFWF` | salt-content budget, S surface BC |
| Zonal wind stress | `TAUX` | momentum surface BC |
| Meridional wind stress | `TAUY` | momentum surface BC |

These are all present as their own time-series directories alongside the
prognostic variables.

**Static geometry / constants** (identical across members, extracted once by
`scripts/build_grid.py` into `data/grid_gx1v6.nc`):

`dz`, `dzw` (layer thicknesses), `z_t`, `z_w_top`, `TAREA`, `UAREA`, `HTN`,
`HTE`, `HUS`, `HUW`, `DXT`, `DYT`, `KMT`, `REGION_MASK`, `TLAT`, `TLONG`,
`ULAT`, `ULONG`, `ANGLE`, plus the scalar constants `cp_sw`, `rho_sw`,
`rho_fw`, `grav`, `omega`, `radius`, `T0_Kelvin`, `ocn_ref_salinity`,
`hflux_factor`, `fwflux_factor`, `salinity_factor`, `sflux_factor`.

### Variables deliberately excluded

The BGC tracers (`DIC`, `O2`, `NO3`, `Fe`, the ecosystem/`diat`/`diaz`/`sp`
groups, CFCs, `IAGE`, …) and the offline diagnostic transports
(`UET`, `VNT`, `WTT`, `ADVT`, `HDIFT`, …) are not prognostic for the
dynamical/thermodynamical core and are omitted from the first emulator. The
advective/diffusive flux diagnostics could later be used as *supervision* for
the learned transport operator, but they are not state variables.

## Normalization

Each variable is standardized **per vertical level** (and globally for the 2-D
η), using mean and standard deviation computed over the training members by
`scripts/compute_stats.py` and stored in `data/stats.nc`. Per-level statistics
matter because temperature variance drops by orders of magnitude from the
mixed layer to the abyss; a single global scale would let the surface dominate
the loss. Velocities are standardized about zero (their mean is ~0). Land cells
are excluded from all statistics.

## Time convention

`time:units = "days since 0000-01-01"`, `calendar = "noleap"`. The emulator
step Δt is one model month; the month-of-year is provided to the network as a
cyclic `(sin, cos)` embedding so it can represent the seasonal cycle of the
forcing and mixed layer.
