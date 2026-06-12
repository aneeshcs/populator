# Skills & Tools — POP Ocean Emulator

What you need to know and have installed to work on this project. For *project
status / what was done*, see `memory.md`.

---

## Environment

- **Machine:** NCAR Casper (GPU nodes: A100). Login/preprocess on CPU; training on GPU.
- **Conda env (use for everything):**
  `/glade/work/acsubram/conda-envs/credit/bin/python`
  - torch 2.10 + cu128, xarray 2026.x, numpy, netCDF4, dask, einops 0.8, timm 1.0, pyyaml, tqdm
  - GPU (`torch.cuda.is_available()`) is False on login nodes, True on Casper GPU nodes.
  - **pytest is NOT in `credit`** — it's in `/glade/work/acsubram/conda-envs/fme/bin`.
- **Modules:** `ncarenv`, `netcdf` (for `ncdump`), `conda`. `cuda/12.9` for GPU.

## Command-line tools used

| Tool | Use |
|------|-----|
| `qsub` / `qstat` (PBS Pro) | submit & monitor jobs on Casper |
| `ncdump -h` / `-v` | inspect CESM-LENS netCDF headers & variables |
| `git` | version control (repo is local; default branch `master`) |
| `gh` | (optional) push to GitHub when ready |

### PBS recipes
```bash
qsub jobs/preprocess.pbs                      # build grid + stats (CPU, htc/casper)
PREP=$(qsub jobs/preprocess.pbs)
qsub -W depend=afterok:$PREP jobs/train.pbs    # train after preprocess succeeds (A100)
qstat -u acsubram                              # my jobs
qstat -f <jobid> | awk -F' = ' '/job_state/{print $2}'   # robust state parse
```
PBS notes: account `UCUB0143`, queue `casper`; `gpu_type=a100` goes **inside** the
`select` chunk (`select=1:ncpus=16:ngpus=1:mem=200GB:gpu_type=a100`).

## Run the pipeline

```bash
CONDA=/glade/work/acsubram/conda-envs/credit/bin
$CONDA/python scripts/smoke_test.py                          # CPU, synthetic, seconds
$CONDA/python scripts/build_grid.py    --config configs/default.yaml
$CONDA/python scripts/compute_stats.py --config configs/default.yaml --max-times 120
$CONDA/python scripts/train.py    --config configs/default.yaml [--resume ckpt.pt] [--max-steps N]
$CONDA/python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/default/final.pt --member 001
```
Run the unit tests (use the `fme` env's pytest, or a fallback loop):
```bash
/glade/work/acsubram/conda-envs/fme/bin/python -m pytest tests/ -q
```

## Domain knowledge required

- **POP2 ocean model**: primitive equations (momentum, hydrostatic, continuity,
  tracer transport), B-grid staggering (tracers at T-cells, velocity at U-cell
  corners, w at layer faces), `gx1v6` displaced-pole/tripole grid, partial bottom
  cells via `KMT`, CGS units, the embedded physical constants (`cp_sw`, `rho_sw`,
  `grav`, `omega`, `ocn_ref_salinity`, flux factors).
- **Equation of state**: MWJF (McDougall, Jackett, Wright & Feistel 2003) 25-term
  rational polynomial; T=potential temp [degC], S [psu], p [dbar] → ρ [kg/m³].
- **Conservation reasoning**: global heat/salt content change equals the surface
  flux integral (interior advection/diffusion integrate to zero); rigid-lid
  volume/continuity; static stability via potential density.
- **ML**: PyTorch (U-Net, GroupNorm, FiLM conditioning, residual/tendency
  prediction, autoregressive rollout, curriculum, bf16/AMP, grad clipping),
  per-level normalization, physics-informed (soft-constraint) losses.
- **Data tooling**: xarray/dask lazy access to large netCDF time series; POP fill
  value handling (`|x| > 1e30` = land → 0, masked in losses).

## Repo map (where things live)

```
configs/default.yaml          experiment config (paths, members, weights, curriculum)
src/pop_emulator/
  constants.py  grid.py  physics.py        # POP constants, geometry, EOS/continuity/conservation
  data.py  normalization.py                # CESM-LENS loader, per-level standardization
  model.py  losses.py  rollout.py          # U-Net, composite loss, autoregressive integration
  testing.py  utils.py                     # synthetic fixtures, checkpoint/seed helpers
scripts/  build_grid · compute_stats · train · evaluate · smoke_test
jobs/     preprocess.pbs · train.pbs       # Casper submission
tests/    test_physics.py
docs/     physics.md · data.md
```
