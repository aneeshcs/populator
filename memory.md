# Project Memory — POP Ocean Emulator

Restart-here notes for the physics-informed ocean emulator. Last updated
2026-06-12. For *how to run / what tools are needed*, see `skills.md`.

---

## What this project is

A physics-informed deep-learning emulator that integrates the 3-D ocean state
(potential temperature, salinity, horizontal velocity, SSH) forward in time on
the POP2 `gx1v6` grid, trained on the CESM-CAM5-BGC Large Ensemble. Goal: a
learned discrete time-integrator that respects POP's primitive equations and
conservation laws, not a generic image regressor.

- Repo: `/glade/work/acsubram/GitRepos/pop-ocean-emulator` (git, default branch `master`)
- Design/physics rationale: `README.md`, `docs/physics.md`, `docs/data.md`

## Data (CESM-LENS, POP2, gx1v6)

- Root: `/glade/campaign/collections/gdex/data/d651027/cesmLE/CESM-CAM5-BGC-LE/ocn/proc/tseries/monthly`
- Grid 384 (nlat) × 320 (nlon) × 60 (z_t). Monthly is the only frequency with
  full 3-D fields (daily is surface-only).
- Members: ~52 historical (`B20TRC5CNBDRD`) + ~73 RCP8.5 (`BRCP85C5CNBDRD`).
  Each 3-D variable is ~23 GB/member → stream lazily, never load whole.
- **Prognostic (predicted):** TEMP, SALT (T-cells), UVEL, VVEL (U-cells), SSH (2-D).
- **Diagnosed, not predicted:** WVEL (from continuity), density (MWJF EOS from T,S).
- **Forcing inputs:** SHF, SHF_QSW, SFWF, TAUX, TAUY.
- Static grid + POP constants (CGS) are embedded in every history file; extracted
  once into `data/grid_gx1v6.nc`.

## Architecture & physics (as built)

- Grid-aware spherical U-Net (`src/pop_emulator/model.py`), **tendency form**
  `x_{t+1} = x_t + Δt·F(x_t, forcing, grid)`, output land-masked. Periodic-lon /
  replicate-lat (tripole) padding. Static channels: Coriolis f, mask, depth, lat/lon.
- Physics-informed loss (`losses.py` + `physics.py`): per-level data MSE +
  continuity (rigid lid) + barotropic/volume + heat & salt budgets + static
  stability; physics weights ramped via warmup curriculum.
- W diagnosed from continuity → incompressible by construction. MWJF-2003 EOS
  (McDougall et al.), pressure in dbar; surface ρ matches EOS-80 to <0.01.

## State of play (2026-06-12)

**Done & validated**
- Full repo scaffolded, committed. `scripts/smoke_test.py` passes; 9 physics
  unit tests pass (`tests/test_physics.py`).
- Real-data integration verified (grid build, slice reads, EOS/wvel/budgets).
- Preprocess job done → `data/grid_gx1v6.nc` (6.6 MB) and `data/stats.nc` (per-level
  norm stats over 19 training members) written. These are gitignored (regenerate
  with the preprocess job if missing).

**Training run (job 4518431) — COMPLETE**
- Submitted to Casper A100; preprocess (job 4518430) ran first via `afterok`.
- Ran the full 12h, PBS-killed at walltime at ~step 60,350. ~1.4 it/s.
- Model is **188.9M params** (default config width=96; the 4.5M figure from earlier
  notes was a width=16 test override, not the production model).
- Rollout curriculum transitioned 1→2 step at ~step 55k (epoch boundary).
- Data loss ~0.10 (1-step) → ~0.2–0.3 (2-step, expected). Continuity/barotropic
  ~1e-6, stability ~1e-5 (healthy). Salt penalty spiked occasionally (see fix below).
- Checkpoint: `checkpoints/default/last.pt` (~2.3 GB, step 60000, gitignored),
  written every 2000 steps. Resume with `--resume checkpoints/default/last.pt`.

**Evaluation (job 4533577, 24-month rollout on held-out member 001) — KEY RESULT**
- **Skill vs persistence ≈ 0 at all lead times** (TEMP/SALT/UVEL/SSH skill in
  [-0.003, +0.005]); `emu_rmse ≈ persist_rmse`. The model learned the trivial
  persistence solution F≈0 (tendency form makes persistence the easy minimum;
  monthly ocean is very persistent so the data loss is already low when copying
  the input).
- **Conservation is excellent**: 24-month heat-content drift +1.2e-3 (0.1%),
  salt-content drift +4e-6 (negligible). Pipeline + physics work as designed;
  the rollout is stable over 2 years.
- Bottom line: infrastructure fully validated end-to-end; **beating persistence
  is the open scientific problem** (addressed in run 2 below).

**Run 2 — LAUNCHED 2026-06-12 (jobs 4533715 preprocess, 4533716 train; held on afterok)**
- `fix/salt-budget-and-rollout` MERGED to master (3a0ae17).
- Config `configs/run2.yaml`: `normalize.tendency_weighted_loss: true`; heat/salt
  weights lowered to 0.02 (they reward F=0); rollout curriculum
  [[0,1],[12000,2],[30000,4]]; `ckpt_dir: checkpoints/run2` (run 1 preserved);
  starts FRESH (not resumed — run-1 weights sit in the persistence basin).
- Tendency-weighted loss makes persistence an O(1)-loss solution (was ~0.0025),
  forcing the model to learn month-to-month evolution. Validated: persistence
  data loss = 1.0; weights 0.8..1e4 (deep T/S clamped — they barely change monthly).
- Preprocess regenerates `data/stats.nc` with `*_tend_std` fields.
- PBS jobs take `CONFIG=` / `CKPT_DIR=` env overrides:
  `qsub -v CONFIG=configs/run2.yaml jobs/preprocess.pbs` then
  `qsub -W depend=afterok:<prep> -v CONFIG=configs/run2.yaml,CKPT_DIR=checkpoints/run2 jobs/train.pbs`.
- **Watch:** does run 2 beat persistence? After it trains, eval with
  `CKPT=checkpoints/run2/last.pt MEMBER=001 qsub jobs/evaluate.pbs`.

**Fixes shipped in run 2 (merged to master 2026-06-12, commit 3a0ae17)**
1. *Spiky salt/heat budget penalty*: was normalized by the flux-implied change
   (`dH_flux` ~0 when net flux small) → squared ratio exploded (salt >1e3, loss
   ~64 at step 40k). Now a **bounded relative closure error** (normalize by the
   magnitude of the change itself; bounded). Physically exact (interior
   advection/diffusion integrate to zero globally).
2. *Rollout curriculum only honored at epoch boundaries*: now re-checked
   per-batch, loader rebuilt on change.
3. *Persistence collapse*: **tendency-weighted data loss** (see run 2 above).

## Next steps (in order)

1. **Beat persistence** — the central problem. The tendency-form model collapsed
   to F≈0 because persistence already gives low normalized data loss. Levers:
   - Train on **anomalies relative to a monthly climatology** (remove the
     persistent mean state so the loss targets the actual evolution).
   - **Weight the loss toward the tendency** (e.g. loss on `Δ = x_{t+1}-x_t`, or a
     skill/ACC-style objective) so copying the input is no longer the easy minimum.
   - More **multi-step rollout** training (curriculum to 4+; only ~5k of 60k steps
     were 2-step) — penalizes persistence over long horizons.
   - Longer training / more members (add RCP8.5 to `train_members`).
2. Merge `fix/salt-budget-and-rollout` into `master` before the next run (bounded
   salt/heat penalty + per-batch rollout curriculum). Resume from `last.pt` or
   start fresh.
3. Eval is fast (~3 min CPU): `qsub jobs/evaluate.pbs` (override `CKPT=`, `MEMBER=`).
4. Scale up once profiled: `model.width`, `data.batch_size`, longer curriculum.
5. Push the repo to GitHub when ready (currently local only).

## Gotchas to remember

- Conda env for everything: `/glade/work/acsubram/conda-envs/credit/bin/python`
  (torch 2.10+cu128, xarray, einops, timm). **pytest is NOT in `credit`** — it's
  in the `fme` env, or run tests via the small fallback loop in this repo's history.
- PBS: `gpu_type` must go *inside* the `select` chunk on Casper, not as a
  job-level `-l` (fixed in `jobs/train.pbs`). Account `UCUB0143`, queue `casper`.
- EOS `mwjf_density`: salinity is floored at 1e-2 before `sqrt(S)` or land cells
  (S=0) produce NaN gradients.
- Member records differ in start year (001 historical = 1850 → 1872 months;
  most others = 1920 → 1032 months); `data.t_start` must be within range or the
  dataset is empty.
- Robust PBS state polling: `qstat -f <id> | awk -F' = ' '/job_state/{print $2}'`
  (the single-job `qstat <id>` column layout is easy to mis-parse).
