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

**Run 2 — RUNNING 2026-06-12 as job 4538136 (preprocess was 4533715)**
- NB: first two train submissions were scrapped — `4533716` ran stale `.pyc`
  bytecode (old code: unbounded salt, unweighted loss); `4537803` then resumed
  from `4533716`'s contaminated checkpoint. Job `4538136` is the clean one:
  fresh start, cleared `__pycache__`, correct code.
- **Persistence baseline (weighted) = data loss ~0.90 at step 0** (was ~0.10
  unweighted). Salt/heat penalties bounded at 1.0 (confirmed).
- **RESULT — run 2 BEATS persistence** (12h, step ~31k, full 1→2→4-step
  curriculum; eval job 4546887, member 001, ckpt checkpoints/run2/last.pt):
  skill = 1 - emu_rmse/persist_rmse, positive = better than persistence.
  | lead | TEMP | SALT | UVEL | SSH |
  |  4mo | 0.62 | 0.50 | 0.52 | 0.47 |
  |  8mo | 0.61 | 0.42 | 0.64 | 0.46 |
  | 12mo | 0.35 | 0.13 | 0.32 | 0.32 |
  | 16mo | 0.48 | 0.28 | 0.60 | 0.39 |
  | 20mo | 0.42 | 0.14 | 0.56 | 0.32 |
  | 24mo | -0.16| -0.40| 0.27 |-0.12 |
  Run 1 was ~0.00 everywhere → the tendency-weighted loss solved the persistence
  collapse. Month-24 dips negative because persistence is artificially strong at
  exactly +24mo (same month-of-year, seasonal recurrence), not a model failure.
  Conservation still excellent (heat drift ~1e-3, salt ~1e-6).

(original launch notes, job ids superseded by 4538136 above)
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

**Run 3 — RUNNING 2026-06-13 as job 4551769 (stability recipe)**
- Implements `docs/training_plan.md` items 1-3 (commit d8d2b89): **2-state history**
  (`model.history=2`), **memory-safe pushforward** (per-step backward on detached
  self-rollout — long rollouts can't OOM), **input-noise injection** (0.1,
  denoiser training; residual anchored on a clean `base` so noise doesn't leak
  into output). All physics constraints kept. Config `configs/run3.yaml`,
  `ckpt_dir=checkpoints/run3`, curriculum to 16-step, fresh start, 189.1M params.
- Goal: extend free-rollout stability from run-2's 5.6 yr toward decades. Validate
  by re-running `jobs/diagnostics.pbs` with `CKPT=checkpoints/run3/last.pt`
  (achieved stable-month count should jump well past 67) and `jobs/evaluate.pbs`.
- Gotcha learned: input noise + tendency-weighted loss blows up the loss (~36) if
  the residual anchors on the *noised* state; anchor on the clean state (model
  `base=` arg). Eval/diag rollouts now keep an H-state history buffer.
- **RESULT — run 3 (job 4551769, completed at 12h walltime, step ~26k; reached
  2/4/8-step curriculum, did NOT reach 12/16-step). Stayed stable the whole run.**
  Eval job 4573198, diagnostics 4573199 (ckpt checkpoints/run3/last.pt, member 001):
  - **Free-rollout stability: 125 mo (10.4 yr)** vs run-2's 67 mo (5.6 yr) — ~2x.
  - Nino3.4 corr 0.75 / std-ratio 0.95; AMO corr 0.67; **PDO PC1 corr 0.84**
    (run 2: 0.67); PDO varfrac emu 0.72 vs POP 0.31 (still over-concentrated).
  - 24-mo skill positive 4-20mo (TEMP 0.29-0.48, UVEL 0.26-0.47, SSH 0.30-0.38);
    slightly less peaked at short lead than run 2 (stability/sharpness tradeoff),
    more consistent across leads. Conservation heat ~1e-3, salt ~1e-4.
  - Takeaway: pushforward+history+noise roughly doubled stable horizon with skill
    maintained. To go further: RESUME run 3 to reach 12/16-step rollout
    (`qsub -v CONFIG=configs/run3.yaml,CKPT_DIR=checkpoints/run3 jobs/train.pbs`
    auto-resumes from checkpoints/run3/last.pt). NB diagnostics figures in
    docs/figures/ were overwritten with run-3's; run-2 figs are in git history.

**Run 3 RESUME — RUNNING 2026-06-14 as job 4573511**
- Resumed cleanly from checkpoints/run3/last.pt at step 26000 (history=2, pushforward,
  noise active). Continuing 8-step rollout, advancing to 12-step at step 45k and
  16-step at 75k. ~step 33k, loss ~1.9, stable, no instability. 12h walltime; at
  ~0.5 it/s it approaches 12-step near the end — expect to need another resume for
  16-step. After it finishes, re-run eval + diagnostics on checkpoints/run3/last.pt
  to measure whether deeper rollout extended the stable horizon past 10.4 yr.

## Next steps (in order)

Persistence is beaten (run 2); run 3 targets long-horizon stability. Remaining:
1. **Long-lead skill / 24-mo degradation**: train longer (run 2 only reached
   step 31k in 12h because 4-step rollout is 4x cost — resume from
   `checkpoints/run2/last.pt`), and/or extend the rollout curriculum past 4-step.
   Consider an anomaly-vs-climatology target to remove the seasonal-recurrence
   artifact that makes persistence look strong at exactly +24mo.
2. **More data**: add RCP8.5 members to `data.train_members` for a bigger corpus.
3. **Scale the model**: raise `model.width`/`depth`, `data.batch_size` once
   profiled; consider longer horizons in the curriculum.
4. **Evaluate more members / leads** for robust skill stats (eval is ~1-3 min:
   `qsub -v CONFIG=configs/run2.yaml,CKPT=checkpoints/run2/last.pt,MEMBER=NNN jobs/evaluate.pbs`).
5. Push the repo to GitHub when ready (currently local only).

## Gotchas to remember

- Conda env for everything: `/glade/work/acsubram/conda-envs/credit/bin/python`
  (torch 2.10+cu128, xarray, einops, timm). **pytest is NOT in `credit`** — it's
  in the `fme` env, or run tests via the small fallback loop in this repo's history.
- PBS: `gpu_type` must go *inside* the `select` chunk on Casper, not as a
  job-level `-l` (fixed in `jobs/train.pbs`). Account `UCUB0143`, queue `casper`.
- EOS `mwjf_density`: salinity is floored at 1e-2 before `sqrt(S)` or land cells
  (S=0) produce NaN gradients.
- **Stale bytecode**: a PBS job once silently ran old `__pycache__/*.pyc` despite
  the working tree being correct (showed unbounded salt + unweighted loss). The
  PBS jobs now clear `src/**/__pycache__` and set `PYTHONDONTWRITEBYTECODE=1` at
  startup. If a job's behavior contradicts the on-disk code, suspect bytecode.
- **Resume gotcha**: `train.pbs` auto-resumes from `$CKPT_DIR/last.pt` if present.
  To force a fresh run, `rm -rf` the ckpt dir first (else it warm-starts, possibly
  from a bad checkpoint).
- Multiple jobs append sections to `jobs/logs/train.log` — read the section after
  the last `Job started` line for the current job.
- Member records differ in start year (001 historical = 1850 → 1872 months;
  most others = 1920 → 1032 months); `data.t_start` must be within range or the
  dataset is empty.
- Robust PBS state polling: `qstat -f <id> | awk -F' = ' '/job_state/{print $2}'`
  (the single-job `qstat <id>` column layout is easy to mis-parse).
