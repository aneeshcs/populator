# Plan: Training a Long-Horizon Stable Ocean Emulator (Run 3)

Goal: extend the emulator's free-running rollout stability from the current
**~5.6 years** (run 2) toward the **decades–centuries** demonstrated by
state-of-the-art ocean emulators, so that decadal modes (PDO, AMO) can be
diagnosed on their native timescales and the model is useful for climate-length
integrations.

This document diagnoses why run 2 destabilizes, summarizes the relevant
literature, and proposes a concrete, prioritized recipe for run 3. It is a
plan only — no training is launched until approved.

---

## 1. Where we are

- Run 2 (tendency-form ConvNeXt-style U-Net, 188.9 M params, monthly step,
  curriculum reached 4-step rollout) **beats persistence** at 4–20 month leads
  (TEMP/UVEL skill 0.3–0.64) and conserves heat/salt well, but the free,
  surface-forced rollout **drifts and blows up after ~67 months (5.6 yr)**
  (`docs/ocean_modes.tex`).
- Symptoms of the instability: (i) numerical blow-up of SST after ~5.6 yr;
  (ii) the leading North Pacific EOF is over-dominant and over-smooth (75% of
  variance vs POP's 28%), i.e. a slowly growing basin-scale drift projects onto
  EOF1. Both point to **error accumulation under autoregressive feedback** plus
  **convolutional damping of small scales**.

## 2. What the literature says drives stability

| Finding | Source |
|---|---|
| Multi-step **unrolled / "pushforward" training** is the primary lever for rollout stability and accuracy; more effective than noise injection alone. | Brandstetter et al. (pushforward); "Exploring Design Choices for Autoregressive DL Climate Models" (arXiv:2505.02506) |
| Instability = **amplification of high-frequency energy**; stable models behave as **denoisers** on perturbed inputs. | arXiv:2505.02506 |
| **Input history of 2 previous states** (2-in/2-out) supports century-scale ocean stability with plain MSE and *no* physics constraints. | Samudra, Dheeshjith et al. 2025 (GRL 2024GL114318) |
| **Tendency / change prediction** reduces error accumulation vs predicting full state. | "Predicting change, not states" (ScienceDirect 2025); arXiv:2505.02506 |
| Larger model capacity sustains accuracy longer (we are at 188 M ≈ Samudra's 135 M, so capacity is likely adequate). | arXiv:2505.02506 |
| Long-term stability + physical consistency achievable for O(1000) ensembles with careful design. | LUCIE (arXiv:2405.16297); ACE2 (npj Clim Atmos Sci 2025) |

Takeaway: stability is mainly a **training-procedure** problem. Samudra is the
closest analog (ConvNeXt-UNet ocean emulator, same variable set T/S/SSH/U/V) and
is stable for centuries using (a) a short **history of 2 states**, (b)
**autoregressive multi-step training**, and (c) plain MSE. We should adopt those,
keep our tendency form and (cheap) physics diagnostics, and add denoiser-style
regularization.

## 3. Proposed interventions (prioritized)

Ordered by expected stability payoff per unit effort. Items 1–3 are the core
recipe; 4–6 are second-wave refinements.

### 1. Much longer multi-step (pushforward) rollout — *highest priority*
- **What:** extend the rollout curriculum well past 4 steps (target 8 → 12 → 16)
  and spend the **majority** of training at long rollout, not 1-step. Run 2 only
  reached 4-step in its final hour.
- **Why:** the #1 stability lever in the literature; directly penalizes the
  error growth that destabilizes free rollouts.
- **How / cost:** full back-prop through 16 steps is ~16× the per-step compute
  and memory. Use **truncated BPTT / detached pushforward**: roll out N steps
  feeding the model its own *detached* predictions, and back-prop only the last
  k (e.g. k=2–4) steps. This caps memory while still exposing the model to its
  own error distribution. **Code change** in `scripts/train.py` `train_step`
  (detach `x` between early steps); optionally add gradient checkpointing.

### 2. Add a 2-state input history (Samudra-style)
- **What:** condition on `x(t-1)` and `x(t)` to predict `x(t+1)` (or 2-in/2-out).
- **Why:** gives the network the local tendency/acceleration, which Samudra uses
  to remain stable for centuries; helps the model distinguish signal from the
  noise it must damp.
- **How / cost:** **code change** in `model.py` (double the state input channels
  or stack a time axis) and `data.py` (return two consecutive input states).
  Modest; ~doubles input channels (241 → 482 + forcing + static).

### 3. Input noise injection (denoiser training)
- **What:** add small Gaussian perturbations to the input state during training
  (optionally as a random walk across unrolled steps).
- **Why:** unstable models amplify high-frequency noise; training the model to
  map noisy inputs back to clean targets makes it a contraction (denoiser),
  which is the empirical signature of stable models.
- **How / cost:** few lines in `train_step` (add scaled noise to `x` before the
  forward). Tune amplitude (~0.05–0.2 in normalized units). Cheap.

### 4. Spectral / gradient regularization
- **What:** activate the existing (currently 0-weight) spectral loss term to
  penalize growth of high-wavenumber energy, or add a small Laplacian/gradient
  penalty.
- **Why:** directly counteracts the high-frequency amplification mechanism and
  the convolutional smoothing that over-broadens the PDO pattern.
- **How / cost:** implement `spectral` term in `losses.py` (FFT of SST/SSH KE
  spectrum slope vs target). Moderate.

### 5. Anomaly-relative-to-climatology framing
- **What:** predict anomalies w.r.t. a monthly climatology instead of full fields
  (or add the climatology as a static input the residual rides on).
- **Why:** removes the large persistent mean state that the basin-scale drift
  rides on (the EOF1-inflation artifact), and removes the seasonal-recurrence
  artifact that flattered persistence at 24-month leads.
- **How / cost:** compute a per-month, per-cell climatology in
  `scripts/compute_stats.py`; subtract/add in the pipeline. Moderate.

### 6. More training data
- **What:** add RCP8.5 members (and more historical members) to
  `data.train_members`.
- **Why:** broader sampling of variability and forcing regimes improves
  generalization of the learned dynamics; cheap insurance against overfitting a
  narrow regime. **Config only.**

## 4. Concrete Run-3 recipe (proposed `configs/run3.yaml`)

Start from `configs/run2.yaml` (keep tendency form, bounded budgets, per-level +
tendency-weighted loss, land mask) and change:

```yaml
model:
  history: 2                 # NEW: condition on x(t-1), x(t)  (code change)
  width: 96                  # keep (capacity adequate vs Samudra 135M)
train:
  # spend most of training at long rollout; detached pushforward (truncated BPTT)
  rollout_curriculum: [[0,2],[8000,4],[20000,8],[40000,12],[70000,16]]
  bptt_window: 3             # NEW: back-prop only last 3 unrolled steps
  noise_injection: 0.1       # NEW: stddev of input noise (normalized units)
  grad_checkpoint: true      # NEW: to fit long rollouts in memory
physics:
  loss_weights: { ..., spectral: 0.02 }   # activate spectral regularizer
data:
  train_members: [ ... historical ... , RCP8.5 members ... ]   # enlarge corpus
```

Required code changes before this config runs:
1. `model.py` — accept a 2-state history input.
2. `scripts/train.py` — detached multi-step pushforward with `bptt_window`;
   optional noise injection and gradient checkpointing.
3. `data.py` — return two consecutive input states.
4. `losses.py` — implement the `spectral` term.
5. `scripts/compute_stats.py` — (item 5, optional) monthly climatology.

## 5. Validation criteria

- **Primary:** free, surface-forced rollout stays physical for **≥ 480 months
  (40 yr)** with bounded global-mean SST/SSH (target: centuries, like Samudra).
  Measured by `scripts/diagnostics.py` (it already detects blow-up and reports
  achieved stable length).
- **Secondary:** re-run `jobs/diagnostics.pbs` to assess decadal PDO/AMO on a
  multi-decade record; check the PDO EOF1 variance fraction approaches POP's
  ~28% (not 75%) and recovers the horseshoe pattern.
- **Skill:** `jobs/evaluate.pbs` 24-month skill should be ≥ run 2 (no regression
  from the stability measures).
- **Conservation:** heat/salt drift remain < ~1% over the long rollout.

## 6. Compute & sequencing

- Each 12 h A100 job currently reaches ~31 k steps at up-to-4-step rollout; at
  8–16 step rollout, expect **2–4× fewer optimizer steps per wall-hour**, so plan
  **3–4 chained 12 h jobs** (resume via `--resume`) to complete the curriculum.
- Run order: (a) implement code changes 1–3 (history, pushforward, noise);
  (b) smoke-test + unit-test; (c) `qsub jobs/preprocess.pbs` only if the
  variable/clim set changes; (d) chain `jobs/train.pbs` with `CONFIG=run3.yaml`;
  (e) validate per §5. Add spectral term (item 4) and anomaly framing (item 5)
  as a follow-on run if §5 is not met.

## 7. Risks & mitigations

- **Memory blow-up at 16-step:** mitigate with `bptt_window` (detach) +
  gradient checkpointing; fall back to 8-step if needed.
- **History doubles input channels → slower/heavier:** acceptable; reduce
  `width` to 80 if memory-bound.
- **Over-stabilization → over-smooth, low variance** (the Samudra "stability vs
  forced-response" tension): monitor the SST-variance ratio in diagnostics; keep
  noise amplitude modest and use the spectral term to preserve small scales.
- **Longer rollout training is slower to converge:** start the curriculum at
  2-step (not 1) to seed, and keep the tendency-weighted loss so the model still
  learns real evolution early.
