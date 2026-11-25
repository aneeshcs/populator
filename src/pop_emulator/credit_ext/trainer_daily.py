"""Daily-step trainer: daily surface supervision, monthly-mean interior constraint.

Daily truth exists for SST and SSH only.  The 3-D state is constrained through
calendar-month means of the daily rollout,

    L = (1/F) sum_d [ w_sfc L_sfc(x_d) + L_phys(x_d) ]
        + (lambda_m / n_M) sum_M L_3D( mean_{d in M} x_d ,  xbar_M ),

with ``F`` rollout days and ``n_M`` complete calendar months in the rollout.

The monthly term couples all days of a month, but the rollout is truncated
between steps (each step's graph is freed after its backward, as in the
monthly trainer).  Each complete month is therefore run twice from the same
state:

  pass 1  no gradient; accumulates the month-mean prediction and target and
          evaluates ``g_M = dL_3D / d(mean)``;
  pass 2  the same days with gradient; day ``d`` back-propagates
          ``L_d + lambda_m <g_M, x_d> / N_M``, whose sum over the month has the
          gradient of ``lambda_m L_3D`` under the same per-step truncation.

The random state is restored between the passes so the input noise is the same.
Days after the last complete month (the rollout length is a fixed number of
days, months are not) receive the surface and physics terms only, in a single
pass; every rank thus performs exactly ``F`` backward calls per sequence, as
DDP requires.

Rollouts start on the first day of a month (``PopLENSDailyDataset``).  Requires
``train_batch_size: 1`` (months differ in length) and data parallelism only
(single GPU or DDP).  Validation is inherited: surface + physics loss at lead
``validation_data.forecast_len`` days.
"""
from __future__ import annotations

import gc
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm

from credit.parallel.collectives import all_reduce_avg, clip_grad_norm_
from credit.postblock import apply_postblocks
from credit.preblock import apply_preblocks
from credit.trainers.rollout_utils import apply_rollout_renames, assemble_rollout_batch
from credit.trainers.trainer_gen2 import _copy_named_input  # pyright: ignore[reportPrivateUsage]
from credit.trainers.utils import cycle

from .dataset_daily import month_of_day
from .loss import PopDailyLoss
from .trainer import PopTrainer

logger = logging.getLogger(__name__)


def month_blocks(first_target_day: int, n_days: int) -> List[Tuple[int, bool]]:
    """Split ``n_days`` consecutive target days into calendar-month blocks.

    Returns ``[(length, complete), ...]``; a block is complete when it covers
    every day of its month.
    """
    blocks: List[Tuple[int, bool]] = []
    day, left = first_target_day, n_days
    while left > 0:
        _, dom, length = month_of_day(day)
        n = min(length - dom, left)
        blocks.append((n, dom == 0 and n == length))
        day, left = day + n, left - n
    return blocks


def _input_day(metadata: dict) -> int:
    """Day index of the newest input state; metadata is nested by source."""
    for value in metadata.values():
        if isinstance(value, dict) and "day" in value:
            return int(torch.as_tensor(value["day"]).flatten()[0])
    if "day" in metadata:
        return int(torch.as_tensor(metadata["day"]).flatten()[0])
    raise KeyError("no 'day' in batch metadata; PopDailyTrainer requires PopLENSDailyDataset")


class PopDailyTrainer(PopTrainer):
    def __init__(self, model, rank: int, conf: dict):
        super().__init__(model, rank, conf)
        if int(conf["trainer"].get("train_batch_size", 1)) != 1:
            raise ValueError("PopDailyTrainer requires trainer.train_batch_size: 1")
        if self.use_batch_axis_ensemble or self._domain_pre_pad is not None:
            raise ValueError("PopDailyTrainer supports neither ensembles nor domain parallelism")

    # ------------------------------------------------------------------ #
    # One model step (the per-step body of TrainerERA5Gen2.train_one_epoch)
    # ------------------------------------------------------------------ #
    def _advance(self, state: Dict[str, Any], batch: dict) -> Dict[str, Any]:
        first = "ic_preprocessed" not in state
        state["x_raw"], state["y_raw"] = batch["input"], batch["target"]
        if first:
            state["ic_raw"] = batch["input"]
            ic = apply_preblocks(self.ic_preblocks, batch, device=self.device)
            state["ic_preprocessed"] = apply_rollout_renames(ic, self.step_preblocks)
            step_input = state["ic_preprocessed"]
        else:
            rollout_batch = apply_rollout_renames(batch, self.ic_preblocks, self.step_preblocks)
            step_input = assemble_rollout_batch(state, rollout_batch, history_len=self.history_len)
        state["x_physical"] = _copy_named_input(step_input["input"])
        state.update(apply_preblocks(self.step_preblocks, step_input, device=self.device))
        if self.history_len > 1:
            state["history_x"] = (state["x"] if first else
                                  self._slide_history_window(state["history_x"], state["x"]))
            state["x"] = state["history_x"]
        self._sharded_forward(state, False)
        return apply_postblocks(self.step_postblocks, state)

    def _rng_state(self):
        cuda = torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None
        return torch.get_rng_state(), cuda

    def _set_rng_state(self, saved) -> None:
        torch.set_rng_state(saved[0])
        if saved[1] is not None:
            torch.cuda.set_rng_state(saved[1], self.device)

    # ------------------------------------------------------------------ #
    # One calendar-month block
    # ------------------------------------------------------------------ #
    def _month_gradient(self, state: Dict[str, Any], batches: List[dict],
                        criterion: PopDailyLoss) -> Tuple[torch.Tensor, float]:
        """Pass 1: ``g_M = dL_3D/d(mean)`` and ``L_3D`` for a complete month."""
        state = dict(state)                       # shallow: the caller's state is untouched
        mean_pred: Optional[torch.Tensor] = None
        mean_target: Optional[torch.Tensor] = None
        with torch.no_grad():
            for batch in batches:
                state = self._advance(state, batch)
                _, pred, target = criterion.day_loss(state)
                mean_pred = pred.clone() if mean_pred is None else mean_pred.add_(pred)
                mean_target = target.clone() if mean_target is None else mean_target.add_(target)
        mean_pred = (mean_pred / len(batches)).requires_grad_(True)
        loss = criterion.monthly_loss(mean_pred, mean_target / len(batches))
        (grad,) = torch.autograd.grad(loss, mean_pred)
        return grad.detach(), float(loss)

    def _train_block(self, state: Dict[str, Any], batches: List[dict], complete: bool,
                     criterion: PopDailyLoss, scaler, day_scale: float, month_scale: float,
                     logs: Dict[str, List[float]]) -> Dict[str, Any]:
        """Run one block with gradients; returns the state after its last day."""
        grad_m = None
        if complete and criterion.monthly_weight > 0:
            rng = self._rng_state()
            grad_m, loss_m = self._month_gradient(state, batches, criterion)
            self._set_rng_state(rng)
            logs["monthly"].append(loss_m)
        for batch in batches:
            state = self._advance(state, batch)
            loss_d, pred, _ = criterion.day_loss(state)
            total = day_scale * loss_d
            if grad_m is not None:
                # Surrogate for the monthly-mean term: linear in this day's state.
                total = total + month_scale * (grad_m * pred).sum() / len(batches)
            scaler.scale(total).backward()
            logs["day"].append(float(loss_d))
            for k, v in criterion.last_var_losses.items():
                logs[f"var/{k}"].append(v)
            self._gather_for_next_step(state)     # detach the carried prediction
            state.pop("y_pred", None)
        return state

    # ------------------------------------------------------------------ #
    # Epoch
    # ------------------------------------------------------------------ #
    def train_one_epoch(self, epoch, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        if not isinstance(criterion, PopDailyLoss):
            raise TypeError("PopDailyTrainer requires the PopDailyLoss criterion")
        loader = self._begin_epoch(epoch, trainloader, criterion)
        if self.use_scheduler and self.scheduler_type == "lambda":
            scheduler.step()
        sampler = getattr(loader, "batch_sampler", None) or getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        n_seq = self.batches_per_epoch
        grad_accum_every = int(self.conf["trainer"].get("grad_accum_every", 1))
        progress = tqdm.tqdm(range(n_seq), total=n_seq, leave=True,
                             disable=not any(h.level <= logging.INFO
                                             for h in logging.getLogger().handlers))
        self.model.train()
        dl = cycle(loader)
        results: Dict[str, List[float]] = defaultdict(list)

        for seq in range(n_seq):
            boundary = (seq + 1) % grad_accum_every == 0 or seq == n_seq - 1
            if grad_accum_every > 1 and hasattr(self.model, "require_backward_grad_sync"):
                self.model.require_backward_grad_sync = boundary

            first = next(dl)
            day0 = _input_day(first["metadata"])
            blocks = month_blocks(day0 + 1, self.forecast_len)
            n_complete = sum(c for _, c in blocks)
            day_scale = 1.0 / (self.forecast_len * grad_accum_every)
            month_scale = criterion.monthly_weight / (max(n_complete, 1) * grad_accum_every)

            logs: Dict[str, List[float]] = defaultdict(list)
            state: Dict[str, Any] = {}
            pending: Optional[dict] = first
            for length, complete in blocks:
                batches = [pending] if pending is not None else []
                pending = None
                batches += [next(dl) for _ in range(length - len(batches))]
                state = self._train_block(state, batches, complete, criterion, scaler,
                                          day_scale, month_scale, logs)
            del state

            if boundary:
                scaler.unscale_(optimizer)
                if not isinstance(self.grad_max_norm, str) and self.grad_max_norm > 0.0:
                    clip_grad_norm_(self.model.parameters(), max_norm=self.grad_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if self.ema is not None:
                    self.ema.update(self.model)

            monthly = float(np.mean(logs["monthly"])) if logs["monthly"] else 0.0
            summary = {"loss": float(np.mean(logs["day"])) + criterion.monthly_weight * monthly,
                       "loss_var/monthly": monthly}
            summary.update({f"loss_{k}": float(np.mean(v)) for k, v in logs.items()
                            if k.startswith("var/")})
            for name, value in summary.items():
                value_t = torch.tensor([value], device=self.device)
                if self.distributed:
                    all_reduce_avg(value_t)
                results[f"train_{name}"].append(value_t[0].item())
            results["train_forecast_len"].append(self.forecast_len)
            results["train_history_len"].append(self.history_len)

            if not np.isfinite(results["train_loss"][-1]) and not self.skip_nan_prune:
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}, sequence {seq}")
            self._log_batch_progress(epoch, results, optimizer, progress, phase="train")

        progress.close()
        torch.cuda.empty_cache()
        gc.collect()
        return results
