"""Loss and metric for the POP emulator under CREDIT Gen 2."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from credit.losses.base import BaseLoss

from ..losses import CompositeLoss
from ..normalization import Normalizer
from ..rollout import _channel_mask
from .common import get_grid, load_pop_config, make_packer
from .postblocks import LOSS_KEY, newest_step, physical_inputs


class PopCompositeLoss(BaseLoss):
    """Masked, tendency-weighted data loss plus the physics penalties.

    Wraps :class:`pop_emulator.losses.CompositeLoss`.  It subclasses CREDIT's
    ``BaseLoss`` only so the Gen 2 trainer passes the full data dict; none of
    ``BaseLoss``'s own machinery is used.  The prediction scored is
    ``y_processed`` (physical units, after the conservation projection), so
    gradients flow through the postblocks, as in the legacy pushforward loop.

    ``rollout_len`` divides the loss so that summing the per-step gradients over
    a rollout gives their mean (the legacy ``loss / rollout_len`` scaling); the
    trainer sets it to the current curriculum length.
    """

    def __init__(self, pop_config: str, validation: bool = False) -> None:
        nn.Module.__init__(self)
        self.pop_config = pop_config
        self.cfg = load_pop_config(pop_config)
        self.packer = make_packer(self.cfg)
        self.state_vars = self.packer.prognostic + self.packer.surface
        self.forcing_vars = self.cfg["data"]["forcing"]
        self.rollout_len = 1
        # Attributes the Gen 2 trainer reads on any BaseLoss.
        self.log_variance = None
        self.last_var_losses: Dict[str, float] = {}
        self._impl: Dict[str, CompositeLoss] = {}

    def _composite(self, device: torch.device) -> CompositeLoss:
        key = str(device)
        if key not in self._impl:
            grid = get_grid(self.pop_config, key)
            normalizer = Normalizer.from_stats(self.cfg["data"]["stats_file"], self.packer).to(device)
            self._impl[key] = CompositeLoss(self.cfg, self.packer, normalizer, grid).to(device)
        return self._impl[key]

    def _prepare(self, full_data_dict: dict):
        """Normalised prediction, previous state and target, plus physical forcing.

        Returns ``(composite, pred_norm, prev_norm, target_norm, forcing_phys)``.
        """
        # LOSS_KEY holds the gradient-attached prediction (see postblocks.py);
        # y_processed itself is detached by the trainer before the loss is called.
        processed = full_data_dict.get(LOSS_KEY) or full_data_dict["y_processed"]
        source, variables = next(iter(processed.items()))
        by_name = {k.rsplit("/", 1)[-1]: v for k, v in variables.items()}
        device = by_name["TEMP"].device
        comp = self._composite(device)

        pred_phys = {v: newest_step(by_name[v].float()) for v in self.state_vars}
        prev_phys, forcing_phys = physical_inputs(
            full_data_dict, source, self.state_vars, self.forcing_vars, device)

        norm = comp.normalizer
        pred_norm = norm.normalize(self.packer.pack(pred_phys)) * comp.chan_mask
        prev_norm = norm.normalize(self.packer.pack(prev_phys))
        target_norm = full_data_dict["y"].float().to(device)
        if target_norm.dim() == 5:
            target_norm = target_norm.flatten(1, 2)

        if torch.is_grad_enabled() and not pred_norm.requires_grad:
            raise RuntimeError(
                f"{type(self).__name__}: the prediction carries no gradient. Use "
                "'detach: false' on the reconstruct postblock and keep PopDenormalize "
                "(and the projection) after it in postblocks.per_step.")
        return comp, pred_norm, prev_norm, target_norm, forcing_phys

    def _log(self, logs: Dict[str, torch.Tensor], full_data_dict: dict) -> None:
        self.last_var_losses = {
            k: float(v) for k, v in logs.items() if k not in ("total", "phys_scale")
        }
        for k, v in (full_data_dict.get("projection_diag") or {}).items():
            self.last_var_losses[k] = float(v)

    def forward(self, full_data_dict: dict) -> torch.Tensor:
        comp, pred_norm, prev_norm, target_norm, forcing_phys = self._prepare(full_data_dict)
        res = comp(pred_norm, target_norm, prev_norm, forcing_phys=forcing_phys)
        self._log(res["logs"], full_data_dict)
        return res["loss"] / self.rollout_len


class PopDailyLoss(PopCompositeLoss):
    """Surface-supervised loss for the daily step (see ``trainer_daily.py``).

    Per day:  ``L_d = w_sfc * L_sfc + L_phys`` where ``L_sfc`` is the ocean-masked
    MSE of SST (TEMP level 0) and SSH against daily truth, each channel weighted
    by ``(sigma_field / sigma_daily_tendency)^2`` so persistence scores O(1), and
    ``L_phys`` is the continuity / barotropic / stability / spectral penalty of
    the legacy loss.  The 3-D data term of the legacy loss is not applied per
    day (no daily 3-D truth exists); it is applied to calendar-month means of
    the rollout through :meth:`monthly_loss`.

    ``data.daily_stats_file`` in the emulator config must hold ``SST_tend_std``
    and ``SSH_tend_std`` (physical units; ``scripts/compute_daily_stats.py``).
    """

    def __init__(self, pop_config: str, surface_weight: float = 1.0,
                 monthly_weight: float = 1.0, validation: bool = False) -> None:
        super().__init__(pop_config, validation)
        self.surface_weight = float(surface_weight)
        self.monthly_weight = float(monthly_weight)
        self._sfc_w: Dict[str, torch.Tensor] = {}

    def _composite(self, device: torch.device) -> CompositeLoss:
        key = str(device)
        if key not in self._impl:
            comp = super()._composite(device)
            # Legacy 3-D data term is handled by monthly_loss(); keep physics only.
            comp.w = dict(comp.w, data=0.0)
        return self._impl[key]

    def _surface_weights(self, comp: CompositeLoss) -> torch.Tensor:
        """``(1, C, 1, 1)`` weights, non-zero on TEMP level 0 and SSH only."""
        key = str(comp.chan_mask.device)
        if key not in self._sfc_w:
            import xarray as xr

            n_chan = comp.chan_mask.shape[1]
            w = torch.zeros(n_chan)
            channel = {"TEMP": self.packer.prognostic.index("TEMP") * self.packer.nlev,
                       "SSH": len(self.packer.prognostic) * self.packer.nlev
                       + self.packer.surface.index("SSH")}
            with xr.open_dataset(self.cfg["data"]["stats_file"]) as st, \
                    xr.open_dataset(self.cfg["data"]["daily_stats_file"]) as daily:
                sigma = {"TEMP": float(st["TEMP_std"].values.reshape(-1)[0]),
                         "SSH": float(st["SSH_std"].values)}
                tend = {"TEMP": float(daily["SST_tend_std"].values),
                        "SSH": float(daily["SSH_tend_std"].values)}
            for v, c in channel.items():
                w[c] = (sigma[v] / tend[v]) ** 2
            self._sfc_w[key] = w.view(1, -1, 1, 1).to(comp.chan_mask.device)
        return self._sfc_w[key]

    def day_loss(self, full_data_dict: dict):
        """Surface + physics loss of one daily step.

        Returns ``(loss, pred_norm, target_norm)``; the normalised fields feed the
        monthly-mean term.
        """
        comp, pred_norm, prev_norm, target_norm, forcing_phys = self._prepare(full_data_dict)
        w = self._surface_weights(comp)
        mask = comp.chan_mask * (w > 0)
        surface = ((pred_norm - target_norm) ** 2 * comp.chan_mask * w).sum() / mask.sum().clamp_min(1.0)
        res = comp(pred_norm, target_norm, prev_norm, forcing_phys=forcing_phys)   # data weight 0
        logs = dict(res["logs"], surface=surface.detach())
        logs.pop("data", None)
        self._log(logs, full_data_dict)
        return self.surface_weight * surface + res["loss"], pred_norm, target_norm

    def monthly_loss(self, mean_pred_norm: torch.Tensor, mean_target_norm: torch.Tensor) -> torch.Tensor:
        """Legacy tendency-weighted masked MSE (monthly tendency units) between the
        calendar-month mean of the daily rollout and the monthly-mean truth."""
        comp = self._composite(mean_pred_norm.device)
        diff2 = (mean_pred_norm - mean_target_norm) ** 2 * comp.chan_mask
        if comp.chan_weight is not None:
            diff2 = diff2 * comp.chan_weight
        return diff2.sum() / comp.chan_mask.sum().clamp_min(1.0)

    def forward(self, full_data_dict: dict) -> torch.Tensor:
        # Used by validation: surface + physics loss at the final lead day.
        return self.day_loss(full_data_dict)[0]


class PopMaskedRMSE(nn.Module):
    """Ocean-masked RMSE of the normalised prediction (training monitor)."""

    def __init__(self, pop_config: str, metric_name: str = "rmse") -> None:
        super().__init__()
        self.pop_config = pop_config
        self.packer = make_packer(load_pop_config(pop_config))
        self._mask: Dict[str, torch.Tensor] = {}

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        key = str(y_pred.device)
        if key not in self._mask:
            self._mask[key] = _channel_mask(self.packer, get_grid(self.pop_config, key))
        mask = self._mask[key]
        y = y.to(y_pred.device)
        if y.dim() == 5:
            y = y.flatten(1, 2)
        if y_pred.dim() == 5:
            y_pred = y_pred.flatten(1, 2)
        se = ((y_pred.detach().float() - y.float()) ** 2 * mask).sum()
        n = mask.sum().clamp_min(1.0) * y_pred.shape[0]
        return {"rmse": float(torch.sqrt(se / n))}
