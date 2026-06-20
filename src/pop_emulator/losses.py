"""Composite data + physics loss for the POP ocean emulator.

The data term is a masked, per-channel MSE in normalized space. The physics
terms (continuity, barotropic/volume, heat & salt budgets, static stability)
operate on the denormalized fields so they are evaluated in physical units and
with POP's own constants. Physics weights are ramped in via a curriculum
(``physics_warmup_steps``) so the model learns the data map before the
conservation constraints tighten.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from . import physics
from .grid import PopGrid
from .normalization import Normalizer, StatePacker


class CompositeLoss:
    def __init__(self, cfg: dict, packer: StatePacker, normalizer: Normalizer,
                 grid: PopGrid):
        self.cfg = cfg
        self.packer = packer
        self.normalizer = normalizer
        self.grid = grid
        self.w = cfg["physics"]["loss_weights"]
        self.warmup = cfg["physics"].get("physics_warmup_steps", 0)

        # Per-channel ocean mask aligned to the packer layout, shape (1, C, J, I).
        mask_chans = []
        for v in packer.prognostic:
            mask_chans.append(grid.mask3d)                       # (Z, J, I)
        for v in packer.surface:
            mask_chans.append(grid.mask2d.unsqueeze(0))          # (1, J, I)
        self.chan_mask = torch.cat(mask_chans, dim=0).unsqueeze(0)

        # Optional per-channel tendency weighting: re-expresses the data loss in
        # tendency units so persistence is no longer the easy minimum. Enabled by
        # normalize.tendency_weighted_loss and the *_tend_std fields in stats.nc.
        self.chan_weight = None
        if cfg.get("normalize", {}).get("tendency_weighted_loss", False):
            from .normalization import tendency_loss_weights
            stats_file = cfg["data"].get("stats_file")
            w = tendency_loss_weights(stats_file, packer) if stats_file else None
            if w is not None:
                self.chan_weight = w.view(1, -1, 1, 1)

    def to(self, device):
        self.chan_mask = self.chan_mask.to(device)
        if self.chan_weight is not None:
            self.chan_weight = self.chan_weight.to(device)
        return self

    def _phys_scale(self, step: int) -> float:
        if self.warmup <= 0:
            return 1.0
        return min(1.0, step / float(self.warmup))

    def __call__(self, pred_norm: torch.Tensor, target_norm: torch.Tensor,
                 input_norm: torch.Tensor,
                 forcing_phys: Optional[Dict[str, torch.Tensor]] = None,
                 step: int = 10 ** 9) -> Dict[str, torch.Tensor]:
        mask = self.chan_mask
        # --- data term: masked per-channel MSE in normalized space --------- #
        # With tendency weighting, each channel is scaled by (σ_field/σ_tend)²,
        # so the squared error is measured in tendency units (persistence -> O(1)).
        diff2 = (pred_norm - target_norm) ** 2 * mask
        if self.chan_weight is not None:
            diff2 = diff2 * self.chan_weight
        denom = mask.sum().clamp_min(1.0)
        data_loss = diff2.sum() / denom

        total = self.w.get("data", 1.0) * data_loss
        logs = {"data": data_loss.detach()}

        scale = self._phys_scale(step)

        # Denormalize to physical units for the physics terms.
        pred_phys = self.packer.unpack(self.normalizer.denormalize(pred_norm))
        prev_phys = self.packer.unpack(self.normalizer.denormalize(input_norm))
        true_phys = self.packer.unpack(self.normalizer.denormalize(target_norm))

        cons = physics.conservation_losses(pred_phys, prev_phys, self.grid,
                                            forcing=forcing_phys)
        for key in ("continuity", "barotropic", "heat", "salt"):
            wk = self.w.get(key, 0.0)
            if wk > 0:
                term = cons[key]
                total = total + scale * wk * term
                logs[key] = term.detach()
        logs["heat_drift_J"] = cons["heat_drift_J"]
        logs["salt_drift"] = cons["salt_drift"]

        w_stab = self.w.get("stability", 0.0)
        if w_stab > 0:
            stab = physics.static_stability_penalty(
                pred_phys["TEMP"], pred_phys["SALT"],
                true_phys["TEMP"], true_phys["SALT"], self.grid)
            total = total + scale * w_stab * stab
            logs["stability"] = stab.detach()

        w_spec = self.w.get("spectral", 0.0)
        if w_spec > 0:
            spec = physics.spectral_penalty(
                pred_phys["UVEL"], pred_phys["VVEL"], self.grid)
            total = total + scale * w_spec * spec
            logs["spectral"] = spec.detach()

        logs["total"] = total.detach()
        logs["phys_scale"] = torch.tensor(scale)
        return {"loss": total, "logs": logs}
