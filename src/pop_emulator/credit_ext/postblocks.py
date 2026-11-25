"""Postblocks for the POP emulator under CREDIT Gen 2.

Chain (``postblocks.per_step``)::

    reconstruct (detach: false)        y_pred -> y_processed (normalised, nested)
    PopDenormalize                     y_processed -> physical units (POP CGS)
    OceanConservationProjection        close the global heat and salt budgets

``y_processed`` leaves the chain in physical units, which is what the trainer
feeds back as the next step's input and what ``PopCompositeLoss`` scores.

Gradient note: ``TrainerERA5Gen2`` detaches the tensors inside ``y_processed``
in place (``_gather_for_next_step``) *before* it evaluates the loss, on every
rollout step but the last.  Each block here therefore also stores a shallow
copy of its output under ``LOSS_KEY``; the copy keeps the gradient-attached
tensors, and ``PopCompositeLoss`` reads it.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from credit.postblock.base import BasePostblock

from .. import physics
from .common import find_key, get_grid, load_pop_config, load_stats, step_seconds, var_name
from .preblocks import standardize

LOSS_KEY = "y_processed_for_loss"


def keep_for_loss(batch_dict: dict, nested: dict) -> None:
    """Store a shallow copy of ``nested`` whose tensors stay attached to the graph."""
    batch_dict[LOSS_KEY] = {source: dict(variables) for source, variables in nested.items()}


class PopDenormalize(BasePostblock):
    """Invert :class:`PopNormalize` on a nested prediction dict."""

    def __init__(self, pop_config: str, key: str = "y_processed") -> None:
        super().__init__()
        self.stats = load_stats(pop_config)
        self.key = key

    def forward(self, batch_dict: dict) -> dict:
        out = {}
        for source, variables in batch_dict[self.key].items():
            out[source] = {
                k: (standardize(v, *self.stats[var_name(k)], inverse=True)
                    if var_name(k) in self.stats else v)
                for k, v in variables.items()
            }
        batch_dict[self.key] = out
        keep_for_loss(batch_dict, out)
        return batch_dict


def newest_step(tensor: torch.Tensor) -> torch.Tensor:
    """``(B, L, T, J, I)`` -> newest time step, ``(B, L, J, I)`` (``(B, J, I)`` if L == 1)."""
    x = tensor[:, :, -1]
    return x[:, 0] if x.shape[1] == 1 else x


def physical_inputs(batch_dict: dict, source: str, state_vars, forcing_vars, device):
    """Previous state and forcing in physical units from ``x_physical``.

    ``x_physical`` is the named input the model consumed this step: the raw
    initial condition at the first rollout step, the previous (projected)
    prediction afterwards.
    """
    x_phys = batch_dict["x_physical"][source]
    prev = {v: newest_step(x_phys[find_key(x_phys, v)]).to(device) for v in state_vars}
    forcing = {v: newest_step(x_phys[find_key(x_phys, v)]).to(device) for v in forcing_vars}
    return prev, forcing


class OceanConservationProjection(BasePostblock):
    """Exact global heat and salt budget closure after each model step.

    Wraps :func:`pop_emulator.physics.conservation_projection`: a spatially
    uniform, differentiable offset is added to TEMP and SALT so the change in
    global heat (salt) content equals ``dt`` times the area-integrated SHF
    (SFWF).  ``dt`` defaults to ``physics.dt_seconds`` in the emulator config
    (one POP month if unset).
    """

    def __init__(self, pop_config: str, dt_seconds: Optional[float] = None,
                 key: str = "y_processed") -> None:
        super().__init__()
        self.pop_config = pop_config
        cfg = load_pop_config(pop_config)
        self.state_vars = cfg["data"]["prognostic"] + cfg["data"]["surface_prognostic"]
        self.forcing_vars = cfg["data"]["forcing"]
        self.dt = float(dt_seconds) if dt_seconds is not None else step_seconds(cfg)
        self.key = key

    def forward(self, batch_dict: dict) -> dict:
        out = {}
        for source, variables in batch_dict[self.key].items():
            keys: Dict[str, str] = {v: find_key(variables, v) for v in self.state_vars}
            device = variables[keys["TEMP"]].device
            grid = get_grid(self.pop_config, str(device))
            pred = {v: newest_step(variables[keys[v]].float()) for v in self.state_vars}
            prev, forcing = physical_inputs(
                batch_dict, source, self.state_vars, self.forcing_vars, device)
            corrected, diag = physics.conservation_projection(
                pred, prev, forcing, grid, dt=self.dt)
            new_vars = dict(variables)
            for v in ("TEMP", "SALT"):
                new_vars[keys[v]] = corrected[v].unsqueeze(2)   # restore the time dim
            out[source] = new_vars
            batch_dict["projection_diag"] = diag
        batch_dict[self.key] = out
        keep_for_loss(batch_dict, out)
        return batch_dict
