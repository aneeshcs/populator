"""Autoregressive integration of the ocean state.

Given an initial condition and a sequence of surface forcing, step the emulator
forward N months. The same routine is used for multi-step training (with
gradients) and for evaluation rollouts (under ``torch.no_grad``).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .model import OceanEmulator
from .normalization import ForcingNormalizer, Normalizer, StatePacker


def month_cond(month_emb: torch.Tensor, forcing_norm: torch.Tensor) -> torch.Tensor:
    """Build the FiLM conditioning vector: month (sin,cos) + global forcing means."""
    gmean = forcing_norm.mean(dim=(-2, -1))           # (B, F)
    return torch.cat([month_emb, gmean], dim=-1)


def _advance_month_emb(month_emb: torch.Tensor) -> torch.Tensor:
    """Advance the cyclic month embedding by one month."""
    sin, cos = month_emb[:, 0], month_emb[:, 1]
    ang = torch.atan2(sin, cos) + 2.0 * np.pi / 12.0
    return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)


def rollout(model: OceanEmulator, packer: StatePacker, normalizer: Normalizer,
            fnorm: ForcingNormalizer,
            init_state: Dict[str, torch.Tensor],
            forcing_seq: List[Dict[str, torch.Tensor]],
            month_emb0: torch.Tensor,
            grid, mask_outputs: bool = True) -> List[Dict[str, torch.Tensor]]:
    """Integrate forward ``len(forcing_seq)`` steps.

    Returns the predicted physical-unit states at each step (excluding the
    initial condition).
    """
    device = month_emb0.device
    x = normalizer.normalize(packer.pack({k: v.to(device) for k, v in init_state.items()}))
    chan_mask = _channel_mask(packer, grid).to(device)
    month_emb = month_emb0
    preds = []

    for forcing in forcing_seq:
        fpack = torch.stack([forcing[v].to(device) for v in fnorm_order(packer, forcing)],
                            dim=1) if isinstance(forcing, dict) else forcing
        fnz = fnorm.normalize(fpack)
        cond = month_cond(month_emb, fnz)
        x = model(x, fnz, cond)
        if mask_outputs:
            x = x * chan_mask
        preds.append(packer.unpack(normalizer.denormalize(x)))
        month_emb = _advance_month_emb(month_emb)

    return preds


def fnorm_order(packer, forcing_dict):
    return list(forcing_dict.keys())


def _channel_mask(packer: StatePacker, grid) -> torch.Tensor:
    chans = []
    for _ in packer.prognostic:
        chans.append(grid.mask3d)
    for _ in packer.surface:
        chans.append(grid.mask2d.unsqueeze(0))
    return torch.cat(chans, dim=0).unsqueeze(0)
