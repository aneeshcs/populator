"""``OceanEmulator`` as a CREDIT Gen 2 model.

CREDIT calls ``model(x)`` with a single tensor assembled by the ``concat``
preblock, ``x: (B, C, T, J, I)`` with ``T = history_len`` and channels ordered

    [ state (prognostic 3-D then 2-D) | forcing | TIME_SIN, TIME_COS ]

This wrapper splits that tensor, builds the FiLM condition, applies the
input-only noise used for pushforward training, and masks land in the output.
The class derives from ``OceanEmulator`` so parameter names (``net.*``) are
identical to the legacy checkpoints.
"""
from __future__ import annotations

from typing import Optional

import torch

from credit.models.base_model import BaseModel

from ..model import OceanEmulator
from ..rollout import _channel_mask, month_cond
from .common import TIME_VARS, get_grid, load_pop_config, make_packer

_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


class PopOceanUNet(OceanEmulator, BaseModel):
    def __init__(self, pop_config: str, noise_injection: float = 0.0,
                 amp_dtype: Optional[str] = "bf16", levels: Optional[int] = None,
                 **_unused) -> None:
        cfg = load_pop_config(pop_config)
        packer = make_packer(cfg)
        if levels is not None and int(levels) != packer.nlev:
            raise ValueError(f"model.levels={levels} but the emulator config has nlev={packer.nlev}")
        grid = get_grid(pop_config, "cpu")
        OceanEmulator.__init__(self, cfg, packer, grid, n_forcing=len(cfg["data"]["forcing"]))
        self.noise_injection = float(noise_injection)
        self.amp_dtype = _AMP_DTYPES[amp_dtype] if amp_dtype else None
        # Land mask per output channel, (1, C, J, I); not part of the checkpoint.
        self.register_buffer("chan_mask", _channel_mask(packer, grid), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dim() == 4:                       # (B, C, J, I): history of one
            x = x.unsqueeze(2)
        n_s, n_f = self.n_state, self.n_forcing
        expected = n_s + n_f + len(TIME_VARS)
        if x.shape[1] != expected:
            raise ValueError(f"expected {expected} input channels "
                             f"({n_s} state + {n_f} forcing + 2 time), got {x.shape[1]}")

        state = x[:, :n_s].permute(0, 2, 1, 3, 4)            # (B, H, C, J, I), oldest first
        forcing = x[:, n_s:n_s + n_f, -1]                    # forcing of the newest input step
        time_emb = x[:, n_s + n_f:, -1].mean(dim=(-2, -1))   # (B, 2): sin, cos of the annual phase
        cond = month_cond(time_emb, forcing)

        base = state[:, -1]                                  # clean residual anchor
        net_in = state
        if self.training and self.noise_injection > 0:
            # Perturb the network input only (pushforward / denoiser training).
            net_in = state + self.noise_injection * torch.randn_like(state)
        if self.history == 1:
            net_in = net_in[:, 0]

        with torch.autocast(device_type=x.device.type, dtype=self.amp_dtype,
                            enabled=self.amp_dtype is not None and x.device.type == "cuda"):
            out = OceanEmulator.forward(self, net_in, forcing, cond, base=base)
        return out.float() * self.chan_mask
