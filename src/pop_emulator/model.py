"""Physics-informed spherical U-Net for the POP ocean emulator.

The network maps the current ocean state (plus surface forcing and static grid
fields) to a *tendency*, which is added to the input to form the next state::

    x(t+Δt) = x(t) + Δt · F_theta(x(t), forcing(t), grid)

Predicting the tendency (rather than the next state directly) mirrors POP's
prognostic time step, is exact in the Δt→0 limit, and makes autoregressive
rollout markedly more stable.

Grid awareness
--------------
* Longitude is periodic -> **circular** padding in the last axis.
* Latitude terminates in a tripole fold -> **replicate** padding (approx.).
* Static fields (Coriolis f, ocean mask, normalized depth, lat/lon encodings)
  are concatenated as input channels so the network can represent
  geostrophic/Coriolis dynamics and respect topography.
* The land mask is reapplied to the output so land cells never carry signal.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Grid-aware padding and convolution.
# --------------------------------------------------------------------------- #
def ocean_pad(x: torch.Tensor, pad: int = 1) -> torch.Tensor:
    """Circular pad in longitude (last axis), replicate pad in latitude."""
    x = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)         # lon wrap
    x = F.pad(x, (0, 0, pad, pad), mode="replicate")               # lat replicate
    return x


class OceanConv2d(nn.Module):
    """3x3 convolution with ocean-aware padding and optional stride-2 downsample."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=0)

    def forward(self, x):
        return self.conv(ocean_pad(x, 1))


class FiLM(nn.Module):
    """Feature-wise linear modulation from a global conditioning vector."""

    def __init__(self, cond_dim, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, channels), nn.SiLU(), nn.Linear(channels, 2 * channels)
        )
        self.channels = channels

    def forward(self, x, cond):
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        g = min(groups, channels)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = OceanConv2d(channels, channels)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = OceanConv2d(channels, channels)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class OceanUNet(nn.Module):
    """U-Net backbone with ocean-aware padding and FiLM conditioning."""

    def __init__(self, cin, cout, width=96, depth=4, blocks=2, cond_dim=8):
        super().__init__()
        self.stem = OceanConv2d(cin, width)

        self.downs = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        ch = width
        chans = [ch]
        for _ in range(depth):
            self.down_blocks.append(nn.Sequential(*[ResBlock(ch) for _ in range(blocks)]))
            self.downs.append(OceanConv2d(ch, ch * 2, stride=2))
            ch *= 2
            chans.append(ch)

        self.mid1 = ResBlock(ch)
        self.film = FiLM(cond_dim, ch)
        self.mid2 = ResBlock(ch)

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(OceanConv2d(ch, ch // 2))
            ch //= 2
            # skip connection doubles the channel count going into the blocks
            self.up_blocks.append(nn.Sequential(
                OceanConv2d(ch * 2, ch), *[ResBlock(ch) for _ in range(blocks)]
            ))

        self.out_norm = nn.GroupNorm(min(8, width), width)
        self.out_conv = OceanConv2d(width, cout)
        nn.init.zeros_(self.out_conv.conv.weight)  # start as identity (zero tendency)
        nn.init.zeros_(self.out_conv.conv.bias)

    def forward(self, x, cond):
        h = self.stem(x)
        skips = []
        for blk, down in zip(self.down_blocks, self.downs):
            h = blk(h)
            skips.append(h)
            h = down(h)

        h = self.mid1(h)
        h = self.film(h, cond)
        h = self.mid2(h)

        for up, blk, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
            h = up(h)
            h = blk(torch.cat([h, skip], dim=1))

        return self.out_conv(F.silu(self.out_norm(h)))


# --------------------------------------------------------------------------- #
# Emulator wrapper: builds static channels, applies the residual update,
# enforces the land mask. Operates in normalized space; physics losses
# denormalize outside.
# --------------------------------------------------------------------------- #
class OceanEmulator(nn.Module):
    def __init__(self, cfg: dict, packer, grid, n_forcing: int):
        super().__init__()
        m = cfg["model"]
        self.packer = packer
        self.n_state = packer.n_channels
        self.n_forcing = n_forcing
        self.residual = m.get("residual_update", True)

        # Static channels: f, mask2d, normalized depth-at-column, sin/cos(lat),
        # sin/cos(lon). Registered as buffers, built lazily on first forward.
        self.register_buffer("static_ch", torch.zeros(1), persistent=False)
        self._static_built = False
        self.grid = grid

        n_static = 7
        cin = self.n_state + n_forcing + n_static
        cond_dim = 2 + n_forcing  # month (sin,cos) + global forcing means
        self.net = OceanUNet(
            cin, self.n_state,
            width=m.get("width", 96), depth=m.get("depth", 4),
            blocks=m.get("blocks_per_stage", 2), cond_dim=cond_dim,
        )

    def _build_static(self, device, dtype):
        g = self.grid
        f = g.fcor
        fn = f / (f.abs().amax().clamp_min(1e-12))
        depth = (g.kmt.float() / g.kmt.float().amax().clamp_min(1.0))
        lat = torch.deg2rad(g.tlat)
        lon = torch.deg2rad(g.tlong)
        ch = torch.stack([
            fn, g.mask2d, depth,
            torch.sin(lat), torch.cos(lat), torch.sin(lon), torch.cos(lon),
        ], dim=0).to(device=device, dtype=dtype)
        self.static_ch = ch.unsqueeze(0)
        self._static_built = True

    def forward(self, x_state_norm: torch.Tensor, forcing_norm: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """``x_state_norm`` (B, C, J, I), ``forcing_norm`` (B, F, J, I),
        ``cond`` (B, cond_dim). Returns next-state in normalized space."""
        if not self._static_built:
            self._build_static(x_state_norm.device, x_state_norm.dtype)
        B = x_state_norm.shape[0]
        static = self.static_ch.expand(B, -1, -1, -1)
        inp = torch.cat([x_state_norm, forcing_norm, static], dim=1)
        out = self.net(inp, cond)
        if self.residual:
            out = x_state_norm + out
        return out
