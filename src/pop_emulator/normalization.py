"""State packing and per-level normalization.

The network operates on a single ``(B, C, J, I)`` tensor whose channels stack
every prognostic variable over its vertical levels. :class:`StatePacker`
converts between the dict-of-fields representation (used by the physics layer
and the data pipeline) and that packed tensor. :class:`Normalizer` standardizes
each channel using statistics computed by ``scripts/compute_stats.py``.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


class StatePacker:
    """Map between ``{var: tensor}`` and a packed ``(B, C, J, I)`` channel tensor.

    Channel order: each 3-D prognostic variable contributes ``nlev`` channels
    (level 0 first), followed by one channel per 2-D surface variable.
    """

    def __init__(self, prognostic: List[str], surface: List[str], nlev: int):
        self.prognostic = list(prognostic)
        self.surface = list(surface)
        self.nlev = nlev
        self.n_channels = len(self.prognostic) * nlev + len(self.surface)

        # Precompute channel slices for each variable.
        self.slices: Dict[str, slice] = {}
        c = 0
        for v in self.prognostic:
            self.slices[v] = slice(c, c + nlev)
            c += nlev
        for v in self.surface:
            self.slices[v] = slice(c, c + 1)
            c += 1

    def pack(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Dict -> (B, C, J, I). 3-D fields are (B, Z, J, I); 2-D are (B, J, I)."""
        chans = []
        for v in self.prognostic:
            chans.append(fields[v])
        for v in self.surface:
            chans.append(fields[v].unsqueeze(1))
        return torch.cat(chans, dim=1)

    def unpack(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """(B, C, J, I) -> dict. Inverse of :meth:`pack`."""
        out: Dict[str, torch.Tensor] = {}
        for v in self.prognostic:
            out[v] = x[:, self.slices[v]]
        for v in self.surface:
            out[v] = x[:, self.slices[v]].squeeze(1)
        return out


class Normalizer:
    """Per-channel standardization ``(x - mean) / std`` aligned to a packer."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        # shape (1, C, 1, 1)
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1).clamp_min(1e-8)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    # --- construction from a stats dataset --------------------------------- #
    @classmethod
    def from_stats(cls, stats_path: str, packer: StatePacker) -> "Normalizer":
        """Build a normalizer from ``stats.nc`` for the packer's channel order.

        Expects per-variable arrays ``{VAR}_mean`` / ``{VAR}_std`` of shape
        ``(nlev,)`` for 3-D variables and scalar for 2-D variables.
        """
        import xarray as xr

        means, stds = [], []
        with xr.open_dataset(stats_path) as ds:
            for v in packer.prognostic:
                means.append(np.asarray(ds[f"{v}_mean"].values).reshape(-1))
                stds.append(np.asarray(ds[f"{v}_std"].values).reshape(-1))
            for v in packer.surface:
                means.append(np.asarray(ds[f"{v}_mean"].values).reshape(-1))
                stds.append(np.asarray(ds[f"{v}_std"].values).reshape(-1))
        mean = torch.as_tensor(np.concatenate(means), dtype=torch.float32)
        std = torch.as_tensor(np.concatenate(stds), dtype=torch.float32)
        assert mean.numel() == packer.n_channels, (
            f"stats give {mean.numel()} channels, packer expects {packer.n_channels}"
        )
        return cls(mean, std)


class ForcingNormalizer:
    """Standardization for the surface forcing channels (each a 2-D field)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1).clamp_min(1e-8)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    @classmethod
    def from_stats(cls, stats_path: str, forcing: List[str]) -> "ForcingNormalizer":
        import xarray as xr

        means, stds = [], []
        with xr.open_dataset(stats_path) as ds:
            for v in forcing:
                means.append(float(np.asarray(ds[f"{v}_mean"].values).reshape(-1)[0]))
                stds.append(float(np.asarray(ds[f"{v}_std"].values).reshape(-1)[0]))
        return cls(torch.tensor(means), torch.tensor(stds))
