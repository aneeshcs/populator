"""Preblocks for the POP emulator under CREDIT Gen 2."""
from __future__ import annotations

import torch

from credit.preblock.base import BasePreblock

from .common import load_stats, var_name


def standardize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                inverse: bool = False) -> torch.Tensor:
    """(De)standardise a ``(B, n_levels, T, J, I)`` field.

    ``mean`` / ``std`` are per-level vectors for 3-D variables and scalars for
    2-D variables.
    """
    mean = mean.to(device=tensor.device, dtype=tensor.dtype)
    std = std.to(device=tensor.device, dtype=tensor.dtype)
    if mean.dim() == 1:
        mean = mean.view(1, -1, 1, 1, 1)
        std = std.view(1, -1, 1, 1, 1)
    return tensor * std + mean if inverse else (tensor - mean) / std


class PopNormalize(BasePreblock):
    """Standardise state and forcing fields with the statistics in ``stats.nc``.

    Identical to ``Normalizer`` / ``ForcingNormalizer`` in the legacy loop, so
    weights trained there see the same inputs.  Variables without statistics
    (the time-of-year channels) pass through unchanged.
    """

    def __init__(self, pop_config: str) -> None:
        super().__init__()
        self.stats = load_stats(pop_config)

    def forward(self, batch: dict) -> dict:
        batch = self._copy_batch(batch)
        for data_type in self.VALID_DATA_TYPES:
            for variables in (batch.get(data_type) or {}).values():
                for key, tensor in variables.items():
                    name = var_name(key)
                    if name in self.stats:
                        variables[key] = standardize(tensor, *self.stats[name])
        return batch
