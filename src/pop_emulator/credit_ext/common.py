"""Shared context for the CREDIT Gen 2 components.

Every component takes a ``pop_config`` argument: the path of a legacy-format
emulator YAML (e.g. ``configs/run5.yaml``).  That file remains the single
source of truth for variables, grid, statistics, architecture and physics
weights, so the CREDIT YAML only adds what CREDIT itself needs.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import torch

from ..grid import PopGrid, load_grid
from ..normalization import StatePacker
from ..utils import load_config, step_seconds  # noqa: F401  (step_seconds re-exported)

# Names of the two time-of-year channels the dataset appends to the forcing.
TIME_VARS: Tuple[str, str] = ("TIME_SIN", "TIME_COS")


def _resolve_path(path: str, config_path: str) -> str:
    """Resolve a path from the legacy config: absolute, relative to the
    working directory, or relative to the repository root (the parent of the
    directory holding the config)."""
    path = os.path.expandvars(path)
    if os.path.isabs(path) or os.path.exists(path):
        return os.path.abspath(path)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    return os.path.join(repo_root, path)


@lru_cache(maxsize=None)
def load_pop_config(pop_config: str) -> dict:
    """Load the legacy emulator config with file paths made absolute."""
    pop_config = os.path.expandvars(pop_config)
    cfg = load_config(pop_config)
    for key in ("grid_file", "stats_file", "daily_stats_file"):
        if key in cfg["data"]:
            cfg["data"][key] = _resolve_path(cfg["data"][key], pop_config)
    return cfg


def make_packer(cfg: dict) -> StatePacker:
    d = cfg["data"]
    return StatePacker(d["prognostic"], d["surface_prognostic"], d["nlev"])


@lru_cache(maxsize=None)
def get_grid(pop_config: str, device: str) -> PopGrid:
    """POP grid on ``device`` (cached per device; the grid is read-only)."""
    cfg = load_pop_config(pop_config)
    return load_grid(cfg["data"]["grid_file"], device=torch.device(device))


@lru_cache(maxsize=None)
def load_stats(pop_config: str) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """``{var: (mean, std)}`` from ``stats.nc``.

    3-D state variables carry per-level vectors of shape ``(nlev,)``; SSH and
    the forcing fields carry 0-d tensors.  ``std`` is floored at 1e-8, matching
    :class:`pop_emulator.normalization.Normalizer`.
    """
    import xarray as xr

    cfg = load_pop_config(pop_config)
    d = cfg["data"]
    names: List[str] = d["prognostic"] + d["surface_prognostic"] + d["forcing"]
    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    with xr.open_dataset(d["stats_file"]) as ds:
        for v in names:
            mean = torch.as_tensor(np.asarray(ds[f"{v}_mean"].values), dtype=torch.float32)
            std = torch.as_tensor(np.asarray(ds[f"{v}_std"].values), dtype=torch.float32)
            out[v] = (mean, std.clamp_min(1e-8))
    return out


def var_name(var_key: str) -> str:
    """``source/field_type/dim/NAME`` -> ``NAME``."""
    return var_key.rsplit("/", 1)[-1]


def find_key(variables: dict, name: str) -> str:
    """Key of variable ``name`` in a ``{var_key: tensor}`` source dict."""
    for key in variables:
        if var_name(key) == name:
            return key
    raise KeyError(f"variable '{name}' not found among {list(variables)}")
