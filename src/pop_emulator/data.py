"""CESM-LENS POP dataset and dataloader.

Each prognostic / forcing variable lives in its own single-variable time-series
directory, e.g.::

    {root}/TEMP/b.e11.{experiment}.f09_g16.{member}.pop.h.TEMP.{period}.nc

A single member's 3-D field is ~23 GB, so files are opened lazily with xarray
and only the needed time slices are materialized. A sample yields the ocean
state at month ``t`` and ``t + step`` (the integration target) plus the surface
forcing at month ``t``.

Returned tensors are in **physical units** (POP CGS); normalization is applied
downstream so the physics layer always sees physical fields.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

POP_FILL = 1.0e30  # values above this are land / missing


def member_files(root: str, experiment: str, member: str, var: str) -> List[str]:
    pat = os.path.join(
        root, var, f"b.e11.{experiment}.f09_g16.{member}.pop.h.{var}.*.nc"
    )
    return sorted(glob.glob(pat))


def _open(root, experiment, member, var):
    import xarray as xr

    files = member_files(root, experiment, member, var)
    if not files:
        raise FileNotFoundError(
            f"no files for var={var} member={member} exp={experiment} under {root}"
        )
    if len(files) == 1:
        return xr.open_dataset(files[0], decode_times=False,
                               chunks={"time": 1})[var]
    return xr.open_mfdataset(files, decode_times=False, combine="by_coords",
                             chunks={"time": 1})[var]


def _clean(arr: np.ndarray) -> np.ndarray:
    """Replace POP fill values with 0 (land cells are masked in the loss)."""
    a = np.asarray(arr, dtype=np.float32)
    a[~np.isfinite(a)] = 0.0
    a[np.abs(a) > POP_FILL] = 0.0
    return a


class PopLENSDataset(Dataset):
    """Pairs of (state_t, forcing_t, state_{t+step}) over CESM-LENS members."""

    def __init__(self, cfg: dict, members: List[str], split: str = "train",
                 rollout_steps: int = 1):
        d = cfg["data"]
        self.root = d["root"]
        self.experiment = d["experiment"]
        self.prognostic = d["prognostic"]
        self.surface = d["surface_prognostic"]
        self.forcing = d["forcing"]
        self.nlev = d["nlev"]
        self.step = d.get("step_months", 1)
        self.rollout = max(1, int(rollout_steps))
        self.members = list(members)
        self.split = split

        self._cache: Dict[tuple, object] = {}  # (member, var) -> DataArray

        # Build the global (member, t) sample index, honoring the time window.
        self.index: List[tuple] = []
        self._ntime: Dict[str, int] = {}
        t_start = d.get("t_start", 0)
        t_end = d.get("t_end", -1)
        span = self.step * self.rollout
        for m in self.members:
            da = self._get(m, self.prognostic[0])
            nt = da.sizes["time"]
            self._ntime[m] = nt
            hi = nt if t_end in (-1, None) else min(t_end, nt)
            lo = max(0, t_start)
            for t in range(lo, hi - span):
                self.index.append((m, t))

    # --- lazy handle cache ------------------------------------------------- #
    def _get(self, member: str, var: str):
        key = (member, var)
        if key not in self._cache:
            self._cache[key] = _open(self.root, self.experiment, member, var)
        return self._cache[key]

    def _read_slice(self, member: str, var: str, t: int) -> torch.Tensor:
        da = self._get(member, var)
        arr = da.isel(time=t).values  # (Z, J, I) or (J, I)
        return torch.from_numpy(_clean(arr))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        member, t = self.index[idx]
        statevars = self.prognostic + self.surface

        # Initial condition at t, then `rollout` targets and forcings.
        state_t = {v: self._read_slice(member, v, t) for v in statevars}
        targets = {v: torch.stack(
            [self._read_slice(member, v, t + self.step * (s + 1))
             for s in range(self.rollout)], dim=0) for v in statevars}
        forcing = {v: torch.stack(
            [self._read_slice(member, v, t + self.step * s)
             for s in range(self.rollout)], dim=0) for v in self.forcing}

        # Month-of-year (noleap) as a cyclic embedding for the seasonal cycle.
        month = t % 12
        ang = 2.0 * np.pi * month / 12.0
        month_emb = torch.tensor([np.sin(ang), np.cos(ang)], dtype=torch.float32)

        return {
            "state_t": state_t,        # {var: (Z,J,I) or (J,I)}
            "targets": targets,        # {var: (R,Z,J,I) or (R,J,I)}
            "forcing": forcing,        # {var: (R,J,I)}
            "month_emb": month_emb,
            "member": member,
            "t": t,
        }


def collate(batch: List[dict]) -> dict:
    """Stack a list of samples into batched dict tensors (batch axis first)."""
    def stack_dict(key):
        keys = batch[0][key].keys()
        return {v: torch.stack([b[key][v] for b in batch], dim=0) for v in keys}

    return {
        "state_t": stack_dict("state_t"),   # {var: (B,Z,J,I)/(B,J,I)}
        "targets": stack_dict("targets"),   # {var: (B,R,Z,J,I)/(B,R,J,I)}
        "forcing": stack_dict("forcing"),   # {var: (B,R,J,I)}
        "month_emb": torch.stack([b["month_emb"] for b in batch], dim=0),
        "member": [b["member"] for b in batch],
        "t": torch.tensor([b["t"] for b in batch]),
    }


def make_loader(cfg: dict, members: List[str], split: str, shuffle: bool,
                rollout_steps: int = 1):
    from torch.utils.data import DataLoader

    ds = PopLENSDataset(cfg, members, split=split, rollout_steps=rollout_steps)
    nw = cfg["train"].get("num_workers", 4)
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=nw,
        collate_fn=collate,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=nw > 0,
    )
