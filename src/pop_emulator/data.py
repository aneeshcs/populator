"""CESM-LENS POP dataset and dataloader.

Each prognostic / forcing variable lives in its own single-variable time-series
directory, e.g.::

    {root}/{var}/b.e11.{experiment}.f09_g16.{member}.pop.h.{var}.{period}.nc

A single member's 3-D field is ~23 GB, so files are opened lazily and only
the needed time window is materialized per sample.  A sample yields the ocean
state at months t0-(H-1)..t0 (history) and t0+1..t0+R (targets) plus the
surface forcing at months t0..t0+R-1.  The full window is loaded in a single
contiguous I/O call per variable rather than one call per time step.

Backend selection
-----------------
- **Zarr** (preferred): set ``data.zarr_root`` in the config.  Stores are
  written by ``scripts/rechunk_to_zarr.py`` with ``time_chunk=300``, so any
  rollout window (≤258 months) touches at most 2 time chunks → ~16 disk reads
  per 3-D variable vs ~1040 for the raw netCDF4 files.
- **netCDF4** (fallback): used when ``zarr_root`` is absent or the store does
  not exist.  Adequate for short rollouts; becomes the bottleneck at rl≥128.

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


def _open_netcdf(root, experiment, member, var):
    """Open via xarray (netCDF4 backend, lazy).  Falls back to mfdataset for
    multi-file members."""
    import xarray as xr

    files = member_files(root, experiment, member, var)
    if not files:
        raise FileNotFoundError(
            f"no files for var={var} member={member} exp={experiment} under {root}"
        )
    if len(files) == 1:
        return xr.open_dataset(files[0], decode_times=False)[var]
    return xr.open_mfdataset(files, decode_times=False, combine="by_coords")[var]


def _open_zarr(zarr_root, experiment, member, var):
    """Open a rechunked Zarr store.  Returns a zarr.Array (fast direct indexing)."""
    import zarr

    path = os.path.join(zarr_root, experiment, member, f"{var}.zarr", var)
    if not os.path.exists(path):
        raise FileNotFoundError(f"zarr store not found: {path}")
    return zarr.open_array(path, mode="r", zarr_format=2)


def _n_time(handle) -> int:
    """Number of time steps from either a zarr.Array or xarray.DataArray."""
    if hasattr(handle, "sizes"):
        return handle.sizes["time"]
    return int(handle.shape[0])


def _clean(arr: np.ndarray) -> np.ndarray:
    """Replace POP fill values with 0 (land cells are masked in the loss)."""
    a = np.asarray(arr, dtype=np.float32)
    a[~np.isfinite(a)] = 0.0
    a[np.abs(a) > POP_FILL] = 0.0
    return a


def _read_slice(handle, t_lo: int, t_hi: int) -> np.ndarray:
    """Read time slice [t_lo, t_hi) from either a zarr.Array or xarray.DataArray."""
    if hasattr(handle, "isel"):
        return _clean(handle.isel(time=slice(t_lo, t_hi)).values)
    return _clean(np.asarray(handle[t_lo:t_hi]))


class PopLENSDataset(Dataset):
    """Pairs of (state_hist, forcing, targets) over CESM-LENS members.

    Each call to __getitem__ reads the entire history+rollout window for every
    variable in a single contiguous I/O call, processing one variable at a time
    to keep peak memory proportional to one window (not all variables at once).

    I/O backend is selected per-variable at init time: Zarr store if available
    (set ``data.zarr_root`` in config), else netCDF4 via xarray.
    """

    def __init__(self, cfg: dict, members: List[str], split: str = "train",
                 rollout_steps: int = 1, history: int = None):
        d = cfg["data"]
        self.root = d["root"]
        self.zarr_root: Optional[str] = d.get("zarr_root", None)
        self.experiment = d["experiment"]
        self.prognostic = d["prognostic"]
        self.surface = d["surface_prognostic"]
        self.forcing = d["forcing"]
        self.nlev = d["nlev"]
        self.step = d.get("step_months", 1)
        self.records_per_year = int(d.get("records_per_year", 12))
        self.rollout = max(1, int(rollout_steps))
        self.history = int(cfg.get("model", {}).get("history", 1)
                           if history is None else history)
        self.members = list(members)
        self.split = split

        self._cache: Dict[tuple, object] = {}  # (member, var) → DataArray or zarr.Array

        # Build the global (member, t0) sample index.
        # t0 is the most recent input month; need history-1 months before it.
        self.index: List[tuple] = []
        self._ntime: Dict[str, int] = {}
        t_start = d.get("t_start", 0)
        t_end = d.get("t_end", -1)
        span = self.step * self.rollout
        for m in self.members:
            handle = self._get(m, self.prognostic[0])
            nt = _n_time(handle)
            self._ntime[m] = nt
            hi = nt if t_end in (-1, None) else min(t_end, nt)
            lo = max(self.step * (self.history - 1), t_start)
            for t in range(lo, hi - span):
                self.index.append((m, t))

    def _get(self, member: str, var: str):
        key = (member, var)
        if key not in self._cache:
            if self.zarr_root is not None:
                try:
                    self._cache[key] = _open_zarr(
                        self.zarr_root, self.experiment, member, var
                    )
                    return self._cache[key]
                except FileNotFoundError:
                    pass  # fall through to netCDF4
            self._cache[key] = _open_netcdf(self.root, self.experiment, member, var)
        return self._cache[key]

    def read_time(self, member: str, var: str, t: int) -> torch.Tensor:
        """Single time record of ``var`` in physical units: (Z,J,I) or (J,I).

        Public accessor for rollout / evaluation scripts, which step through
        time one record at a time rather than reading a training window.
        """
        return torch.from_numpy(_read_slice(self._get(member, var), t, t + 1)[0])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        member, t = self.index[idx]
        statevars = self.prognostic + self.surface
        H = self.history
        R = self.rollout
        step = self.step

        t_lo = t - step * (H - 1)   # oldest history input
        t_hi = t + step * R + 1     # one past the last target

        def local(abs_t: int) -> int:
            return abs_t - t_lo

        # Process one variable at a time so peak RAM = 1 window (not all vars).
        state_hist: dict = {}
        targets: dict = {}
        for v in statevars:
            w = _read_slice(self._get(member, v), t_lo, t_hi)
            state_hist[v] = torch.from_numpy(
                np.stack([w[local(t - step * (H - 1 - h))] for h in range(H)], axis=0)
            )
            targets[v] = torch.from_numpy(
                np.stack([w[local(t + step * (s + 1))] for s in range(R)], axis=0)
            )
            del w  # free the full window before reading the next variable

        forcing = {
            v: torch.from_numpy(
                _read_slice(self._get(member, v), t, t + step * R)
            )
            for v in self.forcing
        }

        # Annual phase of the newest input record (records count from 1 January).
        ang = 2.0 * np.pi * (t % self.records_per_year) / self.records_per_year
        month_emb = torch.tensor([np.sin(ang), np.cos(ang)], dtype=torch.float32)

        return {
            "state_hist": state_hist,  # {var: (H,Z,J,I) or (H,J,I)}
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
        "state_hist": stack_dict("state_hist"),  # {var: (B,H,Z,J,I)/(B,H,J,I)}
        "targets": stack_dict("targets"),        # {var: (B,R,Z,J,I)/(B,R,J,I)}
        "forcing": stack_dict("forcing"),        # {var: (B,R,J,I)}
        "month_emb": torch.stack([b["month_emb"] for b in batch], dim=0),
        "member": [b["member"] for b in batch],
        "t": torch.tensor([b["t"] for b in batch]),
    }


def make_loader(cfg: dict, members: List[str], split: str, shuffle: bool,
                rollout_steps: int = 1):
    from torch.utils.data import DataLoader

    ds = PopLENSDataset(cfg, members, split=split, rollout_steps=rollout_steps)
    nw = cfg["train"].get("num_workers", 4)

    # At large rollout lengths each prefetched batch is ~22 GB (rl=128) —
    # pinning 4 workers × 2 prefetch = 8 batches × 22 GB = 176 GB would
    # exhaust the 200 GB job and cause swap pressure.  Disable pin_memory
    # for rl≥64; the non-pinned CPU→GPU transfer is fast enough relative to
    # the GPU compute time at long rollouts.
    pin = rollout_steps < 64

    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=nw,
        collate_fn=collate,
        pin_memory=pin,
        drop_last=shuffle,
        persistent_workers=nw > 0,
    )
