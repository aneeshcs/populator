#!/usr/bin/env python
"""Daily one-step tendency statistics of SST and SSH for the daily fine-tune.

Writes the ocean-masked standard deviation of ``x(d + 1) - x(d)`` (physical
units: degC, cm) to ``data/stats_daily.nc`` as ``SST_tend_std`` and
``SSH_tend_std``.  ``PopDailyLoss`` uses them to express the surface error in
daily-tendency units.  Field means and standard deviations stay those of
``data/stats.nc`` (monthly), so the pretrained weights see unchanged inputs.

Usage:
    python scripts/compute_daily_stats.py --config configs/run7_daily.yaml \
        --daily-zarr-root /glade/derecho/scratch/$USER/pop_emulator_zarr_daily
"""
from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
import xarray as xr

from pop_emulator.data import _open_zarr, _read_slice
from pop_emulator.grid import load_grid
from pop_emulator.utils import load_config

DAYS_PER_YEAR = 365


def tendency_std(zarr_root: str, experiment: str, members: Sequence[str], var: str,
                 mask: np.ndarray, years: Sequence[int]) -> float:
    """Std over ocean points, members and the sampled years of the daily increment."""
    total, count = 0.0, 0
    for member in members:
        handle = _open_zarr(zarr_root, experiment, member, var)
        for year in years:
            lo = year * DAYS_PER_YEAR
            x = np.asarray(_read_slice(handle, lo, lo + DAYS_PER_YEAR + 1), dtype=np.float64)
            d = np.diff(x, axis=0)[:, mask]            # (364 or 365, n_ocean)
            total += float((d ** 2).sum())             # increments have ~zero mean
            count += d.size
    return float(np.sqrt(total / count))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--daily-zarr-root", required=True)
    ap.add_argument("--members", nargs="+", default=["002", "003", "004"])
    ap.add_argument("--years", nargs="+", type=int, default=[0, 10, 20, 30, 40, 50],
                    help="years since the first record of the daily stores")
    ap.add_argument("--out", default="data/stats_daily.nc")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mask = load_grid(cfg["data"]["grid_file"]).mask2d.cpu().numpy().astype(bool)
    out = xr.Dataset()
    for var in ("SST", "SSH"):
        std = tendency_std(args.daily_zarr_root, cfg["data"]["experiment"], args.members,
                           var, mask, args.years)
        out[f"{var}_tend_std"] = ((), std)
        print(f"[daily stats] {var}: tend_std = {std:.5g}", flush=True)
    out.attrs = {"members": " ".join(args.members), "years": " ".join(map(str, args.years)),
                 "description": "std of the one-day increment over ocean points"}
    out.to_netcdf(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
