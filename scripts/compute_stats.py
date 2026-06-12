#!/usr/bin/env python
"""Compute per-level normalization statistics over the training members.

For each 3-D prognostic variable, mean and std are computed *per vertical level*
(temperature variance spans orders of magnitude with depth, so a single global
scale would let the surface dominate the loss). 2-D variables and surface
forcing get scalar statistics. Land cells are excluded throughout.

A running (Welford-style) accumulation over a subsample of months keeps memory
and I/O bounded. Output: ``data/stats.nc``.

Usage
-----
    python scripts/compute_stats.py --config configs/default.yaml
    python scripts/compute_stats.py --config configs/default.yaml --max-times 60
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pop_emulator.data import member_files, _clean, POP_FILL  # noqa: E402


def level_stats_3d(files_by_member, var, mask3d, times):
    """Mean/std per level (Z,) over the chosen members and time samples."""
    nz = mask3d.shape[0]
    n = np.zeros(nz, dtype=np.float64)
    s1 = np.zeros(nz, dtype=np.float64)
    s2 = np.zeros(nz, dtype=np.float64)
    m2d = mask3d.astype(bool)
    for member, files in files_by_member.items():
        da = _open_var(files, var)
        nt = da.sizes["time"]
        for t in pick_times(nt, times):
            arr = _clean(da.isel(time=t).values)  # (Z,J,I)
            for k in range(nz):
                vals = arr[k][m2d[k]]
                n[k] += vals.size
                s1[k] += vals.sum(dtype=np.float64)
                s2[k] += np.square(vals, dtype=np.float64).sum(dtype=np.float64)
    mean = s1 / np.maximum(n, 1)
    var_ = np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0.0)
    return mean.astype("float32"), np.sqrt(var_).astype("float32")


def scalar_stats_2d(files_by_member, var, mask2d, times):
    m = mask2d.astype(bool)
    n = s1 = s2 = 0.0
    for member, files in files_by_member.items():
        da = _open_var(files, var)
        nt = da.sizes["time"]
        for t in pick_times(nt, times):
            arr = _clean(da.isel(time=t).values)
            vals = arr[m] if arr.ndim == 2 else arr[:, m].reshape(-1)
            n += vals.size
            s1 += vals.sum(dtype=np.float64)
            s2 += np.square(vals, dtype=np.float64).sum(dtype=np.float64)
    mean = s1 / max(n, 1)
    std = np.sqrt(max(s2 / max(n, 1) - mean ** 2, 0.0))
    return np.float32(mean), np.float32(std)


def _open_var(files, var):
    if len(files) == 1:
        return xr.open_dataset(files[0], decode_times=False, chunks={"time": 1})[var]
    return xr.open_mfdataset(files, decode_times=False, combine="by_coords",
                             chunks={"time": 1})[var]


def pick_times(nt, max_times):
    if max_times is None or max_times >= nt:
        return range(nt)
    return np.linspace(0, nt - 1, max_times).astype(int).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-times", type=int, default=120,
                    help="months sampled per member (None = all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]
    out = args.out or d["stats_file"]

    # 3-D ocean mask from the static grid file.
    with xr.open_dataset(d["grid_file"], decode_times=False) as g:
        kmt = np.asarray(g["KMT"].values).astype(int)
        nz = int(g.sizes["z_t"])
    kk = np.arange(nz)[:, None, None]
    mask3d = (kk < kmt[None]).astype("float32")
    mask2d = (kmt > 0).astype("float32")

    members = d["train_members"]
    files_by_member = {}
    for v in d["prognostic"] + d["surface_prognostic"] + d["forcing"]:
        for m in members:
            files_by_member.setdefault(m, member_files(d["root"], d["experiment"], m, v))

    ds_out = xr.Dataset()
    for v in d["prognostic"]:
        fbm = {m: member_files(d["root"], d["experiment"], m, v) for m in members}
        mean, std = level_stats_3d(fbm, v, mask3d, args.max_times)
        ds_out[f"{v}_mean"] = ("z_t", mean)
        ds_out[f"{v}_std"] = ("z_t", std)
        print(f"[stats] {v}: surface mean={mean[0]:.4g} std={std[0]:.4g} | "
              f"deep std={std[-1]:.4g}")

    for v in d["surface_prognostic"] + d["forcing"]:
        fbm = {m: member_files(d["root"], d["experiment"], m, v) for m in members}
        mean, std = scalar_stats_2d(fbm, v, mask2d, args.max_times)
        ds_out[f"{v}_mean"] = ((), mean)
        ds_out[f"{v}_std"] = ((), std)
        print(f"[stats] {v}: mean={mean:.4g} std={std:.4g}")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    ds_out.attrs["members"] = ",".join(members)
    ds_out.attrs["max_times"] = str(args.max_times)
    ds_out.to_netcdf(out)
    print(f"[stats] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
