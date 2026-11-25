#!/usr/bin/env python
"""Rechunk CESM-LENS POP netCDF4 files to Zarr on GLADE scratch.

Source files have internal HDF5 time_chunk=1, making each training sample
require O(rollout_len) disk seeks.  This script writes Zarr stores with
time_chunk=300 so that any rollout window (≤258 months) reads at most 2
time chunks per variable — cutting reads from ~1040 to ~16 per 3D variable.

Output structure:
    {zarr_root}/{experiment}/{member}/{var}.zarr

Run once; training jobs then set data.zarr_root in the config to use the
fast stores.

Usage:
    python scripts/rechunk_to_zarr.py [options]

    --members  002 003 ...   (default: all 001-020)
    --vars     TEMP SALT ...  (default: all 10 variables)
    --zarr-root PATH          (default: /glade/derecho/scratch/acsubram/pop_emulator_zarr)
    --data-root PATH          (default: from configs/run5.yaml)
    --experiment NAME         (default: B20TRC5CNBDRD)
    --dry-run                 Print plan without writing
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numcodecs
import numpy as np
import xarray as xr
import zarr

# ---------------------------------------------------------------------------
# Defaults (match run5.yaml)
# ---------------------------------------------------------------------------
DATA_ROOT = (
    "/glade/campaign/collections/gdex/data/d651027/cesmLE/"
    "CESM-CAM5-BGC-LE/ocn/proc/tseries/monthly"
)
ZARR_ROOT = "/glade/derecho/scratch/acsubram/pop_emulator_zarr"
EXPERIMENT = "B20TRC5CNBDRD"

ALL_MEMBERS = [f"{m:03d}" for m in range(1, 21)]
VARS_3D = ["TEMP", "SALT", "UVEL", "VVEL"]
VARS_2D = ["SSH", "SHF", "SHF_QSW", "SFWF", "TAUX", "TAUY"]
ALL_VARS = VARS_3D + VARS_2D

# Time chunk covers max rollout (256) + max history (2) = 258 months, +42 buffer.
TIME_CHUNK = 300
COMPRESSOR = numcodecs.Blosc(cname="lz4hc", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)

CHUNKS_3D = (TIME_CHUNK, 30, 192, 160)   # (time, z_t, nlat, nlon)
CHUNKS_2D = (TIME_CHUNK, 192, 160)       # (time, nlat, nlon)


def find_files(data_root: str, experiment: str, member: str, var: str) -> list[str]:
    pat = os.path.join(
        data_root, var,
        f"b.e11.{experiment}.f09_g16.{member}.pop.h.{var}.*.nc",
    )
    return sorted(glob.glob(pat))


def rechunk_var(data_root, experiment, member, var, zarr_root, dry_run=False):
    files = find_files(data_root, experiment, member, var)
    if not files:
        print(f"  SKIP {member}/{var}: no source files found", flush=True)
        return

    out_path = os.path.join(zarr_root, experiment, member, f"{var}.zarr")
    if os.path.exists(out_path):
        # Check it's complete by verifying the zarr metadata exists
        if os.path.exists(os.path.join(out_path, ".zarray")) or os.path.exists(
            os.path.join(out_path, var, ".zarray")
        ):
            print(f"  SKIP {member}/{var}: zarr store already exists at {out_path}", flush=True)
            return

    chunks = CHUNKS_3D if var in VARS_3D else CHUNKS_2D

    print(f"  {member}/{var}: {len(files)} source file(s) → {out_path}", flush=True)
    if dry_run:
        return

    t0 = time.time()

    # Open source lazily; use dask chunks matching zarr target so that each
    # dask task produces exactly one zarr write unit (~8 GB uncompressed for
    # 3D vars — fits easily in a 256 GB node).
    open_kw = dict(decode_times=False, chunks={"time": TIME_CHUNK})
    if len(files) == 1:
        ds = xr.open_dataset(files[0], **open_kw)
    else:
        ds = xr.open_mfdataset(files, combine="by_coords", **open_kw)

    da_ds = ds[[var]]  # single-variable Dataset preserves dimension coordinates

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    encoding = {
        var: {
            "chunks": chunks,
            "compressor": COMPRESSOR,
            # Suppress xarray's default coordinate encoding that adds metadata
            "dtype": "float32",
        }
    }

    da_ds.to_zarr(out_path, mode="w", encoding=encoding, zarr_format=2)

    nt = ds[var].shape[0]
    elapsed = time.time() - t0
    print(
        f"  {member}/{var}: done — {nt} months in {elapsed:.0f}s "
        f"({nt/elapsed:.1f} months/s)",
        flush=True,
    )
    ds.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--members", nargs="+", default=ALL_MEMBERS)
    p.add_argument("--vars", nargs="+", default=ALL_VARS)
    p.add_argument("--zarr-root", default=ZARR_ROOT)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--experiment", default=EXPERIMENT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    members = args.members
    variables = args.vars
    total = len(members) * len(variables)

    print(
        f"Rechunking {len(members)} members × {len(variables)} vars = {total} stores",
        flush=True,
    )
    print(f"  data_root : {args.data_root}", flush=True)
    print(f"  zarr_root : {args.zarr_root}", flush=True)
    print(f"  time_chunk: {TIME_CHUNK}", flush=True)
    if args.dry_run:
        print("  [DRY RUN — no files will be written]", flush=True)
    print(flush=True)

    done = 0
    t_wall = time.time()
    for member in members:
        for var in variables:
            done += 1
            print(f"[{done}/{total}] {member}/{var}", flush=True)
            try:
                rechunk_var(
                    args.data_root, args.experiment, member, var,
                    args.zarr_root, dry_run=args.dry_run,
                )
            except Exception as exc:
                print(f"  ERROR {member}/{var}: {exc}", flush=True)

    elapsed = time.time() - t_wall
    print(f"\nFinished {done} stores in {elapsed/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
