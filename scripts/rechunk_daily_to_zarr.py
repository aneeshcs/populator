#!/usr/bin/env python
"""Rechunk CESM-LENS daily POP surface fields (``pop.h.nday1``) to Zarr.

The daily stream holds 2-D fields only.  Those the emulator uses are written
under the emulator's variable names:

    SST    -> SST    surface potential temperature (TEMP level 0)   [degC]
    SSH_2  -> SSH    sea-surface height                             [cm]
    TAUX_2 -> TAUX   zonal wind stress                              [dyne cm-2]
    TAUY_2 -> TAUY   meridional wind stress                         [dyne cm-2]

The source files start on **2 January** of the member's first year (1920 for
members 002+, 1850 for member 001): the coupled ocean starts one day late, so
``time_bound[0]`` begins at day 1.04 of that year and 1 January is absent.
Daily record ``k`` is therefore the mean over day ``k + 1`` of the first year
(noleap calendar).  Stores are trimmed to start on 1 January of
``--start-year`` (which must be later than the first year) so that record 0 is
the same date for every member; the date is recorded in the store attributes.

Output structure (same as the monthly stores, so ``data._open_zarr`` applies):
    {zarr_root}/{experiment}/{member}/{var}.zarr

Usage:
    python scripts/rechunk_daily_to_zarr.py --members 001 002 ... [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time

import numcodecs
import xarray as xr

DATA_ROOT = (
    "/glade/campaign/collections/gdex/data/d651027/cesmLE/"
    "CESM-CAM5-BGC-LE/ocn/proc/tseries/daily"
)
ZARR_ROOT = "/glade/derecho/scratch/acsubram/pop_emulator_zarr_daily"
EXPERIMENT = "B20TRC5CNBDRD"
DAYS_PER_YEAR = 365  # noleap

# source variable -> emulator variable
VAR_MAP = {"SST": "SST", "SSH_2": "SSH", "TAUX_2": "TAUX", "TAUY_2": "TAUY"}
ALL_MEMBERS = [f"{m:03d}" for m in range(1, 7)]

# One model year per time chunk: a monthly training window touches <= 2 chunks.
CHUNKS = (DAYS_PER_YEAR, 192, 160)
COMPRESSOR = numcodecs.Blosc(cname="lz4hc", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)


def find_file(data_root: str, experiment: str, member: str, src_var: str) -> str:
    pat = os.path.join(
        data_root, src_var,
        f"b.e11.{experiment}.f09_g16.{member}.pop.h.nday1.{src_var}.*.nc")
    files = sorted(glob.glob(pat))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one file for {pat}, found {len(files)}")
    return files[0]


def rechunk_var(data_root, experiment, member, src_var, zarr_root, start_year, dry_run=False):
    var = VAR_MAP[src_var]
    path = find_file(data_root, experiment, member, src_var)
    out_path = os.path.join(zarr_root, experiment, member, f"{var}.zarr")
    if os.path.exists(os.path.join(out_path, var, ".zarray")):
        print(f"  SKIP {member}/{var}: {out_path} exists", flush=True)
        return

    # First averaging day from the file name (…nday1.VAR.YYYYMMDD-…): 2 January.
    first = re.search(r"\.(\d{4})(\d{4})-\d{8}\.nc$", path)
    file_year, mmdd = int(first.group(1)), first.group(2)
    if mmdd != "0102":
        raise ValueError(f"{path}: expected the record to start on 2 January")
    if start_year <= file_year:
        raise ValueError(f"start year must be later than the file's first year ({file_year}), "
                         "whose 1 January is missing")
    offset = (start_year - file_year) * DAYS_PER_YEAR - 1   # record 0 is 2 January

    print(f"  {member}/{src_var} -> {out_path} (skip {offset} days)", flush=True)
    if dry_run:
        return

    t0 = time.time()
    ds = xr.open_dataset(path, decode_times=False, chunks={"time": DAYS_PER_YEAR})
    # Re-chunk after trimming so each Zarr time chunk is written by one Dask chunk.
    da = ds[src_var].isel(time=slice(offset, None)).chunk({"time": DAYS_PER_YEAR}).rename(var)
    nt = da.sizes["time"]
    if nt % DAYS_PER_YEAR != 0:
        raise ValueError(f"{member}/{src_var}: {nt} days is not a whole number of noleap years")
    out = da.to_dataset()
    out.attrs = {
        "source_file": path, "source_variable": src_var,
        "record0_date": f"{start_year:04d}-01-01", "calendar": "noleap",
        "records_per_year": DAYS_PER_YEAR,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    encoding = {var: {"chunks": CHUNKS, "compressors": (COMPRESSOR,), "dtype": "float32"}}
    out.to_zarr(out_path, mode="w", encoding=encoding, zarr_format=2)
    ds.close()
    el = time.time() - t0
    print(f"  {member}/{var}: {nt} days ({nt // DAYS_PER_YEAR} years) in {el:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--members", nargs="+", default=ALL_MEMBERS)
    p.add_argument("--vars", nargs="+", default=list(VAR_MAP), help="source variable names")
    p.add_argument("--zarr-root", default=ZARR_ROOT)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--experiment", default=EXPERIMENT)
    p.add_argument("--start-year", type=int, default=1950)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    total = len(args.members) * len(args.vars)
    print(f"Rechunking {total} daily stores from {args.start_year}-01-01 -> {args.zarr_root}",
          flush=True)
    done, errors, t_wall = 0, 0, time.time()
    for member in args.members:
        for src_var in args.vars:
            done += 1
            print(f"[{done}/{total}] {member}/{src_var}", flush=True)
            try:
                rechunk_var(args.data_root, args.experiment, member, src_var,
                            args.zarr_root, args.start_year, dry_run=args.dry_run)
            except Exception as exc:  # keep going; report at the end
                errors += 1
                print(f"  ERROR {member}/{src_var}: {exc}", flush=True)
    print(f"\nFinished {done} stores, {errors} error(s), in {(time.time() - t_wall) / 3600:.2f}h",
          flush=True)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
