#!/usr/bin/env python
"""Extract the static gx1v6 grid geometry and POP constants into a small file.

Every CESM-LENS history file repeats the full set of grid metrics and scalar
constants. This script reads them once from a chosen member's TEMP file and
writes a compact ``grid_gx1v6.nc`` (a few MB) that the data pipeline, physics
layer, and model load instead of re-reading the 23 GB time series.

Usage
-----
    python scripts/build_grid.py --out data/grid_gx1v6.nc
    python scripts/build_grid.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import xarray as xr

# Static fields and scalar constants we keep.
GRID_VARS = [
    "dz", "dzw", "z_t", "z_w", "z_w_top", "z_w_bot",
    "TAREA", "UAREA", "HTN", "HTE", "HUS", "HUW", "DXT", "DYT", "DXU", "DYU",
    "KMT", "KMU", "REGION_MASK", "TLAT", "TLONG", "ULAT", "ULONG",
    "ANGLE", "ANGLET", "HT", "HU",
]
CONST_VARS = [
    "cp_sw", "rho_sw", "rho_fw", "grav", "omega", "radius", "T0_Kelvin",
    "ocn_ref_salinity", "hflux_factor", "fwflux_factor", "salinity_factor",
    "sflux_factor", "sound", "vonkar",
]


def find_source_file(root: str, experiment: str, member: str) -> str:
    pat = os.path.join(
        root, "TEMP", f"b.e11.{experiment}.f09_g16.{member}.pop.h.TEMP.*.nc"
    )
    matches = sorted(glob.glob(pat))
    if not matches:
        raise FileNotFoundError(f"No TEMP file matching: {pat}")
    return matches[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="YAML config (optional)")
    ap.add_argument("--root", default=None, help="monthly tseries root")
    ap.add_argument("--experiment", default="B20TRC5CNBDRD")
    ap.add_argument("--member", default="001")
    ap.add_argument("--src", default=None, help="explicit source netCDF file")
    ap.add_argument("--out", default="data/grid_gx1v6.nc")
    args = ap.parse_args()

    root = args.root
    experiment = args.experiment
    member = args.member
    if args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        d = cfg["data"]
        root = root or d["root"]
        experiment = d.get("experiment", experiment)
        member = d.get("grid_member", member)
        args.out = d.get("grid_file", args.out)

    src = args.src or find_source_file(root, experiment, member)
    print(f"[build_grid] reading static fields from:\n  {src}")

    with xr.open_dataset(src, decode_times=False) as ds:
        keep = [v for v in GRID_VARS + CONST_VARS if v in ds]
        missing = [v for v in GRID_VARS + CONST_VARS if v not in ds]
        sub = ds[keep].load()
    sub.encoding.pop("unlimited_dims", None)
    if "time" in getattr(sub, "encoding", {}).get("unlimited_dims", set()):
        sub.encoding["unlimited_dims"] = set()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in sub.data_vars
           if sub[v].ndim >= 2}
    sub.to_netcdf(args.out, encoding=enc)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[build_grid] wrote {len(keep)} fields -> {args.out} ({size_mb:.1f} MB)")
    if missing:
        print(f"[build_grid] note: {len(missing)} fields not present: {missing}")


if __name__ == "__main__":
    sys.exit(main())
