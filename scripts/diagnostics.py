#!/usr/bin/env python
"""Diagnose the leading ocean modes of variability in the emulator vs POP.

Runs a free-running, surface-forced rollout of the trained emulator from an
initial condition in a held-out member, collects monthly SST (TEMP level 0) and
SSH, and compares the dominant modes of variability against the POP truth over
the same months:

  * ENSO  - Nino 3.4 SST-anomaly index (5S-5N, 170W-120W)
  * PDO   - leading EOF of North Pacific SST anomalies (global-mean removed)
  * AMO   - North Atlantic SST-anomaly index (global-mean removed)
  * SST variability - std of monthly SST anomalies (pattern) + mean bias

Figures are written as PNGs for the LaTeX report; the computed indices are
saved to an .npz for reproducibility.

Usage
-----
    python scripts/diagnostics.py --config configs/run2.yaml \
        --checkpoint checkpoints/run2/last.pt --member 001 \
        --t0 360 --months 480 --out docs/figures
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pop_emulator import data as data_mod              # noqa: E402
from pop_emulator.grid import load_grid                # noqa: E402
from pop_emulator.model import OceanEmulator           # noqa: E402
from pop_emulator.normalization import (               # noqa: E402
    ForcingNormalizer, Normalizer, StatePacker)
from pop_emulator.rollout import _advance_month_emb, month_cond  # noqa: E402
from pop_emulator import utils                         # noqa: E402

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
@torch.no_grad()
def forced_rollout(model, packer, normalizer, fnorm, grid, ds, member,
                   forcing_list, t0, months, device):
    """Free-running rollout with POP surface forcing. Returns emulator and POP
    monthly SST (TEMP k=0) and SSH stacked as (T, J, I), plus the achieved
    length (truncated if the rollout goes unstable)."""
    statevars = packer.prognostic + packer.surface
    ic = {v: ds._read_slice(member, v, t0).unsqueeze(0).to(device) for v in statevars}
    month = t0 % 12
    ang = 2 * np.pi * month / 12.0
    month_emb = torch.tensor([[np.sin(ang), np.cos(ang)]], dtype=torch.float32, device=device)

    chan_mask = _chan_mask(packer, grid).to(device)
    x = normalizer.normalize(packer.pack(ic))
    mask2d = grid.mask2d.bool().cpu().numpy()

    emu_sst, emu_ssh, pop_sst, pop_ssh = [], [], [], []
    sst_idx = packer.slices["TEMP"].start  # channel index of TEMP level 0

    achieved = 0
    for step in range(months):
        t = t0 + step
        forcing = torch.stack(
            [ds._read_slice(member, v, t).to(device) for v in forcing_list], dim=0
        ).unsqueeze(0)
        fnz = fnorm.normalize(forcing)
        cond = month_cond(month_emb, fnz)
        x = model(x, fnz, cond) * chan_mask

        pred = packer.unpack(normalizer.denormalize(x))
        sst = pred["TEMP"][0, 0].cpu().numpy()
        ssh = pred["SSH"][0].cpu().numpy()
        if not np.isfinite(sst[mask2d]).all() or np.nanmax(np.abs(sst[mask2d])) > 60:
            print(f"[diag] rollout went unstable at step {step}; truncating.")
            break

        emu_sst.append(sst)
        emu_ssh.append(ssh)
        pop_sst.append(ds._read_slice(member, "TEMP", t + 1)[0].numpy())
        pop_ssh.append(ds._read_slice(member, "SSH", t + 1).numpy())
        achieved += 1
        month_emb = _advance_month_emb(month_emb)
        if (step + 1) % 60 == 0:
            print(f"[diag] rolled {step+1}/{months} months", flush=True)

    return (np.array(emu_sst), np.array(emu_ssh),
            np.array(pop_sst), np.array(pop_ssh), achieved)


def _chan_mask(packer, grid):
    chans = [grid.mask3d for _ in packer.prognostic]
    chans += [grid.mask2d.unsqueeze(0) for _ in packer.surface]
    return torch.cat(chans, dim=0).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Anomalies, regions, EOF
# --------------------------------------------------------------------------- #
def deseasonalize(field):
    """Remove the monthly climatology (per calendar month) from (T, J, I)."""
    T = field.shape[0]
    clim = np.zeros((12,) + field.shape[1:], dtype=field.dtype)
    for m in range(12):
        clim[m] = field[m::12].mean(axis=0)
    anom = field - clim[np.arange(T) % 12]
    return anom


def region_mask(tlon, tlat, ocean, lon0, lon1, lat0, lat1):
    """Boolean mask for a lon/lat box (lon in 0-360)."""
    lon = tlon % 360
    if lon0 <= lon1:
        in_lon = (lon >= lon0) & (lon <= lon1)
    else:  # wrap across 0
        in_lon = (lon >= lon0) | (lon <= lon1)
    return ocean & in_lon & (tlat >= lat0) & (tlat <= lat1)


def area_index(anom, mask, area):
    """Area-weighted spatial mean over a mask -> time series."""
    w = (area * mask)[None]
    return (anom * w).reshape(anom.shape[0], -1).sum(1) / w.sum()


def linear_detrend(x):
    t = np.arange(len(x))
    a, b = np.polyfit(t, x, 1)
    return x - (a * t + b)


def eof1(anom, mask, area):
    """Leading EOF over a region. Returns (pattern[J,I], pc[T], varfrac).
    Anomalies are area-weighted (sqrt) before SVD; pattern is the regression of
    SST anomalies onto the standardized PC1 (units degC)."""
    idx = np.where(mask.ravel())[0]
    A = anom.reshape(anom.shape[0], -1)[:, idx]          # (T, P)
    A = A - A.mean(0, keepdims=True)
    wsqrt = np.sqrt((area.ravel()[idx]).clip(min=0))
    Aw = A * wsqrt[None]
    U, S, Vt = np.linalg.svd(Aw, full_matrices=False)
    pc1 = U[:, 0] * S[0]
    varfrac = S[0] ** 2 / (S ** 2).sum()
    pc1s = (pc1 - pc1.mean()) / pc1.std()
    patt_region = (A * pc1s[:, None]).mean(0)            # regression, degC
    pattern = np.full(mask.shape, np.nan)
    pattern.ravel()[idx] = patt_region
    return pattern, pc1s, float(varfrac)


def sign_align(ref_pattern, pattern, pc):
    """Flip EOF sign so its pattern correlates positively with a reference."""
    a = ref_pattern[np.isfinite(ref_pattern) & np.isfinite(pattern)]
    b = pattern[np.isfinite(ref_pattern) & np.isfinite(pattern)]
    if np.corrcoef(a, b)[0, 1] < 0:
        return -pattern, -pc
    return pattern, pc


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _map(ax, lon, lat, field, title, vmin, vmax, cmap="RdBu_r"):
    pm = ax.pcolormesh(lon, lat, field, vmin=vmin, vmax=vmax, cmap=cmap, shading="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 360); ax.set_ylim(-80, 80)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    return pm


def plot_index(t_years, emu, pop, name, fname, units="degC"):
    r = np.corrcoef(emu, pop)[0, 1]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t_years, pop, color="k", lw=1.4, label="POP")
    ax.plot(t_years, emu, color="tab:red", lw=1.2, label="Emulator")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_title(f"{name} index  (r = {r:.2f}, std emu/POP = "
                 f"{emu.std()/pop.std():.2f})", fontsize=11)
    ax.set_xlabel("rollout year"); ax.set_ylabel(f"anomaly [{units}]")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)
    return r, emu.std() / pop.std()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--member", default="001")
    ap.add_argument("--t0", type=int, default=360)
    ap.add_argument("--months", type=int, default=480)
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    device = utils.pick_device()
    d = cfg["data"]
    forcing_list = d["forcing"]
    os.makedirs(args.out, exist_ok=True)

    packer = StatePacker(d["prognostic"], d["surface_prognostic"], d["nlev"])
    grid = load_grid(d["grid_file"], device=device)
    normalizer = Normalizer.from_stats(d["stats_file"], packer).to(device)
    fnorm = ForcingNormalizer.from_stats(d["stats_file"], forcing_list).to(device)
    model = OceanEmulator(cfg, packer, grid, n_forcing=len(forcing_list)).to(device)
    utils.load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    ds = data_mod.PopLENSDataset(cfg, [args.member], split="diag", rollout_steps=1)

    print(f"[diag] forced rollout: member {args.member}, t0={args.t0}, "
          f"{args.months} months on {device}")
    emu_sst, emu_ssh, pop_sst, pop_ssh, n = forced_rollout(
        model, packer, normalizer, fnorm, grid, ds, args.member,
        forcing_list, args.t0, args.months, device)
    print(f"[diag] achieved {n} stable months")

    tlon = grid.tlong.cpu().numpy(); tlat = grid.tlat.cpu().numpy()
    area = (grid.tarea * grid.mask2d).cpu().numpy()
    ocean = grid.mask2d.bool().cpu().numpy()
    years = np.arange(n) / 12.0

    # Anomalies (deseasonalized).
    emu_a = deseasonalize(emu_sst); pop_a = deseasonalize(pop_sst)
    # Global-mean SST anomaly (for PDO/AMO removal).
    g_emu = area_index(emu_a, ocean, area); g_pop = area_index(pop_a, ocean, area)

    results = {}

    # --- ENSO / Nino 3.4 -------------------------------------------------- #
    nino = region_mask(tlon, tlat, ocean, 190, 240, -5, 5)
    e = linear_detrend(area_index(emu_a, nino, area))
    p = linear_detrend(area_index(pop_a, nino, area))
    r, sr = plot_index(years, e, p, "Nino 3.4 (ENSO)",
                       f"{args.out}/nino34.png")
    results["nino34"] = (r, sr)

    # --- AMO -------------------------------------------------------------- #
    amo_m = region_mask(tlon, tlat, ocean, 280, 360, 0, 60)
    e = linear_detrend(area_index(emu_a - g_emu[:, None, None], amo_m, area))
    p = linear_detrend(area_index(pop_a - g_pop[:, None, None], amo_m, area))
    r, sr = plot_index(years, e, p, "AMO (N. Atlantic)", f"{args.out}/amo.png")
    results["amo"] = (r, sr)

    # --- PDO (EOF1 of N. Pacific SST anomalies, global mean removed) ------ #
    npac = region_mask(tlon, tlat, ocean, 110, 260, 20, 70)
    emu_p = (emu_a - g_emu[:, None, None])
    pop_p = (pop_a - g_pop[:, None, None])
    patt_pop, pc_pop, vf_pop = eof1(pop_p, npac, area)
    patt_emu, pc_emu, vf_emu = eof1(emu_p, npac, area)
    patt_emu, pc_emu = sign_align(patt_pop, patt_emu, pc_emu)
    # convention: PDO positive = cool central N. Pacific; align POP to that via warmth sign
    rpdo = np.corrcoef(pc_emu, pc_pop)[0, 1]
    results["pdo"] = (rpdo, vf_emu, vf_pop)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    vlim = np.nanmax(np.abs(patt_pop))
    _map(axs[0], tlon, tlat, patt_pop, f"PDO EOF1 — POP ({vf_pop*100:.0f}% var)",
         -vlim, vlim)
    pm = _map(axs[1], tlon, tlat, patt_emu,
              f"PDO EOF1 — Emulator ({vf_emu*100:.0f}% var)", -vlim, vlim)
    for ax in axs:
        ax.set_xlim(110, 260); ax.set_ylim(20, 70)
    fig.colorbar(pm, ax=axs, shrink=0.8, label="degC / std(PC1)")
    fig.suptitle(f"Pacific Decadal Oscillation pattern (PC1 corr r = {rpdo:.2f})",
                 fontsize=11)
    fig.savefig(f"{args.out}/pdo_pattern.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(years, pc_pop, "k", lw=1.4, label="POP")
    ax.plot(years, pc_emu, "tab:red", lw=1.2, label="Emulator")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_title(f"PDO principal component (PC1)  (r = {rpdo:.2f})", fontsize=11)
    ax.set_xlabel("rollout year"); ax.set_ylabel("standardized PC1")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(f"{args.out}/pdo_pc.png", dpi=130); plt.close(fig)

    # --- SST variability pattern + mean bias ------------------------------ #
    std_emu = np.where(ocean, emu_a.std(0), np.nan)
    std_pop = np.where(ocean, pop_a.std(0), np.nan)
    bias = np.where(ocean, emu_sst.mean(0) - pop_sst.mean(0), np.nan)
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.4))
    vlim = np.nanpercentile(std_pop, 99)
    _map(axs[0], tlon, tlat, std_pop, "SST anomaly std — POP", 0, vlim, "viridis")
    pm = _map(axs[1], tlon, tlat, std_emu, "SST anomaly std — Emulator", 0, vlim, "viridis")
    fig.colorbar(pm, ax=axs[:2], shrink=0.8, label="degC")
    bl = np.nanpercentile(np.abs(bias), 98)
    pmb = _map(axs[2], tlon, tlat, bias, "Mean SST bias (Emu - POP)", -bl, bl)
    fig.colorbar(pmb, ax=axs[2], shrink=0.8, label="degC")
    fig.savefig(f"{args.out}/sst_variability.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    np.savez(f"{args.out}/diag_indices.npz",
             years=years, nino_emu=e, nino_pop=p,
             pc_pop=pc_pop, pc_emu=pc_emu, results=str(results), n=n)

    # Summary for the LaTeX text.
    with open(f"{args.out}/diag_summary.txt", "w") as f:
        f.write(f"member={args.member} t0={args.t0} stable_months={n} "
                f"({n/12:.1f} yr)\n")
        f.write(f"Nino3.4: corr={results['nino34'][0]:.2f} "
                f"std_ratio={results['nino34'][1]:.2f}\n")
        f.write(f"AMO: corr={results['amo'][0]:.2f} std_ratio={results['amo'][1]:.2f}\n")
        f.write(f"PDO: PC1 corr={rpdo:.2f} varfrac POP={vf_pop:.2f} "
                f"emu={vf_emu:.2f}\n")
    print("[diag] wrote figures + summary to", args.out)
    for k, v in results.items():
        print("  ", k, v)


if __name__ == "__main__":
    sys.exit(main())
