#!/usr/bin/env python
"""Evaluate a trained emulator: multi-month rollout skill + conservation drift.

Loads a checkpoint, integrates a held-out member forward ``eval.rollout_months``
months from an initial condition, and reports per-variable RMSE against POP
truth alongside a persistence baseline, plus the drift of the global heat and
salt budgets along the rollout.

Usage
-----
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint checkpoints/default/final.pt --member 001
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pop_emulator import data as data_mod              # noqa: E402
from pop_emulator import physics                       # noqa: E402
from pop_emulator.grid import load_grid                # noqa: E402
from pop_emulator.model import OceanEmulator           # noqa: E402
from pop_emulator.normalization import (               # noqa: E402
    ForcingNormalizer, Normalizer, StatePacker)
from pop_emulator.rollout import _advance_month_emb, month_cond  # noqa: E402
from pop_emulator import utils                         # noqa: E402


def masked_rmse(a, b, mask):
    d2 = (a - b) ** 2 * mask
    return torch.sqrt(d2.sum() / mask.sum().clamp_min(1.0))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--member", default=None, help="override eval member")
    ap.add_argument("--t0", type=int, default=None, help="start month index")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    device = utils.pick_device()
    d = cfg["data"]
    forcing_list = d["forcing"]

    packer = StatePacker(d["prognostic"], d["surface_prognostic"], d["nlev"])
    grid = load_grid(d["grid_file"], device=device)
    normalizer = Normalizer.from_stats(d["stats_file"], packer).to(device)
    fnorm = ForcingNormalizer.from_stats(d["stats_file"], forcing_list).to(device)
    model = OceanEmulator(cfg, packer, grid, n_forcing=len(forcing_list)).to(device)
    utils.load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    member = args.member or d["val_members"][0]
    horizon = cfg["eval"]["rollout_months"]

    ds = data_mod.PopLENSDataset(cfg, [member], split="eval", rollout_steps=1)
    t0 = args.t0 if args.t0 is not None else ds.index[len(ds.index) // 2][1]

    # Initial condition history (H states ending at t0) and month embedding.
    statevars = d["prognostic"] + d["surface_prognostic"]
    H = getattr(model, "history", 1)
    hist = [normalizer.normalize(packer.pack(
                {v: ds._read_slice(member, v, t0 - (H - 1 - h)).unsqueeze(0).to(device)
                 for v in statevars})) for h in range(H)]
    ic = {v: ds._read_slice(member, v, t0).unsqueeze(0).to(device) for v in statevars}
    month = t0 % 12
    ang = 2 * np.pi * month / 12.0
    month_emb = torch.tensor([[np.sin(ang), np.cos(ang)]], dtype=torch.float32, device=device)

    chan_mask = _chan_mask(packer, grid).to(device)
    persistence = {v: ic[v].clone() for v in statevars}

    print(f"[eval] member={member} t0={t0} horizon={horizon} months history={H}")
    print(f"{'mo':>3} {'var':>5} {'emu_rmse':>10} {'persist_rmse':>12} {'skill':>7}")
    h0 = physics.heat_content(ic["TEMP"], grid)
    s0 = physics.salt_content(ic["SALT"], grid)

    for step in range(horizon):
        t = t0 + step
        forcing = torch.stack(
            [ds._read_slice(member, v, t).to(device) for v in forcing_list], dim=0
        ).unsqueeze(0)
        fnz = fnorm.normalize(forcing)
        cond = month_cond(month_emb, fnz)
        x_in = hist[-1] if H == 1 else torch.stack(hist, dim=1)
        x = model(x_in, fnz, cond) * chan_mask
        hist.append(x)
        if len(hist) > H:
            hist.pop(0)
        pred = packer.unpack(normalizer.denormalize(x))

        truth = {v: ds._read_slice(member, v, t + 1).unsqueeze(0).to(device)
                 for v in statevars}

        if (step + 1) % max(1, horizon // 6) == 0 or step == horizon - 1:
            for v in ("TEMP", "SALT", "UVEL", "SSH"):
                m = grid.mask3d if pred[v].dim() == 4 else grid.mask2d
                er = masked_rmse(pred[v], truth[v], m)
                pr = masked_rmse(persistence[v], truth[v], m)
                skill = 1 - (er / pr.clamp_min(1e-9))
                print(f"{step+1:>3} {v:>5} {float(er):>10.4f} {float(pr):>12.4f} "
                      f"{float(skill):>7.3f}")
            dH = (physics.heat_content(pred["TEMP"], grid) - h0) / h0.abs().clamp_min(1)
            dS = (physics.salt_content(pred["SALT"], grid) - s0) / s0.abs().clamp_min(1)
            print(f"     heat-content drift {float(dH):+.3e} | "
                  f"salt-content drift {float(dS):+.3e}")
        month_emb = _advance_month_emb(month_emb)

    print("[eval] done")


def _chan_mask(packer, grid):
    chans = [grid.mask3d for _ in packer.prognostic]
    chans += [grid.mask2d.unsqueeze(0) for _ in packer.surface]
    return torch.cat(chans, dim=0).unsqueeze(0)


if __name__ == "__main__":
    sys.exit(main())
