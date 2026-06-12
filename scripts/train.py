#!/usr/bin/env python
"""Train the POP ocean emulator.

Free-running autoregressive training: from the initial condition the model is
stepped forward ``R`` months, each step fed its own previous output, and the
loss is accumulated against the POP targets at every step. ``R`` follows the
rollout curriculum in the config (start at 1, grow to stabilize long rollouts).

Usage
-----
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --resume checkpoints/default/last.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pop_emulator import data as data_mod          # noqa: E402
from pop_emulator.grid import load_grid            # noqa: E402
from pop_emulator.losses import CompositeLoss      # noqa: E402
from pop_emulator.model import OceanEmulator       # noqa: E402
from pop_emulator.normalization import (           # noqa: E402
    ForcingNormalizer, Normalizer, StatePacker)
from pop_emulator.rollout import _advance_month_emb, month_cond  # noqa: E402
from pop_emulator import utils                     # noqa: E402


def build_everything(cfg, device):
    d = cfg["data"]
    packer = StatePacker(d["prognostic"], d["surface_prognostic"], d["nlev"])
    grid = load_grid(d["grid_file"], device=device)
    normalizer = Normalizer.from_stats(d["stats_file"], packer).to(device)
    fnorm = ForcingNormalizer.from_stats(d["stats_file"], d["forcing"]).to(device)
    model = OceanEmulator(cfg, packer, grid, n_forcing=len(d["forcing"])).to(device)
    loss_fn = CompositeLoss(cfg, packer, normalizer, grid).to(device)
    return packer, grid, normalizer, fnorm, model, loss_fn


def cosine_warmup(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, p)))


def pack_forcing(batch_forcing, forcing_list, s, device):
    """(B, F, J, I) at rollout step s, in the config's forcing order."""
    return torch.stack([batch_forcing[v][:, s].to(device) for v in forcing_list], dim=1)


def train_step(batch, model, packer, normalizer, fnorm, loss_fn, forcing_list,
               rollout_len, device, step):
    state0 = {k: v.to(device) for k, v in batch["state_t"].items()}
    x = normalizer.normalize(packer.pack(state0))
    chan_mask = loss_fn.chan_mask
    month_emb = batch["month_emb"].to(device)

    total = 0.0
    last_logs = {}
    for s in range(rollout_len):
        fz_phys = pack_forcing(batch["forcing"], forcing_list, s, device)
        fnz = fnorm.normalize(fz_phys)
        cond = month_cond(month_emb, fnz)

        prev = x
        x = model(x, fnz, cond)
        x = x * chan_mask

        # target at step s
        tgt = {v: batch["targets"][v][:, s].to(device) for v in packer.prognostic}
        tgt.update({v: batch["targets"][v][:, s].to(device) for v in packer.surface})
        tgt_norm = normalizer.normalize(packer.pack(tgt))

        forcing_phys = {v: batch["forcing"][v][:, s].to(device) for v in forcing_list}
        res = loss_fn(x, tgt_norm, prev, forcing_phys=forcing_phys, step=step)
        total = total + res["loss"]
        last_logs = res["logs"]
        month_emb = _advance_month_emb(month_emb)

    total = total / rollout_len
    return total, last_logs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="override for testing")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    utils.set_seed(cfg.get("seed", 0))
    device = utils.pick_device()
    tcfg = cfg["train"]
    forcing_list = cfg["data"]["forcing"]

    packer, grid, normalizer, fnorm, model, loss_fn = build_everything(cfg, device)
    print(f"[train] device={device}  params={utils.count_params(model)/1e6:.1f}M  "
          f"channels={packer.n_channels}")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    use_amp = tcfg.get("amp", True) and device.type == "cuda"
    amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 needs no scaler

    step = 0
    if args.resume and os.path.exists(args.resume):
        step = utils.load_checkpoint(args.resume, model, opt)
        print(f"[train] resumed from {args.resume} at step {step}")

    curriculum = tcfg.get("rollout_curriculum", [[0, tcfg.get("rollout_steps", 1)]])
    total_steps = args.max_steps or (tcfg["epochs"] * 10_000)

    cur_rollout = -1
    loader = None
    log_dir = tcfg.get("log_dir")
    writer = None
    if log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir)
        except Exception as e:  # tensorboard optional
            print(f"[train] tensorboard unavailable: {e}")

    accum = tcfg.get("grad_accum", 1)
    t0 = time.time()
    model.train()
    done = False
    while not done:
        rl = utils.rollout_len_for_step(curriculum, step)
        if rl != cur_rollout:
            cur_rollout = rl
            loader = data_mod.make_loader(cfg, cfg["data"]["train_members"],
                                          "train", shuffle=True, rollout_steps=rl)
            print(f"[train] step {step}: rollout length -> {rl} "
                  f"({len(loader.dataset)} samples)")

        for batch in loader:
            # Honor the rollout curriculum at batch granularity: if the length
            # has changed, break to rebuild the loader rather than waiting for
            # the current epoch to finish (epochs are ~tens of thousands of
            # samples, which would otherwise delay the transition).
            if utils.rollout_len_for_step(curriculum, step) != cur_rollout:
                break

            lr = cosine_warmup(step, tcfg.get("warmup_steps", 0), total_steps, tcfg["lr"])
            for g in opt.param_groups:
                g["lr"] = lr

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, logs = train_step(batch, model, packer, normalizer, fnorm,
                                        loss_fn, forcing_list, rl, device, step)
                loss = loss / accum

            loss.backward()
            if (step + 1) % accum == 0:
                if tcfg.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % tcfg.get("log_every", 50) == 0:
                rate = (step + 1) / (time.time() - t0)
                msg = (f"step {step:>7d} | lr {lr:.2e} | rl {rl} | "
                       f"loss {logs['total'].item():.4f} | data {logs['data'].item():.4f}")
                for k in ("continuity", "barotropic", "heat", "salt", "stability"):
                    if k in logs:
                        msg += f" | {k[:4]} {logs[k].item():.3e}"
                msg += f" | {rate:.2f} it/s"
                print(msg, flush=True)
                if writer:
                    for k, v in logs.items():
                        writer.add_scalar(f"train/{k}", float(v), step)
                    writer.add_scalar("train/lr", lr, step)

            if step > 0 and step % tcfg.get("ckpt_every", 2000) == 0:
                path = os.path.join(tcfg["ckpt_dir"], "last.pt")
                utils.save_checkpoint(path, model, opt, step=step, extra={"cfg": cfg})
                print(f"[train] checkpoint -> {path}")

            step += 1
            if step >= total_steps:
                done = True
                break

    path = os.path.join(tcfg["ckpt_dir"], "final.pt")
    utils.save_checkpoint(path, model, opt, step=step, extra={"cfg": cfg})
    print(f"[train] done at step {step}; final checkpoint -> {path}")


if __name__ == "__main__":
    sys.exit(main())
