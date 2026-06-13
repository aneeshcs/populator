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


def pushforward_step(batch, model, packer, normalizer, fnorm, loss_fn,
                     forcing_list, rollout_len, noise, device, step, accum,
                     use_amp, amp_dtype):
    """Memory-safe pushforward rollout with optional input-noise injection.

    The model is unrolled ``rollout_len`` months feeding its own *detached*
    predictions back as input history (so it learns to correct its own error),
    and the per-step loss is back-propagated immediately. Because each input is
    detached, memory stays at one step's graph regardless of rollout length,
    which lets the curriculum go to long rollouts without OOM. All physics
    constraints are applied at every step via ``loss_fn``.
    """
    statevars = packer.prognostic + packer.surface
    H = model.history
    chan_mask = loss_fn.chan_mask

    # Normalized input history (oldest .. newest), each (B, C, J, I).
    hist = [normalizer.normalize(packer.pack(
                {v: batch["state_hist"][v][:, h].to(device) for v in statevars}))
            for h in range(H)]
    month_emb = batch["month_emb"].to(device)

    total_val = 0.0
    last_logs = {}
    for s in range(rollout_len):
        fnz = fnorm.normalize(pack_forcing(batch["forcing"], forcing_list, s, device))
        cond = month_cond(month_emb, fnz)
        prev = hist[-1]                                   # clean residual anchor
        x_in = hist[-1] if H == 1 else torch.stack(hist, dim=1)
        if noise > 0:
            x_in = x_in + noise * torch.randn_like(x_in)  # perturb network input only

        tgt = {v: batch["targets"][v][:, s].to(device) for v in statevars}
        forcing_phys = {v: batch["forcing"][v][:, s].to(device) for v in forcing_list}

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            x_next = model(x_in, fnz, cond, base=prev) * chan_mask
            tgt_norm = normalizer.normalize(packer.pack(tgt))
            res = loss_fn(x_next, tgt_norm, prev, forcing_phys=forcing_phys, step=step)
            loss_s = res["loss"] / (rollout_len * accum)
        loss_s.backward()

        total_val += float(res["logs"]["total"])
        last_logs = res["logs"]
        hist.append(x_next.detach())          # pushforward: feed own prediction
        if len(hist) > H:
            hist.pop(0)
        month_emb = _advance_month_emb(month_emb)

    return total_val / rollout_len, last_logs


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
    noise = tcfg.get("noise_injection", 0.0)
    if model.history > 1 or noise > 0:
        print(f"[train] history={model.history}  noise_injection={noise}  "
              f"(pushforward, per-step backward)")
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

            # pushforward_step does its own per-step backward (memory-safe).
            _, logs = pushforward_step(
                batch, model, packer, normalizer, fnorm, loss_fn, forcing_list,
                rl, noise, device, step, accum, use_amp, amp_dtype)

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
