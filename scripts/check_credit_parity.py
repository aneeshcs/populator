#!/usr/bin/env python
"""Parity check: CREDIT Gen 2 path vs the legacy loop, same weights and data.

Runs a short deterministic rollout (no input noise, no gradients) twice:

* legacy   — the per-step computation of ``scripts/train.py::pushforward_step``
* CREDIT   — the per-step computation of ``TrainerERA5Gen2`` (preblocks ->
             model -> postblocks -> loss, with ``assemble_rollout_batch``
             feeding the prediction back)

and reports, per step, the loss from each path and the largest difference
between the predicted normalised states.  Differences should be at the level
of float32 round-off (the two paths differ only in operation order).

Usage
-----
    python scripts/check_credit_parity.py --config configs/credit/run6_monthly.yml \
        --checkpoint checkpoints/run5/last.pt --steps 2
"""
from __future__ import annotations

import argparse
import copy

import torch
import yaml
from torch.utils.data import default_collate


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="CREDIT Gen 2 config")
    ap.add_argument("--checkpoint", required=True, help="legacy checkpoint (.pt)")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--index", type=int, default=5, help="index of the initial tick")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        conf = yaml.safe_load(f)
    conf["data"]["forecast_len"] = args.steps
    src_name, src = next(iter(conf["data"]["source"].items()))
    src["members"] = src["members"][:1]
    src["read_ahead"] = args.steps + conf["data"]["history_len"] + 1
    pop_config = conf["model"]["pop_config"]

    from credit.models import load_model
    from credit.losses import load_loss
    from credit.postblock import apply_postblocks, build_postblocks
    from credit.preblock import apply_preblocks, build_preblocks
    from credit.trainers.rollout_utils import assemble_rollout_batch
    from credit.trainers.utils import load_dataset

    from pop_emulator import physics
    from pop_emulator.credit_ext.common import get_grid, load_pop_config, step_seconds
    from pop_emulator.losses import CompositeLoss
    from pop_emulator.model import OceanEmulator
    from pop_emulator.normalization import ForcingNormalizer, Normalizer, StatePacker
    from pop_emulator.rollout import month_cond

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]

    # ---------------- CREDIT Gen 2 components ---------------- #
    dataset = load_dataset(conf, is_train=True)
    model = load_model(copy.deepcopy(conf)).to(device)
    print("load into PopOceanUNet (strict):", model.load_state_dict(state, strict=True))
    model.eval()
    criterion = load_loss(conf)
    step_pre = build_preblocks(conf, phase="per_step")
    step_post = build_postblocks(conf, phase="per_step")

    # ---------------- legacy components ---------------- #
    cfg = load_pop_config(pop_config)
    d = cfg["data"]
    packer = StatePacker(d["prognostic"], d["surface_prognostic"], d["nlev"])
    grid = get_grid(pop_config, str(device))
    normalizer = Normalizer.from_stats(d["stats_file"], packer).to(device)
    fnorm = ForcingNormalizer.from_stats(d["stats_file"], d["forcing"]).to(device)
    legacy = OceanEmulator(cfg, packer, grid, n_forcing=len(d["forcing"])).to(device)
    legacy.load_state_dict(state)
    legacy.eval()
    loss_fn = CompositeLoss(cfg, packer, normalizer, grid).to(device)
    statevars = d["prognostic"] + d["surface_prognostic"]
    project = cfg["physics"].get("conservation_projection", False)
    dt = step_seconds(cfg)

    t0 = dataset.datetimes[args.index]
    member, rec = dataset.datasets[src_name]._locate(t0)
    print(f"member {member}, record {rec}, {args.steps} steps, device {device}")

    def key(field, v):
        return f"{src_name}/{field}/{'3d' if v in d['prognostic'] else '2d'}/{v}"

    full: dict = {}
    hist = None
    for i in range(args.steps):
        batch = default_collate([dataset[(t0 + i * dataset.dt, i)]])
        raw_in, raw_tgt = batch["input"][src_name], batch["target"][src_name]

        # ---- legacy step ---- #
        if i == 0:
            hist = [normalizer.normalize(packer.pack(
                {v: (raw_in[key("prognostic", v)][:, :, h] if v in d["prognostic"]
                     else raw_in[key("prognostic", v)][:, 0, h]).to(device) for v in statevars}))
                for h in range(legacy.history)]
        forcing_phys = {v: raw_in[f"{src_name}/dynamic_forcing/2d/{v}"][:, 0, -1].to(device)
                        for v in d["forcing"]}
        fnz = fnorm.normalize(torch.stack([forcing_phys[v] for v in d["forcing"]], dim=1))
        emb = torch.stack([raw_in[f"{src_name}/dynamic_forcing/2d/TIME_{s}"][:, 0, -1, 0, 0]
                           for s in ("SIN", "COS")], dim=-1).to(device)
        prev = hist[-1]
        x_in = prev if legacy.history == 1 else torch.stack(hist, dim=1)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            x_next = legacy(x_in, fnz, month_cond(emb, fnz), base=prev) * loss_fn.chan_mask
        if project:
            pred_phys, _ = physics.conservation_projection(
                packer.unpack(normalizer.denormalize(x_next.float())),
                packer.unpack(normalizer.denormalize(prev.float())), forcing_phys, grid, dt=dt)
            x_next = normalizer.normalize(packer.pack(pred_phys)) * loss_fn.chan_mask
        tgt = {v: (raw_tgt[key("prognostic", v)][:, :, 0] if v in d["prognostic"]
                   else raw_tgt[key("prognostic", v)][:, 0, 0]).to(device) for v in statevars}
        legacy_loss = loss_fn(x_next, normalizer.normalize(packer.pack(tgt)), prev,
                              forcing_phys=forcing_phys)["loss"]
        hist = (hist + [x_next])[-legacy.history:]

        # ---- CREDIT Gen 2 step (as in TrainerERA5Gen2.train_one_epoch) ---- #
        if i == 0:
            full["ic_preprocessed"] = batch
            step_input = batch
        else:
            step_input = assemble_rollout_batch(full, batch, history_len=legacy.history)
        full["x_physical"] = {s: dict(v) for s, v in step_input["input"].items()}
        full.update(apply_preblocks(step_pre, step_input, device=device))
        if legacy.history > 1:
            if i > 0:
                full["x"] = torch.cat([history_x[:, :, 1:], full["x"][:, :, -1:]], dim=2)
            history_x = full["x"]
        full["y_pred"] = model(full["x"])
        full = apply_postblocks(step_post, full)
        credit_loss = criterion(full)

        y = full["y_processed"][src_name]
        credit_norm = normalizer.normalize(packer.pack(
            {v: (y[key("prognostic", v)][:, :, 0] if v in d["prognostic"]
                 else y[key("prognostic", v)][:, 0, 0]) for v in statevars})) * loss_fn.chan_mask
        diff = (credit_norm - x_next).abs().max().item()
        rel = abs(credit_loss.item() - legacy_loss.item()) / max(abs(legacy_loss.item()), 1e-30)
        print(f"step {i + 1}: legacy loss {legacy_loss.item():.8f} | CREDIT loss "
              f"{credit_loss.item():.8f} | rel diff {rel:.2e} | max |d state| (norm.) {diff:.2e}")


if __name__ == "__main__":
    main()
