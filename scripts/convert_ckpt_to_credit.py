#!/usr/bin/env python
"""Convert a legacy emulator checkpoint to the CREDIT Gen 2 layout.

Legacy (``scripts/train.py``):  ``{"model", "step", "optimizer", "extra"}``
CREDIT Gen 2:                   ``{save_loc}/checkpoint.pt`` with
                                ``{"epoch", "model_state_dict"}``

With ``--cond-step`` the FiLM input weight gains a zero column for the
step-ratio condition (``model.cond_step: true``), which leaves outputs unchanged.

Parameter names are unchanged (``net.*``) because ``PopOceanUNet`` derives from
``OceanEmulator``.  The optimiser state is not carried over: CREDIT builds its
own AdamW, so train with ``load_weights: true`` and ``load_optimizer: false``.

Usage
-----
    python scripts/convert_ckpt_to_credit.py \
        --checkpoint checkpoints/run5/last.pt \
        --config configs/credit/run6_monthly.yml
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="legacy checkpoint (.pt)")
    ap.add_argument("--config", required=True, help="CREDIT Gen 2 config (gives save_loc)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing checkpoint.pt")
    ap.add_argument("--cond-step", action="store_true",
                    help="target model has model.cond_step: true — append a zero column to the "
                         "FiLM input weight for the step-ratio condition (outputs unchanged)")
    ap.add_argument("--reset-epoch", action="store_true",
                    help="store epoch -1 so training starts at epoch 0 (new training stage, "
                         "e.g. the daily fine-tune) instead of continuing the legacy step count")
    args = ap.parse_args()

    with open(args.config) as f:
        conf = yaml.safe_load(f)
    save_loc = os.path.expandvars(conf["save_loc"])
    out = os.path.join(save_loc, "checkpoint.pt")
    if os.path.exists(out) and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = dict(ckpt["model"] if "model" in ckpt else ckpt["model_state_dict"])
    if args.cond_step:
        key = "net.film.net.0.weight"                     # (channels, cond_dim)
        w = state[key]
        state[key] = torch.cat([w, w.new_zeros(w.shape[0], 1)], dim=1)
        print(f"padded {key}: {tuple(w.shape)} -> {tuple(state[key].shape)}")
    step = int(ckpt.get("step", 0))
    bpe = int(conf["trainer"].get("batches_per_epoch", 0))
    start_epoch = step // bpe if bpe > 0 and not args.reset_epoch else 0

    os.makedirs(save_loc, exist_ok=True)
    # "epoch" is the last completed epoch, as CREDIT writes it.
    torch.save({"epoch": start_epoch - 1, "model_state_dict": state}, out)
    n_params = sum(v.numel() for v in state.values())
    print(f"wrote {out}: {len(state)} tensors, {n_params / 1e6:.1f}M values, legacy step {step}")
    print(f"to continue the curriculum from step {step}, set trainer.start_epoch: {start_epoch} "
          f"(batches_per_epoch={bpe})")


if __name__ == "__main__":
    main()
