#!/usr/bin/env python
"""Train the POP emulator with the CREDIT Gen 2 trainer.

Thin launcher around ``credit.applications.train_gen2``: CREDIT resolves
datasets, models, pre/postblocks and losses through ``custom_objects`` in the
config, but trainers come from a fixed table, so ``PopTrainer`` is registered
here before CREDIT's own entry point runs.

When the config has a rollout curriculum, ``data.forecast_len`` is set to the
length of the starting epoch so the first DataLoader CREDIT builds is the one
actually used.

Usage
-----
    python scripts/train_credit.py -c configs/credit/run6_monthly.yml
    torchrun --nproc-per-node 4 scripts/train_credit.py -c configs/credit/run6_monthly.yml
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import yaml


def _start_epoch(conf: dict) -> int:
    """Epoch the run will start at (mirrors CREDIT's ``reload_epoch`` logic)."""
    trainer = conf["trainer"]
    log = os.path.join(os.path.expandvars(conf["save_loc"]), "training_log.csv")
    if trainer.get("load_weights") and trainer.get("reload_epoch") and os.path.exists(log):
        with open(log) as f:
            return max(0, sum(1 for _ in f) - 1)      # rows minus header
    return int(trainer.get("start_epoch", 0))


def main() -> None:
    from pop_emulator.credit_ext import trainer as pop_trainer

    pop_trainer.register()

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("-c", "--config", dest="config", required=True)
    args, rest = ap.parse_known_args()

    with open(args.config) as f:
        conf = yaml.safe_load(f)
    config_path = args.config
    forecast_len = pop_trainer.curriculum_len(conf, _start_epoch(conf))
    if forecast_len != conf["data"]["forecast_len"]:
        conf["data"]["forecast_len"] = forecast_len
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False,
                                          prefix=f"rank{os.environ.get('RANK', '0')}_")
        yaml.safe_dump(conf, tmp, sort_keys=False)
        tmp.close()
        config_path = tmp.name

    from credit.applications.train_gen2 import main_cli

    sys.argv = [sys.argv[0], "-c", config_path, *rest]
    main_cli()


if __name__ == "__main__":
    main()
