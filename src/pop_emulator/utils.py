"""Small utilities: seeding, checkpointing, config loading, logging."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path: str, model, optimizer=None, scheduler=None,
                    step: int = 0, extra: Optional[dict] = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state = {"model": model.state_dict(), "step": step}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None,
                    map_location="cpu") -> int:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("step", 0)


def rollout_len_for_step(curriculum, step: int) -> int:
    """Return the rollout length for the current global step from a schedule
    like ``[[0, 1], [40000, 2], [80000, 4]]``."""
    n = 1
    for thresh, length in curriculum:
        if step >= thresh:
            n = length
    return n
