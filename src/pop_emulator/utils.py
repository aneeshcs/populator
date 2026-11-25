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
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    # Legacy checkpoints store the weights under "model"; CREDIT Gen 2
    # checkpoints (credit_ext) under "model_state_dict" with the same names.
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt["model_state_dict"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("step", 0)


def load_weights_only(path: str, model, map_location="cpu") -> None:
    """Load model weights from a checkpoint, ignoring optimizer state and step.

    Use this for warm-starting a new run from a previously trained checkpoint
    when you want a fresh optimizer and step counter (e.g. run3 -> run4).
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])


def step_seconds(cfg: dict) -> float:
    """Model step length Δt [s]: ``physics.dt_seconds``, else one POP noleap month."""
    from .constants import SECONDS_PER_MONTH

    return float(cfg.get("physics", {}).get("dt_seconds", SECONDS_PER_MONTH))


def records_per_year(cfg: dict) -> int:
    """Records per model year: 12 for monthly means, 365 for daily (noleap)."""
    return int(cfg.get("data", {}).get("records_per_year", 12))


def time_embedding(record: int, per_year: int = 12):
    """(sin, cos) of the annual phase for a record index counted from 1 January."""
    import math

    ang = 2.0 * math.pi * (record % per_year) / per_year
    return math.sin(ang), math.cos(ang)


def rollout_len_for_step(curriculum, step: int) -> int:
    """Return the rollout length for the current global step from a schedule
    like ``[[0, 1], [40000, 2], [80000, 4]]``."""
    n = 1
    for thresh, length in curriculum:
        if step >= thresh:
            n = length
    return n
