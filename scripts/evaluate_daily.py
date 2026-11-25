#!/usr/bin/env python
"""Evaluate a daily-step rollout of the emulator (CREDIT Gen 2 path).

Starting on the last day of a calendar month of the validation member, the
model is integrated for ``--days`` days (no input noise, no gradients) and
compared with what CESM-LE provides at each time scale:

* daily   SST [degC] and SSH [cm]: ocean-masked RMSE of the model and of
          persistence of the initial day, per lead day;
* monthly full 3-D state: RMSE (in units of the field standard deviation) of
          the calendar-month mean of the daily rollout against the monthly-mean
          truth, per variable, with persistence of the initial condition and,
          with ``--monthly-checkpoint``, the one-step prediction of the monthly
          model for the first month;
* global  the uniform TEMP [degC] and SALT offsets applied each day by the
          conservation projection (the budget error of the raw prediction).

Usage
-----
    python scripts/evaluate_daily.py --config configs/credit/run7_daily.yml \
        --checkpoint checkpoints/credit/run7_daily/checkpoint.pt --days 62 \
        [--monthly-config configs/run5.yaml --monthly-checkpoint checkpoints/run5/last.pt]
"""
from __future__ import annotations

import argparse
import copy
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import default_collate


def masked_rmse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    """RMSE over ocean points (``mask`` broadcastable to the fields)."""
    m = mask.expand_as(a)
    return float(torch.sqrt((((a - b) ** 2) * m).sum() / m.sum().clamp_min(1.0)))


def per_variable_rmse(pred: torch.Tensor, truth: torch.Tensor, chan_mask: torch.Tensor,
                      packer) -> Dict[str, float]:
    """Normalised RMSE of each state variable from packed ``(1, C, J, I)`` fields."""
    out, c = {}, 0
    for v in packer.prognostic + packer.surface:
        n = packer.nlev if v in packer.prognostic else 1
        sl = slice(c, c + n)
        out[v] = masked_rmse(pred[:, sl], truth[:, sl], chan_mask[:, sl])
        c += n
    return out


@torch.no_grad()
def monthly_model_step(pop_config: str, checkpoint: str, member: str, record: int,
                       device: torch.device) -> torch.Tensor:
    """One step of the monthly model from monthly records ``(record-2, record-1)``:
    its prediction of record ``record``, normalised and packed."""
    from pop_emulator import physics, utils
    from pop_emulator.credit_ext.common import get_grid, load_pop_config, make_packer
    from pop_emulator.data import _open_zarr, _read_slice
    from pop_emulator.model import OceanEmulator
    from pop_emulator.normalization import ForcingNormalizer, Normalizer
    from pop_emulator.rollout import _channel_mask, month_cond

    cfg = load_pop_config(pop_config)
    d = cfg["data"]
    packer, grid = make_packer(cfg), get_grid(pop_config, str(device))
    normalizer = Normalizer.from_stats(d["stats_file"], packer).to(device)
    fnorm = ForcingNormalizer.from_stats(d["stats_file"], d["forcing"]).to(device)
    model = OceanEmulator(cfg, packer, grid, n_forcing=len(d["forcing"])).to(device).eval()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt["model_state_dict"])

    def read(var: str, rec: int) -> torch.Tensor:
        handle = _open_zarr(d["zarr_root"], d["experiment"], member, var)
        return torch.from_numpy(np.asarray(_read_slice(handle, rec, rec + 1))).to(device)

    state_vars = d["prognostic"] + d["surface_prognostic"]
    hist = [normalizer.normalize(packer.pack({v: read(v, r) for v in state_vars}))
            for r in range(record - model.history, record)]
    forcing = {v: read(v, record - 1) for v in d["forcing"]}
    fnz = fnorm.normalize(torch.stack([forcing[v] for v in d["forcing"]], dim=1))
    sin, cos = utils.time_embedding(record - 1, 12)
    emb = torch.tensor([[sin, cos]], dtype=torch.float32, device=device)
    mask = _channel_mask(packer, grid)
    x_in = hist[-1] if model.history == 1 else torch.stack(hist, dim=1)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        x_next = model(x_in, fnz, month_cond(emb, fnz), base=hist[-1]) * mask
    if cfg["physics"].get("conservation_projection", False):
        pred, _ = physics.conservation_projection(
            packer.unpack(normalizer.denormalize(x_next.float())),
            packer.unpack(normalizer.denormalize(hist[-1].float())), forcing, grid,
            dt=utils.step_seconds(cfg))
        x_next = normalizer.normalize(packer.pack(pred)) * mask
    return x_next.float()


def summarise(cases: List[dict], n_days: int, state_vars: List[str]) -> None:
    """Mean over initial conditions; RMSEs are averaged in quadrature.  The skill
    ratio model/persistence is given with the range over cases (min, max)."""
    done = [c for c in cases if c["n_done"] == n_days]
    print(f"\n=== {len(done)} of {len(cases)} initial conditions with a finite {n_days}-day rollout ===")
    if not done:
        return
    arr = {k: np.asarray([c["series"][k] for c in done]) for k in done[0]["series"]}
    rms = lambda x: np.sqrt((x ** 2).mean(axis=0))
    print(f"{'lead [d]':>8} {'SST RMSE':>10} {'persist.':>10} {'ratio (min, max)':>22} "
          f"{'SSH RMSE':>10} {'persist.':>10} {'ratio (min, max)':>22}")
    for lead in (1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120, 180, 365):
        if lead > n_days:
            break
        row = f"{lead:8d}"
        for name in ("sst", "ssh"):
            m, p = arr[f"{name}_rmse"][:, lead - 1], arr[f"{name}_persistence"][:, lead - 1]
            r = m / p
            row += f" {rms(m):10.4f} {rms(p):10.4f} {rms(m) / rms(p):8.3f} ({r.min():.3f}, {r.max():.3f})"
        print(row)
    print("\nCalendar-month means, normalised RMSE: model (persistence)")
    for k in range(min(len(c["months"]) for c in done)):
        row = " ".join(
            f"{v}: {rms(np.asarray([c['months'][k]['model'][v] for c in done])):.4f} "
            f"({rms(np.asarray([c['months'][k]['persistence'][v] for c in done])):.4f})"
            for v in state_vars)
        print(f"  month {k + 1}: {row}")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="CREDIT Gen 2 daily config")
    ap.add_argument("--checkpoint", required=True,
                    help="CREDIT checkpoint (checkpoint.pt or checkpoint_ema.pt)")
    ap.add_argument("--days", type=int, default=62)
    ap.add_argument("--index", type=int, nargs="+", default=[120],
                    help="indices of the initial month ends in the validation member; "
                         "with several, a mean over initial conditions is also printed")
    ap.add_argument("--monthly-config", default=None, help="legacy config of the monthly model")
    ap.add_argument("--monthly-checkpoint", default=None)
    ap.add_argument("--out", default=None, help="write the per-day series to this .npz")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    from credit.models import load_model
    from credit.postblock import apply_postblocks, build_postblocks
    from credit.preblock import apply_preblocks, build_preblocks
    from credit.trainers.rollout_utils import assemble_rollout_batch
    from credit.trainers.utils import load_dataset

    from pop_emulator.credit_ext.common import get_grid, load_pop_config, make_packer
    from pop_emulator.credit_ext.dataset_daily import PopLENSDailyDataset
    from pop_emulator.credit_ext.postblocks import newest_step
    from pop_emulator.credit_ext.trainer_daily import month_blocks
    from pop_emulator.normalization import Normalizer
    from pop_emulator.rollout import _channel_mask

    pop_config = conf["model"]["pop_config"]
    cfg = load_pop_config(pop_config)
    packer, grid = make_packer(cfg), get_grid(pop_config, str(device))
    normalizer = Normalizer.from_stats(cfg["data"]["stats_file"], packer).to(device)
    chan_mask = _channel_mask(packer, grid)
    mask2d = grid.mask2d
    state_vars = packer.prognostic + packer.surface

    # Validation member through CREDIT's multi-source wrapper (nests fields by source).
    eval_conf = copy.deepcopy(conf)
    eval_conf["validation_data"]["forecast_len"] = args.days
    src_name = next(iter(eval_conf["validation_data"]["source"]))
    multi = load_dataset(eval_conf, is_train=False)
    dataset: PopLENSDailyDataset = multi.datasets[src_name]

    model = load_model(copy.deepcopy(conf)).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = ckpt.get("model_state_dict", ckpt.get("model"))
    if "shadow" in weights:     # CREDIT EMA checkpoint: parameters only, nested under "shadow"
        missing = model.load_state_dict(weights["shadow"], strict=False)
        if missing.unexpected_keys or any(k in dict(model.named_parameters())
                                          for k in missing.missing_keys):
            raise RuntimeError(f"EMA weights do not match the model: {missing}")
        print(f"load EMA weights (step {weights.get('step')}, decay {weights.get('decay')})")
    else:
        print("load weights:", model.load_state_dict(weights, strict=True))
    step_pre = build_preblocks(conf, phase="per_step")
    step_post = build_postblocks(conf, phase="per_step")

    def packed(nested: dict) -> torch.Tensor:
        """Nested physical fields -> normalised packed ``(1, C, J, I)``."""
        by_name = {k.rsplit("/", 1)[-1]: v for k, v in nested.items()}
        phys = {v: newest_step(by_name[v].float().to(device)) for v in state_vars}
        return normalizer.normalize(packer.pack(phys)) * chan_mask

    def surface(nested: dict) -> Dict[str, torch.Tensor]:
        by_name = {k.rsplit("/", 1)[-1]: v for k, v in nested.items()}
        return {"SST": by_name["TEMP"][:, 0, -1].float().to(device),
                "SSH": by_name["SSH"][:, 0, -1].float().to(device)}

    cases: List[dict] = []
    for index in args.index:
        t0 = multi.datetimes[index]
        member, day0 = dataset._locate(t0)
        series: Dict[str, List[float]] = {k: [] for k in (
            "sst_rmse", "sst_persistence", "ssh_rmse", "ssh_persistence", "proj_delta_T", "proj_delta_S")}
        blocks = month_blocks(day0 + 1, args.days)
        ends = np.cumsum([n for n, _ in blocks])
        block, mean_pred, mean_truth, months = 0, None, None, []
        full: dict = {}
        for i in range(args.days):
            batch = default_collate([multi[(t0 + i * dataset.dt, i)]])
            if i == 0:
                full["ic_preprocessed"] = batch
                step_input = batch
                ic_surface = surface(batch["input"][src_name])
                ic_packed = packed(batch["input"][src_name])
            else:
                step_input = assemble_rollout_batch(full, batch, history_len=dataset.history_len)
            full["x_physical"] = {s: dict(v) for s, v in step_input["input"].items()}
            full.update(apply_preblocks(step_pre, step_input, device=device))
            if dataset.history_len > 1:
                if i > 0:
                    full["x"] = torch.cat([history_x[:, :, 1:], full["x"][:, :, -1:]], dim=2)
                history_x = full["x"]
            full["y_pred"] = model(full["x"])
            full = apply_postblocks(step_post, full)

            pred, truth = full["y_processed"][src_name], batch["target"][src_name]
            if not all(torch.isfinite(v).all() for v in pred.values()):
                print(f"non-finite prediction at lead day {i + 1}; stopping")
                break
            p_sfc, t_sfc = surface(pred), surface(truth)
            for name, key in (("sst", "SST"), ("ssh", "SSH")):
                series[f"{name}_rmse"].append(masked_rmse(p_sfc[key], t_sfc[key], mask2d))
                series[f"{name}_persistence"].append(masked_rmse(ic_surface[key], t_sfc[key], mask2d))
            diag = full.get("projection_diag") or {}
            for key in ("proj_delta_T", "proj_delta_S"):
                series[key].append(float(diag.get(key, np.nan)))

            # Calendar-month means of the rollout against the monthly-mean truth.
            p, t = packed(pred), packed(truth)
            mean_pred = p if mean_pred is None else mean_pred + p
            mean_truth = t if mean_truth is None else mean_truth + t
            if i + 1 == ends[block]:
                n, complete = blocks[block]
                if complete:
                    months.append({
                        "first_lead_day": i + 2 - n,
                        "model": per_variable_rmse(mean_pred / n, mean_truth / n, chan_mask, packer),
                        "persistence": per_variable_rmse(ic_packed, mean_truth / n, chan_mask, packer),
                        "mean_pred": mean_pred / n, "mean_truth": mean_truth / n})
                block, mean_pred, mean_truth = block + 1, None, None

        n_done = len(series["sst_rmse"])
        print(f"\nmember {member}, initial day {day0} (noleap day index since the first daily record), "
              f"{n_done} of {args.days} days finite")
        print(f"{'lead [d]':>8} {'SST RMSE':>10} {'persist.':>10} {'SSH RMSE':>10} {'persist.':>10}")
        for lead in (1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120, 180, 365):
            if lead <= n_done:
                print(f"{lead:8d} {series['sst_rmse'][lead - 1]:10.4f} {series['sst_persistence'][lead - 1]:10.4f} "
                      f"{series['ssh_rmse'][lead - 1]:10.4f} {series['ssh_persistence'][lead - 1]:10.4f}")
        if n_done:
            print(f"\nmean |projection offset| per day: TEMP {np.nanmean(series['proj_delta_T']):.3e} degC, "
                  f"SALT {np.nanmean(series['proj_delta_S']):.3e}")
        print("\nCalendar-month means, RMSE in units of the field standard deviation")
        for k, mth in enumerate(months):
            row = " ".join(f"{v}: {mth['model'][v]:.4f} ({mth['persistence'][v]:.4f})" for v in state_vars)
            print(f"  month {k + 1} (lead days {mth['first_lead_day']}+): model (persistence)  {row}")

        if args.monthly_checkpoint and months:
            from pop_emulator.credit_ext.dataset_daily import month_of_day
            record = dataset.month0[member] + month_of_day(day0 + 1)[0]
            x_month = monthly_model_step(args.monthly_config, args.monthly_checkpoint, member, record, device)
            first = months[0]
            vs_truth = per_variable_rmse(x_month, first["mean_truth"], chan_mask, packer)
            vs_daily = per_variable_rmse(x_month, first["mean_pred"], chan_mask, packer)
            print("\nFirst month: monthly model, one step (monthly record %d)" % record)
            print("  monthly model vs truth:       " + " ".join(f"{v}: {vs_truth[v]:.4f}" for v in state_vars))
            print("  monthly model vs daily-mean:  " + " ".join(f"{v}: {vs_daily[v]:.4f}" for v in state_vars))

        cases.append({"day0": day0, "n_done": n_done, "series": series,
                      "months": [{k: m[k] for k in ("model", "persistence")} for m in months]})
        del months, full

    if len(cases) > 1:
        summarise(cases, args.days, state_vars)

    if args.out:
        done = [c for c in cases if c["n_done"] == args.days]      # complete rollouts only
        arrays = {k: np.asarray([c["series"][k] for c in done]) for k in series}   # (case, day)
        for who in ("model", "persistence"):
            for v in state_vars:
                arrays[f"month_{who}_{v}"] = np.asarray(
                    [[m[who][v] for m in c["months"]] for c in done])              # (case, month)
        np.savez(args.out, **arrays, day0=np.asarray([c["day0"] for c in done]))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
