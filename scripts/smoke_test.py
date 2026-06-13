#!/usr/bin/env python
"""End-to-end smoke test on synthetic data (CPU, no GPU or CESM files needed).

Exercises the full path: synthetic grid -> physics operators (EOS, continuity,
conservation) -> packer/normalizer -> model forward -> composite loss ->
backward. Prints a few physical sanity checks. Run before any large job:

    python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pop_emulator import physics                       # noqa: E402
from pop_emulator.losses import CompositeLoss           # noqa: E402
from pop_emulator.model import OceanEmulator            # noqa: E402
from pop_emulator.normalization import StatePacker      # noqa: E402
from pop_emulator.rollout import month_cond             # noqa: E402
from pop_emulator import testing                        # noqa: E402

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"


def check(name, ok, detail=""):
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{tag}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        raise SystemExit(1)


def main():
    torch.manual_seed(0)
    cfg = testing.synthetic_cfg()
    nz = cfg["data"]["nlev"]
    forcing_list = cfg["data"]["forcing"]
    grid = testing.synthetic_grid(nz=nz)
    packer = StatePacker(cfg["data"]["prognostic"], cfg["data"]["surface_prognostic"], nz)
    norm, fnorm = testing.identity_normalizers(packer, len(forcing_list))

    print("1. Equation of state (MWJF 2003)")
    # Surface, T=10C S=35 -> ~1026-1027 kg/m^3.
    rho = physics.mwjf_density(torch.tensor(10.0), torch.tensor(35.0),
                               torch.tensor(0.0))
    check("density in physical range", 1020 < float(rho) < 1030, f"{float(rho):.2f} kg/m^3")
    # Cold water denser than warm at fixed S.
    rho_cold = physics.mwjf_density(torch.tensor(2.0), torch.tensor(35.0), torch.tensor(0.0))
    check("cold water denser than warm", float(rho_cold) > float(rho))

    print("2. Continuity / vertical-velocity diagnosis")
    state = testing.synthetic_state(grid, packer, B=2)
    w = physics.diagnose_wvel(state["UVEL"], state["VVEL"], grid)
    check("wvel shape", tuple(w.shape) == (2, nz, *grid.mask2d.shape))
    # w must vanish at the bottom of each ocean column (rigid floor).
    bottom_k = (grid.kmt.clamp_min(1) - 1).long()
    wb = w[0].gather(0, bottom_k[None]).squeeze(0)[grid.mask2d.bool()]
    check("w = 0 at the ocean floor", torch.allclose(wb, torch.zeros_like(wb), atol=1e-3))

    print("3. Conservation budgets")
    pred = {k: v.clone() for k, v in state.items()}
    pred["TEMP"] = pred["TEMP"] + 0.01 * grid.mask3d[None]   # uniform warming
    cons = physics.conservation_losses(pred, state, grid,
                                       forcing={"SHF": torch.zeros(2, *grid.mask2d.shape),
                                                "SFWF": torch.zeros(2, *grid.mask2d.shape)})
    check("heat budget term finite", torch.isfinite(cons["heat"]))
    check("salt budget term finite", torch.isfinite(cons["salt"]))
    check("barotropic term finite", torch.isfinite(cons["barotropic"]))
    print(f"     heat drift (J) = {float(cons['heat_drift_J']):.3e}")

    print("4. Static stability penalty")
    stab = physics.static_stability_penalty(state["TEMP"], state["SALT"],
                                            state["TEMP"], state["SALT"], grid)
    check("zero penalty when pred == truth", float(stab) == 0.0)

    print("5. Model forward + residual update + land mask")
    model = OceanEmulator(cfg, packer, grid, n_forcing=len(forcing_list))
    x = norm.normalize(packer.pack(state))
    fz = torch.randn(2, len(forcing_list), *grid.mask2d.shape)
    fnz = fnorm.normalize(fz)
    cond = month_cond(torch.tensor([[0.0, 1.0], [0.5, 0.5]]), fnz)
    y = model(x, fnz, cond)
    check("output shape == input shape", y.shape == x.shape)
    # With zero-initialized output conv, residual update starts as identity.
    check("residual update is identity at init", torch.allclose(y, x, atol=1e-5))

    print("6. Composite loss + backward")
    loss_fn = CompositeLoss(cfg, packer, norm, grid)
    target = norm.normalize(packer.pack(pred))
    forcing_phys = {v: fz[:, i] for i, v in enumerate(forcing_list)}
    res = loss_fn(y, target, x, forcing_phys=forcing_phys, step=10 ** 9)
    check("loss is finite scalar", torch.isfinite(res["loss"]) and res["loss"].ndim == 0)
    res["loss"].backward()
    gnorm = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    check("gradients flow to the model", float(gnorm) > 0)

    print("7. History (2-state input) + pushforward rollout")
    cfg2 = testing.synthetic_cfg()
    cfg2["model"]["history"] = 2
    model2 = OceanEmulator(cfg2, packer, grid, n_forcing=len(forcing_list))
    x_hist = torch.stack([x, x], dim=1)  # (B, H=2, C, J, I)
    y2 = model2(x_hist, fnz, cond)
    check("history forward output shape", y2.shape == x.shape)
    check("history residual identity at init", torch.allclose(y2, x, atol=1e-5))

    # mini pushforward: 3 detached steps, per-step backward (as in training)
    loss_fn2 = CompositeLoss(cfg2, packer, norm, grid)
    chan_mask = loss_fn2.chan_mask
    hist = [x, x]
    me = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
    for s in range(3):
        xin = torch.stack(hist, dim=1) + 0.1 * torch.randn_like(torch.stack(hist, dim=1))
        xn = model2(xin, fnz, month_cond(me, fnz)) * chan_mask
        r = loss_fn2(xn, target, hist[-1], forcing_phys=forcing_phys, step=10 ** 9)
        (r["loss"] / 3).backward()
        hist.append(xn.detach()); hist.pop(0)
    g2 = sum(p.grad.abs().sum() for p in model2.parameters() if p.grad is not None)
    check("pushforward gradients finite & flowing",
          bool(torch.isfinite(torch.as_tensor(float(g2)))) and float(g2) > 0)

    print(f"\n{GREEN}All smoke checks passed.{RESET}")


if __name__ == "__main__":
    main()
