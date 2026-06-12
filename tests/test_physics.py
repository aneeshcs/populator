"""Unit tests for the physics operators (run with: pytest)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pop_emulator import physics
from pop_emulator.grid import shift_i, shift_j
from pop_emulator.normalization import StatePacker
from pop_emulator import testing


def test_eos_surface_matches_eos80():
    # Surface seawater density check values (EOS-80 sigma-theta).
    r = physics.mwjf_density(torch.tensor(25.0), torch.tensor(35.0), torch.tensor(0.0))
    assert abs(float(r) - 1023.343) < 0.05
    r2 = physics.mwjf_density(torch.tensor(20.0), torch.tensor(35.0), torch.tensor(0.0))
    assert abs(float(r2) - 1024.763) < 0.05


def test_eos_monotonic_in_temperature():
    p = torch.tensor(0.0)
    cold = physics.mwjf_density(torch.tensor(2.0), torch.tensor(35.0), p)
    warm = physics.mwjf_density(torch.tensor(28.0), torch.tensor(35.0), p)
    assert float(cold) > float(warm)


def test_eos_monotonic_in_salinity_and_pressure():
    fresh = physics.mwjf_density(torch.tensor(10.0), torch.tensor(33.0), torch.tensor(0.0))
    salty = physics.mwjf_density(torch.tensor(10.0), torch.tensor(36.0), torch.tensor(0.0))
    assert float(salty) > float(fresh)
    shallow = physics.mwjf_density(torch.tensor(10.0), torch.tensor(35.0), torch.tensor(0.0))
    deep = physics.mwjf_density(torch.tensor(10.0), torch.tensor(35.0), torch.tensor(2000.0))
    assert float(deep) > float(shallow)


def test_eos_gradients_finite_with_zero_salt():
    s = torch.zeros(4, requires_grad=True)
    t = torch.full((4,), 5.0)
    p = torch.zeros(4)
    rho = physics.mwjf_density(t, s, p)
    rho.sum().backward()
    assert torch.isfinite(s.grad).all()


def test_wvel_vanishes_at_bottom():
    grid = testing.synthetic_grid()
    packer = StatePacker(["TEMP", "SALT", "UVEL", "VVEL"], ["SSH"], grid.shape[0])
    state = testing.synthetic_state(grid, packer, B=2)
    w = physics.diagnose_wvel(state["UVEL"], state["VVEL"], grid)
    bottom_k = (grid.kmt.clamp_min(1) - 1).long()
    wb = w[0].gather(0, bottom_k[None]).squeeze(0)[grid.mask2d.bool()]
    assert torch.allclose(wb, torch.zeros_like(wb), atol=1e-3)


def test_divergence_of_uniform_flow_small():
    # A spatially uniform velocity field has near-zero divergence on a
    # uniform-metric grid (edges/areas constant in the synthetic grid).
    grid = testing.synthetic_grid()
    u = torch.ones(1, grid.shape[0], *grid.mask2d.shape) * grid.mask3d
    v = torch.ones_like(u) * grid.mask3d
    div = grid.divergence(u, v) * grid.mask3d
    interior = div[:, :, 5:-1, :]  # away from the land cap / boundaries
    assert float(interior.abs().max()) < 1e-6


def test_shift_helpers_roundtrip():
    x = torch.randn(2, 3, 8, 8)
    assert torch.allclose(shift_i(shift_i(x, 1), -1), x)
    # replicate shift is not exactly invertible at the edge; check interior.
    y = shift_j(shift_j(x, 1), -1)
    assert torch.allclose(y[:, :, 1:-1], x[:, :, 1:-1])


def test_conservation_runs_and_is_finite():
    grid = testing.synthetic_grid()
    packer = StatePacker(["TEMP", "SALT", "UVEL", "VVEL"], ["SSH"], grid.shape[0])
    state = testing.synthetic_state(grid, packer, B=2)
    pred = {k: v.clone() for k, v in state.items()}
    pred["TEMP"] = pred["TEMP"] + 0.01 * grid.mask3d[None]
    shp = (2, *grid.mask2d.shape)
    cons = physics.conservation_losses(
        pred, state, grid,
        forcing={"SHF": torch.zeros(shp), "SFWF": torch.zeros(shp)})
    for k in ("heat", "salt", "barotropic", "continuity"):
        assert torch.isfinite(cons[k])


def test_static_stability_zero_when_equal():
    grid = testing.synthetic_grid()
    packer = StatePacker(["TEMP", "SALT", "UVEL", "VVEL"], ["SSH"], grid.shape[0])
    state = testing.synthetic_state(grid, packer, B=2)
    pen = physics.static_stability_penalty(
        state["TEMP"], state["SALT"], state["TEMP"], state["SALT"], grid)
    assert float(pen) == 0.0
