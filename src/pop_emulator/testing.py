"""Helpers to build tiny synthetic inputs for smoke tests and unit tests.

These let the full code path (grid -> physics -> model -> loss -> backward) run
on CPU in seconds without touching the 23 GB CESM-LENS files.
"""
from __future__ import annotations

import torch

from .constants import PopConstants
from .grid import PopGrid
from .normalization import ForcingNormalizer, Normalizer, StatePacker


def synthetic_grid(nz=4, J=16, I=16, device="cpu") -> PopGrid:
    """A small, physically sane gx1v6-like grid with a land cap and bowl bottom."""
    dz = torch.linspace(500.0, 2000.0, nz)             # cm
    z_t = torch.cumsum(dz, 0) - dz / 2
    tarea = torch.full((J, I), 1.0e14)                 # cm^2 (~1 deg cell)
    uarea = tarea.clone()
    htn = torch.full((J, I), 1.0e7)                    # cm edge length
    hte = torch.full((J, I), 1.0e7)

    # KMT: full depth except a land strip at the top two rows -> partial cells.
    kmt = torch.full((J, I), nz, dtype=torch.int64)
    kmt[:2] = 0                                        # land
    kmt[2] = max(1, nz - 2)                            # shallow shelf
    kk = torch.arange(nz)[:, None, None]
    mask3d = (kk < kmt[None]).float()
    mask2d = (kmt > 0).float()

    tlat = torch.linspace(-80, 80, J)[:, None].expand(J, I).contiguous()
    tlong = torch.linspace(0, 360, I)[None, :].expand(J, I).contiguous()
    consts = PopConstants()
    fcor = (2 * consts.omega * torch.sin(torch.deg2rad(tlat))).float()
    region = mask2d.long()

    return PopGrid(
        dz=dz, z_t=z_t, tarea=tarea, uarea=uarea, htn=htn, hte=hte,
        kmt=kmt, mask3d=mask3d, mask2d=mask2d, fcor=fcor,
        tlat=tlat, tlong=tlong, region_mask=region, consts=consts,
    ).to(device)


def synthetic_cfg(nz=4, forcing=("SHF", "SFWF", "TAUX")):
    return {
        "seed": 0,
        "data": {
            "prognostic": ["TEMP", "SALT", "UVEL", "VVEL"],
            "surface_prognostic": ["SSH"],
            "forcing": list(forcing),
            "nlev": nz,
        },
        "model": {"arch": "spherical_unet", "width": 16, "depth": 4,
                  "blocks_per_stage": 1, "residual_update": True},
        "physics": {
            "diagnose_wvel": True, "barotropic": "rigid_lid",
            "physics_warmup_steps": 0,
            "loss_weights": {"data": 1.0, "continuity": 0.1, "barotropic": 0.1,
                             "heat": 0.05, "salt": 0.05, "stability": 0.02},
        },
        "train": {"batch_size": 2},
    }


def identity_normalizers(packer: StatePacker, n_forcing: int):
    mean = torch.zeros(packer.n_channels)
    std = torch.ones(packer.n_channels)
    norm = Normalizer(mean, std)
    fnorm = ForcingNormalizer(torch.zeros(n_forcing), torch.ones(n_forcing))
    return norm, fnorm


def synthetic_state(grid: PopGrid, packer: StatePacker, B=2):
    """A plausible stratified ocean state (warm/fresh surface, cold/salty deep)."""
    Z, J, I = grid.shape
    z = grid.z_t / grid.z_t.max()
    temp = (20.0 * (1 - z) + 2.0)[None, :, None, None].expand(B, Z, J, I).clone()
    salt = (34.0 + 1.0 * z)[None, :, None, None].expand(B, Z, J, I).clone()
    uvel = 0.5 * torch.randn(B, Z, J, I)
    vvel = 0.5 * torch.randn(B, Z, J, I)
    ssh = 0.1 * torch.randn(B, J, I)
    fields = {"TEMP": temp, "SALT": salt, "UVEL": uvel, "VVEL": vvel, "SSH": ssh}
    m3 = grid.mask3d[None]
    for k in ("TEMP", "SALT", "UVEL", "VVEL"):
        fields[k] = fields[k] * m3
    fields["SSH"] = fields["SSH"] * grid.mask2d[None]
    return fields
