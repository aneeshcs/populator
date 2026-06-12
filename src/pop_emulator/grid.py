"""gx1v6 grid geometry, masks, Coriolis, and B-grid differential operators.

The :class:`PopGrid` bundles the static fields needed by the physics layer and
the model: layer thicknesses ``dz``, cell areas ``TAREA``, cell-edge lengths
``HTN``/``HTE``, the 3-D ocean mask derived from ``KMT``, and the Coriolis
parameter ``f``. Everything is held as ``torch`` tensors on a chosen device, in
POP's native CGS units.

Horizontal staggering (POP B-grid)
----------------------------------
Tracers (T, S) live at T-cell centers; velocities (u, v) at the NE corner
(U point). With ``(j, i)`` = ``(nlat, nlon)`` index ordering and the U point of
column ``(j, i)`` sitting at the corner shared by T-cells
``(j,i), (j,i+1), (j+1,i), (j+1,i+1)``, the finite-volume divergence of the
velocity field onto a T-cell uses the four surrounding U points (see
:meth:`PopGrid.divergence`). Longitude ``i`` is periodic; latitude ``j`` is
padded by replication (a documented approximation to the tripole fold).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .constants import PopConstants


# --------------------------------------------------------------------------- #
# Padding helpers: longitude periodic, latitude replicate.
# --------------------------------------------------------------------------- #
def shift_i(x: torch.Tensor, s: int) -> torch.Tensor:
    """Shift along longitude (last axis) with periodic wrap. ``s=+1`` brings the
    ``i-1`` neighbour into position ``i`` (i.e. a west shift)."""
    return torch.roll(x, shifts=s, dims=-1)


def shift_j(x: torch.Tensor, s: int) -> torch.Tensor:
    """Shift along latitude (second-to-last axis) with edge replication.
    ``s=+1`` brings the ``j-1`` neighbour into position ``j``."""
    if s == 0:
        return x
    if s > 0:  # bring j-1 -> j : pad one row at the south, drop the north
        pad = x[..., :1, :].expand(*x.shape[:-2], s, x.shape[-1])
        return torch.cat([pad, x[..., :-s, :]], dim=-2)
    s = -s     # bring j+1 -> j : pad at the north, drop the south
    pad = x[..., -1:, :].expand(*x.shape[:-2], s, x.shape[-1])
    return torch.cat([x[..., s:, :], pad], dim=-2)


def pad_circular_replicate(x: torch.Tensor, pad: int = 1) -> torch.Tensor:
    """Pad (..., J, I) with circular padding in I and replicate padding in J.
    Used by the network's convolutions so the longitude seam is seamless."""
    xi = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)           # lon wrap
    top = xi[..., :1, :].expand(*xi.shape[:-2], pad, xi.shape[-1])
    bot = xi[..., -1:, :].expand(*xi.shape[:-2], pad, xi.shape[-1])
    return torch.cat([top, xi, bot], dim=-2)


@dataclass
class PopGrid:
    """Static gx1v6 grid fields as torch tensors (CGS units)."""

    dz: torch.Tensor          # (Z,)            layer thickness            [cm]
    z_t: torch.Tensor         # (Z,)            level-center depth         [cm]
    tarea: torch.Tensor       # (J, I)          T-cell area                [cm^2]
    uarea: torch.Tensor       # (J, I)          U-cell area                [cm^2]
    htn: torch.Tensor         # (J, I)          north-edge length of T     [cm]
    hte: torch.Tensor         # (J, I)          east-edge length of T      [cm]
    kmt: torch.Tensor         # (J, I) int      deepest active level index
    mask3d: torch.Tensor      # (Z, J, I) bool  ocean cell mask
    mask2d: torch.Tensor      # (J, I) bool     surface ocean mask
    fcor: torch.Tensor        # (J, I)          Coriolis parameter         [1/s]
    tlat: torch.Tensor        # (J, I)          T latitude                 [deg]
    tlong: torch.Tensor       # (J, I)          T longitude                [deg]
    region_mask: torch.Tensor # (J, I) int      basin index
    consts: PopConstants

    @property
    def shape(self):
        return self.mask3d.shape  # (Z, J, I)

    @property
    def cell_volume(self) -> torch.Tensor:
        """Ocean cell volume (Z, J, I) = dz * TAREA, zero on land. [cm^3]"""
        vol = self.dz[:, None, None] * self.tarea[None, :, :]
        return vol * self.mask3d

    def to(self, device) -> "PopGrid":
        for f in self.__dataclass_fields__:
            v = getattr(self, f)
            if torch.is_tensor(v):
                setattr(self, f, v.to(device))
        return self

    # ----------------------------------------------------------------- #
    # B-grid finite-volume horizontal divergence of a (u, v) field.
    # u, v are at U points; the result is at T-cell centers.
    # ----------------------------------------------------------------- #
    def divergence(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Discrete horizontal divergence ``∇_h·(u, v)`` at T points.

        ``u, v`` have shape ``(..., Z, J, I)`` (or ``(..., J, I)``) at U points.
        Returns the divergence at T points in CGS [1/s].
        """
        htn = self.htn
        hte = self.hte
        tarea = self.tarea
        # Broadcast 2-D metrics over any leading/vertical axes.
        while htn.dim() < u.dim():
            htn = htn.unsqueeze(0)
            hte = hte.unsqueeze(0)
            tarea = tarea.unsqueeze(0)

        # Face-normal velocities (average of the two U points on each face).
        # East face of T(j,i): U(j,i) [NE] and U(j-1,i) [SE].
        u_e = 0.5 * (u + shift_j(u, +1))
        u_w = shift_i(u_e, +1)                       # west face = east face of i-1
        # North face of T(j,i): U(j,i) [NE] and U(j,i-1) [NW].
        v_n = 0.5 * (v + shift_i(v, +1))
        v_s = shift_j(v_n, +1)                        # south face = north of j-1

        flux_e = u_e * hte
        flux_w = u_w * shift_i(hte, +1)
        flux_n = v_n * htn
        flux_s = v_s * shift_j(htn, +1)

        div = (flux_e - flux_w + flux_n - flux_s) / tarea
        return div

    # ----------------------------------------------------------------- #
    # Vertically integrated (barotropic) divergence: Σ_k dz_k (∇·u)_k.
    # ----------------------------------------------------------------- #
    def barotropic_divergence(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """``Σ_k dz_k (∇_h·u)_k`` at T points, shape ``(..., J, I)`` [cm/s]."""
        div = self.divergence(u, v)                  # (..., Z, J, I)
        dz = self.dz.view(*([1] * (div.dim() - 3)), -1, 1, 1)
        return (div * dz * self.mask3d).sum(dim=-3)


# --------------------------------------------------------------------------- #
# Construction from an xarray dataset (used by build_grid.py and the loader).
# --------------------------------------------------------------------------- #
def _coriolis(tlat_deg: np.ndarray, omega: float) -> np.ndarray:
    return 2.0 * omega * np.sin(np.deg2rad(tlat_deg))


def grid_from_xarray(ds, device="cpu", dtype=torch.float32) -> PopGrid:
    """Build a :class:`PopGrid` from an open POP/grid xarray dataset.

    Accepts either a raw history file (which carries all the static fields) or
    the trimmed ``grid_gx1v6.nc`` produced by ``scripts/build_grid.py``.
    """
    consts = PopConstants.from_dataset(ds)

    def arr(name):
        return np.asarray(ds[name].values)

    dz = arr("dz").astype("float64")             # cm
    z_t = arr("z_t").astype("float64")           # cm
    tarea = arr("TAREA").astype("float64")
    uarea = arr("UAREA").astype("float64")
    htn = arr("HTN").astype("float64")
    hte = arr("HTE").astype("float64")
    kmt = arr("KMT").astype("int64")
    tlat = arr("TLAT").astype("float64")
    tlong = arr("TLONG").astype("float64")
    region = arr("REGION_MASK").astype("int64")

    nz = dz.shape[0]
    J, I = kmt.shape
    # 3-D ocean mask: level k is ocean where k < KMT (KMT counts active cells).
    kk = np.arange(nz)[:, None, None]
    mask3d = (kk < kmt[None, :, :]).astype("float32")
    mask2d = (kmt > 0).astype("float32")

    # Replace fill values in metrics with finite numbers (masked out anyway).
    for a in (tarea, uarea, htn, hte):
        a[~np.isfinite(a)] = 1.0
        a[a > 1e30] = 1.0
    tlat = np.nan_to_num(tlat, nan=0.0)
    tlong = np.nan_to_num(tlong, nan=0.0)

    fcor = _coriolis(tlat, consts.omega).astype("float32")

    def t(x, dt=dtype):
        return torch.as_tensor(x, dtype=dt, device=device)

    return PopGrid(
        dz=t(dz), z_t=t(z_t), tarea=t(tarea), uarea=t(uarea),
        htn=t(htn), hte=t(hte),
        kmt=t(kmt, torch.int64), mask3d=t(mask3d), mask2d=t(mask2d),
        fcor=t(fcor), tlat=t(tlat), tlong=t(tlong),
        region_mask=t(region, torch.int64), consts=consts,
    )


def load_grid(path: str, device="cpu", dtype=torch.float32) -> PopGrid:
    """Load a :class:`PopGrid` from ``grid_gx1v6.nc`` (or any POP file)."""
    import xarray as xr

    with xr.open_dataset(path, decode_times=False) as ds:
        return grid_from_xarray(ds, device=device, dtype=dtype)
