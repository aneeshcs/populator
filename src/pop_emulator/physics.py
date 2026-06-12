"""Physics operators and conservation diagnostics for the POP ocean emulator.

Everything here operates on *physical-unit* fields (the data pipeline
denormalizes before calling these). Tensors are ``(B, Z, J, I)`` for 3-D
variables and ``(B, J, I)`` for 2-D, with ``(J, I) = (nlat, nlon)``.

Contents
--------
mwjf_density            POP's MWJF (2003) equation of state, ρ(T, S, p).
pressure_dbar           POP hydrostatic depth->pressure relation.
diagnose_wvel           vertical velocity from the continuity equation.
heat_content / salt_content   global ocean budgets (SI, Joules / kg·psu).
conservation_losses     ties predicted budget change to surface forcing.
static_stability_penalty   discourages spurious gravitational instability.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from .grid import PopGrid


# --------------------------------------------------------------------------- #
# Equation of state: MWJF (McDougall, Jackett, Wright & Feistel, 2003).
# 25-term rational polynomial ρ = P1(θ,S,p) / P2(θ,S,p), the same EOS POP
# integrates. T in degC (potential temperature), S in psu, pressure in **dbar**;
# returns in-situ density in kg/m^3. Coefficient set from McDougall et al.
# (2003). Surface density matches EOS-80 to <0.01 kg/m^3 (e.g. ρ(35,25,0) =
# 1023.339, ρ(35,20,0) = 1024.763) and the depth compressibility is physically
# reasonable (~0.004-0.0048 kg/m^3/dbar). A small (~0.25%) offset remains in the
# abyssal in-situ density relative to some published check values; because the
# static-stability penalty compares potential density of adjacent cells at a
# common reference pressure, this nearly-uniform offset cancels in the vertical
# density *difference* and does not affect the physics losses.
# --------------------------------------------------------------------------- #
# numerator P1
_A0 = 9.9984085444849347e02
_A1 = 7.3471625860981584e00
_A2 = -5.3211231792841769e-02
_A3 = 3.6492439109814549e-04
_A4 = 2.5880571023991390e00
_A5 = -6.7168282786692355e-03
_A6 = 1.9203202055760151e-03
_A7 = 1.1798263740430364e-02
_A8 = 9.8920219266399117e-08
_A9 = 4.6996642771754730e-06
_A10 = -2.5862187075154352e-08
_A11 = -3.2921414007960662e-12
# denominator P2
_B0 = 1.0
_B1 = 7.2815210113327091e-03
_B2 = -4.4787265461983921e-05
_B3 = 3.3851002965802430e-07
_B4 = 1.3651202389758572e-10
_B5 = 1.7632126669040377e-03
_B6 = -8.8066583251206474e-06
_B7 = -1.8832689434804897e-10
_B8 = 5.7463776745432097e-06
_B9 = 1.4716275472242334e-09
_B10 = 6.7103246285651894e-06
_B11 = -2.4461698007024582e-17
_B12 = -9.1534417604289062e-18


def pressure_dbar(depth_cm: torch.Tensor) -> torch.Tensor:
    """POP hydrostatic pressure from depth. ``depth_cm`` -> pressure in dbar.

    POP's ``pressure()`` returns ``p(d) = 0.059808 (e^{-0.025 d} - 1)
    + 0.100766 d + 2.28405e-7 d^2`` in **bars** with depth ``d`` in metres;
    multiplied by 10 here to give dbar (the units the MWJF coefficients expect).
    """
    d = depth_cm / 100.0  # cm -> m
    bars = 0.059808 * (torch.exp(-0.025 * d) - 1.0) + 0.100766 * d + 2.28405e-7 * d * d
    return 10.0 * bars


def mwjf_density(temp: torch.Tensor, salt: torch.Tensor,
                 pressure: torch.Tensor) -> torch.Tensor:
    """In-situ density [kg/m^3] from potential temperature, salinity, pressure.

    ``temp`` [degC], ``salt`` [psu], ``pressure`` [dbar] are broadcastable
    tensors. This is the same polynomial POP integrates, so densities derived
    from the emulator's (T, S) are consistent with the model's.
    """
    t = temp
    # Floor salinity at a small positive value so the sqrt below has a finite
    # gradient: land/masked cells carry S=0, where d(sqrt S)/dS -> inf would
    # otherwise produce NaN gradients (the forward value is masked out anyway).
    s = salt.clamp_min(1.0e-2)
    p = pressure
    t2 = t * t
    s15 = s * torch.sqrt(s)            # S^{3/2}

    # numerator P1
    num = (_A0 + t * (_A1 + t * (_A2 + _A3 * t))
           + s * (_A4 + _A5 * t + _A6 * s)
           + p * (_A7 + _A8 * t2 + _A9 * s)
           + p * p * (_A10 + _A11 * t2))

    # denominator P2
    den = (_B0 + t * (_B1 + t * (_B2 + t * (_B3 + _B4 * t)))
           + s * (_B5 + _B6 * t + _B7 * t * t2)
           + s15 * (_B8 + _B9 * t2)
           + p * (_B10 + p * p * (_B11 * t * t2) + _B12 * t))

    return num / den


def density_field(temp: torch.Tensor, salt: torch.Tensor, grid: PopGrid,
                  ref_pressure: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Density [kg/m^3] for a (B, Z, J, I) (T, S) field on the grid.

    By default uses in-situ pressure at each level's depth. Pass
    ``ref_pressure`` (a (Z,) or scalar tensor in dbar) to obtain potential
    density referenced to that pressure.
    """
    if ref_pressure is None:
        p = pressure_dbar(grid.z_t).view(1, -1, 1, 1)
    elif ref_pressure.dim() == 0:
        p = ref_pressure.view(1, 1, 1, 1)
    else:
        p = ref_pressure.view(1, -1, 1, 1)
    return mwjf_density(temp, salt, p)


# --------------------------------------------------------------------------- #
# Continuity: diagnose vertical velocity from horizontal divergence.
# --------------------------------------------------------------------------- #
def diagnose_wvel(u: torch.Tensor, v: torch.Tensor, grid: PopGrid) -> torch.Tensor:
    """Vertical velocity [cm/s] at layer top faces from continuity.

    Integrating ``∂w/∂z = -∇_h·u`` upward from a rigid bottom (``w=0`` at the
    floor) gives ``w(k) = w(k+1) + dz(k) (∇_h·u)_k``. Because ``w`` is built
    from the predicted ``(u, v)``, the discrete incompressibility constraint
    holds exactly. Returns ``w`` at the top of each cell, shape ``(B, Z, J, I)``.
    """
    div = grid.divergence(u, v) * grid.mask3d            # (B, Z, J, I) [1/s]
    dz = grid.dz.view(1, -1, 1, 1)
    # cumulative integral from the bottom (k = Z-1) upward
    flux = div * dz                                       # [cm/s] per layer
    w_bottom_up = torch.flip(torch.cumsum(torch.flip(flux, dims=[1]), dim=1), dims=[1])
    return w_bottom_up * grid.mask3d


def surface_continuity_residual(u: torch.Tensor, v: torch.Tensor,
                                grid: PopGrid) -> torch.Tensor:
    """Diagnosed vertical velocity at the surface (k=0); under a rigid lid this
    should vanish. Returns the masked field (B, J, I) [cm/s]."""
    w = diagnose_wvel(u, v, grid)
    return w[:, 0] * grid.mask2d


# --------------------------------------------------------------------------- #
# Global conservation budgets (computed in SI: Joules, kg, psu).
# --------------------------------------------------------------------------- #
def _volume_si(grid: PopGrid) -> torch.Tensor:
    """Ocean cell volume (Z, J, I) in m^3 (zero on land)."""
    return grid.cell_volume * 1.0e-6  # cm^3 -> m^3


def heat_content(temp: torch.Tensor, grid: PopGrid) -> torch.Tensor:
    """Global ocean heat content [J] per batch element, ``Σ ρ0 c_p T dV``."""
    rho0 = grid.consts.rho_sw * 1.0e3          # g/cm^3 -> kg/m^3
    cp = grid.consts.cp_sw * 1.0e-4            # erg/g/K -> J/kg/K
    dV = _volume_si(grid).unsqueeze(0)         # (1, Z, J, I)
    return (rho0 * cp * temp.double() * dV.double()).flatten(1).sum(dim=1)


def salt_content(salt: torch.Tensor, grid: PopGrid) -> torch.Tensor:
    """Global ocean salt content [kg·psu equivalent] per batch element."""
    rho0 = grid.consts.rho_sw * 1.0e3
    dV = _volume_si(grid).unsqueeze(0)
    return (rho0 * salt.double() * dV.double()).flatten(1).sum(dim=1)


def _surface_area_m2(grid: PopGrid) -> torch.Tensor:
    return grid.tarea * 1.0e-4 * grid.mask2d   # cm^2 -> m^2


def conservation_losses(pred: Dict[str, torch.Tensor],
                        prev: Dict[str, torch.Tensor],
                        grid: PopGrid,
                        forcing: Optional[Dict[str, torch.Tensor]] = None,
                        dt: float = None) -> Dict[str, torch.Tensor]:
    """Budget-consistency penalties tying state change to the surface forcing.

    The predicted change in global heat / salt content over one step should
    equal the surface flux integrated over that step; the vertically integrated
    divergence should vanish (rigid-lid volume conservation).

    Parameters
    ----------
    pred, prev : dict with keys TEMP, SALT, UVEL, VVEL (physical units)
    forcing    : dict with SHF [W/m^2] and SFWF (freshwater) if available
    dt         : step length in seconds (defaults to one POP month)

    Returns a dict of scalar losses plus signed budget diagnostics.
    """
    from .constants import SECONDS_PER_MONTH

    dt = SECONDS_PER_MONTH if dt is None else dt
    out: Dict[str, torch.Tensor] = {}

    # --- heat-content budget --------------------------------------------- #
    dH = heat_content(pred["TEMP"], grid) - heat_content(prev["TEMP"], grid)
    dH_flux = torch.zeros_like(dH)
    if forcing is not None and "SHF" in forcing:
        area = _surface_area_m2(grid).unsqueeze(0)
        dH_flux = dt * (forcing["SHF"].double() * area.double()).flatten(1).sum(dim=1)
    scaleH = dH_flux.abs().mean().clamp_min(1.0e15) + 1.0e15
    out["heat"] = (((dH - dH_flux) / scaleH) ** 2).mean()
    out["heat_drift_J"] = (dH - dH_flux).mean().detach()

    # --- salt-content budget --------------------------------------------- #
    dS = salt_content(pred["SALT"], grid) - salt_content(prev["SALT"], grid)
    dS_flux = torch.zeros_like(dS)
    if forcing is not None and "SFWF" in forcing:
        # SFWF is a virtual salt flux; tie change to surface freshwater forcing.
        area = _surface_area_m2(grid).unsqueeze(0)
        c = grid.consts
        sflux = forcing["SFWF"].double() * c.ocn_ref_salinity  # psu * kg/m2/s scale
        dS_flux = dt * (sflux * area.double()).flatten(1).sum(dim=1)
    scaleS = dS_flux.abs().mean().clamp_min(1.0e12) + 1.0e12
    out["salt"] = (((dS - dS_flux) / scaleS) ** 2).mean()
    out["salt_drift"] = (dS - dS_flux).mean().detach()

    # --- barotropic / volume (rigid-lid) --------------------------------- #
    baro = grid.barotropic_divergence(pred["UVEL"], pred["VVEL"])  # (B, J, I)
    out["barotropic"] = (baro ** 2 * grid.mask2d).mean()

    # --- surface continuity (rigid lid) ---------------------------------- #
    wsurf = surface_continuity_residual(pred["UVEL"], pred["VVEL"], grid)
    out["continuity"] = (wsurf ** 2).mean()

    return out


# --------------------------------------------------------------------------- #
# Static stability: discourage spuriously unstable stratification.
# --------------------------------------------------------------------------- #
def static_instability(temp: torch.Tensor, salt: torch.Tensor,
                       grid: PopGrid) -> torch.Tensor:
    """Positive where the column is gravitationally unstable.

    For adjacent levels k (above) and k+1 (below) both referenced to the
    interface pressure, instability means the upper cell is denser:
    ``ρ_k - ρ_{k+1} > 0``. Returns ``relu(ρ_k - ρ_{k+1})`` [kg/m^3] on the
    Z-1 interfaces (B, Z-1, J, I), masked to ocean.
    """
    z_t = grid.z_t
    z_iface = 0.5 * (z_t[:-1] + z_t[1:])                 # (Z-1,) interface depth
    p_iface = pressure_dbar(z_iface).view(1, -1, 1, 1)

    t_up, t_dn = temp[:, :-1], temp[:, 1:]
    s_up, s_dn = salt[:, :-1], salt[:, 1:]
    rho_up = mwjf_density(t_up, s_up, p_iface)
    rho_dn = mwjf_density(t_dn, s_dn, p_iface)

    mask = (grid.mask3d[:-1] * grid.mask3d[1:]).unsqueeze(0)
    return torch.relu(rho_up - rho_dn) * mask


def static_stability_penalty(pred_temp, pred_salt, true_temp, true_salt,
                             grid: PopGrid) -> torch.Tensor:
    """Penalize instability the prediction creates *beyond* what the target has.

    POP convectively adjusts to remove instability, so we only discourage the
    emulator from inventing new instability rather than demanding more
    stability than POP itself maintains.
    """
    inst_pred = static_instability(pred_temp, pred_salt, grid)
    inst_true = static_instability(true_temp, true_salt, grid)
    return torch.relu(inst_pred - inst_true).mean()
