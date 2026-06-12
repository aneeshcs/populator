"""POP physical constants in the model's native CGS units.

These values are taken directly from the CESM-LENS POP history files (scalar
variables stored in every TEMP/SALT/... file), so densities, heat contents and
salt budgets computed here are consistent with POP's own diagnostics.

Units
-----
length    cm
mass      g
time      s
energy    erg  (= g cm^2 / s^2)
temp      degC (potential temperature); T0_Kelvin converts to K
density   g / cm^3
"""
from __future__ import annotations

from dataclasses import dataclass

# --- scalar constants as stored in the POP history files (CGS) -------------
CP_SW = 3.996e7            # specific heat of seawater         [erg / g / K]
RHO_SW = 1.026             # reference seawater density        [g / cm^3]
RHO_FW = 1.0               # freshwater density                [g / cm^3]
GRAV = 980.616             # gravitational acceleration        [cm / s^2]
OMEGA = 7.292123517e-05    # Earth rotation rate               [rad / s]
RADIUS = 6.37122e8         # Earth radius                      [cm]
T0_KELVIN = 273.15         # 0 degC in Kelvin                  [K]
OCN_REF_SALINITY = 34.7    # reference salinity                [psu]
HFLUX_FACTOR = 2.43908625974903e-05   # converts W/m^2 -> degC*cm/s heat flux
FWFLUX_FACTOR = 1.0e-4                 # converts kg/m^2/s -> cm/s freshwater
SALINITY_FACTOR = -0.00347            # virtual salt-flux factor
SFLUX_FACTOR = 0.1                    # converts to kg/m^2/s/(psu)

# Seconds in one POP (noleap) model month, used as the integration step Δt.
DAYS_PER_YEAR = 365.0
SECONDS_PER_DAY = 86400.0
SECONDS_PER_MONTH = DAYS_PER_YEAR / 12.0 * SECONDS_PER_DAY  # ~2.628e6 s

# Unit-conversion helpers for human-readable diagnostics.
ERG_PER_SECOND_TO_PW = 1.0e-7 * 1.0e-15   # erg/s -> W -> PW (1e15 W)
CM3_TO_M3 = 1.0e-6
CM2_TO_M2 = 1.0e-4


@dataclass(frozen=True)
class PopConstants:
    """Container so a run can, if desired, override the built-in defaults with
    values read straight from a specific history file."""

    cp_sw: float = CP_SW
    rho_sw: float = RHO_SW
    rho_fw: float = RHO_FW
    grav: float = GRAV
    omega: float = OMEGA
    radius: float = RADIUS
    t0_kelvin: float = T0_KELVIN
    ocn_ref_salinity: float = OCN_REF_SALINITY
    hflux_factor: float = HFLUX_FACTOR
    fwflux_factor: float = FWFLUX_FACTOR
    salinity_factor: float = SALINITY_FACTOR
    sflux_factor: float = SFLUX_FACTOR

    @classmethod
    def from_dataset(cls, ds) -> "PopConstants":
        """Read the scalar constants from an open xarray POP dataset, falling
        back to the module defaults for any that are absent."""
        def g(name, default):
            return float(ds[name].values) if name in ds else default

        return cls(
            cp_sw=g("cp_sw", CP_SW),
            rho_sw=g("rho_sw", RHO_SW),
            rho_fw=g("rho_fw", RHO_FW),
            grav=g("grav", GRAV),
            omega=g("omega", OMEGA),
            radius=g("radius", RADIUS),
            t0_kelvin=g("T0_Kelvin", T0_KELVIN),
            ocn_ref_salinity=g("ocn_ref_salinity", OCN_REF_SALINITY),
            hflux_factor=g("hflux_factor", HFLUX_FACTOR),
            fwflux_factor=g("fwflux_factor", FWFLUX_FACTOR),
            salinity_factor=g("salinity_factor", SALINITY_FACTOR),
            sflux_factor=g("sflux_factor", SFLUX_FACTOR),
        )
