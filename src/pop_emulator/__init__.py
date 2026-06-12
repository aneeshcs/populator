"""Physics-informed emulator that integrates the POP/CESM-LENS ocean state forward.

Submodules
----------
constants     POP physical constants (CGS), as read from the dataset.
grid          gx1v6 grid geometry, masks, Coriolis, differential operators.
data          CESM-LENS dataset / dataloader.
normalization per-level standardization.
physics       MWJF equation of state, continuity, conservation losses.
losses        composite data + physics loss.
model         physics-informed spherical U-Net (tendency form).
rollout       autoregressive multi-step integration.
"""

__version__ = "0.1.0"
