"""Daily, surface-supervised samples for fine-tuning the emulator to a 1-day step.

CESM-LE provides daily means of SST, SSH and wind stress only; the 3-D state
and the heat/freshwater fluxes exist as monthly means.  A daily sample is
therefore assembled from both streams (record clock as in ``dataset.py``, one
tick = one day, day 0 = 1 January of the daily stores' first year, noleap):

Initial condition (``i == 0``; newest input day ``d0`` is the last day of a month)
    3-D TEMP/SALT/UVEL/VVEL: monthly means interpolated linearly in time to the
    history days (month centres as nodes), with TEMP level 0 replaced by the
    daily SST; SSH: daily SSH.
Forcing at input day ``d`` (every step)
    TAUX, TAUY: daily means.  SHF, SHF_QSW, SFWF: the monthly mean of the month
    containing ``d`` (piecewise constant, so monthly heat and salt budgets are
    exact).  TIME_SIN/TIME_COS: annual phase of day ``d``, shifted by half a
    month so it matches the phase the monthly model saw for a mid-month state.
Target for day ``d + 1``
    TEMP level 0 and SSH: daily SST and SSH.  All other levels and variables:
    the **monthly mean of the month containing the target day**.  These are not
    daily truth; ``PopDailyTrainer`` compares them with the prediction only
    through calendar-month means of the rollout.

Extra source keys (in addition to those of ``PopLENSGen2Dataset``)::

    dataset_type: PopLENSDailyDataset
    daily_zarr_root: <stores written by scripts/rechunk_daily_to_zarr.py>
    records_per_year: 365
    read_ahead: 16            # monthly records buffered (>= window months + 2)
    read_ahead_days: 400      # daily records buffered
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from credit.datasets.gen_2._utils import encode_time  # pyright: ignore[reportPrivateUsage]

from ..data import _n_time, _open_zarr, _read_slice
from .common import TIME_VARS
from .dataset import MEMBER_BLOCK, RECORD_EPOCH, PopLENSGen2Dataset

DAYS_PER_YEAR = 365
MONTH_LEN = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MONTH_START = np.concatenate([[0], np.cumsum(MONTH_LEN)[:-1]])   # day of year of each month's first day

DAILY_SURFACE = {"TEMP": "SST", "SSH": "SSH"}     # state variable -> daily store
DAILY_FORCING = ("TAUX", "TAUY")
# The monthly model saw phase 2*pi*m/12 for the mean of month m, i.e. for a state
# centred half a month later; shift the daily phase so mid-month days match.
PHASE_SHIFT_DAYS = DAYS_PER_YEAR / 24.0


def month_of_day(day: int) -> Tuple[int, int, int]:
    """Day index -> (global month index, day within month, month length)."""
    year, doy = divmod(int(day), DAYS_PER_YEAR)
    m = int(np.searchsorted(MONTH_START, doy, side="right") - 1)
    return year * 12 + m, doy - int(MONTH_START[m]), int(MONTH_LEN[m])


def month_first_day(month: int) -> int:
    """Global month index -> day index of its first day."""
    year, m = divmod(int(month), 12)
    return year * DAYS_PER_YEAR + int(MONTH_START[m])


def month_centre(month: int) -> float:
    """Time [days] of the centre of a month (its mean is assigned to this time)."""
    return month_first_day(month) + MONTH_LEN[int(month) % 12] / 2.0


def interpolation_months(day: int, n_months: int) -> Tuple[int, int, float]:
    """Bracketing months and weight for linear interpolation to the centre of ``day``:
    ``x(day) = (1 - w) * x[a] + w * x[b]``.  Clamped at the ends of the record."""
    tau = day + 0.5
    k, _, _ = month_of_day(day)
    a = k if tau >= month_centre(k) else k - 1
    a = min(max(a, 0), n_months - 2)
    w = (tau - month_centre(a)) / (month_centre(a + 1) - month_centre(a))
    return a, a + 1, float(min(max(w, 0.0), 1.0))


class PopLENSDailyDataset(PopLENSGen2Dataset):
    def __init__(self, data_config: Dict[str, Any], return_target: bool = False) -> None:
        self._daily_cache: Dict[Tuple[str, str], Any] = {}
        self._daily_buffer: Dict[str, Tuple[str, int, int, np.ndarray]] = {}
        super().__init__(data_config, return_target)
        self.dataset_type = "PopLENSDailyDataset"

    # ------------------------------------------------------------------ #
    # Clock: initial days are month ends
    # ------------------------------------------------------------------ #
    def _build_timestamps(self) -> pd.DatetimeIndex:
        src = self._configure()
        self.daily_zarr_root: str = src["daily_zarr_root"]
        self.read_ahead_days = max(1, int(src.get("read_ahead_days", 400)))
        if self.records_per_year != DAYS_PER_YEAR:
            raise ValueError("PopLENSDailyDataset requires records_per_year: 365")
        d_start = int(src.get("t_start", 0))
        d_end = src.get("t_end", -1)

        prog = src["variables"]["prognostic"]
        first_3d = prog["vars_3D"][0]
        ticks: List[np.ndarray] = []
        self.n_days: Dict[str, int] = {}
        self.month0: Dict[str, int] = {}     # monthly record of the daily stores' first month
        for mi, member in enumerate(self.members):
            nd = _n_time(self._daily_handle(member, "SST"))
            nm = _n_time(self._handle(member, first_3d))
            if nd % DAYS_PER_YEAR != 0 or nd > MEMBER_BLOCK:
                raise ValueError(f"member {member}: {nd} daily records is not a whole number of years")
            n_months = nd // DAYS_PER_YEAR * 12
            self.n_days[member], self.n_records[member] = nd, nm
            # Both streams end in the same month, so align them from the end.
            self.month0[member] = nm - n_months
            if self.month0[member] < 0:
                raise ValueError(f"member {member}: monthly record shorter than the daily record")
            hi = nd if d_end in (-1, None) else min(int(d_end), nd)
            d0 = np.array([month_first_day(m) - 1 for m in range(1, n_months)])
            ok = (d0 - (self.history_len - 1) >= d_start) & (d0 + self.num_forecast_steps <= hi - 1)
            ticks.append(mi * MEMBER_BLOCK + d0[ok])
        self._cache.clear()
        self._daily_cache.clear()
        offsets = np.concatenate(ticks) if ticks else np.empty(0, dtype=int)
        return pd.DatetimeIndex(RECORD_EPOCH + offsets * self.dt)

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    def _daily_handle(self, member: str, var: str):
        key = (member, var)
        if key not in self._daily_cache:
            self._daily_cache[key] = _open_zarr(self.daily_zarr_root, self.experiment, member, var)
        return self._daily_cache[key]

    def __getstate__(self):
        state = super().__getstate__()
        state["_daily_cache"] = {}
        state["_daily_buffer"] = {}
        return state

    def _daily(self, member: str, var: str, lo: int, hi: int) -> np.ndarray:
        """Daily records ``[lo, hi)`` of ``var`` (read-ahead buffered), ``(T, J, I)``."""
        buf = self._daily_buffer.get(var)
        if buf is None or buf[0] != member or lo < buf[1] or hi > buf[2]:
            b_hi = min(max(hi, lo + self.read_ahead_days), self.n_days[member])
            buf = (member, lo, b_hi, _read_slice(self._daily_handle(member, var), lo, b_hi))
            self._daily_buffer[var] = buf
        return buf[3][lo - buf[1]:hi - buf[1]]

    def _monthly(self, member: str, var: str, month: int) -> np.ndarray:
        """Monthly mean of global month ``month``: ``(Z, J, I)`` or ``(J, I)``."""
        rec = self.month0[member] + month
        return self._records(member, var, rec, rec + 1)[0]

    # ------------------------------------------------------------------ #
    # Sample assembly
    # ------------------------------------------------------------------ #
    def _forcing(self, member: str, days: Sequence[int]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        lo, hi = days[0], days[-1] + 1
        shape2d = None
        for v in self.var_dict["dynamic_forcing"]["vars_2D"]:
            if v in TIME_VARS:
                continue
            if v in DAILY_FORCING:
                arr = np.array(self._daily(member, v, lo, hi))
            else:   # monthly-mean flux, constant within the month
                arr = np.stack([self._monthly(member, v, month_of_day(d)[0]) for d in days])
            shape2d = arr.shape[1:]
            out[self._get_field_name("dynamic_forcing", "2d", v)] = torch.from_numpy(arr).unsqueeze(0)
        ang = 2.0 * np.pi * (np.asarray(days) + 0.5 - PHASE_SHIFT_DAYS) / DAYS_PER_YEAR
        for v, fn in zip(TIME_VARS, (np.sin, np.cos)):
            if v in self.var_dict["dynamic_forcing"]["vars_2D"]:
                val = torch.as_tensor(fn(ang), dtype=torch.float32).view(1, -1, 1, 1)
                out[self._get_field_name("dynamic_forcing", "2d", v)] = val.expand(1, -1, *shape2d).clone()
        return out

    def _state(self, member: str, days: Sequence[int], interpolate: bool) -> Dict[str, torch.Tensor]:
        """Prognostic fields for ``days``: 3-D from monthly means (interpolated in
        time for inputs, month-constant for targets), surface from daily means."""
        out: Dict[str, torch.Tensor] = {}
        names = self.var_dict["prognostic"]
        n_months = self.n_days[member] // DAYS_PER_YEAR * 12
        lo, hi = days[0], days[-1] + 1
        for v in names["vars_3D"]:
            steps = []
            for d in days:
                if interpolate:
                    a, b, w = interpolation_months(d, n_months)
                    x = (1.0 - w) * self._monthly(member, v, a) + w * self._monthly(member, v, b)
                else:
                    x = self._monthly(member, v, month_of_day(d)[0]).copy()
                steps.append(x.astype(np.float32, copy=False))
            arr = np.stack(steps, axis=1)                                   # (Z, T, J, I)
            if v in DAILY_SURFACE:
                arr[0] = self._daily(member, DAILY_SURFACE[v], lo, hi)     # level 0 <- daily SST
            out[self._get_field_name("prognostic", "3d", v)] = torch.from_numpy(arr)
        for v in names["vars_2D"]:
            arr = np.array(self._daily(member, DAILY_SURFACE[v], lo, hi))  # (T, J, I)
            out[self._get_field_name("prognostic", "2d", v)] = torch.from_numpy(arr).unsqueeze(0)
        return out

    def _load_sample(self, t, i: int) -> Dict[str, Any]:
        member, day = self._locate(t)
        if i == 0:
            days = list(range(day - self.history_len + 1, day + 1))
            inputs = self._forcing(member, days)
            inputs.update(self._state(member, days, interpolate=True))
        else:
            inputs = self._forcing(member, [day])
        sample: Dict[str, Any] = {
            "input": inputs,
            "metadata": {"input_datetime": encode_time(pd.Timestamp(t)), "day": day},
        }
        if self.return_target:
            sample["target"] = self._state(member, [day + 1], interpolate=False)
            sample["metadata"]["target_datetime"] = encode_time(pd.Timestamp(t) + self.dt)
        return sample
