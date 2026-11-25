"""CESM-LE POP members as a CREDIT Gen 2 dataset.

CREDIT Gen 2 samples by timestamp on a clock with a fixed ``pd.Timedelta``
step and has no ensemble-member dimension.  Neither fits this data directly:
calendar months are not a fixed timedelta, and all members share the same
model dates.  The dataset therefore exposes a **record clock**:

    tick(member_index, record) = RECORD_EPOCH + (member_index * MEMBER_BLOCK + record) * dt

where ``dt`` is ``data.timestep`` (use ``"1h"``; one tick is one record, i.e.
one model month or one model day, *not* one hour of model time).  Members
occupy disjoint blocks, and valid initial ticks are trimmed per member so a
history window or rollout never crosses a member boundary.  The physical time
of year is computed from the record index and delivered to the model as two
extra forcing channels (``TIME_SIN``, ``TIME_COS``).

Source config (``data.source.<name>``)::

    dataset_type: PopLENSGen2Dataset
    root: <monthly netCDF root>          # fallback when a Zarr store is absent
    zarr_root: <rechunked Zarr root>     # optional, preferred
    experiment: B20TRC5CNBDRD
    members: ["002", "003", ...]
    t_start: 360                         # first usable record (per member)
    t_end: -1
    records_per_year: 12                 # 12 monthly, 365 daily (noleap)
    read_ahead: 64                       # records buffered per variable (see below)
    variables:
      prognostic:      {vars_3D: [TEMP, SALT, UVEL, VVEL], vars_2D: [SSH]}
      dynamic_forcing: {vars_2D: [SHF, SHF_QSW, SFWF, TAUX, TAUY, TIME_SIN, TIME_COS]}

The Gen 2 sampler requests one record per rollout step, whereas the Zarr
stores are chunked for windowed reads (300 records per chunk), so a single
record costs a full chunk decompression.  Each worker therefore keeps a
read-ahead buffer per variable: a miss reads ``read_ahead`` consecutive records
in one call and later steps of the same rollout are served from memory
(about 118 MB per buffered record for the full state).

Tensors are returned in physical units (POP CGS) with shapes
``(n_levels, n_time, J, I)`` for 3-D and ``(1, n_time, J, I)`` for 2-D fields.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from credit.datasets.gen_2.base_dataset import BaseDataset

from ..data import _n_time, _open_netcdf, _open_zarr, _read_slice
from .common import TIME_VARS

RECORD_EPOCH = pd.Timestamp("2000-01-01")
MEMBER_BLOCK = 40_000  # ticks reserved per member (> 31,389 daily records)


def record_tick(member_index: int, record: int, dt: pd.Timedelta) -> pd.Timestamp:
    return RECORD_EPOCH + (member_index * MEMBER_BLOCK + record) * dt


def time_of_year_angle(record: np.ndarray, records_per_year: int) -> np.ndarray:
    """Phase of the annual cycle for record indices that start on 1 January."""
    return 2.0 * np.pi * (np.asarray(record) % records_per_year) / records_per_year


class PopLENSGen2Dataset(BaseDataset):
    def __init__(self, data_config: Dict[str, Any], return_target: bool = False) -> None:
        self.dataset_type = "PopLENSGen2Dataset"
        self._cache: Dict[Tuple[str, str], Any] = {}
        # var -> (member, lo, hi, array): read-ahead buffer, one window per variable.
        self._buffer: Dict[str, Tuple[str, int, int, np.ndarray]] = {}
        super().__init__(data_config, return_target)
        self.init_register_all_fields()

    # ------------------------------------------------------------------ #
    # Clock
    # ------------------------------------------------------------------ #
    def _configure(self) -> Dict[str, Any]:
        """Read the store settings shared by all POP datasets; returns the source config."""
        src = self.curr_source_cfg
        self.root: str = src["root"]
        self.zarr_root = src.get("zarr_root")
        self.experiment: str = src["experiment"]
        self.members: List[str] = [str(m) for m in src["members"]]
        self.records_per_year = int(src.get("records_per_year", 12))
        self.read_ahead = max(1, int(src.get("read_ahead", 64)))
        self.n_records: Dict[str, int] = {}
        return src

    def _build_timestamps(self) -> pd.DatetimeIndex:
        """Valid initial ticks: per member, records ``[lo, hi - forecast_len)``.

        ``lo`` leaves room for the history window; the upper bound leaves room
        for the target of the last rollout step (same rule as
        ``PopLENSDataset``).  Called from ``BaseDataset.__init__``.
        """
        src = self._configure()
        t_start = int(src.get("t_start", 0))
        t_end = src.get("t_end", -1)

        first_var = (src["variables"]["prognostic"].get("vars_3D")
                     or src["variables"]["prognostic"]["vars_2D"])[0]
        ticks: List[np.ndarray] = []
        for mi, member in enumerate(self.members):
            nt = _n_time(self._handle(member, first_var))
            if nt > MEMBER_BLOCK:
                raise ValueError(f"member {member} has {nt} records > MEMBER_BLOCK={MEMBER_BLOCK}")
            self.n_records[member] = nt
            hi = nt if t_end in (-1, None) else min(int(t_end), nt)
            lo = max(self.history_len - 1, t_start)
            recs = np.arange(lo, hi - self.num_forecast_steps)
            ticks.append(mi * MEMBER_BLOCK + recs)
        self._cache.clear()  # handles are reopened lazily in each worker
        offsets = np.concatenate(ticks) if ticks else np.empty(0, dtype=int)
        return pd.DatetimeIndex(RECORD_EPOCH + offsets * self.dt)

    def _locate(self, t) -> Tuple[str, int]:
        """Tick -> (member, record)."""
        n = int(round((pd.Timestamp(t) - RECORD_EPOCH) / self.dt))
        mi, rec = divmod(n, MEMBER_BLOCK)
        return self.members[mi], rec

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    def _handle(self, member: str, var: str):
        key = (member, var)
        if key not in self._cache:
            handle = None
            if self.zarr_root is not None:
                try:
                    handle = _open_zarr(self.zarr_root, self.experiment, member, var)
                except FileNotFoundError:
                    handle = None  # fall back to netCDF
            if handle is None:
                handle = _open_netcdf(self.root, self.experiment, member, var)
            self._cache[key] = handle
        return self._cache[key]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}   # open file handles are per-process
        state["_buffer"] = {}
        return state

    def _records(self, member: str, var: str, lo: int, hi: int) -> np.ndarray:
        """Records ``[lo, hi)`` of ``var``, served from the read-ahead buffer."""
        buf = self._buffer.get(var)
        if buf is None or buf[0] != member or lo < buf[1] or hi > buf[2]:
            b_hi = min(max(hi, lo + self.read_ahead), self.n_records[member])
            buf = (member, lo, b_hi, _read_slice(self._handle(member, var), lo, b_hi))
            self._buffer[var] = buf
        return buf[3][lo - buf[1]:hi - buf[1]].copy()

    def _get_file_source(self, field_config: Dict[str, Any]) -> bool:
        return True  # files are resolved per (member, variable) in _handle

    def _read(self, field_type: str, member: str, lo: int, hi: int, sample: Dict[str, Any]) -> None:
        """Records ``[lo, hi)`` of every variable of ``field_type``."""
        if field_type not in self.var_dict:
            return
        names = self.var_dict[field_type]
        for v in names["vars_3D"]:
            arr = self._records(member, v, lo, hi)                       # (T, Z, J, I)
            sample[self._get_field_name(field_type, "3d", v)] = (
                torch.from_numpy(arr).permute(1, 0, 2, 3).contiguous())  # (Z, T, J, I)
        shape2d = None
        for v in names["vars_2D"]:
            if v in TIME_VARS:
                continue
            arr = self._records(member, v, lo, hi)                       # (T, J, I)
            shape2d = arr.shape[1:]
            sample[self._get_field_name(field_type, "2d", v)] = torch.from_numpy(arr).unsqueeze(0)
        if any(v in TIME_VARS for v in names["vars_2D"]):
            if shape2d is None:
                raise ValueError("TIME_SIN/TIME_COS must accompany at least one 2-D forcing field")
            # Spatially uniform channels; the model reduces them to a scalar pair.
            ang = time_of_year_angle(np.arange(lo, hi), self.records_per_year)
            for v, fn in zip(TIME_VARS, (np.sin, np.cos)):
                if v in names["vars_2D"]:
                    val = torch.as_tensor(fn(ang), dtype=torch.float32).view(1, -1, 1, 1)
                    sample[self._get_field_name(field_type, "2d", v)] = val.expand(1, -1, *shape2d).clone()

    def _extract_field(self, field_type, t, sample: Dict[str, Any]) -> None:
        member, rec = self._locate(t)
        self._read(field_type, member, rec, rec + 1, sample)

    def _extract_field_window(self, field_type, t_history, sample: Dict[str, Any]) -> None:
        member, lo = self._locate(t_history[0])
        member_hi, hi = self._locate(t_history[-1])
        if member != member_hi or hi - lo + 1 != len(t_history):
            raise ValueError("history window is not contiguous within one member")
        self._read(field_type, member, lo, hi + 1, sample)
