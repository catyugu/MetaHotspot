"""High-level wrapper for ``mhs_solution_t`` — simulation results."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from metahotspot._handle import OwnedHandle
import metahotspot._native as _native


class ProbeTrace(NamedTuple):
    """A single probe trace — name plus time series."""

    name: str
    times: np.ndarray
    values: np.ndarray


class Solution(OwnedHandle):
    """Read-only simulation result.

    Do not instantiate directly — use ``Compiled.solve()``.  ``state_history``
    contains every C++ output snapshot, including the initial transient state
    and final accepted state.
    """

    def __init__(self) -> None:
        self._time = 0.0
        self._state = np.empty(0, dtype=np.float64)
        self._temperature = np.empty(0, dtype=np.float64)
        self._history_times = np.empty(0, dtype=np.float64)
        self._state_history = np.empty((0, 0), dtype=np.float64)
        self._temperature_history = np.empty((0, 0), dtype=np.float64)
        self._fvm_count = 0
        self._probes: list[ProbeTrace] = []
        super().__init__(None, None)

    @classmethod
    def _solve_compiled(
        cls,
        compiled,
        state: np.ndarray | None = None,
        opts=None,
    ) -> Solution:
        self = cls()
        self._dll = compiled._dll
        self._destroy_fn = self._dll.mhs_solution_destroy
        overrides = opts._overrides() if opts is not None else {}
        self._handle = _native.solve(self._dll, compiled._handle, state, overrides)
        self._load_data()
        return self

    @classmethod
    def _from_handle(
        cls,
        dll,
        destroy_fn,
        handle,
    ) -> Solution:
        self = cls()
        self._dll = dll
        self._destroy_fn = destroy_fn
        self._handle = handle
        self._load_data()
        return self

    def _load_data(self) -> None:
        """Snapshot native results into independent Python-owned arrays."""
        data = _native.solution_snapshot(self._dll, self._handle)
        self._time = data["time"]
        self._state = data["state"]
        self._history_times = data["history_times"]
        self._state_history = data["state_history"]
        self._fvm_count = data["fvm_count"]
        self._temperature = self._state[: self._fvm_count]
        self._temperature_history = self._state_history[:, : self._fvm_count]
        self._probes = [
            ProbeTrace(name, times, values) for name, times, values in data["probes"]
        ]

    @property
    def temperature(self) -> np.ndarray:
        """Final FVM temperature field [fvm_count] (view of ``state``)."""
        return self._temperature

    @property
    def time(self) -> float:
        """Final simulation time."""
        return self._time

    @property
    def state(self) -> np.ndarray:
        """Final full state, including retained external modes."""
        return self._state

    @property
    def history_times(self) -> np.ndarray:
        """C++ output times [record_count]."""
        return self._history_times

    @property
    def state_history(self) -> np.ndarray:
        """C++ output states [record_count, state_count], row-major."""
        return self._state_history

    @property
    def temperature_history(self) -> np.ndarray:
        """FVM temperature snapshots [record_count, fvm_count] (view of ``state_history``)."""
        return self._temperature_history

    @property
    def probes(self) -> list[ProbeTrace]:
        """Return all probe traces as high-level named tuples."""
        return self._probes

    def write_vtu(self, path: str) -> None:
        """Export the final FVM temperature field to a VTU file."""
        _native.write_vtu(self._dll, self._handle, str(path))
