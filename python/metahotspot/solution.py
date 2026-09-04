"""High-level wrapper for ``mhs_solution_t`` — simulation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from metahotspot._handle import OwnedHandle
import metahotspot._native_solution as _native_solution


class ProbeTrace(NamedTuple):
    """A single probe trace — name plus time series."""

    name: str
    times: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class _SolutionData:
    time: float
    fvm_count: int
    state: np.ndarray
    history_times: np.ndarray
    state_history: np.ndarray
    probes: list[ProbeTrace]


class Solution(OwnedHandle):
    """Read-only simulation result.

    Do not instantiate directly — use ``Compiled.solve()``.  ``state_history``
    contains every C++ output snapshot, including the initial transient state
    and final accepted state.
    """

    def __init__(self) -> None:
        self._data: _SolutionData | None = None
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
        self._handle = _native_solution.solve(
            self._dll, compiled._handle, state, overrides
        )
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
        data = _native_solution.solution_snapshot(self._dll, self._handle)
        self._data = _SolutionData(
            time=data.time,
            fvm_count=data.fvm_count,
            state=data.state,
            history_times=data.history_times,
            state_history=data.state_history,
            probes=[
                ProbeTrace(probe.name, probe.times, probe.values)
                for probe in data.probes
            ],
        )

    @property
    def temperature(self) -> np.ndarray:
        """Final FVM temperature field [fvm_count] (view of ``state``)."""
        return self.state[: self.fvm_count]

    @property
    def time(self) -> float:
        """Final simulation time."""
        return self._data.time

    @property
    def state(self) -> np.ndarray:
        """Final full state, including retained external modes."""
        return self._data.state

    @property
    def history_times(self) -> np.ndarray:
        """C++ output times [record_count]."""
        return self._data.history_times

    @property
    def state_history(self) -> np.ndarray:
        """C++ output states [record_count, state_count], row-major."""
        return self._data.state_history

    @property
    def temperature_history(self) -> np.ndarray:
        """FVM temperature snapshots [record_count, fvm_count] (view of ``state_history``)."""
        return self.state_history[:, : self.fvm_count]

    @property
    def probes(self) -> list[ProbeTrace]:
        """Return all probe traces as high-level named tuples."""
        return self._data.probes

    @property
    def fvm_count(self) -> int:
        """Number of FVM temperatures at the front of the full state."""
        return self._data.fvm_count

    def write_vtu(self, path: str) -> None:
        """Export the final FVM temperature field to a VTU file."""
        _native_solution.write_vtu(self._dll, self._handle, str(path))
