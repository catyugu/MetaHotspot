"""High-level wrapper for ``mhs_solution_t`` — simulation results."""

from __future__ import annotations

import ctypes
from typing import NamedTuple

import numpy as np

from metahotspot._error import check
from metahotspot._dll_interface import _opts_ptr, copy_array
from metahotspot._handle import OwnedHandle
from metahotspot.types import MhsSolution, MhsSolutionInfo


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
        pp = ctypes.POINTER(MhsSolution)()

        opts_ptr = _opts_ptr(opts, self._dll)

        state_ptr = (
            state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if state is not None
            else None
        )
        state_count = state.size if state is not None else 0
        check(
            self._dll.mhs_compiled_solve(
                compiled._handle, state_ptr, state_count, opts_ptr, ctypes.byref(pp)
            ),
            "solve",
        )
        self._handle = pp
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
        info = MhsSolutionInfo()
        check(
            self._dll.mhs_solution_get_info(self._handle, ctypes.byref(info)),
            "solution_info",
        )
        self._time = float(info.time)
        self._state = np.empty(info.state_count, dtype=np.float64)
        self._history_times = np.empty(info.record_count, dtype=np.float64)
        history = np.empty(info.record_count * info.state_count, dtype=np.float64)

        copy_array(
            self._dll.mhs_solution_copy_state,
            self._handle,
            self._state,
            ctypes.c_double,
            "solution_copy_state",
        )
        copy_array(
            self._dll.mhs_solution_copy_history_times,
            self._handle,
            self._history_times,
            ctypes.c_double,
            "solution_copy_history_times",
        )
        copy_array(
            self._dll.mhs_solution_copy_history_states,
            self._handle,
            history,
            ctypes.c_double,
            "solution_copy_history_states",
        )
        self._fvm_count = int(info.fvm_count)
        self._state_history = history.reshape(
            (int(info.record_count), int(info.state_count))
        )
        self._temperature = self._state[: self._fvm_count]
        self._temperature_history = self._state_history[:, : self._fvm_count]

        self._probes = []
        for index in range(info.probe_count):
            name_size = ctypes.c_size_t()
            record_count = ctypes.c_size_t()
            check(
                self._dll.mhs_solution_probe_get_info(
                    self._handle,
                    index,
                    ctypes.byref(name_size),
                    ctypes.byref(record_count),
                ),
                "solution_probe_info",
            )
            name = ctypes.create_string_buffer(name_size.value)
            times = np.empty(record_count.value, dtype=np.float64)
            values = np.empty(record_count.value, dtype=np.float64)
            check(
                self._dll.mhs_solution_copy_probe(
                    self._handle,
                    index,
                    name,
                    name_size.value,
                    times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    record_count.value,
                ),
                "solution_copy_probe",
            )
            self._probes.append(ProbeTrace(name.value.decode("utf-8"), times, values))

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
        check(
            self._dll.mhs_solution_write_vtu(self._handle, str(path).encode("utf-8")),
            "write_vtu",
        )
