"""High-level wrapper for ``mhs_solution_t`` — simulation results."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot.types import MhsSolution, MhsSolutionInfo, _SolveOptionsCStruct


# ---- read-only result snapshots -----------------------------------------


@dataclass(frozen=True)
class ProbeSnapshot:
    name: str
    times: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class SolutionSnapshot:
    time: float
    fvm_count: int
    state: np.ndarray
    history_times: np.ndarray
    state_history: np.ndarray
    probes: list[ProbeSnapshot]


# ---- low-level ctypes marshalling helpers --------------------------------


def _text(value: str | None) -> bytes | None:
    return None if value is None else value.encode("utf-8")


def _double_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _solve_options(dll, overrides: dict):
    """Build a C solve-options struct with *overrides* applied over defaults.

    Returns the struct object; callers must keep it alive until the matching
    solve call completes.
    """
    opts = _SolveOptionsCStruct()
    dll.mhs_solve_options_default(ctypes.byref(opts))
    for name, value in overrides.items():
        setattr(opts, name, value)
    return opts


def _solve(dll, compiled_handle, state, opts_overrides: dict):
    opts = _solve_options(dll, opts_overrides)
    state_ptr = _double_ptr(state) if state is not None else None
    state_count = state.size if state is not None else 0
    out = ctypes.POINTER(MhsSolution)()
    check(
        dll.mhs_compiled_solve(
            compiled_handle,
            state_ptr,
            state_count,
            ctypes.byref(opts),
            ctypes.byref(out),
        ),
        "solve",
    )
    return out


def _solution_snapshot(dll, handle) -> SolutionSnapshot:
    """Snapshot a native solution into Python-owned arrays."""
    info = MhsSolutionInfo()
    check(dll.mhs_solution_get_info(handle, ctypes.byref(info)), "solution_info")

    state = np.empty(int(info.state_count), dtype=np.float64)
    history_times = np.empty(int(info.record_count), dtype=np.float64)
    history_states = np.empty(
        int(info.record_count) * int(info.state_count), dtype=np.float64
    )
    check(
        dll.mhs_solution_copy_state(handle, _double_ptr(state), state.size),
        "solution_copy_state",
    )
    check(
        dll.mhs_solution_copy_history_times(
            handle, _double_ptr(history_times), history_times.size
        ),
        "solution_copy_history_times",
    )
    check(
        dll.mhs_solution_copy_history_states(
            handle, _double_ptr(history_states), history_states.size
        ),
        "solution_copy_history_states",
    )

    probes: list[ProbeSnapshot] = []
    for index in range(int(info.probe_count)):
        name_size = ctypes.c_size_t()
        record_count = ctypes.c_size_t()
        check(
            dll.mhs_solution_probe_get_info(
                handle, index, ctypes.byref(name_size), ctypes.byref(record_count)
            ),
            "solution_probe_info",
        )
        name = ctypes.create_string_buffer(name_size.value)
        times = np.empty(record_count.value, dtype=np.float64)
        values = np.empty(record_count.value, dtype=np.float64)
        check(
            dll.mhs_solution_copy_probe(
                handle,
                index,
                name,
                name_size.value,
                _double_ptr(times),
                _double_ptr(values),
                record_count.value,
            ),
            "solution_copy_probe",
        )
        probes.append(ProbeSnapshot(name.value.decode("utf-8"), times, values))

    return SolutionSnapshot(
        time=float(info.time),
        fvm_count=int(info.fvm_count),
        state=state,
        history_times=history_times,
        state_history=history_states.reshape(
            int(info.record_count), int(info.state_count)
        ),
        probes=probes,
    )


# ---- public wrapper ------------------------------------------------------


class Solution(OwnedHandle):
    """Read-only simulation result.

    Do not instantiate directly — use ``Compiled.solve()``.  ``state_history``
    contains every C++ output snapshot, including the initial transient state
    and final accepted state.
    """

    def __init__(self, dll, handle) -> None:
        super().__init__(dll, handle, dll.mhs_solution_destroy)

    @classmethod
    def _solve_compiled(
        cls,
        compiled,
        state: np.ndarray | None = None,
        opts=None,
    ) -> Solution:
        overrides = opts._overrides() if opts is not None else {}
        handle = _solve(compiled._dll, compiled._handle, state, overrides)
        return cls(compiled._dll, handle)

    @cached_property
    def data(self) -> SolutionSnapshot:
        """Snapshot native results into independent Python-owned arrays."""
        return _solution_snapshot(self._dll, self._handle)

    @property
    def temperature(self) -> np.ndarray:
        """Final FVM temperature field [fvm_count] (view of ``state``)."""
        return self.state[: self.fvm_count]

    @property
    def time(self) -> float:
        """Final simulation time."""
        return self.data.time

    @property
    def state(self) -> np.ndarray:
        """Final full state, including retained external modes."""
        return self.data.state

    @property
    def history_times(self) -> np.ndarray:
        """C++ output times [record_count]."""
        return self.data.history_times

    @property
    def state_history(self) -> np.ndarray:
        """C++ output states [record_count, state_count], row-major."""
        return self.data.state_history

    @property
    def temperature_history(self) -> np.ndarray:
        """FVM temperature snapshots [record_count, fvm_count] (view of ``state_history``)."""
        return self.state_history[:, : self.fvm_count]

    @property
    def probes(self) -> list[ProbeSnapshot]:
        """Return all probe traces as Python-owned snapshots."""
        return self.data.probes

    @property
    def fvm_count(self) -> int:
        """Number of FVM temperatures at the front of the full state."""
        return self.data.fvm_count

    def write_vtu(self, path: str) -> None:
        """Export the final FVM temperature field to a VTU file."""
        self._call("mhs_solution_write_vtu", _text(str(path)), ctx="write_vtu")
