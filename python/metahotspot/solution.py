"""High-level wrapper for ``mhs_solution_t`` — simulation results."""

from __future__ import annotations

import ctypes
from typing import NamedTuple

import numpy as np

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot.types import MhsSolution, SolutionView, ProbeView


class _SolutionHistoryView(ctypes.Structure):
    _fields_ = [
        ("times", ctypes.POINTER(ctypes.c_double)),
        ("states", ctypes.POINTER(ctypes.c_double)),
        ("record_count", ctypes.c_size_t),
        ("state_count", ctypes.c_size_t),
    ]


class ProbeTrace(NamedTuple):
    """A single probe trace — name plus time series."""

    name: str
    times: np.ndarray
    values: np.ndarray


class Solution(OwnedHandle):
    """Read-only simulation result.

    Do not instantiate directly — use ``Compiled.solve()`` or the macromodel
    extension.  ``state_history`` contains every C++ output snapshot, including
    the initial transient state and final accepted state.
    """

    def __init__(self) -> None:
        self._compiled = None
        self._view_cache: SolutionView | None = None
        self._history_cache: _SolutionHistoryView | None = None
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

        opts_ptr = None
        if opts is not None:
            c_opts = (
                opts._to_c_struct(self._dll) if hasattr(opts, "_to_c_struct") else opts
            )
            opts_ptr = ctypes.byref(c_opts)

        normalized_state = None
        if state is not None:
            normalized_state = np.ascontiguousarray(state, dtype=np.float64)
        state_ptr = (
            normalized_state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if normalized_state is not None
            else None
        )
        state_count = normalized_state.size if normalized_state is not None else 0
        check(
            self._dll.mhs_compiled_solve(
                compiled._handle, state_ptr, state_count, opts_ptr, ctypes.byref(pp)
            ),
            "solve",
        )
        self._handle = pp
        self._compiled = compiled
        return self

    @classmethod
    def _from_handle(
        cls,
        dll,
        destroy_fn,
        handle,
        compiled=None,
    ) -> Solution:
        self = cls()
        self._dll = dll
        self._destroy_fn = destroy_fn
        self._handle = handle
        self._compiled = compiled
        return self

    def _fetch_view(self) -> SolutionView:
        if self._view_cache is None:
            view = SolutionView()
            check(
                self._dll.mhs_solution_view(self._handle, ctypes.byref(view)),
                "solution_view",
            )
            self._view_cache = view
        return self._view_cache

    def _fetch_history(self) -> _SolutionHistoryView:
        if self._history_cache is None:
            try:
                function = self._dll.mhs_solution_history_view
            except AttributeError as exc:
                raise RuntimeError(
                    "the loaded MetaHotspot C API does not expose solution history; "
                    "rebuild mhs_c_api from the current source tree"
                ) from exc
            function.restype = ctypes.c_int32
            function.argtypes = [
                ctypes.POINTER(MhsSolution),
                ctypes.POINTER(_SolutionHistoryView),
            ]
            view = _SolutionHistoryView()
            check(function(self._handle, ctypes.byref(view)), "solution_history_view")
            self._history_cache = view
        return self._history_cache

    @property
    def temperature(self) -> np.ndarray:
        """Final FVM temperature field [fvm_count]."""
        view = self._fetch_view()
        return np.ctypeslib.as_array(view.state, shape=(view.fvm_count,))

    @property
    def time(self) -> float:
        """Final simulation time."""
        return self._fetch_view().time

    @property
    def state(self) -> np.ndarray:
        """Final full state, including retained external modes."""
        view = self._fetch_view()
        return np.ctypeslib.as_array(view.state, shape=(view.state_count,))

    @property
    def history_times(self) -> np.ndarray:
        """C++ output times [record_count]."""
        view = self._fetch_history()
        if view.record_count == 0:
            return np.empty(0, dtype=np.float64)
        return np.ctypeslib.as_array(view.times, shape=(view.record_count,))

    @property
    def state_history(self) -> np.ndarray:
        """C++ output states [record_count, state_count], row-major."""
        view = self._fetch_history()
        if view.record_count == 0:
            return np.empty((0, view.state_count), dtype=np.float64)
        flat = np.ctypeslib.as_array(
            view.states, shape=(view.record_count * view.state_count,)
        )
        return flat.reshape((view.record_count, view.state_count))

    @property
    def temperature_history(self) -> np.ndarray:
        """FVM temperature snapshots [record_count, fvm_count]."""
        fvm_count = self._fetch_view().fvm_count
        return self.state_history[:, :fvm_count]

    @property
    def probes(self) -> list[ProbeTrace]:
        """Return all probe traces as high-level named tuples."""
        count = self._dll.mhs_solution_probe_count(self._handle)
        result: list[ProbeTrace] = []
        for index in range(count):
            name, times, values, _ = self._probe_view(index)
            result.append(ProbeTrace(name, times, values))
        return result

    def _probe_view(self, index: int):
        probe = ProbeView()
        check(
            self._dll.mhs_solution_probe_view(self._handle, index, ctypes.byref(probe)),
            "solution_probe_view",
        )
        name = probe.name.decode("utf-8") if probe.name else ""
        times = (
            np.ctypeslib.as_array(probe.times, shape=(probe.record_count,))
            if probe.times
            else np.empty(0, dtype=np.float64)
        )
        values = (
            np.ctypeslib.as_array(probe.values, shape=(probe.record_count,))
            if probe.values
            else np.empty(0, dtype=np.float64)
        )
        return name, times, values, probe.record_count

    def write_vtu(self, path: str) -> None:
        """Export the final FVM temperature field to a VTU file."""
        if self._compiled is None:
            raise RuntimeError("write_vtu requires a solution from Compiled.solve().")
        check(
            self._dll.mhs_compiled_write_vtu(
                self._compiled._handle,
                self._handle,
                str(path).encode("utf-8"),
            ),
            "write_vtu",
        )
