"""High-level wrapper for ``mhs_solution_t`` — simulation results."""

from __future__ import annotations

import ctypes

import numpy as np

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot.types import MhsSolution, SolutionView, ProbeView


class Solution(OwnedHandle):
    """Read-only simulation result.

    Do not instantiate directly — use ``Compiled.solve()``.

    Access solution data via properties::

        sol = compiled.solve()
        T = sol.temperature          # ndarray of cell temperatures
        t = sol.time                 # simulation time
    """

    def __init__(self) -> None:
        self._compiled = None  # strong reference to Compiled
        self._view_cache: SolutionView | None = None
        super().__init__(None, None)

    @classmethod
    def _solve_compiled(
        cls,
        compiled,
        state: np.ndarray | None = None,
        opts=None,
    ) -> Solution:
        """Solve a compiled model and wrap the result."""
        self = cls()
        self._dll = compiled._dll
        self._destroy_fn = self._dll.mhs_solution_destroy
        pp = ctypes.POINTER(MhsSolution)()
        opts_ptr = ctypes.byref(opts) if opts is not None else None
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
        self._compiled = compiled  # strong reference prevents GC
        return self

    @classmethod
    def _from_handle(
        cls,
        dll,
        destroy_fn,
        handle,
        compiled=None,
    ) -> Solution:
        """Wrap an already-obtained ``mhs_solution_t`` pointer."""
        self = cls()
        self._dll = dll
        self._destroy_fn = destroy_fn
        self._handle = handle
        self._compiled = compiled
        return self

    # ---- Lazy view caching ----

    def _fetch_view(self) -> SolutionView:
        if self._view_cache is None:
            v = SolutionView()
            check(
                self._dll.mhs_solution_view(self._handle, ctypes.byref(v)),
                "solution_view",
            )
            self._view_cache = v
        return self._view_cache

    @property
    def temperature(self) -> np.ndarray:
        """Temperature field [cell_count]."""
        v = self._fetch_view()
        return np.ctypeslib.as_array(v.state, shape=(v.fvm_count,))

    @property
    def time(self) -> float:
        """Simulation end time."""
        return self._fetch_view().time

    @property
    def state(self) -> np.ndarray:
        """Full system state, including any retained external modes."""
        view = self._fetch_view()
        return np.ctypeslib.as_array(
            view.state,
            shape=(view.state_count,),
        )

    # ---- Probes ----

    def probe_count(self) -> int:
        return self._dll.mhs_solution_probe_count(self._handle)

    def probe_view(self, index: int):
        """Return (name, times, values, record_count) for a probe trace."""
        pv = ProbeView()
        check(
            self._dll.mhs_solution_probe_view(self._handle, index, ctypes.byref(pv)),
            "probe_view",
        )
        name = pv.name.decode("utf-8") if pv.name else ""
        times = (
            np.ctypeslib.as_array(pv.times, shape=(pv.record_count,))
            if pv.times
            else None
        )
        values = (
            np.ctypeslib.as_array(pv.values, shape=(pv.record_count,))
            if pv.values
            else None
        )
        return name, times, values, pv.record_count

    # ---- VTU export ----

    def write_vtu(self, path: str) -> None:
        """Export the temperature field to a VTU file."""
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
