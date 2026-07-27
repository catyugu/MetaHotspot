"""High-level wrapper for ``mhs_solution_t`` — simulation results."""

from __future__ import annotations

import ctypes
from typing import NamedTuple

import numpy as np

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot.types import MhsSolution, SolutionView, SolverOpts, ProbeView


SolutionViewData = NamedTuple(
    "SolutionViewData",
    [
        ("cell_count", int),
        ("state_count", int),
        ("time", float),
        ("cell_temperatures", np.ndarray),
        ("states", np.ndarray),
    ],
)


ProbeTraceData = NamedTuple(
    "ProbeTraceData",
    [
        ("name", str),
        ("times", np.ndarray | None),
        ("values", np.ndarray | None),
        ("record_count", int),
    ],
)


class Solution(OwnedHandle):
    """Read-only simulation result.

    Do not instantiate directly — use ``Compiled.solve()``.
    """

    def __init__(self) -> None:
        self._compiled = None  # strong reference to Compiled
        super().__init__(None, None)

    @classmethod
    def _solve_compiled(
        cls,
        compiled,
        state: np.ndarray | None = None,
        opts: SolverOpts | None = None,
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

    # ---- View (single C call replaces ~7 individual accessors) ----

    def view(self) -> SolutionViewData:
        """Return all solution bulk data in one call to the C layer."""
        v = SolutionView()
        check(
            self._dll.mhs_solution_view(self._handle, ctypes.byref(v)), "solution_view"
        )
        return SolutionViewData(
            cell_count=v.cell_count,
            state_count=v.state_count,
            time=v.time,
            cell_temperatures=np.ctypeslib.as_array(
                v.cell_temperatures, shape=(v.cell_count,)
            ),
            states=np.ctypeslib.as_array(v.states, shape=(v.state_count,)),
        )

    # ---- Probes ----

    def probe_count(self) -> int:
        return self._dll.mhs_solution_probe_count(self._handle)

    def probe_view(self, index: int) -> ProbeTraceData:
        """Return name, times, values, record_count for a probe trace."""
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
        return ProbeTraceData(name, times, values, pv.record_count)

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
