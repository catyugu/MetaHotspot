"""High-level wrapper for ``mhs_solution_t`` — simulation results."""

from __future__ import annotations

import ctypes
from typing import NamedTuple

import numpy as np

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot.types import MhsSolution, SolutionView, SolverOpts


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


class Solution(OwnedHandle):
    """Read-only simulation result.

    Do not instantiate directly — use ``Compiled.solve()``.
    """

    def __init__(self) -> None:
        self._compiled_handle = None  # optional ref for write_vtu
        super().__init__(None, None)

    @classmethod
    def _solve_compiled(
        cls,
        dll,
        compiled_handle,
        state: np.ndarray | None = None,
        opts: SolverOpts | None = None,
    ) -> Solution:
        """Solve a compiled model and wrap the result."""
        self = cls()
        self._dll = dll
        self._destroy_fn = dll.mhs_solution_destroy
        pp = ctypes.POINTER(MhsSolution)()
        opts_ptr = ctypes.byref(opts) if opts is not None else None
        state_ptr = (
            state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if state is not None
            else None
        )
        state_count = state.size if state is not None else 0
        check(
            dll.mhs_compiled_solve(
                compiled_handle, state_ptr, state_count, opts_ptr, ctypes.byref(pp)
            ),
            "solve",
        )
        self._handle = pp
        self._compiled_handle = compiled_handle  # keep for write_vtu
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

    def probe_name(self, index: int) -> str:
        ptr = self._dll.mhs_solution_probe_name(self._handle, index)
        return ptr.decode("utf-8") if ptr else ""

    def probe_record_count(self, probe_index: int) -> int:
        return self._dll.mhs_solution_probe_record_count(self._handle, probe_index)

    def probe_times(self, probe_index: int) -> np.ndarray | None:
        """Time vector for a probe (None for steady-state)."""
        ptr = self._dll.mhs_solution_probe_times(self._handle, probe_index)
        if not ptr:
            return None
        n = self.probe_record_count(probe_index)
        return np.ctypeslib.as_array(ptr, shape=(n,))

    def probe_values(self, probe_index: int) -> np.ndarray:
        n = self.probe_record_count(probe_index)
        ptr = self._dll.mhs_solution_probe_values(self._handle, probe_index)
        return np.ctypeslib.as_array(ptr, shape=(n,))

    # ---- VTU export ----

    def write_vtu(self, path: str) -> None:
        """Export the temperature field to a VTU file."""
        if self._compiled_handle is None:
            raise RuntimeError("write_vtu requires a solution from Compiled.solve().")
        check(
            self._dll.mhs_compiled_write_vtu(
                self._compiled_handle,
                self._handle,
                str(path).encode("utf-8"),
            ),
            "write_vtu",
        )
