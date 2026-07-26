"""High-level wrapper for ``mhs_solution_t`` — simulation results."""

from __future__ import annotations

import ctypes
from typing import NamedTuple

import numpy as np

from metahotspot._error import check
from metahotspot.types import MhsSolution, SolutionView, SolverOpts


SolutionViewData = NamedTuple(
    "SolutionViewData",
    [
        ("cell_count", int),
        ("state_count", int),
        ("node_count", int),
        ("time", float),
        ("cell_temperatures", np.ndarray),
        ("states", np.ndarray),
        ("node_temperatures", np.ndarray),
    ],
)


class Solution:
    """Read-only simulation result.

    Do not instantiate directly — use ``Compiled.solve()``.
    """

    def __init__(self) -> None:
        self._dll = None
        self._handle: MhsSolution | None = None
        self._compiled_handle = None  # optional ref for write_vtu
        self._owned = True

    @classmethod
    def _solve_compiled(
        cls, dll, compiled_handle, opts: SolverOpts | None = None
    ) -> Solution:
        """Solve a compiled model and wrap the result."""
        self = cls()
        self._dll = dll
        pp = ctypes.POINTER(MhsSolution)()
        opts_ptr = ctypes.byref(opts) if opts is not None else None
        check(
            dll.mhs_compiled_solve(compiled_handle, opts_ptr, ctypes.byref(pp)), "solve"
        )
        self._handle = pp
        self._compiled_handle = compiled_handle  # keep for write_vtu
        return self

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._owned and self._handle is not None:
            self._dll.mhs_solution_destroy(self._handle)
            self._handle = None

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
            node_count=v.node_count,
            time=v.time,
            cell_temperatures=np.ctypeslib.as_array(
                v.cell_temperatures, shape=(v.cell_count,)
            ),
            states=np.ctypeslib.as_array(v.states, shape=(v.state_count,)),
            node_temperatures=np.ctypeslib.as_array(
                v.node_temperatures, shape=(v.node_count,)
            ),
        )

    # ---- Shortcuts (convenience wrappers around view()) ----

    def state_count(self) -> int:
        return self.view().state_count

    def cell_count(self) -> int:
        return self.view().cell_count

    def node_count(self) -> int:
        return self.view().node_count

    def time(self) -> float:
        return self.view().time

    def states(self) -> np.ndarray:
        """Complete system state (read-only view)."""
        return self.view().states

    def cell_temperatures(self) -> np.ndarray:
        """Cell-centroid temperature field (read-only view)."""
        return self.view().cell_temperatures

    def node_temperatures(self) -> np.ndarray:
        """Node temperature field (read-only view)."""
        return self.view().node_temperatures

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
