"""High-level wrapper for ``mhs_solution_t`` — simulation results."""

from __future__ import annotations

import ctypes

import numpy as np

from metahotspot._error import check
from metahotspot.types import MhsSolution, SolverOpts


class Solution:
    """Read-only simulation result.

    Do not instantiate directly — use ``Compiled.solve()`` or ``Model.solve()``.
    """

    def __init__(self) -> None:
        self._dll = None
        self._handle: MhsSolution | None = None
        self._owned = True

    @classmethod
    def _solve_compiled(cls, dll, compiled_handle, opts: SolverOpts | None = None) -> Solution:
        """Solve a compiled model and wrap the result."""
        self = cls()
        self._dll = dll
        pp = ctypes.POINTER(MhsSolution)()
        opts_ptr = ctypes.byref(opts) if opts is not None else None
        check(dll.mhs_compiled_solve(compiled_handle, opts_ptr, ctypes.byref(pp)), "solve")
        self._handle = pp
        return self

    @classmethod
    def _solve_model(cls, dll, model_handle, opts: SolverOpts | None = None) -> Solution:
        """Compile-and-solve a model, wrapping the result."""
        self = cls()
        self._dll = dll
        pp = ctypes.POINTER(MhsSolution)()
        opts_ptr = ctypes.byref(opts) if opts is not None else None
        check(dll.mhs_solve(model_handle, opts_ptr, ctypes.byref(pp)), "solve")
        self._handle = pp
        return self

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._owned and self._handle is not None:
            self._dll.mhs_solution_destroy(self._handle)
            self._handle = None

    # ---- Accessors ----

    def cell_count(self) -> int:
        return self._dll.mhs_solution_cell_count(self._handle)

    def node_count(self) -> int:
        return self._dll.mhs_solution_node_count(self._handle)

    def time(self) -> float:
        return self._dll.mhs_solution_time(self._handle)

    def cell_temperatures(self) -> np.ndarray:
        """Cell-centroid temperature field (read-only view)."""
        n = self.cell_count()
        ptr = self._dll.mhs_solution_cell_temperatures(self._handle)
        return np.ctypeslib.as_array(ptr, shape=(n,))

    def node_temperatures(self) -> np.ndarray:
        """Node temperature field from cell-to-node interpolation (read-only view)."""
        n = self.node_count()
        ptr = self._dll.mhs_solution_node_temperatures(self._handle)
        return np.ctypeslib.as_array(ptr, shape=(n,))

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
