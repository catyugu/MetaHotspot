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

    @classmethod
    def _solve_model(
        cls, dll, model_handle, opts: SolverOpts | None = None
    ) -> Solution:
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

    def state_count(self) -> int:
        return self._dll.mhs_solution_state_count(self._handle)

    def cell_count(self) -> int:
        return self._dll.mhs_solution_cell_count(self._handle)

    def node_count(self) -> int:
        return self._dll.mhs_solution_node_count(self._handle)

    def time(self) -> float:
        return self._dll.mhs_solution_time(self._handle)

    def states(self) -> np.ndarray:
        """Complete system state (read-only view). Entries may not be temperatures."""
        n = self.state_count()
        ptr = self._dll.mhs_solution_states(self._handle)
        return np.ctypeslib.as_array(ptr, shape=(n,))

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

    # ---- VTU export ----

    def write_vtu(self, path: str) -> None:
        """Export the temperature field to a VTU file.

        Writes an unstructured-grid VTU with hexahedral cells matching the
        active subset of the simulation mesh.  Only active cells (those
        belonging to a layer/block) are included; inactive grid cells are
        omitted.

        Parameters
        ----------
        path : str or Path
            Output ``.vtu`` file path.  Parent directories are created
            automatically.
        """
        if self._compiled_handle is None:
            raise RuntimeError(
                "write_vtu requires a solution from Compiled.solve(). "
                "Solutions from Model.solve() do not retain the compiled handle."
            )
        check(
            self._dll.mhs_compiled_write_vtu(
                self._compiled_handle,
                self._handle,
                str(path).encode("utf-8"),
            ),
            "write_vtu",
        )
