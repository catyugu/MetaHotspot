"""High-level wrapper for ``mhs_compiled_t`` — compiled runtime model."""

from __future__ import annotations

import ctypes

import numpy as np

from metahotspot._error import check
from metahotspot.types import MhsCompiled, SolverOpts


class Compiled:
    """Read-only compiled runtime model.

    Do not instantiate directly — use ``Model.compile()``.
    """

    def __init__(self) -> None:
        # Internal constructor; use _from_model() instead.
        self._dll = None
        self._handle: MhsCompiled | None = None
        self._owned = True

    @classmethod
    def _from_model(cls, dll, model_handle) -> Compiled:
        """Compile *model_handle* and return a new Compiled instance."""
        self = cls()
        self._dll = dll
        pp = ctypes.POINTER(MhsCompiled)()
        check(dll.mhs_model_compile(model_handle, ctypes.byref(pp)), "compile")
        self._handle = pp
        return self

    @classmethod
    def _from_ptr(cls, dll, handle) -> Compiled:
        """Wrap an existing compiled handle (not owned by this session)."""
        self = cls()
        self._dll = dll
        self._handle = handle
        self._owned = True
        return self

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._owned and self._handle is not None:
            self._dll.mhs_compiled_destroy(self._handle)
            self._handle = None

    # ---- Introspection ----

    def cell_count(self) -> int:
        return self._dll.mhs_compiled_cell_count(self._handle)

    def state_count(self) -> int:
        return self._dll.mhs_compiled_state_count(self._handle)

    def node_count(self) -> int:
        return self._dll.mhs_compiled_node_count(self._handle)

    def initial_temperature(self) -> float:
        return self._dll.mhs_compiled_initial_temperature(self._handle)

    def study_type(self) -> int:
        return self._dll.mhs_compiled_study_type(self._handle)

    def layer_count(self) -> int:
        return self._dll.mhs_compiled_layer_count(self._handle)

    def block_count(self, layer: int) -> int:
        return self._dll.mhs_compiled_block_count(self._handle, layer)

    def layer_ids(self) -> np.ndarray:
        """Per-cell layer IDs as a read-only numpy array."""
        ptr = self._dll.mhs_compiled_layer_ids(self._handle)
        n = self.cell_count()
        return np.ctypeslib.as_array(ptr, shape=(n,))

    def block_ids(self) -> np.ndarray:
        """Per-cell block IDs as a read-only numpy array."""
        ptr = self._dll.mhs_compiled_block_ids(self._handle)
        n = self.cell_count()
        return np.ctypeslib.as_array(ptr, shape=(n,))

    # ---- Grid topology ----

    def grid_count(self) -> int:
        """Total number of cells in the Cartesian grid (nx * ny * nz)."""
        return self._dll.mhs_compiled_grid_count(self._handle)

    def grid_to_cell(self) -> np.ndarray:
        """Map from linear grid index (ix-iy-iz) to active-cell index.

        Returns a ``size_t`` array (``np.uintp``) of length grid_count().
        Entry ``SIZE_MAX`` (``np.iinfo(np.uintp).max``) means the grid cell
        is inactive (not part of any layer/block).  Non-negative values are
        indices into cell-level arrays such as *cell_temperatures*,
        *layer_ids*, *block_ids*, etc.

        The linear grid index follows the ix-iy-iz convention:
            idx = ix * (ny * nz) + iy * nz + iz
        """
        ptr = self._dll.mhs_compiled_grid_to_cell(self._handle)
        n = self.grid_count()
        return np.ctypeslib.as_array(ptr, shape=(n,))

    # ---- Solve ----

    def solve(self, opts: SolverOpts | None = None) -> Solution:
        """Solve the compiled model."""
        from metahotspot.solution import Solution

        return Solution._solve_compiled(self._dll, self._handle, opts)

    # ---- Assembly ----

    def assemble(
        self,
        state: np.ndarray | None = None,
        time: float = 0.0,
    ) -> Assembly:
        """Assemble K, f at a given temperature field and time.

        Parameters
        ----------
        state : ndarray or None
            Complete system state. None = use the initial state.
        time : float
            Current simulation time.
        """
        from metahotspot.assembly import Assembly

        return Assembly._assemble(self._dll, self._handle, state, time)
