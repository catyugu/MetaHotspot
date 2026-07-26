"""High-level wrapper for ``mhs_compiled_t`` — compiled runtime model."""

from __future__ import annotations

import ctypes

import numpy as np

from metahotspot._error import check
from metahotspot.types import MhsCompiled, MhsMeshInfo, MhsStepInfo, SolverOpts


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

    # ---- Pre-solve configuration ----

    def set_initial_state(self, state: np.ndarray) -> None:
        """Override the initial state from a previous solution.

        Useful for chaining steady-state → transient: solve steady, extract
        ``solution.states()``, then set it here before the transient solve.

        Parameters
        ----------
        state : ndarray
            Full system state, length ``state_count()``.
        """
        ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        check(
            self._dll.mhs_compiled_set_initial_state(self._handle, ptr, state.size),
            "set_initial_state",
        )

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

    # ---- Mesh geometry ----

    def mesh(self) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
        """Return the structured mesh geometry in SI units.

        Returns
        -------
        nx, ny, nz : int
            Number of cells along each axis.
        x_verts, y_verts, z_verts : ndarray
            Length ``(nx+1)``, ``(ny+1)``, ``(nz+1)`` arrays of vertex coordinates
            (read-only views into the compiled model's internal storage).
        """
        info = MhsMeshInfo()
        check(self._dll.mhs_compiled_mesh(self._handle, ctypes.byref(info)), "mesh")
        nx = int(info.nx)
        ny = int(info.ny)
        nz = int(info.nz)
        xv = np.ctypeslib.as_array(info.x_verts, shape=(nx + 1,))
        yv = np.ctypeslib.as_array(info.y_verts, shape=(ny + 1,))
        zv = np.ctypeslib.as_array(info.z_verts, shape=(nz + 1,))
        return nx, ny, nz, xv, yv, zv

    # ---- Single transient step ----

    def step(
        self,
        state: np.ndarray,
        time: float,
        dt: float,
        opts: SolverOpts | None = None,
    ) -> tuple[np.ndarray, dict | None]:
        """Execute a single transient time step (BDF1).

        The compiled model must have ``study = TRANSIENT``.

        Parameters
        ----------
        state : ndarray of shape ``(state_count(),)``
            State at time *t* (T\ :sup:`n`\ ).
        time : float
            Current simulation time (t\ :sup:`n`\ ).
        dt : float
            Time step size.
        opts : SolverOpts or None
            Solver options (underrelaxation, tolerances, …).
            Pass ``None`` for defaults.

        Returns
        -------
        new_state : ndarray
            State at time ``time + dt`` (T\ :sup:`n+1`\ ).
        info : dict or None
            Optional diagnostics (error ratio, step accepted flag, …).
        """
        n = self.state_count()
        assert len(state) == n, f"state length {len(state)} != {n}"
        out_state = np.empty(n, dtype=np.float64)
        step_info = MhsStepInfo()
        opts_ptr = ctypes.byref(opts) if opts is not None else None
        check(
            self._dll.mhs_compiled_step(
                self._handle,
                state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                time,
                dt,
                out_state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.byref(step_info),
                opts_ptr,
            ),
            "step",
        )
        if step_info.error_ratio == 0.0 and step_info.accepted == 0:
            # Info wasn't filled (shouldn't happen, but be safe)
            return out_state, None
        return out_state, {
            "error_ratio": step_info.error_ratio,
            "suggested_dt_factor": step_info.suggested_dt_factor,
            "nonlinear_iterations": step_info.nonlinear_iterations,
            "accepted": bool(step_info.accepted),
        }
