"""High-level wrapper for ``mhs_compiled_t`` — compiled runtime model."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot.types import (
    MhsCompiled,
    MhsCompiledMetadataView,
)


class Operators(NamedTuple):
    """K, C, f of the linearised system: C * dx/dt + K * x = f."""

    K: object  # scipy.sparse.csc_matrix
    C: object  # scipy.sparse.csc_matrix
    f: np.ndarray


@dataclass
class SolveOptions:
    """Solver configuration options.

    All fields have sensible defaults.  Pass to ``Compiled.solve()``.
    """

    # Linear solver
    linear_solver: str = "Pardiso"  # "Pardiso", "EigenSparseLU", "EigenBiCGSTAB"
    linear_tolerance: float = 1e-8
    linear_max_iterations: int = 1000

    # Non-linear solver
    underrelaxation: float = 1.0
    nonlinear_max_iterations: int = 200
    nonlinear_relative_tolerance: float = 1e-6
    nonlinear_absolute_tolerance: float = 1e-12

    # Time integration
    integrator: str = "Bdf1"  # "Bdf1", "Bdf2"
    step_strategy: str = "Adaptive"  # "Adaptive", "Fixed"
    error_abs_tol: float = 1e-4
    error_safety: float = 0.9
    min_dt: float = 1e-12
    max_dt: float = 1.0
    fixed_dt: float = 1.0

    @staticmethod
    def default() -> SolveOptions:
        """Return sensible defaults."""
        return SolveOptions()

    def _to_c_struct(self, dll):
        """Convert to the C API's mhs_solve_options_t."""
        from metahotspot.types import _SolveOptionsCStruct

        c_opts = _SolveOptionsCStruct()
        dll.mhs_solve_options_default(ctypes.byref(c_opts))

        # Map string enums to ints
        solver_map = {
            "Pardiso": 0,
            "EigenSparseLU": 1,
            "EigenBiCGSTAB": 2,
        }
        integrator_map = {"Bdf1": 0, "Bdf2": 1}
        strategy_map = {"Adaptive": 0, "Fixed": 1}

        c_opts.solver_type = solver_map.get(self.linear_solver, 0)
        c_opts.linear_tolerance = self.linear_tolerance
        c_opts.linear_max_iterations = self.linear_max_iterations
        c_opts.underrelaxation = self.underrelaxation
        c_opts.nonlinear_max_iterations = self.nonlinear_max_iterations
        c_opts.nonlinear_relative_tolerance = self.nonlinear_relative_tolerance
        c_opts.nonlinear_absolute_tolerance = self.nonlinear_absolute_tolerance
        c_opts.integrator = integrator_map.get(self.integrator, 0)
        c_opts.step_strategy = strategy_map.get(self.step_strategy, 0)
        c_opts.error_abs_tol = self.error_abs_tol
        c_opts.error_safety = self.error_safety
        c_opts.min_dt = self.min_dt
        c_opts.max_dt = self.max_dt
        c_opts.fixed_dt = self.fixed_dt
        return c_opts


class Compiled(OwnedHandle):
    """Read-only compiled runtime model.

    Do not instantiate directly — use ``Model.compile()``.
    """

    def __init__(self) -> None:
        super().__init__(None, None)
        self._metadata_cache: dict | None = None

    @classmethod
    def _from_model(cls, dll, model_handle) -> Compiled:
        """Compile *model_handle* and return a new Compiled instance."""
        self = cls()
        self._dll = dll
        self._destroy_fn = dll.mhs_compiled_destroy
        pp = ctypes.POINTER(MhsCompiled)()
        check(dll.mhs_model_compile(model_handle, ctypes.byref(pp)), "compile")
        self._handle = pp
        return self

    # ---- Metadata (cached) ----

    def _fetch_metadata(self) -> MhsCompiledMetadataView:
        if self._metadata_cache is not None:
            return self._metadata_cache
        view = MhsCompiledMetadataView()
        check(
            self._dll.mhs_compiled_metadata(self._handle, ctypes.byref(view)),
            "compiled_metadata",
        )
        self._metadata_cache = view
        return view

    @property
    def cell_count(self) -> int:
        """Number of active FVM cells."""
        return self._fetch_metadata().cell_count

    @property
    def study_type(self) -> int:
        """Study type constant (Study.STEADY or Study.TRANSIENT)."""
        return self._fetch_metadata().study_type

    @property
    def initial_temperature(self) -> float:
        """Default initial temperature."""
        return self._fetch_metadata().initial_temperature

    # ---- Metadata array properties ----

    @property
    def nx(self) -> int:
        """Grid x-dimension (number of cells along x)."""
        return self._fetch_metadata().nx

    @property
    def ny(self) -> int:
        """Grid y-dimension."""
        return self._fetch_metadata().ny

    @property
    def nz(self) -> int:
        """Grid z-dimension."""
        return self._fetch_metadata().nz

    @property
    def grid_to_cell(self) -> np.ndarray:
        """Map from grid index to active-cell index [nx*ny*nz].

        Inactive/dead cells have the value SIZE_MAX.
        """
        meta = self._fetch_metadata()
        total = meta.nx * meta.ny * meta.nz
        return np.ctypeslib.as_array(meta.grid_to_cell, shape=(total,))

    @property
    def layer_ids(self) -> np.ndarray:
        """Per-cell layer ID [cell_count]."""
        meta = self._fetch_metadata()
        return np.ctypeslib.as_array(meta.layer_ids, shape=(meta.cell_count,))

    @property
    def block_ids(self) -> np.ndarray:
        """Per-cell block ID [cell_count]."""
        meta = self._fetch_metadata()
        return np.ctypeslib.as_array(meta.block_ids, shape=(meta.cell_count,))

    # ---- Default state ----

    def default_state(self) -> np.ndarray:
        """Return a uniform-temperature initial state vector [cell_count]."""
        return np.full(self.cell_count, self.initial_temperature, dtype=np.float64)

    # ---- Assembly ----

    def assemble(self, state: np.ndarray | None = None, time: float = 0.0) -> Operators:
        """Assemble K, C, f at *state* and *time*.

        Parameters
        ----------
        state : ndarray | None
            Temperature state [cell_count].  ``None`` = uniform initial_temperature.
        time : float
            Evaluation time (default 0.0).

        Returns
        -------
        Operators(K, C, f)
        """
        from metahotspot.types import MhsOperatorsView

        if state is None:
            state = self.default_state()
        state = np.ascontiguousarray(state, dtype=np.float64)
        if state.size != self.cell_count:
            raise ValueError(
                f"state size ({state.size}) != cell_count ({self.cell_count})"
            )

        c_op = MhsOperatorsView()
        check(
            self._dll.mhs_compiled_assemble(
                self._handle,
                state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                state.size,
                time,
                ctypes.byref(c_op),
            ),
            "assemble",
        )

        import scipy.sparse

        def _view_to_csc(v) -> scipy.sparse.csc_matrix:
            n_rows = v.rows
            n_cols = v.columns
            nnz = v.nnz
            outer = np.ctypeslib.as_array(v.outer_indices, shape=(n_cols + 1,)).copy()
            inner = np.ctypeslib.as_array(v.inner_indices, shape=(nnz,)).copy()
            vals = np.ctypeslib.as_array(v.values, shape=(nnz,)).copy()
            return scipy.sparse.csc_matrix((vals, inner, outer), shape=(n_rows, n_cols))

        K = _view_to_csc(c_op.K)
        C = _view_to_csc(c_op.C)
        f = np.ctypeslib.as_array(c_op.rhs, shape=(c_op.n,)).copy()
        return Operators(K, C, f)

    # ---- Solve ----

    def solve(
        self,
        state: np.ndarray | None = None,
        opts: SolveOptions | None = None,
    ) -> Solution:
        """Solve the compiled model.

        Parameters
        ----------
        state : ndarray | None
            Initial state vector.  ``None`` = uniform initial_temperature.
        opts : SolveOptions | None
            Solver options.  ``None`` = defaults.

        Returns
        -------
        Solution
        """
        from metahotspot.solution import Solution

        return Solution._solve_compiled(self, state, opts)
