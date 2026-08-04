"""High-level wrapper for ``mhs_compiled_t`` — compiled runtime model."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot.types import MhsCompiled, MhsCompiledMetadataView


class Operators(NamedTuple):
    """K, C, f of the linearised system: C * dx/dt + K * x = f."""

    K: object
    C: object
    f: np.ndarray


@dataclass
class SolveOptions:
    """Solver configuration options."""

    linear_solver: str = "Pardiso"
    linear_tolerance: float = 1e-8
    linear_max_iterations: int = 1000
    underrelaxation: float = 1.0
    nonlinear_max_iterations: int = 200
    nonlinear_relative_tolerance: float = 1e-6
    nonlinear_absolute_tolerance: float = 1e-12
    integrator: str = "Bdf1"
    step_strategy: str = "Adaptive"
    error_rel_tol: float = 1e-4
    error_safety: float = 0.9
    min_dt: float = 1e-12
    max_dt: float = 1.0
    fixed_dt: float = 1.0

    @staticmethod
    def default() -> SolveOptions:
        return SolveOptions()

    def _to_c_struct(self, dll):
        from metahotspot.types import _SolveOptionsCStruct

        c_opts = _SolveOptionsCStruct()
        dll.mhs_solve_options_default(ctypes.byref(c_opts))
        c_opts.solver_type = {
            "Pardiso": 0,
            "EigenSparseLU": 1,
            "EigenBiCGSTAB": 2,
        }.get(self.linear_solver, 0)
        c_opts.linear_tolerance = self.linear_tolerance
        c_opts.linear_max_iterations = self.linear_max_iterations
        c_opts.underrelaxation = self.underrelaxation
        c_opts.nonlinear_max_iterations = self.nonlinear_max_iterations
        c_opts.nonlinear_relative_tolerance = self.nonlinear_relative_tolerance
        c_opts.nonlinear_absolute_tolerance = self.nonlinear_absolute_tolerance
        c_opts.integrator = {"Bdf1": 0, "Bdf2": 1}.get(self.integrator, 0)
        c_opts.step_strategy = {"Adaptive": 0, "Fixed": 1}.get(self.step_strategy, 0)
        c_opts.error_rel_tol = self.error_rel_tol
        c_opts.error_safety = self.error_safety
        c_opts.min_dt = self.min_dt
        c_opts.max_dt = self.max_dt
        c_opts.fixed_dt = self.fixed_dt
        return c_opts


class Compiled(OwnedHandle):
    """Read-only compiled runtime model. Use ``Model.compile()``."""

    def __init__(self) -> None:
        super().__init__(None, None)
        self._metadata_cache = None

    @classmethod
    def _from_model(cls, dll, model_handle) -> Compiled:
        self = cls()
        self._dll = dll
        self._destroy_fn = dll.mhs_compiled_destroy
        handle = ctypes.POINTER(MhsCompiled)()
        check(dll.mhs_model_compile(model_handle, ctypes.byref(handle)), "compile")
        self._handle = handle
        return self

    def _fetch_metadata(self) -> MhsCompiledMetadataView:
        if self._metadata_cache is None:
            view = MhsCompiledMetadataView()
            check(
                self._dll.mhs_compiled_metadata(self._handle, ctypes.byref(view)),
                "compiled_metadata",
            )
            self._metadata_cache = view
        return self._metadata_cache

    @property
    def cell_count(self) -> int:
        return self._fetch_metadata().cell_count

    @property
    def study_type(self) -> int:
        return self._fetch_metadata().study_type

    @property
    def initial_temperature(self) -> float:
        return self._fetch_metadata().initial_temperature

    @property
    def nx(self) -> int:
        return self._fetch_metadata().nx

    @property
    def ny(self) -> int:
        return self._fetch_metadata().ny

    @property
    def nz(self) -> int:
        return self._fetch_metadata().nz

    @property
    def grid_to_cell(self) -> np.ndarray:
        metadata = self._fetch_metadata()
        indices = np.ctypeslib.as_array(
            metadata.grid_to_cell, shape=(metadata.nx * metadata.ny * metadata.nz,)
        )
        # C++ stores cell indices as size_t and uses SIZE_MAX for invalid cells.
        # Reinterpret the pointer-sized unsigned view as signed so the sentinel is
        # exposed consistently as -1 on Windows and Unix without copying.
        return indices.view(np.intp)

    @property
    def layer_ids(self) -> np.ndarray:
        metadata = self._fetch_metadata()
        return np.ctypeslib.as_array(metadata.layer_ids, shape=(metadata.cell_count,))

    @property
    def block_ids(self) -> np.ndarray:
        metadata = self._fetch_metadata()
        return np.ctypeslib.as_array(metadata.block_ids, shape=(metadata.cell_count,))

    def default_state(self) -> np.ndarray:
        return np.full(self.cell_count, self.initial_temperature, dtype=np.float64)

    def assemble(self, state: np.ndarray | None = None, time: float = 0.0) -> Operators:
        """Assemble K, C, f at a state and time."""
        from metahotspot.types import MhsOperatorsView
        import scipy.sparse

        if state is None:
            state = self.default_state()
        state = np.ascontiguousarray(state, dtype=np.float64)
        if state.size != self.cell_count:
            raise ValueError(
                f"state size ({state.size}) != cell_count ({self.cell_count})"
            )

        view = MhsOperatorsView()
        check(
            self._dll.mhs_compiled_assemble(
                self._handle,
                state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                state.size,
                time,
                ctypes.byref(view),
            ),
            "assemble",
        )

        def to_csc(matrix_view):
            columns = matrix_view.columns
            nnz = matrix_view.nnz
            outer = np.ctypeslib.as_array(
                matrix_view.outer_indices, shape=(columns + 1,)
            ).copy()
            inner = np.ctypeslib.as_array(
                matrix_view.inner_indices, shape=(nnz,)
            ).copy()
            values = np.ctypeslib.as_array(matrix_view.values, shape=(nnz,)).copy()
            return scipy.sparse.csc_matrix(
                (values, inner, outer),
                shape=(matrix_view.rows, matrix_view.columns),
            )

        return Operators(
            to_csc(view.K),
            to_csc(view.C),
            np.ctypeslib.as_array(view.rhs, shape=(view.n,)).copy(),
        )

    def solve(
        self,
        state: np.ndarray | None = None,
        opts: SolveOptions | None = None,
    ):
        from metahotspot.solution import Solution

        return Solution._solve_compiled(self, state, opts)
