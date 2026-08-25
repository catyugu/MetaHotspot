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
    MhsCellFields,
    MhsMaterialValues,
    MhsCompiledInfo,
    MhsOperators,
    MhsOperatorsInfo,
)


class Operators(NamedTuple):
    """K, C, f of the linearised system: C * dx/dt + K * x = f."""

    K: object
    C: object
    f: np.ndarray


# Single declaration of the CellFields contract: (field name, ctypes element
# type, numpy dtype, count source on MhsCompiledInfo).  Drives buffer
# allocation, the native mhs_cell_fields_t wiring and the CellFields view.
_CELL_FIELD_SPECS: tuple[tuple[str, type, type, str], ...] = (
    ("grid_to_cell", ctypes.c_size_t, np.intp, "grid_count"),
    ("cell_to_grid", ctypes.c_size_t, np.intp, "cell_count"),
    ("dx", ctypes.c_double, np.float64, "nx"),
    ("dy", ctypes.c_double, np.float64, "ny"),
    ("dz", ctypes.c_double, np.float64, "nz"),
    ("cx", ctypes.c_double, np.float64, "nx"),
    ("cy", ctypes.c_double, np.float64, "ny"),
    ("cz", ctypes.c_double, np.float64, "nz"),
    ("layer_id", ctypes.c_uint32, np.uint32, "cell_count"),
    ("block_id", ctypes.c_uint32, np.uint32, "cell_count"),
    ("material_id", ctypes.c_uint32, np.uint32, "cell_count"),
    ("heat_source_idx", ctypes.c_uint32, np.uint32, "cell_count"),
)


@dataclass(frozen=True)
class CellFields:
    grid_to_cell: np.ndarray
    cell_to_grid: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    cz: np.ndarray
    layer_id: np.ndarray
    block_id: np.ndarray
    material_id: np.ndarray
    heat_source_idx: np.ndarray

    # Face directions in exposed_face_mask bit order (bit 0..5 = XM, XP, YM,
    # YP, ZM, ZP), matching metahotspot.enums.Face.
    _FACE_STEPS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))

    @staticmethod
    def _vertices(centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
        return np.concatenate(([centers[0] - 0.5 * widths[0]], centers + 0.5 * widths))

    @property
    def x_vertices(self) -> np.ndarray:
        return self._vertices(self.cx, self.dx)

    @property
    def y_vertices(self) -> np.ndarray:
        return self._vertices(self.cy, self.dy)

    @property
    def z_vertices(self) -> np.ndarray:
        return self._vertices(self.cz, self.dz)

    @property
    def ijk(self) -> np.ndarray:
        """Per-compact-cell ``(ix, iy, iz)`` grid coordinates, decoded once."""
        return self._ijk

    @property
    def exposed_face_mask(self) -> np.ndarray:
        """Per-compact-cell ``(N, 6)`` uint8 mask: bit ``Face`` set when that
        face has no active neighbour (out of bounds or empty cell)."""
        return self._exposed_face_mask

    @property
    def cell_sizes(self) -> np.ndarray:
        ijk = self._ijk
        return np.column_stack(
            (self.dx[ijk[:, 0]], self.dy[ijk[:, 1]], self.dz[ijk[:, 2]])
        )

    @property
    def centers(self) -> np.ndarray:
        ijk = self._ijk
        return np.column_stack(
            (self.cx[ijk[:, 0]], self.cy[ijk[:, 1]], self.cz[ijk[:, 2]])
        )

    @property
    def half_sizes(self) -> np.ndarray:
        return self.cell_sizes * 0.5

    @property
    def volumes(self) -> np.ndarray:
        sizes = self.cell_sizes
        return sizes[:, 0] * sizes[:, 1] * sizes[:, 2]

    @property
    def nx(self) -> int:
        return self.dx.size

    @property
    def ny(self) -> int:
        return self.dy.size

    @property
    def nz(self) -> int:
        return self.dz.size

    def _compute_exposed_face_mask(self) -> np.ndarray:
        grid = self._grid3d
        invalid = np.iinfo(self.grid_to_cell.dtype).max
        # Pad with an invalid border so out-of-bounds neighbours read as
        # inactive, then compare each of the six shifted views to `invalid`.
        padded = np.full(
            (self.nx + 2, self.ny + 2, self.nz + 2), invalid, dtype=grid.dtype
        )
        padded[1:-1, 1:-1, 1:-1] = grid
        layers = [
            padded[
                1 + dx : self.nx + 1 + dx,
                1 + dy : self.ny + 1 + dy,
                1 + dz : self.nz + 1 + dz,
            ]
            == invalid
            for dx, dy, dz in self._FACE_STEPS
        ]
        exposed = np.stack(layers, axis=-1)  # (nx, ny, nz, 6), grid order
        bits = np.sum(
            exposed.reshape(-1, 6) * (1 << np.arange(6, dtype=np.uint8)), axis=1
        )
        return bits[self.cell_to_grid].astype(np.uint8)

    def __post_init__(self) -> None:
        yz = self.ny * self.nz
        grid = self.cell_to_grid
        ijk = np.empty((grid.size, 3), dtype=np.intp)
        ijk[:, 0] = grid // yz
        ijk[:, 1] = (grid % yz) // self.nz
        ijk[:, 2] = grid % self.nz
        object.__setattr__(self, "_ijk", ijk)
        object.__setattr__(self, "_grid3d", self.grid_to_cell.reshape(self.nx, self.ny, self.nz))
        object.__setattr__(self, "_exposed_face_mask", self._compute_exposed_face_mask())
        for value in self.__dict__.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)


def _operators_from_handle(dll, handle) -> Operators:
    """Copy a native operator handle into Python-owned SciPy/NumPy storage."""
    import scipy.sparse

    try:
        info = MhsOperatorsInfo()
        check(dll.mhs_operators_get_info(handle, ctypes.byref(info)), "operators_info")
        n = int(info.state_count)

        def copy_matrix(function, nnz: int):
            outer = np.empty(n + 1, dtype=np.int32)
            inner = np.empty(nnz, dtype=np.int32)
            values = np.empty(nnz, dtype=np.float64)
            check(
                function(
                    handle,
                    outer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    outer.size,
                    inner.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    inner.size,
                    values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    values.size,
                ),
                "operators_copy",
            )
            return scipy.sparse.csc_matrix((values, inner, outer), shape=(n, n))

        rhs = np.empty(n, dtype=np.float64)
        check(
            dll.mhs_operators_copy_rhs(
                handle,
                rhs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                rhs.size,
            ),
            "operators_copy_rhs",
        )
        return Operators(
            copy_matrix(dll.mhs_operators_copy_k, int(info.k_nnz)),
            copy_matrix(dll.mhs_operators_copy_c, int(info.c_nnz)),
            rhs,
        )
    finally:
        dll.mhs_operators_destroy(handle)


@dataclass
class SolveOptions:
    """Solver configuration options."""

    linear_solver: str = "AmgCg"
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
            "AmgCg": 1,
        }.get(self.linear_solver, 1)
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
        self._info = None
        self._cells = None

    @classmethod
    def _from_model(cls, dll, model_handle) -> Compiled:
        self = cls()
        self._dll = dll
        self._destroy_fn = dll.mhs_compiled_destroy
        handle = ctypes.POINTER(MhsCompiled)()
        check(dll.mhs_model_compile(model_handle, ctypes.byref(handle)), "compile")
        self._handle = handle
        self._fetch_metadata()
        return self

    def _fetch_metadata(self) -> CellFields:
        if self._cells is None:
            info = MhsCompiledInfo()
            check(
                self._dll.mhs_compiled_get_info(self._handle, ctypes.byref(info)),
                "compiled_info",
            )
            arrays = {
                name: np.empty(int(getattr(info, count)), dtype=dtype)
                for name, _, dtype, count in _CELL_FIELD_SPECS
            }
            native = MhsCellFields(
                **{
                    name: arrays[name].ctypes.data_as(ctypes.POINTER(ctype))
                    for name, ctype, _, _ in _CELL_FIELD_SPECS
                },
                **{
                    count: arrays[name].size
                    for name, _, _, count in _CELL_FIELD_SPECS
                },
            )
            check(
                self._dll.mhs_compiled_copy_cell_fields(
                    self._handle, ctypes.byref(native)
                ),
                "cell_fields",
            )
            self._info = info
            self._cells = CellFields(**arrays)
        return self._cells

    @property
    def cell_count(self) -> int:
        self._fetch_metadata()
        return int(self._info.cell_count)

    @property
    def study_type(self) -> int:
        self._fetch_metadata()
        return int(self._info.study_type)

    @property
    def initial_temperature(self) -> float:
        self._fetch_metadata()
        return float(self._info.initial_temperature)

    @property
    def nx(self) -> int:
        self._fetch_metadata()
        return int(self._info.nx)

    @property
    def ny(self) -> int:
        self._fetch_metadata()
        return int(self._info.ny)

    @property
    def nz(self) -> int:
        self._fetch_metadata()
        return int(self._info.nz)

    @property
    def cells(self) -> CellFields:
        return self._fetch_metadata()

    def eval_materials(self, state: np.ndarray | None = None, time: float = 0.0):
        """Evaluate material laws for every compact cell at ``state`` and ``time``."""
        if state is None:
            state = self.default_state()
        state = np.ascontiguousarray(state, dtype=np.float64)
        if state.size != self.cell_count:
            raise ValueError(f"state size ({state.size}) != cell_count ({self.cell_count})")
        values = {name: np.empty(state.size, dtype=np.float64) for name in (
            "conductivity_x", "conductivity_y", "conductivity_z", "density", "specific_heat"
        )}
        native = MhsMaterialValues(
            **{name: values[name].ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for name in values},
            count=state.size,
        )
        check(
            self._dll.mhs_compiled_eval_materials(
                self._handle,
                state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                state.size,
                time,
                ctypes.byref(native),
            ),
            "eval_materials",
        )
        return values

    def default_state(self) -> np.ndarray:
        return np.full(self.cell_count, self.initial_temperature, dtype=np.float64)

    def assemble(self, state: np.ndarray | None = None, time: float = 0.0) -> Operators:
        """Assemble K, C, f at a state and time."""
        if state is None:
            state = self.default_state()
        state = np.ascontiguousarray(state, dtype=np.float64)
        if state.size != self.cell_count:
            raise ValueError(
                f"state size ({state.size}) != cell_count ({self.cell_count})"
            )

        handle = ctypes.POINTER(MhsOperators)()
        check(
            self._dll.mhs_compiled_assemble(
                self._handle,
                state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                state.size,
                time,
                ctypes.byref(handle),
            ),
            "assemble",
        )

        return _operators_from_handle(self._dll, handle)

    def solve(
        self,
        state: np.ndarray | None = None,
        opts: SolveOptions | None = None,
    ):
        from metahotspot.solution import Solution

        return Solution._solve_compiled(self, state, opts)
