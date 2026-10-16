"""High-level wrapper for ``mhs_compiled_t`` — compiled runtime model."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, fields
from functools import cached_property

import numpy as np

from metahotspot._compiled_data import CellFields, Operators
from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot.enums import IntegratorKind, SolverType, StepStrategy
from metahotspot.types import (
    MhsCellFields,
    MhsCompiled,
    MhsCompiledInfo,
    MhsMaterialValues,
    MhsOperators,
    MhsOperatorsInfo,
)


# ---- SolveOptions --------------------------------------------------------

_SOLVER_VALUES = {"Pardiso": SolverType.PARDISO, "AmgCg": SolverType.AMG}
_INTEGRATOR_VALUES = {"Bdf1": IntegratorKind.BDF1, "Bdf2": IntegratorKind.BDF2}
_STEP_STRATEGY_VALUES = {"Adaptive": StepStrategy.ADAPTIVE, "Fixed": StepStrategy.FIXED}

# SolveOptions dataclass field -> C struct field name, with an enum coercer for
# the three enum-typed fields.
_ENUM_FIELDS = {
    "linear_solver": (SolverType, _SOLVER_VALUES, "solver_type"),
    "integrator": (IntegratorKind, _INTEGRATOR_VALUES, "integrator"),
    "step_strategy": (StepStrategy, _STEP_STRATEGY_VALUES, "step_strategy"),
}


def _enum_value(value, enum_type, string_values, field_name: str) -> int:
    if isinstance(value, enum_type):
        return int(value)
    if isinstance(value, str):
        try:
            return int(string_values[value])
        except KeyError as exc:
            raise ValueError(f"unknown {field_name}: {value!r}") from exc
    raise TypeError(f"{field_name} must be {enum_type.__name__} or str")


@dataclass
class SolveOptions:
    """Solver configuration options.

    ``None`` fields fall back to the C++ defaults (via ``mhs_solve_options_default``).
    """

    linear_solver: SolverType | str | None = None
    linear_tolerance: float | None = None
    linear_max_iterations: int | None = None
    underrelaxation: float | None = None
    nonlinear_max_iterations: int | None = None
    nonlinear_relative_tolerance: float | None = None
    nonlinear_absolute_tolerance: float | None = None
    integrator: IntegratorKind | str | None = None
    step_strategy: StepStrategy | str | None = None
    error_rel_tol: float | None = None
    error_safety: float | None = None
    min_dt: float | None = None
    max_dt: float | None = None
    fixed_dt: float | None = None

    def _overrides(self) -> dict:
        """Present non-``None`` values as C solve-options struct field overrides."""
        overrides = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if f.name in _ENUM_FIELDS:
                enum_type, string_values, c_name = _ENUM_FIELDS[f.name]
                overrides[c_name] = _enum_value(value, enum_type, string_values, f.name)
            else:
                overrides[f.name] = value
        return overrides


# ---- compiled metadata / snapshot helpers --------------------------------


@dataclass(frozen=True)
class MaterialValues:
    conductivity_x: np.ndarray
    conductivity_y: np.ndarray
    conductivity_z: np.ndarray
    density: np.ndarray
    specific_heat: np.ndarray


@dataclass(frozen=True)
class CompiledMetadata:
    cell_count: int
    face_count: int
    state_count: int
    projection_count: int
    source_count: int
    study_type: int
    initial_temperature: float
    cell_fields: CellFields


def _double_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _int32_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


# (field name, ctypes element type, numpy dtype, count source on MhsCompiledInfo)
_CELL_FIELD_SPECS = (
    ("cell_min", ctypes.c_double, np.float64, "cell_count", 3),
    ("cell_max", ctypes.c_double, np.float64, "cell_count", 3),
    ("material_id", ctypes.c_uint32, np.uint32, "cell_count", 1),
    ("face_owner", ctypes.c_size_t, np.uintp, "face_count", 1),
    ("face_neighbor", ctypes.c_size_t, np.uintp, "face_count", 1),
    ("face_axis", ctypes.c_int32, np.int32, "face_count", 1),
    ("face_sign", ctypes.c_int32, np.int32, "face_count", 1),
    ("face_min", ctypes.c_double, np.float64, "face_count", 3),
    ("face_max", ctypes.c_double, np.float64, "face_count", 3),
    ("face_area", ctypes.c_double, np.float64, "face_count", 1),
    ("owner_distance", ctypes.c_double, np.float64, "face_count", 1),
    ("neighbor_distance", ctypes.c_double, np.float64, "face_count", 1),
    ("projection_cell", ctypes.c_size_t, np.uintp, "projection_count", 1),
    ("projection_source", ctypes.c_uint32, np.uint32, "projection_count", 1),
    ("projection_volume", ctypes.c_double, np.float64, "projection_count", 1),
    ("projection_center", ctypes.c_double, np.float64, "projection_count", 3),
)


def _compile_model(dll, model_handle):
    handle = ctypes.POINTER(MhsCompiled)()
    check(dll.mhs_model_compile(model_handle, ctypes.byref(handle)), "compile")
    return handle


def _compiled_metadata(dll, handle) -> CompiledMetadata:
    info = MhsCompiledInfo()
    check(dll.mhs_compiled_get_info(handle, ctypes.byref(info)), "compiled_info")

    arrays = {
        name: np.empty(
            (int(getattr(info, count)), 3) if dim == 3 else int(getattr(info, count)),
            dtype=dtype,
        )
        for name, _, dtype, count, dim in _CELL_FIELD_SPECS
    }
    native = MhsCellFields(
        cell_count=info.cell_count,
        face_count=info.face_count,
        projection_count=info.projection_count,
        **{
            name: arrays[name].ctypes.data_as(ctypes.POINTER(ctype))
            for name, ctype, _, _, _ in _CELL_FIELD_SPECS
        },
    )
    check(
        dll.mhs_compiled_copy_cell_fields(handle, ctypes.byref(native)), "cell_fields"
    )
    return CompiledMetadata(
        cell_count=int(info.cell_count),
        face_count=int(info.face_count),
        state_count=int(info.state_count),
        projection_count=int(info.projection_count),
        source_count=int(info.source_count),
        study_type=int(info.study_type),
        initial_temperature=float(info.initial_temperature),
        cell_fields=CellFields(**arrays),
    )


def _eval_materials(dll, handle, state: np.ndarray, time: float) -> MaterialValues:
    values = {
        name: np.empty(state.size, dtype=np.float64)
        for name in (
            "conductivity_x",
            "conductivity_y",
            "conductivity_z",
            "density",
            "specific_heat",
        )
    }
    native = MhsMaterialValues(
        **{
            name: values[name].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            for name in values
        },
        count=state.size,
    )
    check(
        dll.mhs_compiled_eval_materials(
            handle, _double_ptr(state), state.size, time, ctypes.byref(native)
        ),
        "eval_materials",
    )
    return MaterialValues(**values)


def _assembled_operators(dll, handle, state: np.ndarray, time: float):
    """Assemble and copy ``K, C, f`` at ``state``/``time`` into SciPy/NumPy."""
    import scipy.sparse

    out = ctypes.POINTER(MhsOperators)()
    check(
        dll.mhs_compiled_assemble(
            handle, _double_ptr(state), state.size, time, ctypes.byref(out)
        ),
        "assemble",
    )
    try:
        info = MhsOperatorsInfo()
        check(dll.mhs_operators_get_info(out, ctypes.byref(info)), "operators_info")
        n = int(info.state_count)

        def copy_matrix(function, nnz: int):
            outer = np.empty(n + 1, dtype=np.int32)
            inner = np.empty(nnz, dtype=np.int32)
            values = np.empty(nnz, dtype=np.float64)
            check(
                function(
                    out,
                    _int32_ptr(outer),
                    outer.size,
                    _int32_ptr(inner),
                    inner.size,
                    _double_ptr(values),
                    values.size,
                ),
                "operators_copy",
            )
            return scipy.sparse.csc_matrix((values, inner, outer), shape=(n, n))

        rhs = np.empty(n, dtype=np.float64)
        check(
            dll.mhs_operators_copy_rhs(out, _double_ptr(rhs), rhs.size),
            "operators_copy_rhs",
        )
        return (
            copy_matrix(dll.mhs_operators_copy_k, int(info.k_nnz)),
            copy_matrix(dll.mhs_operators_copy_c, int(info.c_nnz)),
            rhs,
        )
    finally:
        dll.mhs_operators_destroy(out)


# ---- public wrapper ------------------------------------------------------


class Compiled(OwnedHandle):
    """Read-only compiled runtime model. Use ``Model.compile()``."""

    def __init__(self, dll, handle) -> None:
        super().__init__(dll, handle, dll.mhs_compiled_destroy)

    @classmethod
    def _from_model(cls, dll, model_handle) -> Compiled:
        handle = _compile_model(dll, model_handle)
        return cls(dll, handle)

    @cached_property
    def metadata(self) -> CompiledMetadata:
        return _compiled_metadata(self._dll, self._handle)

    @property
    def cell_count(self) -> int:
        return self.metadata.cell_count

    @property
    def cells(self) -> CellFields:
        return self.metadata.cell_fields

    def source_operator(self, state=None, time=0.0):
        """Per-source integrated powers in algebraic state order (sparse CSC).

        Columns retain physical source identity even when regions share a cell.
        """
        import scipy.sparse as sp

        if state is None:
            state = self.default_state()
        state = np.ascontiguousarray(state, dtype=np.float64)
        values = np.empty(self.metadata.projection_count, dtype=np.float64)
        check(
            self._dll.mhs_compiled_eval_source_terms(
                self._handle,
                _double_ptr(state),
                state.size,
                time,
                _double_ptr(values),
                values.size,
            ),
            "eval_source_terms",
        )
        fields = self.cells
        result = sp.csc_matrix(
            (
                values,
                (fields.projection_cell, fields.projection_source),
            ),
            shape=(self.metadata.state_count, self.metadata.source_count),
        )

        result.eliminate_zeros()
        return result

    def eval_materials(self, state: np.ndarray | None = None, time: float = 0.0):
        """Evaluate material laws for every compact cell at ``state`` and ``time``."""
        if state is None:
            state = self.default_state()
        return _eval_materials(self._dll, self._handle, state, time)

    def default_state(self) -> np.ndarray:
        return np.full(
            self.cell_count, self.metadata.initial_temperature, dtype=np.float64
        )

    def assemble(self, state: np.ndarray | None = None, time: float = 0.0) -> Operators:
        """Assemble K, C, f at a state and time."""
        if state is None:
            state = self.default_state()
        return Operators(*_assembled_operators(self._dll, self._handle, state, time))

    def solve(
        self,
        state: np.ndarray | None = None,
        opts: SolveOptions | None = None,
    ):
        from metahotspot.solution import Solution

        return Solution._solve_compiled(self, state, opts)
