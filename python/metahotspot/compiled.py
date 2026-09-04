"""High-level wrapper for ``mhs_compiled_t`` — compiled runtime model."""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cached_property

import numpy as np

from metahotspot._compiled_data import CellFields, Operators
from metahotspot._handle import OwnedHandle
from metahotspot.enums import IntegratorKind, SolverType, StepStrategy
import metahotspot._native_compiled as _native_compiled


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


class Compiled(OwnedHandle):
    """Read-only compiled runtime model. Use ``Model.compile()``."""

    def __init__(self, dll, handle) -> None:
        super().__init__(dll, handle, dll.mhs_compiled_destroy)

    @classmethod
    def _from_model(cls, dll, model_handle) -> Compiled:
        handle = _native_compiled.compile_model(dll, model_handle)
        return cls(dll, handle)

    @cached_property
    def metadata(self) -> _native_compiled.CompiledMetadata:
        return _native_compiled.compiled_metadata(self._dll, self._handle)

    @property
    def cell_count(self) -> int:
        return self.metadata.cell_count

    @property
    def cells(self) -> CellFields:
        return self.metadata.cell_fields

    def eval_materials(self, state: np.ndarray | None = None, time: float = 0.0):
        """Evaluate material laws for every compact cell at ``state`` and ``time``."""
        if state is None:
            state = self.default_state()
        return _native_compiled.eval_materials(self._dll, self._handle, state, time)

    def default_state(self) -> np.ndarray:
        return np.full(
            self.cell_count, self.metadata.initial_temperature, dtype=np.float64
        )

    def assemble(self, state: np.ndarray | None = None, time: float = 0.0) -> Operators:
        """Assemble K, C, f at a state and time."""
        if state is None:
            state = self.default_state()
        return Operators(
            *_native_compiled.assembled_operators(self._dll, self._handle, state, time)
        )

    def solve(
        self,
        state: np.ndarray | None = None,
        opts: SolveOptions | None = None,
    ):
        from metahotspot.solution import Solution

        return Solution._solve_compiled(self, state, opts)
