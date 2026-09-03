"""Low-level ctypes adapter for the public Python objects.

Only this module touches ``ctypes`` and the C ABI details.  The high-level
classes in ``metahotspot/`` deal in plain Python values and NumPy/SciPy data.
"""

from __future__ import annotations

import ctypes

import numpy as np

from metahotspot._error import check
from metahotspot.types import (
    MhsCellFields,
    MhsCompiled,
    MhsCompiledInfo,
    MhsFaceRegion,
    MhsMaterialValues,
    MhsModel,
    MhsOperators,
    MhsOperatorsInfo,
    MhsSolution,
    MhsSolutionInfo,
    Point2D,
    Rect2D,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _text(value: str | None) -> bytes | None:
    return None if value is None else value.encode("utf-8")


def _double_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _int32_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def create_model(dll):
    handle = ctypes.POINTER(MhsModel)()
    check(dll.mhs_model_create(ctypes.byref(handle)), "create")
    return handle


def set_settings(
    dll, handle, study, length_unit, initial_temperature, duration, output_interval
) -> None:
    check(
        dll.mhs_model_set_settings(
            handle,
            int(study),
            int(length_unit),
            initial_temperature,
            duration,
            output_interval,
        ),
        "set_settings",
    )


def call(dll, name: str, handle, *args) -> None:
    args = tuple(_text(arg) if isinstance(arg, str) else arg for arg in args)
    check(getattr(dll, name)(handle, *args), name.removeprefix("mhs_model_"))


def set_mesh(dll, handle, x, y, z) -> None:
    arrays = (x, y, z)
    args = []
    for array in arrays:
        args.extend(
            (
                0 if array is None else len(array),
                None if array is None else _double_ptr(array),
            )
        )
    check(dll.mhs_model_set_mesh(handle, *args), "set_mesh")


def add_id(dll, name: str, handle, *args) -> int:
    identifier = ctypes.c_uint32()
    call_args = tuple(_text(arg) if isinstance(arg, str) else arg for arg in args)
    check(
        getattr(dll, name)(handle, *call_args, ctypes.byref(identifier)),
        name.removeprefix("mhs_model_"),
    )
    return identifier.value


def face_regions(regions) -> tuple[object, int]:
    values = []
    for axis, coordinate, a_min, a_max, b_min, b_max in regions or ():
        values.append(
            MhsFaceRegion(int(axis), coordinate, Rect2D(a_min, a_max, b_min, b_max))
        )
    return ((MhsFaceRegion * len(values))(*values) if values else None), len(values)


def add_boundary(dll, name: str, handle, regions, *expressions) -> None:
    native_regions, count = face_regions(regions)
    args = tuple(_text(expression) for expression in expressions)
    check(
        getattr(dll, name)(handle, native_regions, count, *args),
        name.removeprefix("mhs_model_"),
    )


def add_piecewise(dll, handle, name: str, points: np.ndarray) -> None:
    native = (Point2D * len(points))(*[Point2D(x, y) for x, y in points])
    check(
        dll.mhs_model_add_function_piecewise(handle, _text(name), native, len(points)),
        "add_function_piecewise",
    )


def add_periodic_constant(
    dll, handle, name: str, values: np.ndarray, period: float
) -> None:
    native = (ctypes.c_double * len(values))(*values)
    check(
        dll.mhs_model_add_function_periodic_piecewise_constant(
            handle, _text(name), native, len(values), period
        ),
        "add_function_periodic_piecewise_constant",
    )


def add_fluid_boundary(
    dll, handle, axis, coordinate, region, kind, value, inlet_temperature
) -> None:
    rect = Rect2D(*region)
    check(
        dll.mhs_model_add_fluid_boundary(
            handle, int(axis), coordinate, rect, int(kind), value, inlet_temperature
        ),
        "add_fluid_boundary",
    )


# ---------------------------------------------------------------------------
# Compiled model
# ---------------------------------------------------------------------------

# (field name, ctypes element type, numpy dtype, count source on MhsCompiledInfo)
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


def compile_model(dll, model_handle):
    handle = ctypes.POINTER(MhsCompiled)()
    check(dll.mhs_model_compile(model_handle, ctypes.byref(handle)), "compile")
    return handle


def compiled_metadata(dll, handle) -> tuple[dict, dict]:
    """Return ``(info, cell_fields)`` as plain dicts of Python values/arrays."""
    info = MhsCompiledInfo()
    check(dll.mhs_compiled_get_info(handle, ctypes.byref(info)), "compiled_info")

    arrays = {
        name: np.empty(int(getattr(info, count)), dtype=dtype)
        for name, _, dtype, count in _CELL_FIELD_SPECS
    }
    native = MhsCellFields(
        **{
            name: arrays[name].ctypes.data_as(ctypes.POINTER(ctype))
            for name, ctype, _, _ in _CELL_FIELD_SPECS
        },
        **{count: arrays[name].size for name, _, _, count in _CELL_FIELD_SPECS},
    )
    check(
        dll.mhs_compiled_copy_cell_fields(handle, ctypes.byref(native)), "cell_fields"
    )

    info_dict = {
        "cell_count": int(info.cell_count),
        "grid_count": int(info.grid_count),
        "study_type": int(info.study_type),
        "initial_temperature": float(info.initial_temperature),
        "nx": int(info.nx),
        "ny": int(info.ny),
        "nz": int(info.nz),
    }
    return info_dict, arrays


def eval_materials(dll, handle, state: np.ndarray, time: float) -> dict:
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
    return values


def assembled_operators(dll, handle, state: np.ndarray, time: float):
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


# ---------------------------------------------------------------------------
# Solve options and solution
# ---------------------------------------------------------------------------


def solve_options(dll, overrides: dict):
    """Build a C solve-options struct with *overrides* applied over defaults.

    Returns the struct object; callers must keep it alive until the matching
    solve call completes.
    """
    from metahotspot.types import _SolveOptionsCStruct

    opts = _SolveOptionsCStruct()
    dll.mhs_solve_options_default(ctypes.byref(opts))
    for name, value in overrides.items():
        setattr(opts, name, value)
    return opts


def solve(dll, compiled_handle, state, opts_overrides: dict):
    opts = solve_options(dll, opts_overrides)
    state_ptr = _double_ptr(state) if state is not None else None
    state_count = state.size if state is not None else 0
    out = ctypes.POINTER(MhsSolution)()
    check(
        dll.mhs_compiled_solve(
            compiled_handle,
            state_ptr,
            state_count,
            ctypes.byref(opts),
            ctypes.byref(out),
        ),
        "solve",
    )
    return out


def solution_snapshot(dll, handle) -> dict:
    """Snapshot a native solution into Python-owned arrays."""
    info = MhsSolutionInfo()
    check(dll.mhs_solution_get_info(handle, ctypes.byref(info)), "solution_info")

    state = np.empty(int(info.state_count), dtype=np.float64)
    history_times = np.empty(int(info.record_count), dtype=np.float64)
    history_states = np.empty(
        int(info.record_count) * int(info.state_count), dtype=np.float64
    )
    check(
        dll.mhs_solution_copy_state(handle, _double_ptr(state), state.size),
        "solution_copy_state",
    )
    check(
        dll.mhs_solution_copy_history_times(
            handle, _double_ptr(history_times), history_times.size
        ),
        "solution_copy_history_times",
    )
    check(
        dll.mhs_solution_copy_history_states(
            handle, _double_ptr(history_states), history_states.size
        ),
        "solution_copy_history_states",
    )

    probes = []
    for index in range(int(info.probe_count)):
        name_size = ctypes.c_size_t()
        record_count = ctypes.c_size_t()
        check(
            dll.mhs_solution_probe_get_info(
                handle, index, ctypes.byref(name_size), ctypes.byref(record_count)
            ),
            "solution_probe_info",
        )
        name = ctypes.create_string_buffer(name_size.value)
        times = np.empty(record_count.value, dtype=np.float64)
        values = np.empty(record_count.value, dtype=np.float64)
        check(
            dll.mhs_solution_copy_probe(
                handle,
                index,
                name,
                name_size.value,
                _double_ptr(times),
                _double_ptr(values),
                record_count.value,
            ),
            "solution_copy_probe",
        )
        probes.append((name.value.decode("utf-8"), times, values))

    return {
        "time": float(info.time),
        "fvm_count": int(info.fvm_count),
        "state": state,
        "history_times": history_times,
        "state_history": history_states.reshape(
            int(info.record_count), int(info.state_count)
        ),
        "probes": probes,
    }


def write_vtu(dll, handle, path: str) -> None:
    check(dll.mhs_solution_write_vtu(handle, _text(path)), "write_vtu")
