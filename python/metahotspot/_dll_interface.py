"""Configure ctypes function signatures on a loaded CDLL object.

This module is internal — call ``configure_dll(dll)`` once after loading.
The macromodel extension DLL is configured separately by
:mod:`metahotspot.macromodel` (its signatures are hand-registered there).
"""

from __future__ import annotations

import ctypes

from metahotspot.types import (
    MhsModel,
    MhsCompiled,
    MhsSolution,
    MhsOperators,
    Rect2D,
    Point2D,
    MhsFaceRegion,
    MhsCompiledInfo,
    MhsOperatorsInfo,
    MhsSolutionInfo,
    _SolveOptionsCStruct,
)

# ---- Core C API function signatures ----
# (name, restype, argtypes) table — drives configure_dll().
_CORE_FUNC_SIGS: list[tuple[str, type | None, list]] = [
    # ---- Global helpers ----
    ("mhs_solve_options_default", None, [ctypes.POINTER(_SolveOptionsCStruct)]),
    ("mhs_last_error", ctypes.c_char_p, []),
    # ---- Model life-cycle ----
    ("mhs_model_create", ctypes.c_int32, [ctypes.POINTER(ctypes.POINTER(MhsModel))]),
    ("mhs_model_destroy", None, [ctypes.POINTER(MhsModel)]),
    ("mhs_model_read_xml", ctypes.c_int32, [ctypes.POINTER(MhsModel), ctypes.c_char_p]),
    # ---- Settings / mesh / variables ----
    (
        "mhs_model_set_settings",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ],
    ),
    (
        "mhs_model_set_mesh",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
        ],
    ),
    (
        "mhs_model_add_variable",
        ctypes.c_int32,
        [ctypes.POINTER(MhsModel), ctypes.c_char_p, ctypes.c_char_p],
    ),
    # ---- Materials / layers / blocks / rects ----
    (
        "mhs_model_add_material",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ],
    ),
    (
        "mhs_model_add_layer",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint32),
        ],
    ),
    (
        "mhs_model_add_block",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint32),
        ],
    ),
    (
        "mhs_model_add_rect",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ],
    ),
    # ---- Atomic boundary conditions ----
    (
        "mhs_model_add_dirichlet",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.POINTER(MhsFaceRegion),
            ctypes.c_size_t,
            ctypes.c_char_p,
        ],
    ),
    (
        "mhs_model_add_neumann",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.POINTER(MhsFaceRegion),
            ctypes.c_size_t,
            ctypes.c_char_p,
        ],
    ),
    (
        "mhs_model_add_convection",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.POINTER(MhsFaceRegion),
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ],
    ),
    (
        "mhs_model_set_default_dirichlet",
        ctypes.c_int32,
        [ctypes.POINTER(MhsModel), ctypes.c_char_p],
    ),
    (
        "mhs_model_set_default_neumann",
        ctypes.c_int32,
        [ctypes.POINTER(MhsModel), ctypes.c_char_p],
    ),
    (
        "mhs_model_set_default_convection",
        ctypes.c_int32,
        [ctypes.POINTER(MhsModel), ctypes.c_char_p, ctypes.c_char_p],
    ),
    # ---- Functions ----
    (
        "mhs_model_add_function_expr",
        ctypes.c_int32,
        [ctypes.POINTER(MhsModel), ctypes.c_char_p, ctypes.c_char_p],
    ),
    (
        "mhs_model_add_function_gauss",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_char_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ],
    ),
    (
        "mhs_model_add_function_sine",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_char_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ],
    ),
    (
        "mhs_model_add_function_double_exponential",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_char_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ],
    ),
    (
        "mhs_model_add_function_piecewise",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_char_p,
            ctypes.POINTER(Point2D),
            ctypes.c_size_t,
        ],
    ),
    (
        "mhs_model_add_function_periodic_piecewise_constant",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
        ],
    ),
    # ---- Probes & fluid boundaries ----
    (
        "mhs_model_add_probe",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_char_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ],
    ),
    (
        "mhs_model_add_fluid_boundary",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_int32,
            ctypes.c_double,
            Rect2D,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
        ],
    ),
    # ---- Compilation ----
    (
        "mhs_model_compile",
        ctypes.c_int32,
        [ctypes.POINTER(MhsModel), ctypes.POINTER(ctypes.POINTER(MhsCompiled))],
    ),
    ("mhs_compiled_destroy", None, [ctypes.POINTER(MhsCompiled)]),
    # ---- Compiled metadata ----
    (
        "mhs_compiled_get_info",
        ctypes.c_int32,
        [ctypes.POINTER(MhsCompiled), ctypes.POINTER(MhsCompiledInfo)],
    ),
    (
        "mhs_compiled_copy_grid_to_cell",
        ctypes.c_int32,
        [ctypes.POINTER(MhsCompiled), ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t],
    ),
    (
        "mhs_compiled_copy_layer_ids",
        ctypes.c_int32,
        [ctypes.POINTER(MhsCompiled), ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t],
    ),
    (
        "mhs_compiled_copy_block_ids",
        ctypes.c_int32,
        [ctypes.POINTER(MhsCompiled), ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t],
    ),
    # ---- Assembly ----
    (
        "mhs_compiled_assemble",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsCompiled),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.POINTER(ctypes.POINTER(MhsOperators)),
        ],
    ),
    ("mhs_operators_destroy", None, [ctypes.POINTER(MhsOperators)]),
    (
        "mhs_operators_get_info",
        ctypes.c_int32,
        [ctypes.POINTER(MhsOperators), ctypes.POINTER(MhsOperatorsInfo)],
    ),
    (
        "mhs_operators_copy_k",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsOperators),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ],
    ),
    (
        "mhs_operators_copy_c",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsOperators),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ],
    ),
    (
        "mhs_operators_copy_rhs",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsOperators),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ],
    ),
    (
        "mhs_operators_create",
        ctypes.c_int32,
        [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.POINTER(MhsOperators)),
        ],
    ),
    # ---- Half-conductance ----
    (
        "mhs_compiled_half_conductance",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsCompiled),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ],
    ),
    # ---- Solve ----
    (
        "mhs_compiled_solve",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsCompiled),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(_SolveOptionsCStruct),
            ctypes.POINTER(ctypes.POINTER(MhsSolution)),
        ],
    ),
    ("mhs_solution_destroy", None, [ctypes.POINTER(MhsSolution)]),
    # ---- VTU ----
    (
        "mhs_solution_write_vtu",
        ctypes.c_int32,
        [ctypes.POINTER(MhsSolution), ctypes.c_char_p],
    ),
    # ---- Solution copy-out accessors ----
    (
        "mhs_solution_get_info",
        ctypes.c_int32,
        [ctypes.POINTER(MhsSolution), ctypes.POINTER(MhsSolutionInfo)],
    ),
    (
        "mhs_solution_copy_state",
        ctypes.c_int32,
        [ctypes.POINTER(MhsSolution), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t],
    ),
    (
        "mhs_solution_copy_history_times",
        ctypes.c_int32,
        [ctypes.POINTER(MhsSolution), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t],
    ),
    (
        "mhs_solution_copy_history_states",
        ctypes.c_int32,
        [ctypes.POINTER(MhsSolution), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t],
    ),
    (
        "mhs_solution_probe_get_info",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsSolution),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ],
    ),
    (
        "mhs_solution_copy_probe",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsSolution),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ],
    ),
]


def configure_dll(dll: ctypes.CDLL) -> None:
    """Set *argtypes* and *restype* on every core C API function in *dll*."""
    for name, restype, argtypes in _CORE_FUNC_SIGS:
        fn = getattr(dll, name)
        fn.restype = restype
        fn.argtypes = argtypes


def copy_array(fn, handle, array, c_type, label) -> None:
    """Copy a native array out of *handle* into the caller-owned NumPy *array*.

    Wraps the simple ``mhs_*_copy_*`` C calls, which take
    ``(handle, buffer, count)`` and fill the pre-allocated buffer.
    """
    from metahotspot._error import check

    check(
        fn(handle, array.ctypes.data_as(ctypes.POINTER(c_type)), array.size),
        label,
    )


def _opts_ptr(opts, dll):
    """Return a ctypes pointer to the C struct form of *opts* (or None).

    *opts* is a :class:`SolveOptions`; anything already C-struct shaped is
    passed through unchanged.
    """
    if opts is None:
        return None
    c_opts = opts._to_c_struct(dll) if hasattr(opts, "_to_c_struct") else opts
    return ctypes.byref(c_opts)
