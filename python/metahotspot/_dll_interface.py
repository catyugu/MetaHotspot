"""Configure ctypes function signatures on a loaded CDLL object.

This module is internal — call ``configure_dll(dll)`` once after loading.
"""

from __future__ import annotations

import ctypes

from metahotspot.types import (
    MhsModel,
    MhsCompiled,
    MhsSolution,
    MhsAssembly,
    MhsAssemblyView,
    ModalPortView,
    CscView,
    SolverOpts,
    Rect2D,
    Point2D,
    MhsFaceRegion,
    CompiledMetadataView,
    SolutionView,
    ProbeView,
)

# (name, restype, argtypes) table — drives configure_dll().
# Pass-by-value structs (Rect2D, ...) appear directly in argtypes.
_FUNC_SIGS: list[tuple[str, type | None, list]] = [
    # ---- Global helpers ----
    ("mhs_solver_opts_default", None, [ctypes.POINTER(SolverOpts)]),
    ("mhs_status_string", ctypes.c_char_p, [ctypes.c_int32]),
    ("mhs_last_error", ctypes.c_char_p, []),
    # ---- Model life-cycle ----
    ("mhs_model_create", ctypes.c_int32, [ctypes.POINTER(ctypes.POINTER(MhsModel))]),
    ("mhs_model_destroy", ctypes.c_int32, [ctypes.POINTER(MhsModel)]),
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
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_char_p,
            ctypes.c_char_p,
        ],
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
        ctypes.c_uint32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ],
    ),
    (
        "mhs_model_add_block",
        ctypes.c_uint32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
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
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_char_p,
        ],
    ),
    (
        "mhs_model_set_default_neumann",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_char_p,
        ],
    ),
    (
        "mhs_model_set_default_convection",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_char_p,
            ctypes.c_char_p,
        ],
    ),
    # ---- Functions ----
    (
        "mhs_model_add_function_expr",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsModel),
            ctypes.c_char_p,
            ctypes.c_char_p,
        ],
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
        [
            ctypes.POINTER(MhsModel),
            ctypes.POINTER(ctypes.POINTER(MhsCompiled)),
        ],
    ),
    ("mhs_compiled_destroy", ctypes.c_int32, [ctypes.POINTER(MhsCompiled)]),
    # ---- Compiled metadata ----
    (
        "mhs_compiled_metadata",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsCompiled),
            ctypes.POINTER(CompiledMetadataView),
        ],
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
            ctypes.POINTER(ctypes.POINTER(MhsAssembly)),
        ],
    ),
    ("mhs_assembly_destroy", ctypes.c_int32, [ctypes.POINTER(MhsAssembly)]),
    (
        "mhs_assembly_view",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsAssembly),
            ctypes.POINTER(MhsAssemblyView),
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
            ctypes.POINTER(SolverOpts),
            ctypes.POINTER(ctypes.POINTER(MhsSolution)),
        ],
    ),
    (
        "mhs_compiled_solve_modal_port",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsCompiled),
            ctypes.POINTER(ModalPortView),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(SolverOpts),
            ctypes.POINTER(ctypes.POINTER(MhsSolution)),
        ],
    ),
    ("mhs_solution_destroy", ctypes.c_int32, [ctypes.POINTER(MhsSolution)]),
    # ---- VTU ----
    (
        "mhs_compiled_write_vtu",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsCompiled),
            ctypes.POINTER(MhsSolution),
            ctypes.c_char_p,
        ],
    ),
    # ---- Solution view ----
    (
        "mhs_solution_view",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsSolution),
            ctypes.POINTER(SolutionView),
        ],
    ),
    # ---- Probe accessors ----
    ("mhs_solution_probe_count", ctypes.c_size_t, [ctypes.POINTER(MhsSolution)]),
    (
        "mhs_solution_probe_view",
        ctypes.c_int32,
        [
            ctypes.POINTER(MhsSolution),
            ctypes.c_size_t,
            ctypes.POINTER(ProbeView),
        ],
    ),
]


def configure_dll(dll: ctypes.CDLL) -> None:
    """Set *argtypes* and *restype* on every function in *dll*."""
    for name, restype, argtypes in _FUNC_SIGS:
        fn = getattr(dll, name)
        fn.restype = restype
        fn.argtypes = argtypes
