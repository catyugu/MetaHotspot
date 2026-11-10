"""Configure ctypes function signatures on a loaded CDLL object.

This module is internal — call ``configure_dll(dll)`` once after loading.
"""

from __future__ import annotations

import ctypes

from metahotspot.types import (
    MhsCompiled,
    MhsCompiledFields,
    MhsCompiledInfo,
    MhsMaterialValues,
    MhsModel,
    MhsOperators,
    MhsOperatorsInfo,
    MhsSolution,
    MhsSolutionInfo,
    MhsStudy,
    MhsSurfaceRegion,
    Point2D,
    _SolveOptionsCStruct,
)

# Exact C ABI signatures; P is ctypes.POINTER, including opaque handle pointers.
P = ctypes.POINTER
I32 = ctypes.c_int32
F64 = ctypes.c_double
Size = ctypes.c_size_t
Text = ctypes.c_char_p

_CORE_FUNC_SIGS = [
    ("mhs_log_set_level", I32, [I32]),
    ("mhs_log_get_level", I32, []),
    ("mhs_study_default", None, [P(MhsStudy)]),
    ("mhs_solve_options_default", None, [P(_SolveOptionsCStruct)]),
    ("mhs_last_error", Text, []),
    ("mhs_version_string", Text, []),
    (
        "mhs_compiled_eval_source_terms",
        I32,
        [P(MhsCompiled), P(F64), Size, F64, P(F64), Size],
    ),
    ("mhs_model_create", I32, [P(P(MhsModel))]),
    ("mhs_model_destroy", None, [P(MhsModel)]),
    ("mhs_model_set_reference_temperature", I32, [P(MhsModel), F64]),
    (
        "mhs_model_set_cartesian_mesh",
        I32,
        [P(MhsModel), Size, P(F64), Size, P(F64), Size, P(F64)],
    ),
    (
        "mhs_model_add_material",
        I32,
        [
            P(MhsModel),
            Text,
            Text,
            Text,
            Text,
            Text,
            Text,
            I32,
            Text,
            P(ctypes.c_uint32),
        ],
    ),
    ("mhs_model_add_source", I32, [P(MhsModel), Text, P(ctypes.c_uint32)]),
    (
        "mhs_model_add_volume",
        I32,
        [P(MhsModel), ctypes.c_uint32, ctypes.c_uint32, P(ctypes.c_uint32)],
    ),
    (
        "mhs_model_volume_add_box",
        I32,
        [P(MhsModel), ctypes.c_uint32, I32, P(F64), P(F64)],
    ),
    ("mhs_model_add_dirichlet", I32, [P(MhsModel), P(MhsSurfaceRegion), Size, Text]),
    ("mhs_model_add_neumann", I32, [P(MhsModel), P(MhsSurfaceRegion), Size, Text]),
    (
        "mhs_model_add_convection",
        I32,
        [P(MhsModel), P(MhsSurfaceRegion), Size, Text, Text],
    ),
    ("mhs_model_set_default_dirichlet", I32, [P(MhsModel), Text]),
    ("mhs_model_set_default_neumann", I32, [P(MhsModel), Text]),
    ("mhs_model_set_default_convection", I32, [P(MhsModel), Text, Text]),
    ("mhs_model_add_function_expr", I32, [P(MhsModel), Text, Text]),
    ("mhs_model_add_function_gauss", I32, [P(MhsModel), Text, F64, F64, F64]),
    ("mhs_model_add_function_sine", I32, [P(MhsModel), Text, F64, F64, F64]),
    (
        "mhs_model_add_function_double_exponential",
        I32,
        [P(MhsModel), Text, F64, F64, F64],
    ),
    ("mhs_model_add_function_piecewise", I32, [P(MhsModel), Text, P(Point2D), Size]),
    (
        "mhs_model_add_function_periodic_piecewise_constant",
        I32,
        [P(MhsModel), Text, P(F64), Size, F64],
    ),
    ("mhs_model_add_point_probe", I32, [P(MhsModel), Text, P(F64)]),
    (
        "mhs_model_add_fluid_boundary",
        I32,
        [P(MhsModel), P(MhsSurfaceRegion), Size, I32, F64, P(F64)],
    ),
    ("mhs_model_compile", I32, [P(MhsModel), P(P(MhsCompiled))]),
    ("mhs_compiled_destroy", None, [P(MhsCompiled)]),
    ("mhs_compiled_get_info", I32, [P(MhsCompiled), P(MhsCompiledInfo)]),
    ("mhs_compiled_copy_fields", I32, [P(MhsCompiled), P(MhsCompiledFields)]),
    (
        "mhs_compiled_eval_materials",
        I32,
        [P(MhsCompiled), P(F64), Size, F64, P(MhsMaterialValues)],
    ),
    (
        "mhs_compiled_assemble",
        I32,
        [P(MhsCompiled), P(F64), Size, F64, P(P(MhsOperators))],
    ),
    ("mhs_operators_destroy", None, [P(MhsOperators)]),
    ("mhs_operators_get_info", I32, [P(MhsOperators), P(MhsOperatorsInfo)]),
    (
        "mhs_operators_copy_k",
        I32,
        [P(MhsOperators), P(I32), Size, P(I32), Size, P(F64), Size],
    ),
    (
        "mhs_operators_copy_c",
        I32,
        [P(MhsOperators), P(I32), Size, P(I32), Size, P(F64), Size],
    ),
    ("mhs_operators_copy_rhs", I32, [P(MhsOperators), P(F64), Size]),
    (
        "mhs_compiled_solve",
        I32,
        [
            P(MhsCompiled),
            P(MhsStudy),
            P(F64),
            Size,
            P(_SolveOptionsCStruct),
            P(P(MhsSolution)),
        ],
    ),
    ("mhs_solution_destroy", None, [P(MhsSolution)]),
    ("mhs_solution_get_info", I32, [P(MhsSolution), P(MhsSolutionInfo)]),
    ("mhs_solution_copy_state", I32, [P(MhsSolution), P(F64), Size]),
    ("mhs_solution_copy_history_times", I32, [P(MhsSolution), P(F64), Size]),
    ("mhs_solution_copy_history_states", I32, [P(MhsSolution), P(F64), Size]),
    ("mhs_solution_probe_get_info", I32, [P(MhsSolution), Size, P(Size), P(Size)]),
    (
        "mhs_solution_copy_probe",
        I32,
        [P(MhsSolution), Size, P(ctypes.c_char), Size, P(F64), Size],
    ),
]


def configure_dll(dll: ctypes.CDLL) -> None:
    """Set *argtypes* and *restype* on every core C API function in *dll*."""
    for name, restype, argtypes in _CORE_FUNC_SIGS:
        fn = getattr(dll, name)
        fn.restype = restype
        fn.argtypes = argtypes
