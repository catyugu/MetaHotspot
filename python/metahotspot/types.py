"""ctypes type definitions for the MetaHotspot C API.

Contains opaque handle types, value types (structs passed by value).
"""

from __future__ import annotations

import ctypes

# ---- Opaque handle types ----


class MhsModel(ctypes.Structure):
    pass


class MhsCompiled(ctypes.Structure):
    pass


class MhsSolution(ctypes.Structure):
    pass


class MhsOperators(ctypes.Structure):
    pass


# ---- Value types ----


class Rect2D(ctypes.Structure):
    _fields_ = [
        ("a_min", ctypes.c_double),
        ("a_max", ctypes.c_double),
        ("b_min", ctypes.c_double),
        ("b_max", ctypes.c_double),
    ]


class Point2D(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
    ]


class MhsFaceRegion(ctypes.Structure):
    _fields_ = [
        ("axis", ctypes.c_int32),
        ("coordinate", ctypes.c_double),
        ("rectangle", Rect2D),
    ]


class MhsStudyOptions(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int32),
        ("duration", ctypes.c_double),
        ("output_interval", ctypes.c_double),
    ]


class MhsCompiledInfo(ctypes.Structure):
    _fields_ = [
        ("cell_count", ctypes.c_size_t),
        ("face_count", ctypes.c_size_t),
        ("projection_count", ctypes.c_size_t),
        ("source_count", ctypes.c_size_t),
        ("initial_temperature", ctypes.c_double),
    ]


class MhsCellFields(ctypes.Structure):
    _fields_ = [
        ("cell_count", ctypes.c_size_t),
        ("face_count", ctypes.c_size_t),
        ("projection_count", ctypes.c_size_t),
        ("cell_min", ctypes.POINTER(ctypes.c_double)),
        ("cell_max", ctypes.POINTER(ctypes.c_double)),
        ("material_id", ctypes.POINTER(ctypes.c_uint32)),
        ("face_minus", ctypes.POINTER(ctypes.c_uint32)),
        ("face_plus", ctypes.POINTER(ctypes.c_uint32)),
        ("face_axis", ctypes.POINTER(ctypes.c_int32)),
        ("face_bounds", ctypes.POINTER(ctypes.c_double)),
        ("projection_cell", ctypes.POINTER(ctypes.c_uint32)),
        ("projection_source", ctypes.POINTER(ctypes.c_uint32)),
        ("projection_volume", ctypes.POINTER(ctypes.c_double)),
        ("projection_center", ctypes.POINTER(ctypes.c_double)),
    ]


class MhsMaterialValues(ctypes.Structure):
    _fields_ = [
        ("conductivity_x", ctypes.POINTER(ctypes.c_double)),
        ("conductivity_y", ctypes.POINTER(ctypes.c_double)),
        ("conductivity_z", ctypes.POINTER(ctypes.c_double)),
        ("density", ctypes.POINTER(ctypes.c_double)),
        ("specific_heat", ctypes.POINTER(ctypes.c_double)),
        ("count", ctypes.c_size_t),
    ]


class MhsOperatorsInfo(ctypes.Structure):
    _fields_ = [
        ("state_count", ctypes.c_size_t),
        ("k_nnz", ctypes.c_size_t),
        ("c_nnz", ctypes.c_size_t),
    ]


class MhsSolutionInfo(ctypes.Structure):
    _fields_ = [
        ("state_count", ctypes.c_size_t),
        ("record_count", ctypes.c_size_t),
        ("probe_count", ctypes.c_size_t),
        ("time", ctypes.c_double),
        ("converged", ctypes.c_int32),
    ]


# ---- Private C struct for SolveOptions (used internally by compiled.py) ----


class _SolveOptionsCStruct(ctypes.Structure):
    _fields_ = [
        ("solver_type", ctypes.c_int32),
        ("linear_tolerance", ctypes.c_double),
        ("linear_max_iterations", ctypes.c_int32),
        ("underrelaxation", ctypes.c_double),
        ("nonlinear_max_iterations", ctypes.c_int32),
        ("nonlinear_relative_tolerance", ctypes.c_double),
        ("nonlinear_absolute_tolerance", ctypes.c_double),
        ("time_tolerance", ctypes.c_double),
        ("max_time_step", ctypes.c_double),
    ]
