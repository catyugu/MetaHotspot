"""ctypes type definitions for the MetaHotspot C API.

Contains opaque handle types, value types (structs passed by value), and
invalid-ID sentinels.
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


class MhsAssembly(ctypes.Structure):
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


class SolverOpts(ctypes.Structure):
    _fields_ = [
        ("solver_type", ctypes.c_int32),
        ("linear_tolerance", ctypes.c_double),
        ("linear_max_iterations", ctypes.c_int32),
        ("underrelaxation", ctypes.c_double),
        ("nonlinear_max_iterations", ctypes.c_int32),
        ("nonlinear_relative_tolerance", ctypes.c_double),
        ("nonlinear_absolute_tolerance", ctypes.c_double),
    ]


class CscView(ctypes.Structure):
    _fields_ = [
        ("rows", ctypes.c_int32),
        ("columns", ctypes.c_int32),
        ("nnz", ctypes.c_int32),
        ("outer_indices", ctypes.POINTER(ctypes.c_int32)),
        ("inner_indices", ctypes.POINTER(ctypes.c_int32)),
        ("values", ctypes.POINTER(ctypes.c_double)),
    ]


class MhsStepInfo(ctypes.Structure):
    _fields_ = [
        ("accepted", ctypes.c_int32),
        ("error_ratio", ctypes.c_double),
        ("suggested_dt_factor", ctypes.c_double),
        ("nonlinear_iterations", ctypes.c_int32),
    ]


class CompiledMetadataView(ctypes.Structure):
    _fields_ = [
        ("cell_count", ctypes.c_size_t),
        ("state_count", ctypes.c_size_t),
        ("node_count", ctypes.c_size_t),
        ("grid_count", ctypes.c_size_t),
        ("study_type", ctypes.c_int32),
        ("initial_temperature", ctypes.c_double),
        ("layer_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("block_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("grid_to_cell", ctypes.POINTER(ctypes.c_size_t)),
        ("nx", ctypes.c_size_t),
        ("ny", ctypes.c_size_t),
        ("nz", ctypes.c_size_t),
        ("x_verts", ctypes.POINTER(ctypes.c_double)),
        ("y_verts", ctypes.POINTER(ctypes.c_double)),
        ("z_verts", ctypes.POINTER(ctypes.c_double)),
    ]


class SolutionView(ctypes.Structure):
    _fields_ = [
        ("cell_count", ctypes.c_size_t),
        ("state_count", ctypes.c_size_t),
        ("node_count", ctypes.c_size_t),
        ("time", ctypes.c_double),
        ("cell_temperatures", ctypes.POINTER(ctypes.c_double)),
        ("states", ctypes.POINTER(ctypes.c_double)),
        ("node_temperatures", ctypes.POINTER(ctypes.c_double)),
    ]


class ProbeMetadata(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_size_t),
        ("names", ctypes.POINTER(ctypes.c_char_p)),
        ("record_counts", ctypes.POINTER(ctypes.c_size_t)),
    ]


# ---- Invalid-ID sentinels ----

MHS_LAYER_ID_INVALID: int = 0xFFFFFFFF
MHS_BLOCK_ID_INVALID: int = 0xFFFFFFFF
MHS_MATERIAL_ID_INVALID: int = 0xFFFFFFFF
MHS_FUNCTION_ID_INVALID: int = 0xFFFFFFFF
MHS_PROBE_ID_INVALID: int = 0xFFFFFFFF
