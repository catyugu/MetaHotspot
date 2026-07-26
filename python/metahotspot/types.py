"""Composite C-struct wrappers for the MetaHotspot C API."""

from __future__ import annotations

import ctypes

# ---------------------------------------------------------------------------
# Opaque handle forward declarations
# ---------------------------------------------------------------------------


class MhsModel(ctypes.Structure):
    pass


class MhsCompiled(ctypes.Structure):
    pass


class MhsSolution(ctypes.Structure):
    pass


class MhsAssembly(ctypes.Structure):
    pass


# ---------------------------------------------------------------------------
# Composite types
# ---------------------------------------------------------------------------


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


class MhsMeshInfo(ctypes.Structure):
    _fields_ = [
        ("nx", ctypes.c_size_t),
        ("ny", ctypes.c_size_t),
        ("nz", ctypes.c_size_t),
        ("x_verts", ctypes.POINTER(ctypes.c_double)),
        ("y_verts", ctypes.POINTER(ctypes.c_double)),
        ("z_verts", ctypes.POINTER(ctypes.c_double)),
    ]


class MhsStepInfo(ctypes.Structure):
    _fields_ = [
        ("error_ratio", ctypes.c_double),
        ("suggested_dt_factor", ctypes.c_double),
        ("nonlinear_iterations", ctypes.c_int32),
        ("accepted", ctypes.c_int32),
    ]


# ---------------------------------------------------------------------------
# ID types are uint32_t (UINT32_MAX = invalid)
# ---------------------------------------------------------------------------

MHS_LAYER_ID_INVALID = 0xFFFFFFFF
MHS_BLOCK_ID_INVALID = 0xFFFFFFFF
MHS_MATERIAL_ID_INVALID = 0xFFFFFFFF
MHS_BOUNDARY_ID_INVALID = 0xFFFFFFFF
MHS_FUNCTION_ID_INVALID = 0xFFFFFFFF
MHS_PROBE_ID_INVALID = 0xFFFFFFFF
