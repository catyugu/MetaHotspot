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


# ---------------------------------------------------------------------------
# ID types are plain int32_t
# ---------------------------------------------------------------------------

MHS_LAYER_ID_INVALID = -1
MHS_BLOCK_ID_INVALID = -1
MHS_MATERIAL_ID_INVALID = -1
MHS_BOUNDARY_ID_INVALID = -1
MHS_FUNCTION_ID_INVALID = -1
MHS_PROBE_ID_INVALID = -1
