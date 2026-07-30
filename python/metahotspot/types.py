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

class SolveOptions(ctypes.Structure):
    _fields_ = [
        ("solver_type", ctypes.c_int32),
        ("linear_tolerance", ctypes.c_double),
        ("linear_max_iterations", ctypes.c_int32),
        ("underrelaxation", ctypes.c_double),
        ("nonlinear_max_iterations", ctypes.c_int32),
        ("nonlinear_relative_tolerance", ctypes.c_double),
        ("nonlinear_absolute_tolerance", ctypes.c_double),
        ("integrator", ctypes.c_int32),
        ("step_strategy", ctypes.c_int32),
        ("error_abs_tol", ctypes.c_double),
        ("error_safety", ctypes.c_double),
        ("min_dt", ctypes.c_double),
        ("max_dt", ctypes.c_double),
        ("fixed_dt", ctypes.c_double),
    ]

    @staticmethod
    def default() -> "SolveOptions":
        """Return a SolveOptions filled with sensible defaults (Pardiso, 1e-8, ...)."""
        opts = SolveOptions()
        from metahotspot._lib import get_dll
        get_dll().mhs_solve_options_default(ctypes.byref(opts))
        return opts

class CscView(ctypes.Structure):
    _fields_ = [
        ("rows", ctypes.c_int32),
        ("columns", ctypes.c_int32),
        ("nnz", ctypes.c_int32),
        ("outer_indices", ctypes.POINTER(ctypes.c_int32)),
        ("inner_indices", ctypes.POINTER(ctypes.c_int32)),
        ("values", ctypes.POINTER(ctypes.c_double)),
    ]

class MhsOperatorsView(ctypes.Structure):
    _fields_ = [
        ("K", CscView),
        ("C", CscView),
        ("rhs", ctypes.POINTER(ctypes.c_double)),
        ("n", ctypes.c_size_t),
    ]

class CompiledMetadataView(ctypes.Structure):
    _fields_ = [
        ("cell_count", ctypes.c_size_t),
        ("study_type", ctypes.c_int32),
        ("initial_temperature", ctypes.c_double),
        ("layer_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("block_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("grid_to_cell", ctypes.POINTER(ctypes.c_size_t)),
        ("nx", ctypes.c_size_t),
        ("ny", ctypes.c_size_t),
        ("nz", ctypes.c_size_t),
    ]

class SolutionView(ctypes.Structure):
    _fields_ = [
        ("fvm_count", ctypes.c_size_t),
        ("state_count", ctypes.c_size_t),
        ("time", ctypes.c_double),
        ("state", ctypes.POINTER(ctypes.c_double)),
    ]

class ProbeView(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("times", ctypes.POINTER(ctypes.c_double)),
        ("values", ctypes.POINTER(ctypes.c_double)),
        ("record_count", ctypes.c_size_t),
    ]

class MhsMacroPortModel(ctypes.Structure):
    """C-side ``mhs_macro_port_model_t`` mirror — macro-model extension."""
    _fields_ = [
        ("operators", MhsOperatorsView),
        ("basis", ctypes.POINTER(ctypes.c_double)),
        ("physical_port_count", ctypes.c_size_t),
        ("model_cells", ctypes.POINTER(ctypes.c_size_t)),
        ("model_face", ctypes.c_int32),
        ("exterior_half_conductance", ctypes.POINTER(ctypes.c_double)),
    ]


# ---- Invalid-ID sentinels (remaining public API) ----

MHS_LAYER_ID_INVALID: int = 0xFFFFFFFF
MHS_BLOCK_ID_INVALID: int = 0xFFFFFFFF
