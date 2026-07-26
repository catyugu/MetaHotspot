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
    CscView,
    SolverOpts,
    Rect2D,
    Point2D,
    MhsFaceRegion,
    CompiledMetadataView,
    SolutionView,
    ProbeMetadata,
    MhsStepInfo,
)


def configure_dll(dll: ctypes.CDLL) -> None:
    """Set *argtypes* and *restype* on every function in *dll*."""

    # ---- Global helpers ----
    dll.mhs_solver_opts_default.restype = None
    dll.mhs_solver_opts_default.argtypes = [ctypes.POINTER(SolverOpts)]

    dll.mhs_status_string.restype = ctypes.c_char_p
    dll.mhs_status_string.argtypes = [ctypes.c_int32]

    dll.mhs_last_error.restype = ctypes.c_char_p
    dll.mhs_last_error.argtypes = []

    # ---- Model life-cycle ----
    dll.mhs_model_create.restype = ctypes.c_int32
    dll.mhs_model_create.argtypes = [ctypes.POINTER(ctypes.POINTER(MhsModel))]

    dll.mhs_model_destroy.restype = ctypes.c_int32
    dll.mhs_model_destroy.argtypes = [ctypes.POINTER(MhsModel)]

    dll.mhs_model_read_xml.restype = ctypes.c_int32
    dll.mhs_model_read_xml.argtypes = [ctypes.POINTER(MhsModel), ctypes.c_char_p]

    # ---- Settings / mesh / variables ----
    dll.mhs_model_set_settings.restype = ctypes.c_int32
    dll.mhs_model_set_settings.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_int32,  # study
        ctypes.c_int32,  # length_unit
        ctypes.c_double,  # initial_temperature_K
        ctypes.c_double,  # duration
        ctypes.c_double,  # output_interval
    ]

    dll.mhs_model_set_mesh.restype = ctypes.c_int32
    dll.mhs_model_set_mesh.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_size_t,  # nx
        ctypes.POINTER(ctypes.c_double),  # x
        ctypes.c_size_t,  # ny
        ctypes.POINTER(ctypes.c_double),  # y
        ctypes.c_size_t,  # nz
        ctypes.POINTER(ctypes.c_double),  # z
    ]

    dll.mhs_model_add_variable.restype = ctypes.c_int32
    dll.mhs_model_add_variable.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]

    # ---- Materials / layers / blocks / rects ----
    dll.mhs_model_add_material.restype = ctypes.c_uint32
    dll.mhs_model_add_material.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,  # kx, ky, kz
        ctypes.c_char_p,  # rho
        ctypes.c_char_p,  # c
        ctypes.c_char_p,  # dynamic_viscosity
    ]

    dll.mhs_model_add_layer.restype = ctypes.c_uint32
    dll.mhs_model_add_layer.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,  # thickness
        ctypes.c_char_p,  # x_offset
        ctypes.c_char_p,  # y_offset
    ]

    dll.mhs_model_add_block.restype = ctypes.c_uint32
    dll.mhs_model_add_block.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_uint32,  # layer
        ctypes.c_char_p,  # material_name
        ctypes.c_char_p,  # heat_source
        ctypes.c_char_p,  # x_offset
        ctypes.c_char_p,  # y_offset
        ctypes.c_char_p,  # thickness (or None)
    ]

    dll.mhs_model_add_rect.restype = ctypes.c_int32
    dll.mhs_model_add_rect.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_uint32,  # block
        ctypes.c_int32,  # op
        ctypes.c_char_p,  # x
        ctypes.c_char_p,  # y
        ctypes.c_char_p,  # width
        ctypes.c_char_p,  # height
    ]

    # ---- Atomic boundary conditions ----
    dll.mhs_model_add_dirichlet.restype = ctypes.c_int32
    dll.mhs_model_add_dirichlet.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.POINTER(MhsFaceRegion),  # regions
        ctypes.c_size_t,  # n_regions
        ctypes.c_char_p,  # temperature
    ]

    dll.mhs_model_add_neumann.restype = ctypes.c_int32
    dll.mhs_model_add_neumann.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.POINTER(MhsFaceRegion),
        ctypes.c_size_t,
        ctypes.c_char_p,
    ]

    dll.mhs_model_add_convection.restype = ctypes.c_int32
    dll.mhs_model_add_convection.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.POINTER(MhsFaceRegion),
        ctypes.c_size_t,
        ctypes.c_char_p,  # coefficient
        ctypes.c_char_p,  # ambient_temperature
    ]

    dll.mhs_model_set_default_dirichlet.restype = ctypes.c_int32
    dll.mhs_model_set_default_dirichlet.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
    ]

    dll.mhs_model_set_default_neumann.restype = ctypes.c_int32
    dll.mhs_model_set_default_neumann.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
    ]

    dll.mhs_model_set_default_convection.restype = ctypes.c_int32
    dll.mhs_model_set_default_convection.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]

    # ---- Functions ----
    dll.mhs_model_add_function_expr.restype = ctypes.c_uint32
    dll.mhs_model_add_function_expr.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]

    dll.mhs_model_add_function_gauss.restype = ctypes.c_uint32
    dll.mhs_model_add_function_gauss.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]

    dll.mhs_model_add_function_sine.restype = ctypes.c_uint32
    dll.mhs_model_add_function_sine.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]

    dll.mhs_model_add_function_double_exponential.restype = ctypes.c_uint32
    dll.mhs_model_add_function_double_exponential.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]

    dll.mhs_model_add_function_piecewise.restype = ctypes.c_uint32
    dll.mhs_model_add_function_piecewise.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.POINTER(Point2D),
        ctypes.c_size_t,
    ]

    dll.mhs_model_add_function_periodic_piecewise_constant.restype = ctypes.c_uint32
    dll.mhs_model_add_function_periodic_piecewise_constant.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_double,
    ]

    # ---- Probes & fluid boundaries ----
    dll.mhs_model_add_probe.restype = ctypes.c_uint32
    dll.mhs_model_add_probe.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]

    dll.mhs_model_add_fluid_boundary.restype = ctypes.c_int32
    dll.mhs_model_add_fluid_boundary.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_int32,  # axis
        ctypes.c_double,  # coordinate
        Rect2D,
        ctypes.c_int32,  # kind
        ctypes.c_double,  # value
        ctypes.c_double,  # inlet_temperature
    ]

    # ---- Compiled metadata ----
    dll.mhs_model_compile.restype = ctypes.c_int32
    dll.mhs_model_compile.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.POINTER(ctypes.POINTER(MhsCompiled)),
    ]

    dll.mhs_compiled_destroy.restype = ctypes.c_int32
    dll.mhs_compiled_destroy.argtypes = [ctypes.POINTER(MhsCompiled)]

    dll.mhs_compiled_metadata.restype = ctypes.c_int32
    dll.mhs_compiled_metadata.argtypes = [
        ctypes.POINTER(MhsCompiled),
        ctypes.POINTER(CompiledMetadataView),
    ]

    # ---- Model introspection ----
    dll.mhs_model_material_name.restype = ctypes.c_char_p
    dll.mhs_model_material_name.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_size_t,
    ]

    dll.mhs_model_material_count.restype = ctypes.c_size_t
    dll.mhs_model_material_count.argtypes = [ctypes.POINTER(MhsModel)]

    # ---- Assembly ----
    dll.mhs_compiled_assemble.restype = ctypes.c_int32
    dll.mhs_compiled_assemble.argtypes = [
        ctypes.POINTER(MhsCompiled),
        ctypes.POINTER(ctypes.c_double),  # state or None
        ctypes.c_double,  # time
        ctypes.POINTER(ctypes.POINTER(MhsAssembly)),
    ]

    dll.mhs_assembly_destroy.restype = ctypes.c_int32
    dll.mhs_assembly_destroy.argtypes = [ctypes.POINTER(MhsAssembly)]

    dll.mhs_assembly_n.restype = ctypes.c_size_t
    dll.mhs_assembly_n.argtypes = [ctypes.POINTER(MhsAssembly)]

    dll.mhs_assembly_rhs.restype = ctypes.POINTER(ctypes.c_double)
    dll.mhs_assembly_rhs.argtypes = [ctypes.POINTER(MhsAssembly)]

    dll.mhs_assembly_matrix.restype = ctypes.c_int32
    dll.mhs_assembly_matrix.argtypes = [
        ctypes.POINTER(MhsAssembly),
        ctypes.c_int32,
        ctypes.POINTER(CscView),
    ]

    # ---- Pre-solve ----
    dll.mhs_compiled_set_initial_state.restype = ctypes.c_int32
    dll.mhs_compiled_set_initial_state.argtypes = [
        ctypes.POINTER(MhsCompiled),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]

    # ---- Solve ----
    dll.mhs_compiled_solve.restype = ctypes.c_int32
    dll.mhs_compiled_solve.argtypes = [
        ctypes.POINTER(MhsCompiled),
        ctypes.POINTER(SolverOpts),  # or None
        ctypes.POINTER(ctypes.POINTER(MhsSolution)),
    ]

    dll.mhs_solution_destroy.restype = ctypes.c_int32
    dll.mhs_solution_destroy.argtypes = [ctypes.POINTER(MhsSolution)]

    # ---- Single step ----
    dll.mhs_compiled_step.restype = ctypes.c_int32
    dll.mhs_compiled_step.argtypes = [
        ctypes.POINTER(MhsCompiled),
        ctypes.POINTER(ctypes.c_double),  # state
        ctypes.c_double,  # time
        ctypes.c_double,  # dt
        ctypes.POINTER(ctypes.c_double),  # out_state
        ctypes.POINTER(MhsStepInfo),  # info or None
        ctypes.POINTER(SolverOpts),  # or None
    ]

    # ---- VTU ----
    dll.mhs_compiled_write_vtu.restype = ctypes.c_int32
    dll.mhs_compiled_write_vtu.argtypes = [
        ctypes.POINTER(MhsCompiled),
        ctypes.POINTER(MhsSolution),
        ctypes.c_char_p,
    ]

    # ---- Solution view ----
    dll.mhs_solution_view.restype = ctypes.c_int32
    dll.mhs_solution_view.argtypes = [
        ctypes.POINTER(MhsSolution),
        ctypes.POINTER(SolutionView),
    ]

    # ---- Probe accessors ----
    dll.mhs_solution_probe_count.restype = ctypes.c_size_t
    dll.mhs_solution_probe_count.argtypes = [ctypes.POINTER(MhsSolution)]

    dll.mhs_solution_probe_name.restype = ctypes.c_char_p
    dll.mhs_solution_probe_name.argtypes = [
        ctypes.POINTER(MhsSolution),
        ctypes.c_size_t,
    ]

    dll.mhs_solution_probe_record_count.restype = ctypes.c_size_t
    dll.mhs_solution_probe_record_count.argtypes = [
        ctypes.POINTER(MhsSolution),
        ctypes.c_size_t,
    ]

    dll.mhs_solution_probe_times.restype = ctypes.POINTER(ctypes.c_double)
    dll.mhs_solution_probe_times.argtypes = [
        ctypes.POINTER(MhsSolution),
        ctypes.c_size_t,
    ]

    dll.mhs_solution_probe_values.restype = ctypes.POINTER(ctypes.c_double)
    dll.mhs_solution_probe_values.argtypes = [
        ctypes.POINTER(MhsSolution),
        ctypes.c_size_t,
    ]

    dll.mhs_solution_probe_metadata.restype = ctypes.c_int32
    dll.mhs_solution_probe_metadata.argtypes = [
        ctypes.POINTER(MhsSolution),
        ctypes.POINTER(ProbeMetadata),
    ]

    dll.mhs_solution_probe_metadata_free.restype = ctypes.c_int32
    dll.mhs_solution_probe_metadata_free.argtypes = [
        ctypes.POINTER(ProbeMetadata),
    ]
