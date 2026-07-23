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
    SolverOpts,
    Rect2D,
    Point2D,
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
        ctypes.c_int32, ctypes.POINTER(ctypes.c_double),  # nx, x
        ctypes.c_int32, ctypes.POINTER(ctypes.c_double),  # ny, y
        ctypes.c_int32, ctypes.POINTER(ctypes.c_double),  # nz, z
    ]

    dll.mhs_model_add_variable.restype = ctypes.c_int32
    dll.mhs_model_add_variable.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]

    # ---- Materials / layers / blocks / rects ----
    dll.mhs_model_add_material.restype = ctypes.c_int32
    dll.mhs_model_add_material.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]

    dll.mhs_model_add_layer.restype = ctypes.c_int32
    dll.mhs_model_add_layer.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]

    dll.mhs_model_add_block.restype = ctypes.c_int32
    dll.mhs_model_add_block.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_int32,  # layer
        ctypes.c_char_p,  # material_name
        ctypes.c_char_p,  # heat_source
        ctypes.c_char_p,  # x_offset
        ctypes.c_char_p,  # y_offset
        ctypes.c_char_p,  # thickness (NULL = inherit)
    ]

    dll.mhs_model_add_rect.restype = ctypes.c_int32
    dll.mhs_model_add_rect.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_int32,
        ctypes.c_int32,  # op
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]

    # ---- Boundary conditions ----
    dll.mhs_model_add_boundary.restype = ctypes.c_int32
    dll.mhs_model_add_boundary.argtypes = [ctypes.POINTER(MhsModel)]

    dll.mhs_boundary_set_dirichlet.restype = ctypes.c_int32
    dll.mhs_boundary_set_dirichlet.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_int32,
        ctypes.c_char_p,
    ]

    dll.mhs_boundary_set_neumann.restype = ctypes.c_int32
    dll.mhs_boundary_set_neumann.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_int32,
        ctypes.c_char_p,
    ]

    dll.mhs_boundary_set_convection.restype = ctypes.c_int32
    dll.mhs_boundary_set_convection.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]

    dll.mhs_boundary_add_face_region.restype = ctypes.c_int32
    dll.mhs_boundary_add_face_region.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        Rect2D,
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
    dll.mhs_model_add_function_expr.restype = ctypes.c_int32
    dll.mhs_model_add_function_expr.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]

    dll.mhs_model_add_function_gauss.restype = ctypes.c_int32
    dll.mhs_model_add_function_gauss.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]

    dll.mhs_model_add_function_sine.restype = ctypes.c_int32
    dll.mhs_model_add_function_sine.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]

    dll.mhs_model_add_function_double_exponential.restype = ctypes.c_int32
    dll.mhs_model_add_function_double_exponential.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]

    dll.mhs_model_add_function_piecewise.restype = ctypes.c_int32
    dll.mhs_model_add_function_piecewise.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_char_p,
        ctypes.POINTER(Point2D),
        ctypes.c_int32,
    ]

    # ---- Probes / fluid boundaries ----
    dll.mhs_model_add_probe.restype = ctypes.c_int32
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
        ctypes.c_int32,
        ctypes.c_double,
        Rect2D,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_double,
    ]

    # ---- Compilation ----
    dll.mhs_model_compile.restype = ctypes.c_int32
    dll.mhs_model_compile.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.POINTER(ctypes.POINTER(MhsCompiled)),
    ]

    dll.mhs_compiled_destroy.restype = ctypes.c_int32
    dll.mhs_compiled_destroy.argtypes = [ctypes.POINTER(MhsCompiled)]

    dll.mhs_compiled_cell_count.restype = ctypes.c_int32
    dll.mhs_compiled_cell_count.argtypes = [ctypes.POINTER(MhsCompiled)]

    dll.mhs_compiled_node_count.restype = ctypes.c_int32
    dll.mhs_compiled_node_count.argtypes = [ctypes.POINTER(MhsCompiled)]

    dll.mhs_compiled_initial_temperature.restype = ctypes.c_double
    dll.mhs_compiled_initial_temperature.argtypes = [ctypes.POINTER(MhsCompiled)]

    dll.mhs_compiled_study_type.restype = ctypes.c_int32
    dll.mhs_compiled_study_type.argtypes = [ctypes.POINTER(MhsCompiled)]

    dll.mhs_compiled_layer_count.restype = ctypes.c_uint32
    dll.mhs_compiled_layer_count.argtypes = [ctypes.POINTER(MhsCompiled)]

    dll.mhs_compiled_block_count.restype = ctypes.c_uint32
    dll.mhs_compiled_block_count.argtypes = [
        ctypes.POINTER(MhsCompiled),
        ctypes.c_uint32,
    ]

    dll.mhs_compiled_layer_ids.restype = ctypes.POINTER(ctypes.c_uint32)
    dll.mhs_compiled_layer_ids.argtypes = [ctypes.POINTER(MhsCompiled)]

    dll.mhs_compiled_block_ids.restype = ctypes.POINTER(ctypes.c_uint32)
    dll.mhs_compiled_block_ids.argtypes = [ctypes.POINTER(MhsCompiled)]

    # ---- Model introspection ----
    dll.mhs_model_material_name.restype = ctypes.c_char_p
    dll.mhs_model_material_name.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.c_int32,
    ]

    dll.mhs_model_material_count.restype = ctypes.c_int32
    dll.mhs_model_material_count.argtypes = [ctypes.POINTER(MhsModel)]

    # ---- Solve ----
    dll.mhs_compiled_solve.restype = ctypes.c_int32
    dll.mhs_compiled_solve.argtypes = [
        ctypes.POINTER(MhsCompiled),
        ctypes.POINTER(SolverOpts),
        ctypes.POINTER(ctypes.POINTER(MhsSolution)),
    ]

    dll.mhs_solve.restype = ctypes.c_int32
    dll.mhs_solve.argtypes = [
        ctypes.POINTER(MhsModel),
        ctypes.POINTER(SolverOpts),
        ctypes.POINTER(ctypes.POINTER(MhsSolution)),
    ]

    dll.mhs_solution_destroy.restype = ctypes.c_int32
    dll.mhs_solution_destroy.argtypes = [ctypes.POINTER(MhsSolution)]

    # ---- Assembly ----
    dll.mhs_compiled_assemble.restype = ctypes.c_int32
    dll.mhs_compiled_assemble.argtypes = [
        ctypes.POINTER(MhsCompiled),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.POINTER(ctypes.POINTER(MhsAssembly)),
    ]

    dll.mhs_assembly_destroy.restype = ctypes.c_int32
    dll.mhs_assembly_destroy.argtypes = [ctypes.POINTER(MhsAssembly)]

    dll.mhs_assembly_n.restype = ctypes.c_int32
    dll.mhs_assembly_n.argtypes = [ctypes.POINTER(MhsAssembly)]

    dll.mhs_assembly_nnz.restype = ctypes.c_int32
    dll.mhs_assembly_nnz.argtypes = [ctypes.POINTER(MhsAssembly)]

    dll.mhs_assembly_outer_indices.restype = ctypes.POINTER(ctypes.c_int32)
    dll.mhs_assembly_outer_indices.argtypes = [ctypes.POINTER(MhsAssembly)]

    dll.mhs_assembly_inner_indices.restype = ctypes.POINTER(ctypes.c_int32)
    dll.mhs_assembly_inner_indices.argtypes = [ctypes.POINTER(MhsAssembly)]

    dll.mhs_assembly_values.restype = ctypes.POINTER(ctypes.c_double)
    dll.mhs_assembly_values.argtypes = [ctypes.POINTER(MhsAssembly)]

    dll.mhs_assembly_rhs.restype = ctypes.POINTER(ctypes.c_double)
    dll.mhs_assembly_rhs.argtypes = [ctypes.POINTER(MhsAssembly)]

    dll.mhs_assembly_mass_diagonal.restype = ctypes.POINTER(ctypes.c_double)
    dll.mhs_assembly_mass_diagonal.argtypes = [ctypes.POINTER(MhsAssembly)]

    # ---- Solution accessors ----
    dll.mhs_solution_cell_count.restype = ctypes.c_int32
    dll.mhs_solution_cell_count.argtypes = [ctypes.POINTER(MhsSolution)]

    dll.mhs_solution_node_count.restype = ctypes.c_int32
    dll.mhs_solution_node_count.argtypes = [ctypes.POINTER(MhsSolution)]

    dll.mhs_solution_time.restype = ctypes.c_double
    dll.mhs_solution_time.argtypes = [ctypes.POINTER(MhsSolution)]

    dll.mhs_solution_cell_temperatures.restype = ctypes.POINTER(ctypes.c_double)
    dll.mhs_solution_cell_temperatures.argtypes = [ctypes.POINTER(MhsSolution)]

    dll.mhs_solution_node_temperatures.restype = ctypes.POINTER(ctypes.c_double)
    dll.mhs_solution_node_temperatures.argtypes = [ctypes.POINTER(MhsSolution)]

    # ---- Probe accessors ----
    dll.mhs_solution_probe_count.restype = ctypes.c_int32
    dll.mhs_solution_probe_count.argtypes = [ctypes.POINTER(MhsSolution)]

    dll.mhs_solution_probe_name.restype = ctypes.c_char_p
    dll.mhs_solution_probe_name.argtypes = [
        ctypes.POINTER(MhsSolution),
        ctypes.c_int32,
    ]

    dll.mhs_solution_probe_record_count.restype = ctypes.c_int32
    dll.mhs_solution_probe_record_count.argtypes = [
        ctypes.POINTER(MhsSolution),
        ctypes.c_int32,
    ]

    dll.mhs_solution_probe_times.restype = ctypes.POINTER(ctypes.c_double)
    dll.mhs_solution_probe_times.argtypes = [
        ctypes.POINTER(MhsSolution),
        ctypes.c_int32,
    ]

    dll.mhs_solution_probe_values.restype = ctypes.POINTER(ctypes.c_double)
    dll.mhs_solution_probe_values.argtypes = [
        ctypes.POINTER(MhsSolution),
        ctypes.c_int32,
    ]
