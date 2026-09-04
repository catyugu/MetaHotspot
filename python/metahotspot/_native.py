"""Internal native adapters, grouped by runtime object."""

from metahotspot._native_model import (
    add_block,
    add_convection,
    add_dirichlet,
    add_fluid_boundary,
    add_function_double_exponential,
    add_function_expr,
    add_function_gauss,
    add_function_sine,
    add_layer,
    add_material,
    add_neumann,
    add_periodic_constant,
    add_piecewise,
    add_probe,
    add_rect,
    add_variable,
    create_model,
    read_xml,
    set_default_convection,
    set_default_dirichlet,
    set_default_neumann,
    set_mesh,
    set_settings,
)
from metahotspot._native_compiled import (
    assembled_operators,
    compile_model,
    compiled_metadata,
    eval_materials,
)
from metahotspot._native_solution import (
    solution_snapshot,
    solve,
    solve_options,
    write_vtu,
)
