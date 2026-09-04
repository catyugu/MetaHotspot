"""Internal native adapters, grouped by runtime object."""

from metahotspot._native_model import (
    add_boundary,
    add_fluid_boundary,
    add_id,
    add_periodic_constant,
    add_piecewise,
    call,
    create_model,
    face_regions,
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
