"""MetaHotspot — Python bindings for the MetaHotspot C API thermal simulation library.

Usage::

    import metahotspot

    model = metahotspot.Model()
    model.read_xml("case.xml")
    compiled = model.compile()
    solution = compiled.solve()
    T_cells = solution.temperature

    # Access the assembled linear system for custom workflows
    K, C, f = compiled.assemble(compiled.default_state())

    # Macro-model coupled solve (optional plugin):
    from metahotspot.macromodel import solve, PortModel, PortCoupling
    solution = solve(compiled, port_model, coupling, state)
"""

from metahotspot.model import Model
from metahotspot.macromodel import MhsMacroPortModel
from metahotspot.compiled import Compiled
from metahotspot.solution import Solution
from metahotspot.assembly import Operators
from metahotspot import enums, types
from metahotspot._error import MetaHotspotError
from metahotspot._lib import get_dll, load_library
from metahotspot.types import SolveOptions

__all__ = [
    "Model",
    "Compiled",
    "Solution",
    "Operators",
    "enums",
    "types",
    "MetaHotspotError",
    "SolveOptions",
    "MhsMacroPortModel"
]
