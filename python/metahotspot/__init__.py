"""MetaHotspot — Python bindings for the MetaHotspot C API thermal simulation library.

Usage::

    import metahotspot

    model = metahotspot.Model()
    model.read_xml("case.xml")
    compiled = model.compile()
    solution = compiled.solve()
    T_cells = solution.temperature

    # Macro-model coupled solve (optional plugin):
    from metahotspot.macromodel import solve, PortModel, PortCoupling
    solution = solve(compiled, port_model, coupling, state)
"""

from metahotspot.model import Model
from metahotspot.compiled import Compiled, SolveOptions, Operators
from metahotspot.solution import Solution
from metahotspot import enums
from metahotspot._error import MetaHotspotError

__all__ = [
    "Model",
    "Compiled",
    "Solution",
    "SolveOptions",
    "Operators",
    "enums",
    "MetaHotspotError",
]
