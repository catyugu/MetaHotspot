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
"""

from metahotspot.model import Model
from metahotspot.compiled import Compiled
from metahotspot.solution import Solution
from metahotspot.assembly import Operators
from metahotspot import enums, types
from metahotspot._error import MetaHotspotError
from metahotspot._lib import get_dll, load_library
from metahotspot.types import SolverOpts

__all__ = [
    "Model",
    "Compiled",
    "Solution",
    "Operators",
    "enums",
    "types",
    "MetaHotspotError",
    "SolverOpts",
]
