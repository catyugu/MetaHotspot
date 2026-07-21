"""MetaHotspot — Python bindings for the MetaHotspot C API thermal simulation library.

Usage::

    import metahotspot

    model = metahotspot.Model()
    model.read_xml("case.xml")
    compiled = model.compile()
    solution = compiled.solve()
    T_cells = solution.cell_temperatures()

    # Access the assembled linear system for custom workflows
    assembly = compiled.assemble()
    K, f = assembly.as_csc()
"""

from metahotspot.model import Model
from metahotspot.compiled import Compiled
from metahotspot.solution import Solution
from metahotspot.assembly import Assembly
from metahotspot import enums, types
from metahotspot._error import MetaHotspotError
from metahotspot._lib import get_dll, load_library

__all__ = [
    "Model",
    "Compiled",
    "Solution",
    "Assembly",
    "enums",
    "types",
    "MetaHotspotError",
]
