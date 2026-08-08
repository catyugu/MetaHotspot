"""Python bindings for the MetaHotspot C API thermal simulation library.

Usage::

    import metahotspot

    model = metahotspot.Model()
    model.read_xml("case.xml")
    compiled = model.compile()
    solution = compiled.solve()
    temperatures = solution.temperature
"""

from metahotspot import enums
from metahotspot._error import MetaHotspotError
from metahotspot.compiled import Compiled, Operators, SolveOptions
from metahotspot.model import Model
from metahotspot.solution import Solution

__all__ = [
    "Model",
    "Compiled",
    "Solution",
    "SolveOptions",
    "Operators",
    "enums",
    "MetaHotspotError",
]
