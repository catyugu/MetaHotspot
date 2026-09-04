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
from metahotspot._compiled_data import Operators
from metahotspot.compiled import Compiled, SolveOptions
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
