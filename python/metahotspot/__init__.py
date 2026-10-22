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
from metahotspot._compiled_data import Operators
from metahotspot._error import MetaHotspotError
from metahotspot._lib import library_version
from metahotspot.compiled import Compiled, SolveOptions
from metahotspot.model import Model
from metahotspot.solution import Solution

# Reported by the library this module binds, so it can never drift from it.
__version__ = library_version()

__all__ = [
    "Compiled",
    "MetaHotspotError",
    "Model",
    "Operators",
    "Solution",
    "SolveOptions",
    "enums",
]
