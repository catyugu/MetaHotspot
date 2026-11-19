"""Python bindings for the MetaHotspot C API thermal simulation library.

Usage::

    import metahotspot

    model = metahotspot.Fom()
    # Construct SI volumes and mesh axes before compiling.
    compiled = model.compile()
    solution = compiled.solve()
    temperatures = solution.state
"""

from metahotspot import enums
from metahotspot._compiled_data import Operators
from metahotspot._error import MetaHotspotError
from metahotspot._lib import library_version
from metahotspot.compiled import Compiled, SolveOptions
from metahotspot.fom import Fom, SurfaceRegion
from metahotspot.logging import get_log_level, set_log_level
from metahotspot.solution import Solution, Study

# Reported by the library this module binds, so it can never drift from it.
__version__ = library_version()

__all__ = [
    "Compiled",
    "Fom",
    "MetaHotspotError",
    "Operators",
    "Solution",
    "SolveOptions",
    "Study",
    "SurfaceRegion",
    "enums",
    "get_log_level",
    "set_log_level",
]
