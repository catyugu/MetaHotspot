"""Affine parametric thermal models: concrete case adapters only.

This playground package holds the *concrete* model implementations
(``_bci_pop``, ``_chiplet_stack``) that register themselves with the
model-agnostic registry in :mod:`metahotspot.macromodel.affine`.  All shared
mechanism — the :class:`~metahotspot.macromodel.affine.AffineParametricModel`
base, plain-data satellites, factory, and algorithms — lives in the installed
``metahotspot.macromodel`` library and is imported from there.

Experiment code calls :func:`metahotspot.macromodel.create` by name and works
identically against any registered model.
"""

from metahotspot.macromodel.affine import (
    AffineParametricModel,
    AffineSolveResult,
    BoundaryGroup,
    CellLayout,
    SourcePort,
    create,
    register,
    registered_names,
    surface_exposed_cells,
)

# Explicit registration (no auto-registration on import, so merely importing
# this package costs nothing beyond the base contract).  Each model brings its
# own quick-mode config recipe, so an experiment only ever asks the factory
# *whether* it is quick — never what that means for a given model.
from affine_parametric_models import _chiplet_stack, _bci_pop

register(
    "chiplet_stack",
    _chiplet_stack._builder,
    quick_overrides=_chiplet_stack.QUICK_OVERRIDES,
)
register(
    "bci_pop",
    _bci_pop._builder,
    quick_overrides=_bci_pop.QUICK_OVERRIDES,
)

__all__ = [
    "AffineParametricModel",
    "AffineSolveResult",
    "BoundaryGroup",
    "CellLayout",
    "SourcePort",
    "surface_exposed_cells",
    "create",
    "register",
    "registered_names",
]
