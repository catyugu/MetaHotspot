"""Affine parametric thermal models: shared base + factory only.

This package deliberately exposes **no concrete model implementation**.  The
public surface is the :class:`AffineParametricModel` base (plus its plain-data
satellites :class:`BoundaryGroup` and :class:`AffineSolveResult`) and the
factory :func:`create` / :func:`register` / :func:`registered_names`.
Models are private modules registered here;
experiment code calls :func:`create` by name and works identically against any of them.

Design rationale (FANTASTIC 2014 + BCI matrix reduction 2015): reduction
methods operate only on ``(K, C, f)`` DtN operators and per-boundary-group
``(cells, g, areas)`` data; the model is whatever supplies those, parametrized
by one heat-exchange coefficient ``h`` per group.  Keeping implementations
private is what makes the experiments model-agnostic: they can never reach into
a concrete config.
"""

from affine_parametric_models._interfaces import (
    AffineParametricModel,
    AffineSolveResult,
    BoundaryGroup,
)
from affine_parametric_models._registry import create, register, registered_names

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
    "create",
    "register",
    "registered_names",
]
