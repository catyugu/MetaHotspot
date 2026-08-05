"""Affine parametric thermal models: abstract interface + factory only.

This package deliberately exposes **no concrete model implementation**.  The
public surface is the abstract contract :class:`AffineParametricModel` (plus
its plain-data satellites :class:`BoundaryGroup` and :class:`AffineSolveResult`)
and the factory :func:`create` / :func:`register` / :func:`registered_names`.
Concrete models (``chiplet_stack``, ``bci_pkg``, ``toy_1d``) are private
modules registered here; experiment code calls :func:`create` by name and works
identically against any of them.

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
    StateLayout,
)
from affine_parametric_models._registry import create, register, registered_names

# Explicit registration (no auto-registration on import, so merely importing
# this package costs nothing beyond the abstract contract).
from affine_parametric_models import _chiplet_stack, _bci_pkg, _toy_1d

register("chiplet_stack", _chiplet_stack._builder)
register("bci_pkg", _bci_pkg._builder)
register("toy_1d", _toy_1d._builder)

__all__ = [
    "AffineParametricModel",
    "AffineSolveResult",
    "BoundaryGroup",
    "StateLayout",
    "create",
    "register",
    "registered_names",
]
