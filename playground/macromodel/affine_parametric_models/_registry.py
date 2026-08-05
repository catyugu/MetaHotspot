"""Factory/registry for affine parametric thermal models.

The registry is the *only* way experiment code obtains a concrete
:class:`AffineParametricModel`.  Concrete implementations are private modules
inside this package; the factory returns instances typed as the abstract base,
so no concrete class name leaks into experiment code.  This is the decoupling
guarantee: an experiment asks ``create("chiplet_stack")`` and
works identically against any registered implementation.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from affine_parametric_models._interfaces import AffineParametricModel


class _Entry:
    """Registered builder plus its quick-mode config overrides.

    ``builder`` is called as ``builder(overrides: dict | None = None, **kw)``
    and must return an :class:`AffineParametricModel`.  ``quick_overrides`` is
    the model's own recipe for a fast smoke experiment; the factory applies it
    when ``create(..., quick=True)`` is used, so the experiment only has to
    say *whether* it is quick, never *what* that means for a given model.
    """

    __slots__ = ("builder", "quick_overrides")

    def __init__(self, builder, quick_overrides=None):
        self.builder = builder
        self.quick_overrides = quick_overrides


_REGISTRY: dict[str, _Entry] = {}


def register(
    name: str,
    builder: Callable[..., AffineParametricModel],
    *,
    quick_overrides: Optional[dict] = None,
) -> None:
    """Register a concrete model under ``name``.

    ``builder`` is called as ``builder(overrides: dict | None = None, **kw)``
    and must return an :class:`AffineParametricModel`.  ``quick_overrides`` is
    the mapping of scalar config fields the model applies in quick mode (its
    own smoke-experiment recipe).  Re-registering a name replaces the previous
    entry.
    """
    if not name or not name.isidentifier():
        raise ValueError(f"invalid model name: {name!r}")
    _REGISTRY[name] = _Entry(builder, quick_overrides)


def create(name: str, *, quick: bool = False, **kwargs: Any) -> AffineParametricModel:
    """Instantiate the registered model ``name``.

    ``quick`` toggles the model's own quick-mode overrides (``True`` applies
    them); additional ``**kwargs`` are forwarded to the registered builder as
    config overrides and take precedence.  Raises ``KeyError`` for unknown
    names.  The returned value is typed as the abstract base; the concrete
    class is private to this package.
    """
    try:
        entry = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown affine parametric model {name!r}; "
            f"registered: {sorted(_REGISTRY)}"
        ) from None
    overrides = entry.quick_overrides if quick else None
    return entry.builder(overrides=overrides, **kwargs)


def registered_names() -> list[str]:
    """Sorted names of all registered models."""
    return sorted(_REGISTRY)
