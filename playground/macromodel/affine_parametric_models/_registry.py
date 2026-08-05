"""Factory/registry for affine parametric thermal models.

The registry is the *only* way experiment code obtains a concrete
:class:`AffineParametricModel`.  Concrete implementations are private modules
inside this package; the factory returns instances typed as the abstract base,
so no concrete class name leaks into experiment code.  This is the decoupling
guarantee: an experiment asks ``create("chiplet_stack")`` and
works identically against any registered implementation.
"""

from __future__ import annotations

from typing import Any, Callable

from affine_parametric_models._interfaces import AffineParametricModel

_REGISTRY: dict[str, Callable[..., AffineParametricModel]] = {}


def register(name: str, builder: Callable[..., AffineParametricModel]) -> None:
    """Register a concrete model under ``name``.

    ``builder`` is called as ``builder(overrides: dict | None = None, **kw)``
    and must return an :class:`AffineParametricModel`.  Re-registering a name
    replaces the previous builder.
    """
    if not name or not name.isidentifier():
        raise ValueError(f"invalid model name: {name!r}")
    _REGISTRY[name] = builder


def create(
    name: str, *, overrides: dict | None = None, **kwargs: Any
) -> AffineParametricModel:
    """Instantiate the registered model ``name``.

    ``overrides`` (mapping of scalar config field -> value) and ``**kwargs``
    are forwarded to the registered builder.  Raises ``KeyError`` for unknown
    names.  The returned value is typed as the abstract base; the concrete
    class is private to this package.
    """
    try:
        builder = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown affine parametric model {name!r}; "
            f"registered: {sorted(_REGISTRY)}"
        ) from None
    return builder(overrides=overrides, **kwargs)


def registered_names() -> list[str]:
    """Sorted names of all registered models."""
    return sorted(_REGISTRY)
