"""Regression tests for native-owned solve option defaults."""

from __future__ import annotations

import ctypes

from metahotspot._lib import get_dll
from metahotspot.compiled import SolveOptions
from metahotspot.solution import _solve_options
from metahotspot.types import _SolveOptionsCStruct


def _native_defaults(dll):
    options = _SolveOptionsCStruct()
    dll.mhs_solve_options_default(ctypes.byref(options))
    return options


def _build(dll, opts: SolveOptions | None):
    return _solve_options(dll, opts._overrides() if opts is not None else {})


def test_empty_solve_options_match_native_defaults():
    dll = get_dll()
    expected = _native_defaults(dll)
    actual = _build(dll, SolveOptions())

    for name, _ in _SolveOptionsCStruct._fields_:
        assert getattr(actual, name) == getattr(expected, name), name


def test_solve_options_only_override_explicit_fields():
    dll = get_dll()
    expected = _native_defaults(dll)
    actual = _build(dll, SolveOptions(linear_tolerance=2.5e-10, fixed_dt=0.25))

    assert actual.linear_tolerance == 2.5e-10
    assert actual.fixed_dt == 0.25
    for name, _ in _SolveOptionsCStruct._fields_:
        if name not in {"linear_tolerance", "fixed_dt"}:
            assert getattr(actual, name) == getattr(expected, name), name
