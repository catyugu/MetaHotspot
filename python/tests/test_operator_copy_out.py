"""Regression tests for C API operator copy-out boundaries."""

from __future__ import annotations

import ctypes

import numpy as np

from metahotspot._lib import get_dll
from metahotspot.types import MhsOperators, MhsOperatorsInfo


def _make_operator(dll):
    outer = (ctypes.c_int32 * 2)(0, 1)
    inner = (ctypes.c_int32 * 1)(0)
    values = (ctypes.c_double * 1)(2.0)
    rhs = (ctypes.c_double * 1)(3.0)
    handle = ctypes.POINTER(MhsOperators)()
    status = dll.mhs_operators_create(
        1,
        outer,
        inner,
        values,
        1,
        outer,
        inner,
        values,
        1,
        rhs,
        ctypes.byref(handle),
    )
    assert status == 0
    return handle


def test_operator_copy_rhs_rejects_oversized_count_without_reading_past_rhs():
    dll = get_dll()
    handle = _make_operator(dll)
    try:
        info = MhsOperatorsInfo()
        assert dll.mhs_operators_get_info(handle, ctypes.byref(info)) == 0

        output = np.full(info.state_count + 1, np.nan)
        status = dll.mhs_operators_copy_rhs(
            handle,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            output.size,
        )

        assert status == -1
        assert np.all(np.isnan(output))
    finally:
        dll.mhs_operators_destroy(handle)
