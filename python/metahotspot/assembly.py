"""Assembled operators — plain data, no lifecycle management.

``Compiled.assemble()`` now returns an ``Operators`` namedtuple
instead of a managed ``Assembly`` handle.  The C handle is created,
read, and destroyed inside the call.
"""

from __future__ import annotations

import ctypes
from typing import NamedTuple

import numpy as np
from scipy.sparse import csc_matrix

from metahotspot._error import check
from metahotspot.types import CscView, MhsAssembly, MhsAssemblyView


def _csc_from_view(view: CscView) -> csc_matrix:
    outer = np.ctypeslib.as_array(
        view.outer_indices,
        shape=(view.columns + 1,),
    ).copy()
    inner = np.ctypeslib.as_array(
        view.inner_indices,
        shape=(view.nnz,),
    ).copy()
    values = np.ctypeslib.as_array(
        view.values,
        shape=(view.nnz,),
    ).copy()
    return csc_matrix(
        (values, inner, outer),
        shape=(view.rows, view.columns),
    )


class Operators(NamedTuple):
    """``C * dx/dt + K * x = f`` — thermal system operators.

    Returned by ``Compiled.assemble()``.  Unpack directly::

        K, C, f = compiled.assemble(state)
    """

    K: csc_matrix
    """Stiffness / conductance matrix."""

    C: csc_matrix
    """Capacity (mass) matrix."""

    f: np.ndarray
    """Right-hand side vector."""


def _assemble_operators(
    dll, compiled_handle, state: np.ndarray, time: float = 0.0
) -> Operators:
    """Call the C assembly routine, copy results, destroy handle immediately."""
    pp = ctypes.POINTER(MhsAssembly)()
    state_ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    check(
        dll.mhs_compiled_assemble(
            compiled_handle, state_ptr, len(state), time, ctypes.byref(pp)
        ),
        "assemble",
    )

    # Read the view while the handle is alive.
    view = MhsAssemblyView()
    check(dll.mhs_assembly_view(pp, ctypes.byref(view)), "assembly_view")
    K = _csc_from_view(view.K)
    C = _csc_from_view(view.C)
    f_arr = np.ctypeslib.as_array(view.rhs, shape=(view.n,)).copy()

    # Destroy the C handle immediately.
    dll.mhs_assembly_destroy(pp)
    return Operators(K, C, f_arr)
