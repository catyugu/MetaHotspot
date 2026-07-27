"""High-level wrapper for assembled thermal operators."""

from __future__ import annotations

import ctypes

import numpy as np
from scipy.sparse import csc_matrix

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
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


class Assembly(OwnedHandle):
    """Assembled operators ``C * dx/dt + K * x = f``.

    Do not instantiate directly — use ``Compiled.assemble()``.

    Access matrices via the ``.K``, ``.C``, ``.f`` properties (lazily fetched
    from the C layer on first access)::

        K, C, f = assembly.K, assembly.C, assembly.f
    """

    def __init__(self) -> None:
        super().__init__(None, None)
        self._view: MhsAssemblyView | None = None

    @classmethod
    def _assemble(
        cls, dll, compiled_handle, state: np.ndarray, time: float = 0.0
    ) -> Assembly:
        self = cls()
        self._dll = dll
        self._destroy_fn = dll.mhs_assembly_destroy
        pp = ctypes.POINTER(MhsAssembly)()
        state_ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        check(
            dll.mhs_compiled_assemble(
                compiled_handle, state_ptr, len(state), time, ctypes.byref(pp)
            ),
            "assemble",
        )
        self._handle = pp
        return self

    def _fetch_view(self) -> MhsAssemblyView:
        if self._view is None:
            self._view = MhsAssemblyView()
            check(
                self._dll.mhs_assembly_view(self._handle, ctypes.byref(self._view)),
                "assembly_view",
            )
        return self._view

    @property
    def K(self) -> csc_matrix:
        """Stiffness / conductance matrix."""
        return _csc_from_view(self._fetch_view().K)

    @property
    def C(self) -> csc_matrix:
        """Capacity (mass) matrix."""
        return _csc_from_view(self._fetch_view().C)

    @property
    def f(self) -> np.ndarray:
        """Right-hand side vector (copy)."""
        v = self._fetch_view()
        return np.ctypeslib.as_array(v.rhs, shape=(v.n,)).copy()
