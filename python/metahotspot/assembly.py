"""High-level wrapper for assembled thermal operators."""

from __future__ import annotations

import ctypes

import numpy as np
from scipy.sparse import csc_matrix

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot.enums import Operator
from metahotspot.types import CscView, MhsAssembly


class Assembly(OwnedHandle):
    """Assembled operators ``C * dx/dt + K * x = f``.

    Do not instantiate directly — use ``Compiled.assemble()``.
    """

    def __init__(self) -> None:
        super().__init__(None, None)

    @classmethod
    def _assemble(cls, dll, compiled_handle, state=None, time=0.0) -> Assembly:
        self = cls()
        self._dll = dll
        self._destroy_fn = dll.mhs_assembly_destroy
        pp = ctypes.POINTER(MhsAssembly)()
        state_ptr = (
            state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if state is not None
            else None
        )
        check(
            dll.mhs_compiled_assemble(
                compiled_handle, state_ptr, time, ctypes.byref(pp)
            ),
            "assemble",
        )
        self._handle = pp
        return self

    def n(self) -> int:
        """Operator dimension (number of global states)."""
        return self._dll.mhs_assembly_n(self._handle)

    def _matrix(self, which: Operator) -> csc_matrix:
        view = CscView()
        check(
            self._dll.mhs_assembly_matrix(self._handle, which, ctypes.byref(view)),
            "assembly_matrix",
        )
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

    def stiffness_matrix(self) -> csc_matrix:
        """Return a copy of the stiffness/conductance matrix K."""
        return self._matrix(Operator.STIFFNESS)

    def capacity_matrix(self) -> csc_matrix:
        """Return a copy of the capacity matrix C."""
        return self._matrix(Operator.CAPACITY)

    def rhs(self) -> np.ndarray:
        """Return a copy of the right-hand side f."""
        return np.ctypeslib.as_array(
            self._dll.mhs_assembly_rhs(self._handle),
            shape=(self.n(),),
        ).copy()

    def operators(self) -> tuple[csc_matrix, csc_matrix, np.ndarray]:
        """Return copies of ``(K, C, f)``."""
        return self.stiffness_matrix(), self.capacity_matrix(), self.rhs()
