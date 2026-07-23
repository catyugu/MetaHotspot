"""High-level wrapper for ``mhs_assembly_t`` — assembled linear system."""

from __future__ import annotations

import ctypes

import numpy as np
from scipy.sparse import csc_matrix

from metahotspot._error import check
from metahotspot.types import MhsAssembly


class Assembly:
    """Assembled linear system ``K * x = f`` in CSC format.

    Do not instantiate directly — use ``Compiled.assemble()``.
    """

    def __init__(self) -> None:
        self._dll = None
        self._handle: MhsAssembly | None = None
        self._owned = True

    @classmethod
    def _assemble(cls, dll, compiled_handle, T=None, time=0.0) -> Assembly:
        self = cls()
        self._dll = dll
        pp = ctypes.POINTER(MhsAssembly)()
        T_ptr = T.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if T is not None else None
        check(
            dll.mhs_compiled_assemble(compiled_handle, T_ptr, time, ctypes.byref(pp)),
            "assemble",
        )
        self._handle = pp
        return self

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._owned and self._handle is not None:
            self._dll.mhs_assembly_destroy(self._handle)
            self._handle = None

    # ---- Accessors ----

    def n(self) -> int:
        """Matrix dimension (number of active cells)."""
        return self._dll.mhs_assembly_n(self._handle)

    def nnz(self) -> int:
        """Number of non-zero entries."""
        return self._dll.mhs_assembly_nnz(self._handle)

    def as_csc(self) -> csc_matrix:
        """Return ``(K, f)`` — the stiffness matrix as ``scipy.sparse.csc_matrix``
        and the right-hand side as a 1-D ndarray (copies, not views)."""  # noqa: E501
        n = self.n()
        nnz = self.nnz()

        outer = np.ctypeslib.as_array(
            self._dll.mhs_assembly_outer_indices(self._handle),
            shape=(n + 1,),
        ).copy()
        inner = np.ctypeslib.as_array(
            self._dll.mhs_assembly_inner_indices(self._handle),
            shape=(nnz,),
        ).copy()
        vals = np.ctypeslib.as_array(
            self._dll.mhs_assembly_values(self._handle),
            shape=(nnz,),
        ).copy()
        rhs = np.ctypeslib.as_array(
            self._dll.mhs_assembly_rhs(self._handle),
            shape=(n,),
        ).copy()

        K = csc_matrix((vals, inner, outer), shape=(n, n))
        return K, rhs

    def mass_diagonal(self) -> np.ndarray:
        """Mass diagonal vector (copy)."""
        n = self.n()
        ptr = self._dll.mhs_assembly_mass_diagonal(self._handle)
        return np.ctypeslib.as_array(ptr, shape=(n,)).copy()
