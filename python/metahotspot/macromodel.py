"""Macromodel plugin — modal-port coupled solve for MetaHotspot.

This module is loaded on demand, not when ``import metahotspot`` is executed.
SciPy is imported lazily — it's only needed if you construct operators.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from metahotspot._error import check
from metahotspot._lib import get_dll
from metahotspot.assembly import Operators
from metahotspot.solution import Solution
from metahotspot.types import (
    CscView,
    MhsMacroPortModel,
    MhsOperatorsView,
    MhsSolution,
    SolveOptions,
)


# ---- High-level types ----

class PortModel(NamedTuple):
    """Macro port model: Operators + optional basis.

    Parameters
    ----------
    operators : Operators
        Macro K, C, f with dimension macro_state_count × macro_state_count.
    basis : ndarray | None
        Row-major [physical_port_count × macro_state_count] matrix.
        None means unit basis (physical_port_count == macro_state_count).
    physical_port_count : int
        Number of physical interface ports.
    """
    operators: Operators
    basis: np.ndarray | None
    physical_port_count: int


@dataclass
class PortCoupling:
    """Coupling between FVM interface cells and macro physical ports.

    Parameters
    ----------
    model_cells : ndarray
        FVM cell index for each physical port [physical_port_count].
    model_face : int
        Interface face direction (enums.Face value).
    exterior_half_conductance : ndarray
        Macro-side half conductance for each physical port [physical_port_count].
    """
    model_cells: np.ndarray
    model_face: int
    exterior_half_conductance: np.ndarray


def solve(
    compiled,
    port_model: PortModel,
    coupling: PortCoupling,
    state: np.ndarray,
    opts: SolveOptions | None = None,
) -> Solution:
    """Solve an FVM model coupled to a macro port model.

    Parameters
    ----------
    compiled : Compiled
        The compiled FVM model.
    port_model : PortModel
        Macro operators with optional basis.
    coupling : PortCoupling
        Interface geometry and conductances.
    state : ndarray
        Initial state [FVM temps, macro states].
    opts : SolveOptions | None
        Solver options (defaults used if None).

    Returns
    -------
    Solution
        Solution with .state = [FVM temps, macro states].
    """
    # Normalize all arrays
    state = np.ascontiguousarray(state, dtype=np.float64)
    coupling.model_cells = np.ascontiguousarray(
        coupling.model_cells, dtype=np.uintp
    )
    coupling.exterior_half_conductance = np.ascontiguousarray(
        coupling.exterior_half_conductance, dtype=np.float64
    )

    macro_state_count = port_model.operators.f.size
    meta = compiled.metadata()

    # Validate dimensions
    if state.size != meta.cell_count + macro_state_count:
        raise ValueError(
            f"state size ({state.size}) must equal cell_count ({meta.cell_count}) "
            f"+ macro_state_count ({macro_state_count})"
        )
    if coupling.model_cells.size != port_model.physical_port_count:
        raise ValueError("model_cells size must match physical_port_count")
    if coupling.exterior_half_conductance.size != port_model.physical_port_count:
        raise ValueError("exterior_half_conductance size must match physical_port_count")

    has_basis = port_model.basis is not None
    if has_basis:
        basis = np.ascontiguousarray(port_model.basis, dtype=np.float64)
        if basis.ndim != 2:
            raise ValueError("basis must be a 2-D matrix")
        if basis.shape[0] != port_model.physical_port_count:
            raise ValueError("basis rows must equal physical_port_count")
        if basis.shape[1] != macro_state_count:
            raise ValueError("basis cols must equal macro_state_count")
        basis_ptr = basis.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        if port_model.physical_port_count != macro_state_count:
            raise ValueError(
                "unit-basis requires physical_port_count == macro_state_count"
            )
        basis_ptr = None
        basis = None

    # Convert macro Operators to CSC views
    def _csc_input_view(matrix):
        import scipy.sparse
        normalized = scipy.sparse.csc_matrix(matrix, dtype=np.float64)
        normalized.sort_indices()
        if normalized.indices.dtype != np.int32 or normalized.indptr.dtype != np.int32:
            normalized = scipy.sparse.csc_matrix(
                (
                    np.ascontiguousarray(normalized.data, dtype=np.float64),
                    np.ascontiguousarray(normalized.indices, dtype=np.int32),
                    np.ascontiguousarray(normalized.indptr, dtype=np.int32),
                ),
                shape=normalized.shape,
            )
        view = CscView(
            rows=normalized.shape[0],
            columns=normalized.shape[1],
            nnz=normalized.nnz,
            outer_indices=normalized.indptr.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int32)
            ),
            inner_indices=normalized.indices.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int32)
            ),
            values=normalized.data.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
        )
        return normalized, view

    rhs = np.ascontiguousarray(port_model.operators.f, dtype=np.float64)

    normalized_k, k_view = _csc_input_view(port_model.operators.K)
    normalized_c, c_view = _csc_input_view(port_model.operators.C)

    operators_view = MhsOperatorsView(
        K=k_view,
        C=c_view,
        rhs=rhs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n=macro_state_count,
    )
    macro_view = MhsMacroPortModel(
        operators=operators_view,
        basis=basis_ptr,
        physical_port_count=port_model.physical_port_count,
        model_cells=coupling.model_cells.ctypes.data_as(
            ctypes.POINTER(ctypes.c_size_t)
        ),
        model_face=int(coupling.model_face),
        exterior_half_conductance=coupling.exterior_half_conductance.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        ),
    )

    # Keep owners alive until C call returns
    _ = normalized_k, normalized_c, basis, rhs

    dll = get_dll()
    pp = ctypes.POINTER(MhsSolution)()
    opts_ptr = ctypes.byref(opts) if opts is not None else None
    state_ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    check(
        dll.mhs_macromodel_solve(
            compiled._handle,
            ctypes.byref(macro_view),
            state_ptr,
            state.size,
            opts_ptr,
            ctypes.byref(pp),
        ),
        "macromodel_solve",
    )
    return Solution._from_handle(dll, dll.mhs_solution_destroy, pp, compiled)
