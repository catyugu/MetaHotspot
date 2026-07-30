"""Macromodel plugin — modal-port coupled solve for MetaHotspot.

Usage::

    from metahotspot.macromodel import solve
    solution = solve(compiled, macro, basis, model_cells, model_face,
                     exterior_half_conductance, state, opts=None)
"""

from __future__ import annotations

import ctypes

import numpy as np
from scipy.sparse import csc_matrix

from metahotspot._error import check
from metahotspot._lib import get_dll
from metahotspot.assembly import Operators
from metahotspot.solution import Solution
from metahotspot.types import CscView, MhsAssemblyView, ModalPortView, MhsSolution, SolverOpts


def _csc_input_view(matrix) -> tuple[csc_matrix, CscView]:
    """Normalize a CSC matrix and return a borrowed C view into it."""
    normalized = csc_matrix(matrix, dtype=np.float64)
    normalized.sort_indices()
    if normalized.indices.dtype != np.int32 or normalized.indptr.dtype != np.int32:
        normalized = csc_matrix(
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


def solve(
    compiled,
    macro: Operators,
    basis: np.ndarray,
    model_cells: np.ndarray,
    model_face: int,
    exterior_half_conductance: np.ndarray,
    state: np.ndarray,
    opts: SolverOpts | None = None,
) -> Solution:
    """Solve an FVM model coupled to a modal macro port.

    ``basis`` maps retained modal coefficients to physical port
    temperatures. The model-side half conductance is reevaluated by C++
    during every nonlinear iteration.

    Returns a ``Solution`` whose ``.state`` holds [FVM temps, mode coeffs].
    """
    basis = np.ascontiguousarray(basis, dtype=np.float64)
    model_cells = np.ascontiguousarray(model_cells, dtype=np.uintp)
    exterior_half_conductance = np.ascontiguousarray(
        exterior_half_conductance, dtype=np.float64
    )
    state = np.ascontiguousarray(state, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("basis must be a 2-D physical-port by mode matrix")
    physical_port_count, mode_count = basis.shape
    if model_cells.size != physical_port_count:
        raise ValueError("model_cells size must match basis rows")
    if exterior_half_conductance.size != physical_port_count:
        raise ValueError("exterior_half_conductance size must match basis rows")
    if macro.K.shape != (mode_count, mode_count):
        raise ValueError("macro.K shape must match retained mode count")
    if macro.C.shape != (mode_count, mode_count):
        raise ValueError("macro.C shape must match retained mode count")

    dll = get_dll()
    rhs = np.ascontiguousarray(macro.f, dtype=np.float64)
    if rhs.size != mode_count:
        raise ValueError("macro.f size must match retained mode count")
    meta = compiled.metadata()
    if state.size != meta.cell_count + mode_count:
        raise ValueError("state size must equal cell count + mode count")

    normalized_k, k_view = _csc_input_view(macro.K)
    normalized_c, c_view = _csc_input_view(macro.C)
    operators_view = MhsAssemblyView(
        K=k_view,
        C=c_view,
        rhs=rhs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n=mode_count,
    )
    modal_view = ModalPortView(
        operators=operators_view,
        basis=basis.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        physical_port_count=physical_port_count,
        mode_count=mode_count,
        model_cells=model_cells.ctypes.data_as(
            ctypes.POINTER(ctypes.c_size_t)
        ),
        model_face=int(model_face),
        exterior_half_conductance=exterior_half_conductance.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        ),
    )

    # Keep normalized CSC owners alive until the synchronous C call returns.
    _ = normalized_k, normalized_c

    pp = ctypes.POINTER(MhsSolution)()
    opts_ptr = ctypes.byref(opts) if opts is not None else None
    state_ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    check(
        dll.mhs_compiled_solve_modal_port(
            compiled._handle,
            ctypes.byref(modal_view),
            state_ptr,
            state.size,
            opts_ptr,
            ctypes.byref(pp),
        ),
        "solve_modal_port",
    )
    return Solution._from_handle(dll, dll.mhs_solution_destroy, pp, compiled)
