"""Dirichlet-to-Neumann macromodel coupling for MetaHotspot.

Physical port temperatures are the leading states of every DtN operator. Any
reduced internal coordinates follow those exact physical port states.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from metahotspot._error import check
from metahotspot._dll_interface import _opts_ptr
from metahotspot._handle import OwnedHandle
from metahotspot._lib import get_ext_dll
from metahotspot.compiled import Operators, _operators_from_handle
from metahotspot.solution import Solution
from metahotspot.types import (
    MhsCompiled,
    MhsOperators,
    MhsSolution,
    Rect2D,
    _SolveOptionsCStruct,
)


class MhsMacroPortMap(ctypes.Structure):
    pass


class MhsMacroPortPatch(ctypes.Structure):
    _fields_ = [
        ("face", ctypes.c_int32),
        ("coordinate", ctypes.c_double),
        ("rectangle", Rect2D),
    ]


_configured_dll_ids: set[int] = set()


def _get_dll():
    dll = get_ext_dll()
    key = id(dll)
    if key in _configured_dll_ids:
        return dll

    dll.mhs_macromodel_port_map_create.restype = ctypes.c_int32
    dll.mhs_macromodel_port_map_create.argtypes = [
        ctypes.POINTER(MhsCompiled),
        ctypes.POINTER(MhsMacroPortPatch),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.POINTER(MhsMacroPortMap)),
    ]
    dll.mhs_macromodel_port_map_destroy.restype = None
    dll.mhs_macromodel_port_map_destroy.argtypes = [ctypes.POINTER(MhsMacroPortMap)]
    dll.mhs_macromodel_assemble_dtn.restype = ctypes.c_int32
    dll.mhs_macromodel_assemble_dtn.argtypes = [
        ctypes.POINTER(MhsMacroPortMap),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.POINTER(ctypes.POINTER(MhsOperators)),
    ]
    dll.mhs_macromodel_solve.restype = ctypes.c_int32
    dll.mhs_macromodel_solve.argtypes = [
        ctypes.POINTER(MhsMacroPortMap),
        ctypes.POINTER(MhsOperators),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(_SolveOptionsCStruct),
        ctypes.POINTER(ctypes.POINTER(MhsSolution)),
    ]
    _configured_dll_ids.add(key)
    return dll


@dataclass(frozen=True)
class PortPatch:
    """One geometric boundary patch and therefore one physical DtN port.

    Coordinates use SI units after model compilation. Rectangle coordinates are
    (y, z) for X faces, (x, z) for Y faces, and (x, y) for Z faces.
    """

    face: int
    coordinate: float
    rectangle: tuple[float, float, float, float]

    def _to_c(self) -> MhsMacroPortPatch:
        a_min, a_max, b_min, b_max = self.rectangle
        return MhsMacroPortPatch(
            int(self.face),
            float(self.coordinate),
            Rect2D(float(a_min), float(a_max), float(b_min), float(b_max)),
        )


def _csc_arrays(matrix):
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
    return normalized


def _create_operator_handle(dll, operators: Operators):
    K, C, f = operators
    rhs = np.ascontiguousarray(f, dtype=np.float64)
    normalized_k = _csc_arrays(K)
    normalized_c = _csc_arrays(C)
    expected_shape = (rhs.size, rhs.size)
    if normalized_k.shape != expected_shape:
        raise ValueError("DtN K dimension must match f")
    if normalized_c.shape != expected_shape:
        raise ValueError("DtN C dimension must match f")

    handle = ctypes.POINTER(MhsOperators)()
    check(
        dll.mhs_operators_create(
            rhs.size,
            normalized_k.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            normalized_k.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            normalized_k.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            normalized_k.nnz,
            normalized_c.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            normalized_c.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            normalized_c.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            normalized_c.nnz,
            rhs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(handle),
        ),
        "operators_create",
    )
    return handle


class PortMap(OwnedHandle):
    """C++-compiled mapping from geometric patches to exposed FVM faces."""

    def __init__(self, compiled, patches: Sequence[PortPatch]):
        dll = _get_dll()
        super().__init__(dll.mhs_macromodel_port_map_destroy, dll)
        if not patches:
            raise ValueError("at least one port patch is required")
        c_patches = (MhsMacroPortPatch * len(patches))(
            *(patch._to_c() for patch in patches)
        )
        handle = ctypes.POINTER(MhsMacroPortMap)()
        check(
            dll.mhs_macromodel_port_map_create(
                compiled._handle, c_patches, len(patches), ctypes.byref(handle)
            ),
            "macromodel_port_map_create",
        )
        self._handle = handle
        self._compiled = compiled
        self._port_count = len(patches)

    @property
    def port_count(self) -> int:
        return self._port_count

    def assemble(self, state: np.ndarray | None = None, time: float = 0.0) -> Operators:
        """Assemble the isolated component as [port temperatures, FVM states]."""
        if state is None:
            state = self._compiled.default_state()
        state = np.ascontiguousarray(state, dtype=np.float64)
        if state.size != self._compiled.cell_count:
            raise ValueError("state size must equal compiled.cell_count")
        handle = ctypes.POINTER(MhsOperators)()
        check(
            self._dll.mhs_macromodel_assemble_dtn(
                self._handle,
                state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                state.size,
                float(time),
                ctypes.byref(handle),
            ),
            "macromodel_assemble_dtn",
        )
        return _operators_from_handle(self._dll, handle)


def solve(
    operators: Operators,
    ports: PortMap,
    state: np.ndarray,
    opts=None,
) -> Solution:
    """Solve an FVM model coupled to sparse, exact-port DtN operators."""
    K, C, f = operators
    rhs = np.ascontiguousarray(f, dtype=np.float64)
    state = np.ascontiguousarray(state, dtype=np.float64)
    dtn_state_count = rhs.size
    if dtn_state_count < ports.port_count:
        raise ValueError("DtN states must begin with one state per physical port")
    if state.size != ports._compiled.cell_count + dtn_state_count:
        raise ValueError("state size must equal cell_count + DtN state count")

    operator_handle = _create_operator_handle(ports._dll, Operators(K, C, rhs))
    opts_ptr = _opts_ptr(opts, ports._dll)

    solution = ctypes.POINTER(MhsSolution)()
    try:
        check(
            ports._dll.mhs_macromodel_solve(
                ports._handle,
                operator_handle,
                state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                state.size,
                opts_ptr,
                ctypes.byref(solution),
            ),
            "macromodel_solve",
        )
    finally:
        ports._dll.mhs_operators_destroy(operator_handle)
    return Solution._from_handle(ports._dll, ports._dll.mhs_solution_destroy, solution)
