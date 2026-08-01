"""Dirichlet-to-Neumann macromodel coupling for MetaHotspot.

Physical port temperatures are always the leading states of a DtN model. This
keeps the coupling graph sparse and removes the former dense physical-port basis
from both the Python and C APIs. Any reduced internal coordinates follow the
exact physical port states.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot._lib import get_ext_dll
from metahotspot.compiled import Operators
from metahotspot.solution import Solution
from metahotspot.types import (
    CscView,
    MhsCompiled,
    MhsOperatorsView,
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


class MhsMacroDtNModel(ctypes.Structure):
    _fields_ = [("operators", MhsOperatorsView)]


_configured_dll_ids: set[int] = set()


def _get_dll():
    dll = get_ext_dll()
    key = id(dll)
    if key not in _configured_dll_ids:
        dll.mhs_macromodel_port_map_create.restype = ctypes.c_int32
        dll.mhs_macromodel_port_map_create.argtypes = [
            ctypes.POINTER(MhsCompiled),
            ctypes.POINTER(MhsMacroPortPatch),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(MhsMacroPortMap)),
        ]
        dll.mhs_macromodel_port_map_destroy.restype = None
        dll.mhs_macromodel_port_map_destroy.argtypes = [ctypes.POINTER(MhsMacroPortMap)]
        dll.mhs_macromodel_port_count.restype = ctypes.c_size_t
        dll.mhs_macromodel_port_count.argtypes = [ctypes.POINTER(MhsMacroPortMap)]
        dll.mhs_macromodel_assemble_dtn.restype = ctypes.c_int32
        dll.mhs_macromodel_assemble_dtn.argtypes = [
            ctypes.POINTER(MhsCompiled),
            ctypes.POINTER(MhsMacroPortMap),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.POINTER(MhsOperatorsView),
        ]
        dll.mhs_macromodel_solve.restype = ctypes.c_int32
        dll.mhs_macromodel_solve.argtypes = [
            ctypes.POINTER(MhsCompiled),
            ctypes.POINTER(MhsMacroPortMap),
            ctypes.POINTER(MhsMacroDtNModel),
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

    Coordinates use SI units after model compilation. The rectangle coordinates
    are (y, z) for X faces, (x, z) for Y faces, and (x, y) for Z faces.
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


class DtNModel(NamedTuple):
    """DtN operators whose leading states are exact physical ports."""

    operators: tuple


PortModel = DtNModel


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
        outer_indices=normalized.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        inner_indices=normalized.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        values=normalized.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return normalized, view


def _csc_output(view: CscView):
    import scipy.sparse

    rows = int(view.rows)
    columns = int(view.columns)
    nnz = int(view.nnz)
    outer = np.ctypeslib.as_array(view.outer_indices, shape=(columns + 1,)).copy()
    inner = np.ctypeslib.as_array(view.inner_indices, shape=(nnz,)).copy()
    values = np.ctypeslib.as_array(view.values, shape=(nnz,)).copy()
    return scipy.sparse.csc_matrix((values, inner, outer), shape=(rows, columns))


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
        self._patches = tuple(patches)

    @property
    def port_count(self) -> int:
        return int(self._dll.mhs_macromodel_port_count(self._handle))

    @property
    def patches(self) -> tuple[PortPatch, ...]:
        return self._patches

    def assemble(self, state: np.ndarray | None = None, time: float = 0.0) -> Operators:
        """Assemble the isolated component as [port temperatures, FVM states]."""
        compiled = self._compiled
        if state is None:
            state = compiled.default_state()
        state = np.ascontiguousarray(state, dtype=np.float64)
        if state.size != compiled.cell_count:
            raise ValueError("state size must equal compiled.cell_count")
        view = MhsOperatorsView()
        check(
            self._dll.mhs_macromodel_assemble_dtn(
                compiled._handle,
                self._handle,
                state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                state.size,
                float(time),
                ctypes.byref(view),
            ),
            "macromodel_assemble_dtn",
        )
        return Operators(
            _csc_output(view.K),
            _csc_output(view.C),
            np.ctypeslib.as_array(view.rhs, shape=(view.n,)).copy(),
        )


def solve(
    compiled, dtn: DtNModel, ports: PortMap, state: np.ndarray, opts=None
) -> Solution:
    """Solve an FVM model coupled to a sparse, exact-port DtN model."""
    if ports._compiled is not compiled:
        raise ValueError("ports were compiled for a different model")

    K, C, f = dtn.operators
    rhs = np.ascontiguousarray(f, dtype=np.float64)
    state = np.ascontiguousarray(state, dtype=np.float64)
    dtn_state_count = rhs.size
    if dtn_state_count < ports.port_count:
        raise ValueError("DtN states must begin with one state per physical port")
    if state.size != compiled.cell_count + dtn_state_count:
        raise ValueError("state size must equal cell_count + DtN state count")

    normalized_k, k_view = _csc_input_view(K)
    normalized_c, c_view = _csc_input_view(C)
    if normalized_k.shape != (dtn_state_count, dtn_state_count):
        raise ValueError("DtN K dimension must match f")
    if normalized_c.shape != (dtn_state_count, dtn_state_count):
        raise ValueError("DtN C dimension must match f")

    operators = MhsOperatorsView(
        K=k_view,
        C=c_view,
        rhs=rhs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n=dtn_state_count,
    )
    model = MhsMacroDtNModel(operators=operators)

    opts_ptr = None
    if opts is not None:
        c_opts = (
            opts._to_c_struct(ports._dll) if hasattr(opts, "_to_c_struct") else opts
        )
        opts_ptr = ctypes.byref(c_opts)

    solution = ctypes.POINTER(MhsSolution)()
    check(
        ports._dll.mhs_macromodel_solve(
            compiled._handle,
            ports._handle,
            ctypes.byref(model),
            state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            state.size,
            opts_ptr,
            ctypes.byref(solution),
        ),
        "macromodel_solve",
    )
    _ = normalized_k, normalized_c
    return Solution._from_handle(
        ports._dll, ports._dll.mhs_solution_destroy, solution, compiled
    )
