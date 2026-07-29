"""High-level wrapper for ``mhs_compiled_t`` — compiled runtime model."""

from __future__ import annotations

import ctypes
from typing import NamedTuple

import numpy as np
from scipy.sparse import csc_matrix

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot.assembly import Operators, _assemble_operators
from metahotspot.types import (
    CscView,
    MhsCompiled,
    CompiledMetadataView,
    MhsAssemblyView,
    ModalPortView,
    SolverOpts,
)


CompiledMetadata = NamedTuple(
    "CompiledMetadata",
    [
        ("cell_count", int),
        ("study_type", int),
        ("initial_temperature", float),
        ("layer_ids", np.ndarray),
        ("block_ids", np.ndarray),
        ("grid_to_cell", np.ndarray),
        ("nx", int),
        ("ny", int),
        ("nz", int),
    ],
)


def _csc_input_view(matrix) -> tuple[csc_matrix, CscView]:
    """Return a normalized CSC matrix and a borrowed C view into it."""
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


class Compiled(OwnedHandle):
    """Read-only compiled runtime model.

    Do not instantiate directly — use ``Model.compile()``.
    """

    def __init__(self) -> None:
        super().__init__(None, None)
        self._metadata_cache: CompiledMetadata | None = None

    @classmethod
    def _from_model(cls, dll, model_handle) -> Compiled:
        """Compile *model_handle* and return a new Compiled instance."""
        self = cls()
        self._dll = dll
        self._destroy_fn = dll.mhs_compiled_destroy
        pp = ctypes.POINTER(MhsCompiled)()
        check(dll.mhs_model_compile(model_handle, ctypes.byref(pp)), "compile")
        self._handle = pp
        return self

    # ---- Metadata view (cached after first call) ----

    def metadata(self) -> CompiledMetadata:
        """Return all compiled-model metadata in one call to the C layer."""
        if self._metadata_cache is not None:
            return self._metadata_cache
        view = CompiledMetadataView()
        check(
            self._dll.mhs_compiled_metadata(self._handle, ctypes.byref(view)),
            "compiled_metadata",
        )
        cell_count = view.cell_count
        result = CompiledMetadata(
            cell_count=cell_count,
            study_type=view.study_type,
            initial_temperature=view.initial_temperature,
            layer_ids=np.ctypeslib.as_array(view.layer_ids, shape=(cell_count,)),
            block_ids=np.ctypeslib.as_array(view.block_ids, shape=(cell_count,)),
            grid_to_cell=np.ctypeslib.as_array(
                view.grid_to_cell,
                shape=(view.nx * view.ny * view.nz,),
            ),
            nx=view.nx,
            ny=view.ny,
            nz=view.nz,
        )
        self._metadata_cache = result
        return result

    # ---- Solve ----

    def default_state(self) -> np.ndarray:
        """Return a uniform state vector filled with ``initial_temperature``."""
        meta = self.metadata()
        return np.full(meta.cell_count, meta.initial_temperature, dtype=np.float64)

    def solve(
        self,
        state: np.ndarray | None = None,
        opts: SolverOpts | None = None,
    ) -> Solution:
        """Solve the compiled model.

        If *state* is provided, it is used as the initial condition.
        Otherwise, the model's initial_temperature is used.
        """
        from metahotspot.solution import Solution

        return Solution._solve_compiled(self, state, opts)

    def solve_modal_port(
        self,
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
        """
        from metahotspot.solution import Solution

        basis = np.ascontiguousarray(basis, dtype=np.float64)
        model_cells = np.ascontiguousarray(model_cells, dtype=np.uintp)
        exterior_half_conductance = np.ascontiguousarray(
            exterior_half_conductance,
            dtype=np.float64,
        )
        state = np.ascontiguousarray(state, dtype=np.float64)
        if basis.ndim != 2:
            raise ValueError("basis must be a 2-D physical-port by mode matrix")
        physical_port_count, mode_count = basis.shape
        if model_cells.size != physical_port_count:
            raise ValueError("model_cells size must match basis rows")
        if exterior_half_conductance.size != physical_port_count:
            raise ValueError(
                "exterior_half_conductance size must match basis rows"
            )
        if macro.K.shape != (mode_count, mode_count):
            raise ValueError("macro.K shape must match retained mode count")
        if macro.C.shape != (mode_count, mode_count):
            raise ValueError("macro.C shape must match retained mode count")

        rhs = np.ascontiguousarray(macro.f, dtype=np.float64)
        if rhs.size != mode_count:
            raise ValueError("macro.f size must match retained mode count")
        if state.size != self.metadata().cell_count + mode_count:
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
        return Solution._solve_modal_port(self, modal_view, state, opts)

    # ---- Assembly ----

    def assemble(
        self,
        state: np.ndarray,
        time: float = 0.0,
    ) -> Operators:
        """Assemble K, C, f at a given temperature field and time.

        *state* is required — use ``Compiled.default_state()`` or any
        compatible state vector.

        Returns an ``Operators`` namedtuple with ``.K``, ``.C``, ``.f``.
        """
        return _assemble_operators(self._dll, self._handle, state, time)
