"""High-level wrapper for ``mhs_compiled_t`` — compiled runtime model."""

from __future__ import annotations

import ctypes
from typing import NamedTuple

import numpy as np

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot.assembly import Operators, _assemble_operators
from metahotspot.types import (
    CscView,
    MhsCompiled,
    CompiledMetadataView,
    MhsOperatorsView,
    SolveOptions,
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
        opts: SolveOptions | None = None,
    ) -> Solution:
        """Solve the compiled model.

        If *state* is provided, it is used as the initial condition.
        Otherwise, the model's initial_temperature is used.
        """
        from metahotspot.solution import Solution

        return Solution._solve_compiled(self, state, opts)

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
