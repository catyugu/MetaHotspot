"""High-level wrapper for ``mhs_compiled_t`` — compiled runtime model."""

from __future__ import annotations

import ctypes
from typing import NamedTuple

import numpy as np

from metahotspot._error import check
from metahotspot._handle import OwnedHandle
from metahotspot.types import (
    MhsCompiled,
    CompiledMetadataView,
    MhsStepInfo,
    SolverOpts,
)


CompiledMetadata = NamedTuple(
    "CompiledMetadata",
    [
        ("cell_count", int),
        ("state_count", int),
        ("node_count", int),
        ("grid_count", int),
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
        result = CompiledMetadata(
            cell_count=view.cell_count,
            state_count=view.state_count,
            node_count=view.node_count,
            grid_count=view.grid_count,
            study_type=view.study_type,
            initial_temperature=view.initial_temperature,
            layer_ids=np.ctypeslib.as_array(view.layer_ids, shape=(view.cell_count,)),
            block_ids=np.ctypeslib.as_array(view.block_ids, shape=(view.cell_count,)),
            grid_to_cell=np.ctypeslib.as_array(
                view.grid_to_cell, shape=(view.grid_count,)
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
        return np.full(meta.state_count, meta.initial_temperature, dtype=np.float64)

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

    # ---- Assembly ----

    def assemble(
        self,
        state: np.ndarray,
        time: float = 0.0,
    ) -> Assembly:
        """Assemble K, C, f at a given temperature field and time.

        *state* is required — use ``Compiled.default_state()`` or any
        compatible state vector.
        """
        from metahotspot.assembly import Assembly

        return Assembly._assemble(self._dll, self._handle, state, time)

    # ---- Single transient step ----

    def step(
        self,
        state: np.ndarray,
        time: float,
        dt: float,
        opts: SolverOpts | None = None,
    ) -> tuple[np.ndarray, dict | None]:
        """Execute a single transient time step."""
        n = self.metadata().state_count
        if len(state) != n:
            raise ValueError(f"state length {len(state)} != {n}")
        out_state = np.empty(n, dtype=np.float64)
        step_info = MhsStepInfo()
        opts_ptr = ctypes.byref(opts) if opts is not None else None
        check(
            self._dll.mhs_compiled_step(
                self._handle,
                state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                time,
                dt,
                out_state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.byref(step_info),
                opts_ptr,
            ),
            "step",
        )
        if step_info.error_ratio == 0.0 and step_info.accepted == 0:
            return out_state, None
        return out_state, {
            "error_ratio": step_info.error_ratio,
            "suggested_dt_factor": step_info.suggested_dt_factor,
            "nonlinear_iterations": step_info.nonlinear_iterations,
            "accepted": bool(step_info.accepted),
        }
