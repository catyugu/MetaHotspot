"""Abstract contract for an affine parametric thermal model.

This module defines the *only* surface experiment code is allowed to see.  A
concrete :class:`AffineParametricModel` is an opaque handle produced by the
factory (:func:`affine_parametric_models.create`); experiment scripts never
name a concrete implementation or reach into a config dataclass.

The contract is deliberately model-agnostic in the spirit of the MOR papers:

* **FANTASTIC (Therminic 2014)** reduces *internal* dynamics only: everything
  after ``(K, C, f, ports)`` is pure linear algebra, so a reduction method
  needs no knowledge of the geometry that produced the operators.
* **BCI Matrix Reduction (Therminic 2015)** makes the boundary-parameter
  structure explicit: ``K(p) = K0 + sum_k p_k K_k`` with one scalar ``h`` per
  boundary group.  The operators here are h-free; ``h`` enters only through
  the boundary-port saturation closure ``g*h*A/(g+h*A)`` that the reduction
  math builds from the per-group ``(cells, g, areas)`` data below.

Every method returns plain numpy/scipy data or ``metahotspot`` handles; no
concrete model setting appears anywhere in the signatures.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from metahotspot.compiled import Operators, SolveOptions

if TYPE_CHECKING:
    from metahotspot.macromodel import PortMap

@dataclass(frozen=True)
class BoundaryGroup:
    """One affine parameter: the boundary-port data a single ``h`` controls.

    ``cells`` are 0-based indices into the macro FVM order (the ``n_cell``
    internal block of the core DtN operators).  ``g`` is the port conductance
    ``k*A/half`` recovered from the merged DtN operator; ``areas`` the SI face
    area of each boundary port.  Together with a scalar ``h`` they define the
    exact saturation closure ``g*h*A/(g+h*A)`` added to the cell diagonal.
    ``h_range`` is the admissible coefficient range used for training/holdout.
    """

    name: str
    cells: np.ndarray  # int64, boundary-port cells in the macro FVM order
    g: np.ndarray  # float64, port conductances k*A/half (W/K)
    areas: np.ndarray  # float64, SI face area of each boundary port (m^2)
    h_default: float  # nominal heat-exchange coefficient (W/m^2 K)
    h_range: tuple[float, float] = (1.0, 1.0e6)  # admissible coefficient range


@dataclass(frozen=True)
class StateLayout:
    """Layout of a reduced state vector ``[detail | port | internal]``.

    Experiments use this to build the initial reduced state and to index
    reduced results without knowing anything about the concrete model.
    """

    detail_count: int  # detail-domain cell states
    port_count: int  # exact physical port states
    internal_count: int  # reduced internal (macro) coordinates

    @property
    def total(self) -> int:
        return self.detail_count + self.port_count + self.internal_count


@dataclass(frozen=True)
class AffineSolveResult:
    """Native full-model steady+transient reference at one parameter vector."""

    steady_temperature: np.ndarray  # full-layout temperature field (K)
    times: np.ndarray  # transient output times (s)
    history: np.ndarray  # (n_times, full_cell_count) temperature history (K)
    compile_s: float  # geometry compile wall-clock (s)
    steady_s: float  # steady solve wall-clock (s)
    transient_s: float  # transient solve wall-clock (s)
    full_order: int  # full model cell count


class AffineParametricModel(ABC):
    """Abstract affine-parametric thermal model.

    A model is a domain whose boundary is partitioned into one or more groups
    (top / side / ...), each controlled by a scalar heat-exchange coefficient
    ``h``.  The model exposes h-free DtN operators plus the per-group boundary
    data, and knows how to run a native reference solve and map reduced
    results back onto its own geometry.
    """

    # ------------------------------------------------------------------ identity

    @property
    @abstractmethod
    def name(self) -> str:
        """Registered model name (opaque label, not a concrete class name)."""

    @property
    @abstractmethod
    def ambient_K(self) -> float:
        """Single ambient temperature used for boundary conditions."""

    # --------------------------------------------------- DtN core extraction

    @property
    @abstractmethod
    def port_count(self) -> int:
        """Number of physical interface ports (leading DtN states)."""

    @abstractmethod
    def core_operators(self) -> Operators:
        """h-free DtN operators ``[interface ports | macro cells]`` (cached)."""

    @abstractmethod
    def merged_operators(self, group_sizes: Sequence[int]) -> Operators:
        """``[interface ports | boundary group ports | macro cells]`` DtN.

        Used once by implementations to recover per-group boundary data via
        :func:`utils.extract_boundary_groups`.
        """

    @abstractmethod
    def boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        """One :class:`BoundaryGroup` per affine parameter (cached)."""

    @property
    @abstractmethod
    def macro_cell_count(self) -> int:
        """Internal cell count of the macro block (``core`` internal order)."""

    @property
    @abstractmethod
    def macro_nx(self) -> int:
        """Macro block lateral resolution (x cells)."""

    @property
    @abstractmethod
    def macro_ny(self) -> int:
        """Macro block lateral resolution (y cells)."""

    @abstractmethod
    def macro_grid(self) -> np.ndarray:
        """Macro block ``grid_to_cell`` reshaped to ``(nx, ny, nz)``.

        Column (ix, iy) holds the z-ordered internal cell ids (``-1`` where no
        cell exists).  Used by per-column reduction strategies.
        """

    @property
    @abstractmethod
    def dt(self) -> float:
        """Transient output interval (s); the model's natural time scale."""

    @abstractmethod
    def port_lookup(self) -> dict[tuple[int, int], int]:
        """Lateral (ix, iy) -> interface port index (cached)."""

    # --------------------------------------- native reference + recovery

    @abstractmethod
    def full_reference(self, h_vec: Sequence[float]) -> AffineSolveResult:
        """Native steady+transient reference at parameter vector ``h_vec``.

        One coefficient per :meth:`boundary_groups` group.  Geometry is
        compiled once and cached; per-call only h-dependent BC assembly and
        solves run.
        """

    @abstractmethod
    def state_layout(self, internal_count: int) -> StateLayout:
        """Describe ``[detail | port | internal]`` at a given internal order."""

    @abstractmethod
    def solve_reduced(self, operators: Operators, state, transient: bool):
        """Solve a reduced ``Operators`` coupled to the physical ports.

        Returns ``(state,)`` for steady or ``(times, state_history)`` for
        transient.  The implementation decides how to couple the ports (a
        metahotspot :class:`PortMap` for DLL-backed models, analytic assembly
        for the toy), so experiments need not know.
        """

    @abstractmethod
    def recover_temperature(
        self,
        states,
        *,
        basis,
        ports: int,
        ambient_K: float | None,
    ) -> np.ndarray:
        """Map reduced ``[detail | port | internal]`` states to a full field.

        ``basis`` is the macro internal basis used to lift internal
        coordinates back to macro cells.  Returns a full-layout temperature
        field (or history when ``states`` is 2-D).
        """

    @abstractmethod
    def monitor_cells(self) -> np.ndarray:
        """Per-monitor detail-model cell indices (e.g. die-top junctions)."""

    @abstractmethod
    def monitor_full(self, detail_cells: np.ndarray) -> np.ndarray:
        """Map detail monitor cells to full-layout indices."""

    @abstractmethod
    def report_dict(self) -> dict:
        """Opaque scalar configuration, dumped verbatim into result JSON."""

    @abstractmethod
    def solver_options(self, transient: bool) -> SolveOptions:
        """Shared fixed-step BDF1 solve options for this model."""


def native_solve_timing(
    steady_compiled,
    transient_compiled,
    solver_options,
    compile_s: float = 0.0,
):
    """Shared native steady+transient reference runner.

    Compiles are assumed already done by the caller; this runs the two solves,
    times them, and returns an :class:`AffineSolveResult`.  Model-agnostic:
    works on any pair of compiled steady/transient models with a uniform
    ``solver_options``.
    """
    started = time.perf_counter()
    with steady_compiled.solve(opts=solver_options(False)) as solution:
        steady_temperature = solution.temperature
    steady_s = time.perf_counter() - started

    started = time.perf_counter()
    with transient_compiled.solve(opts=solver_options(True)) as solution:
        times = solution.history_times
        history = solution.temperature_history
    transient_s = time.perf_counter() - started

    return AffineSolveResult(
        steady_temperature=steady_temperature,
        times=times,
        history=history,
        compile_s=compile_s,
        steady_s=steady_s,
        transient_s=transient_s,
        full_order=transient_compiled.cell_count,
    )


def apply_boundary_convection(
    model,
    regions_by_group: dict[str, tuple],
    boundary_h: dict[str, float],
    ambient_K: float,
    z_offset: float,
) -> None:
    """Apply per-group convection from a boundary-group region table.

    ``regions_by_group`` maps a group name to its face regions expressed in the
    *macro frame* (z = 0 at the macro block base): each region is an
    ``(axis, coordinate, a_min, a_max, b_min, b_max)`` tuple exactly as passed
    to :meth:`metahotspot.Model.add_convection`.  ``boundary_h`` maps group
    name -> coefficient; groups with a ``0.0`` value stay insulated.  ``h`` is
    never hard-coded to a specific face here — the region table is the model's
    parameterization, so a model with one, two, or more groups reuses this
    helper unchanged.
    """
    for group, h in boundary_h.items():
        if group not in regions_by_group:
            raise ValueError(f"unknown boundary group {group!r}")
        if not h:
            continue
        regions = []
        for axis, coord, a_min, a_max, b_min, b_max in regions_by_group[group]:
            if axis == 2:  # Z face: coord is the face z, offset it
                coord += z_offset
            else:  # X/Y side face: tangential extents are (., z); offset z
                b_min += z_offset
                b_max += z_offset
            regions.append((axis, coord, a_min, a_max, b_min, b_max))
        model.add_convection(str(float(h)), str(ambient_K), regions)
