"""Base contract and shared mechanism for an affine parametric thermal model.

A concrete :class:`AffineParametricModel` is an opaque handle produced by the
factory (:func:`affine_parametric_models.create`); experiment scripts never
name a concrete implementation or reach into a config dataclass.

The class is deliberately light: it is a **concrete base** that carries all the
shared mechanism — DtN/PortMap plumbing for metahotspot-backed geometries,
temperature recovery, solve options, initial state, counters — and leaves only
the genuinely model-specific hooks (geometry construction, patch layout,
native reference, reduced solve) to each implementation.  Almost every model is
metahotspot-backed, so the DtN plumbing lives here directly.

This mirrors the MOR papers' separation:

* **FANTASTIC (Therminic 2014)** reduces *internal* dynamics only: everything
  after ``(K, C, f, ports)`` is pure linear algebra, so a reduction method
  needs no knowledge of the geometry that produced the operators.
* **BCI Matrix Reduction (Therminic 2015)** makes the boundary-parameter
  structure explicit: ``K(p) = K0 + sum_k p_k K_k`` with one scalar ``h`` per
  boundary group.  The operators here are h-free; ``h`` enters only through
  the boundary-port saturation closure ``g*h*A/(g+h*A)`` that the reduction
  math builds from the per-group ``(cells, g, areas)`` data below.

Every method returns plain numpy/scipy data; no concrete model setting appears
in the shared signatures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from metahotspot.compiled import Operators, SolveOptions
from metahotspot.enums import Study
from metahotspot.macromodel import PortMap, PortPatch, solve as solve_macro

from utils import (
    coordinate_map,
    extract_boundary_groups,
    normalized_operators,
)


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

    cells: np.ndarray  # int64, boundary-port cells in the macro FVM order
    g: np.ndarray  # float64, port conductances k*A/half (W/K)
    areas: np.ndarray  # float64, SI face area of each boundary port (m^2)
    h_range: tuple[float, float] = (1.0, 1.0e6)  # admissible coefficient range


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


class AffineParametricModel:
    """Affine-parametric thermal model: shared DtN plumbing + model hooks.

    A model is a domain whose boundary is partitioned into one or more groups
    (top / side / ...), each controlled by a scalar heat-exchange coefficient
    ``h``.  The model exposes h-free DtN operators plus the per-group boundary
    data, and knows how to run a native reference solve and map reduced
    results back onto its own geometry.

    Subclasses must provide a frozen dataclass ``config`` (with ``ambient_K``,
    ``dt_s``, ``duration_s``, ``nx`` and a ``report_dict()``) and implement the
    geometry hooks: ``name``, ``build_geometry``, ``interface_patches``,
    ``boundary_patch_groups``, ``boundary_h``, ``group_h_ranges``,
    ``detail_interface_patches``, ``detail_nz``, ``monitor_cells``,
    ``port_lookup``.  Everything else — DtN assembly, boundary-group recovery,
    native reference, reduced solve, temperature recovery — is shared here.
    ``parameter_points`` has a default; override it when a model wants its own
    parameter-space sampling (e.g. a product grid over several boundary
    groups).
    """

    # ------------------------------------------------------------------ config

    @property
    def name(self) -> str:
        """Registered model name (opaque label, not a concrete class name)."""
        raise NotImplementedError

    config = None  # frozen dataclass with ambient_K / dt_s / duration_s / nx

    @property
    def ambient_K(self) -> float:
        """Single ambient temperature used for boundary conditions."""
        return self.config.ambient_K

    @property
    def dt(self) -> float:
        """Transient output interval (s); the model's natural time scale."""
        return self.config.dt_s

    # ------------------------------------------------- geometry hooks (required)

    def build_geometry(self, study, *, detail, macro, boundary_h=None):
        """Assemble a metahotspot ``Model`` for the given study/domain split.

        ``boundary_h`` (if given) maps boundary-group name -> coefficient and
        applies the group's convection (used only for the native reference).
        """
        raise NotImplementedError

    def interface_patches(self) -> list[PortPatch]:
        """Interface ``PortPatch`` list on the macro block bottom face."""
        raise NotImplementedError

    def boundary_patch_groups(self):
        """Return ``(groups, areas)`` — boundary port patches + SI areas, in
        boundary-group order."""
        raise NotImplementedError

    def boundary_h(self, h_vec) -> dict[str, float]:
        """Map a parameter vector (one h per group, in order) to group names."""
        raise NotImplementedError

    def group_h_ranges(self) -> tuple[tuple[float, float], ...]:
        """Admissible coefficient range per boundary group, in order."""
        raise NotImplementedError

    def parameter_points(self, count: int = 5) -> list[tuple[float, ...]]:
        """Parameter-space points (one h-vector per boundary group) to validate.

        Default: sweep the first boundary group's admissible range at ``count``
        geometrically spaced points and anchor every remaining group at the
        geometric mean of its own range — for a single-group model this is
        exactly the scalar h sweep.  A model overrides this to describe its own
        parameterization (e.g. the product grid over two independent groups),
        so an experiment never needs to know how many affine parameters a model
        has.
        """
        ranges = self.group_h_ranges()
        if not ranges:
            return []
        first = ranges[0]
        axis = np.geomspace(first[0], first[1], count)
        anchors = tuple(float(np.sqrt(lo * hi)) for lo, hi in ranges[1:])
        return [tuple((float(h), *anchors)) for h in axis]

    def detail_interface_patches(self) -> list[PortPatch]:
        """Interface patches expressed on the detail model (die top)."""
        raise NotImplementedError

    @property
    def detail_nz(self) -> int:
        """Detail-domain z-cell count (macro z-offset in the full layout)."""
        raise NotImplementedError

    def monitor_cells(self) -> np.ndarray:
        """Per-monitor detail-model cell indices (e.g. die-top junctions)."""
        raise NotImplementedError

    def port_lookup(self) -> dict[tuple[int, int], int]:
        """Lateral (ix, iy) -> interface port index (cached)."""
        raise NotImplementedError

    @property
    def macro_nx(self) -> int:
        """Macro block lateral resolution (x cells)."""
        return self.config.nx

    @property
    def macro_ny(self) -> int:
        """Macro block lateral resolution (y cells)."""
        return self.config.nx

    # --------------------------------------------------- DtN core extraction

    @cached_property
    def _macro(self):
        return self.build_geometry(Study.STEADY, detail=False, macro=True).compile()

    @property
    def _interface(self) -> list[PortPatch]:
        return self.interface_patches()

    @cached_property
    def _core(self) -> Operators:
        pm_core = PortMap(self._macro, self._interface)
        return normalized_operators(*pm_core.assemble())

    def core_operators(self) -> Operators:
        """h-free DtN operators ``[interface ports | macro cells]`` (cached)."""
        return self._core

    @property
    def port_count(self) -> int:
        """Number of physical interface ports (leading DtN states)."""
        return len(self._interface)

    def merged_operators(self, group_sizes) -> Operators:
        """``[interface ports | boundary group ports | macro cells]`` DtN.

        Used once to recover per-group boundary data via
        :func:`utils.extract_boundary_groups`.
        """
        groups, _areas = self.boundary_patch_groups()
        all_boundary = [p for g in groups for p in g]
        pm_merged = PortMap(self._macro, self._interface + all_boundary)
        return normalized_operators(*pm_merged.assemble())

    @cached_property
    def _boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        groups, group_areas = self.boundary_patch_groups()
        group_sizes = [len(g) for g in groups]
        merged = self.merged_operators(group_sizes)
        extracted = extract_boundary_groups(merged, len(self._interface), group_sizes)
        return tuple(
            BoundaryGroup(
                cells=extracted[k][0],
                g=extracted[k][1],
                areas=group_areas[k],
                h_range=self.group_h_ranges()[k],
            )
            for k in range(len(groups))
        )

    def boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        """One :class:`BoundaryGroup` per affine parameter (cached)."""
        return self._boundary_groups

    @cached_property
    def macro_grid(self) -> np.ndarray:
        """Macro block ``grid_to_cell`` reshaped to ``(nx, ny, nz)``.

        Column (ix, iy) holds the z-ordered internal cell ids (``-1`` where no
        cell exists).  Used by per-column reduction strategies.
        """
        return self._macro.grid_to_cell.reshape(
            self._macro.nx, self._macro.ny, self._macro.nz
        )

    # --------------------------------------- native reference + recovery

    @cached_property
    def _full_layout(self):
        return self.build_geometry(Study.STEADY, detail=True, macro=True).compile()

    @cached_property
    def _detail_steady(self):
        return self.build_geometry(Study.STEADY, detail=True, macro=False).compile()

    @cached_property
    def _detail_transient(self):
        return self.build_geometry(Study.TRANSIENT, detail=True, macro=False).compile()

    def full_reference(self, h_vec) -> AffineSolveResult:
        """Native steady+transient reference at parameter vector ``h_vec``.

        One coefficient per boundary group.  Geometry is compiled once and
        cached; per-call only h-dependent BC assembly and solves run.
        """
        boundary_h = self.boundary_h(h_vec)
        started = time.perf_counter()
        steady = self.build_geometry(
            Study.STEADY, detail=True, macro=True, boundary_h=boundary_h
        ).compile()
        transient = self.build_geometry(
            Study.TRANSIENT, detail=True, macro=True, boundary_h=boundary_h
        ).compile()
        compile_s = time.perf_counter() - started
        return native_solve_timing(steady, transient, self.solver_options, compile_s)

    @cached_property
    def _detail_ports(self) -> tuple[PortMap, PortMap]:
        detail_patches = self.detail_interface_patches()
        return (
            PortMap(self._detail_steady, detail_patches),
            PortMap(self._detail_transient, detail_patches),
        )

    def solve_reduced(self, operators: Operators, state, transient: bool):
        """Solve a reduced ``Operators`` coupled to the physical ports.

        Returns ``(state, elapsed_s)`` for steady or ``(times, history,
        elapsed_s)`` for transient.
        """
        detail_ports_steady, detail_ports_transient = self._detail_ports
        if transient:
            started = time.perf_counter()
            with solve_macro(
                operators,
                detail_ports_transient,
                state,
                self.solver_options(True),
            ) as solution:
                elapsed = time.perf_counter() - started
                return solution.history_times, solution.state_history, elapsed
        started = time.perf_counter()
        with solve_macro(
            operators,
            detail_ports_steady,
            state,
            self.solver_options(False),
        ) as solution:
            elapsed = time.perf_counter() - started
            return solution.state, elapsed

    def initial_state(self, internal_count: int, internal=None) -> np.ndarray:
        """Reduced initial state ``[detail | port | internal]`` at ambient.

        ``internal_count`` is the reduced internal (macro) coordinate count;
        ``internal`` optionally overrides the internal block (default zeros).
        """
        if internal is None:
            internal = np.zeros(internal_count)
        return np.r_[
            np.full(self.detail_cell_count + self.port_count, self.ambient_K),
            np.asarray(internal, dtype=np.float64),
        ]

    @cached_property
    def _detail_to_full(self) -> np.ndarray:
        mapping = coordinate_map(
            self._detail_steady, self._full_layout, 0, "detail/full"
        )
        if not np.array_equal(
            mapping,
            coordinate_map(
                self._detail_transient, self._full_layout, 0, "transient/full"
            ),
        ):
            raise RuntimeError("steady and transient detail orderings differ")
        return mapping

    @cached_property
    def _macro_to_full(self) -> np.ndarray:
        return coordinate_map(
            self._macro, self._full_layout, self.detail_nz, "macro/full"
        )

    def detail_to_full(self) -> np.ndarray:
        """int64 map from detail-model cells to full-layout cell ids."""
        return self._detail_to_full

    def macro_to_full(self) -> np.ndarray:
        """int64 map from macro-model cells to full-layout cell ids."""
        return self._macro_to_full

    @property
    def full_cell_count(self) -> int:
        return self.detail_cell_count + self.macro_cell_count

    @property
    def detail_cell_count(self) -> int:
        return self.detail_to_full().size

    @property
    def macro_cell_count(self) -> int:
        return self.macro_to_full().size

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
        states = np.atleast_2d(states)
        temperature = np.empty((states.shape[0], self.full_cell_count))
        temperature[:, self.detail_to_full()] = states[:, : self.detail_cell_count]
        internal = (basis @ states[:, self.detail_cell_count + ports :].T).T
        temperature[:, self.macro_to_full()] = (
            internal if ambient_K is None else ambient_K + internal
        )
        return temperature

    def monitor_full(self, detail_cells: np.ndarray) -> np.ndarray:
        """Map detail monitor cells to full-layout indices."""
        return self.detail_to_full()[np.asarray(detail_cells, dtype=np.int64)]

    def report_dict(self) -> dict:
        """Opaque scalar configuration, dumped verbatim into result JSON."""
        return self.config.report_dict()

    def solver_options(self, transient: bool) -> SolveOptions:
        """Shared fixed-step BDF1 solve options for this model."""
        dt = self.config.dt_s if transient else 1.0
        return SolveOptions(
            linear_solver="EigenSparseLU",
            linear_tolerance=1.0e-12,
            linear_max_iterations=5000,
            nonlinear_max_iterations=30,
            nonlinear_relative_tolerance=1.0e-11,
            nonlinear_absolute_tolerance=1.0e-11,
            integrator="Bdf1",
            step_strategy="Fixed",
            error_rel_tol=1.0e-3,
            min_dt=dt,
            max_dt=dt,
            fixed_dt=dt,
        )


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
