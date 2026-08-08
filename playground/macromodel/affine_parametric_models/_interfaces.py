"""Base contract and shared mechanism for an affine parametric thermal model.

A concrete :class:`AffineParametricModel` is an opaque handle produced by the
factory (:func:`affine_parametric_models.create`); experiment scripts never
name a concrete implementation or reach into a config dataclass.

The class is deliberately light: it is a **concrete base** that carries all the
shared mechanism — full-domain DtN-free operator assembly, heat-source shape
extraction, boundary-group affine terms, the native (unreduced) linear
reference, and the reduced solve — and leaves only the genuinely
model-specific hooks (geometry construction, source/boundary layout) to each
implementation.

This mirrors the MOR papers' separation:

* **FANTASTIC (Therminic 2014)** reduces the *whole* package as a single FEM
  domain, driven by the real power inputs: ``(σM + K) X = g_i`` for every
  source port i (MPMM).  Ports are heat-source regions — their power shapes
  ``g_i`` and the junction temperatures ``T = T0 + Gᵀ x``.
* **BCI Matrix Reduction (Therminic 2015)** makes the boundary-parameter
  structure explicit: ``K(p) = K0 + Σ_k p_k K_k`` with one scalar ``h`` per
  boundary group, and exposes each group as a boundary port ``Ĝ`` so the
  model is boundary-condition independent.  ``h`` enters linearly through the
  per-group affine term ``Ĥ_k = Vᵀ H_k V`` (BCI 2015 eq. 7), never through a
  fixed closure.

Every method returns plain numpy/scipy data; no concrete model setting appears
in the shared signatures.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import numpy as np
import scipy.sparse as sp

from metahotspot.compiled import Operators, SolveOptions
from metahotspot.enums import Study

from utils import (
    normalized_operators,
    solve_rom_steady,
    solve_rom_transient,
)


@dataclass(frozen=True)
class BoundaryGroup:
    """One affine parameter: the boundary data a single scalar ``h`` controls.

    ``cells`` are 0-based indices into the full-domain FVM order; ``areas`` the
    SI exposed face area (m²) of each of those cells on this group's boundary
    surface.  Together with a scalar ``h`` they define the linear affine Robin
    term ``H_k = diag(areas)`` (BCI 2015 eq. 7).  ``h_range`` is the admissible
    coefficient range used for training/holdout.
    """

    cells: np.ndarray  # int64, boundary cells in the full-domain FVM order
    areas: np.ndarray  # float64, SI exposed face area of each cell (m^2)
    h_range: tuple[float, float] = (1.0, 1.0e6)  # admissible coefficient range


@dataclass(frozen=True)
class SourcePort:
    """One heat-source port: a region plus its power waveform.

    ``cells`` are the full-domain FVM indices of the source region,
    ``power_W`` its nominal total power (W), and ``activity`` an optional
    dimensionless time factor ``t -> factor`` (``None`` = constant).  The
    port's power shape is its unit-power distribution ``G_src[:, k]`` and its
    power input is ``P_k(t) = power_W * activity(t)``.
    """

    cells: np.ndarray  # int64, source-region cells in the full-domain order
    power_W: float  # nominal total power (W)
    activity: Callable[[float], float] | None = None  # t -> dimensionless factor


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


def surface_exposed_cells(
    grid_to_cell, x_verts, y_verts, z_verts, face, coord, z_range=None
):
    """Exposed-surface cells + SI face areas for one flat face region.

    ``grid_to_cell`` is the compiled ``(nx, ny, nz)`` occupancy grid (``-1``
    where empty), ``x_verts/y_verts/z_verts`` the SI vertex arrays along each
    axis, ``face`` a :class:`Face` and ``coord`` the face's SI coordinate.  A
    cell is on the face if it touches that surface and has no active neighbour
    across it (i.e. it is truly exposed).  ``z_range`` (optional) restricts the
    cells to those whose z-centre falls inside ``(zmin, zmax)``.  Returns
    ``(cells, areas)``: the full-domain FVM indices of the exposed cells and
    their SI face area (m²).  Only the four lateral faces (X/Y) and the two Z
    faces are supported.
    """
    nx, ny, nz = grid_to_cell.shape
    cells, areas = [], []
    z_center = 0.5 * (np.asarray(z_verts)[:-1] + np.asarray(z_verts)[1:])

    def in_z(iz):
        if z_range is None:
            return True
        return z_range[0] - 1.0e-9 <= z_center[iz] <= z_range[1] + 1.0e-9

    def face_area(ix, iy, iz):
        return (y_verts[iy + 1] - y_verts[iy]) * (z_verts[iz + 1] - z_verts[iz])

    def inside(a, lo, hi):
        return lo - 1.0e-9 <= a <= hi + 1.0e-9

    def neighbour(ix, iy, iz):
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            return grid_to_cell[ix, iy, iz]
        return -1

    if face in (4, 5):  # ZM/ZP
        iz = 0 if face == 4 else nz - 1
        z_face = z_verts[iz] if face == 4 else z_verts[iz + 1]
        if not inside(z_face, coord, coord):
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        above = -1 if face == 4 else 1  # ZM exposed across iz-1, ZP across iz+1
        for ix in range(nx):
            for iy in range(ny):
                cell = grid_to_cell[ix, iy, iz]
                if cell < 0:
                    continue
                nz_ = iz + above
                if 0 <= nz_ < nz and grid_to_cell[ix, iy, nz_] >= 0:
                    continue  # not exposed
                cells.append(cell)
                areas.append(
                    (x_verts[ix + 1] - x_verts[ix]) * (y_verts[iy + 1] - y_verts[iy])
                )
    else:  # XM(0)/XP(1)/YM(2)/YP(3)
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    cell = grid_to_cell[ix, iy, iz]
                    if cell < 0 or not in_z(iz):
                        continue
                    if face in (0, 1):  # X faces
                        x_face = x_verts[ix] if face == 0 else x_verts[ix + 1]
                        if not inside(x_face, coord, coord):
                            continue
                        nix = ix - 1 if face == 0 else ix + 1
                        if 0 <= nix < nx and grid_to_cell[nix, iy, iz] >= 0:
                            continue
                        cells.append(cell)
                        areas.append(face_area(ix, iy, iz))
                    else:  # Y faces (2, 3)
                        y_face = y_verts[iy] if face == 2 else y_verts[iy + 1]
                        if not inside(y_face, coord, coord):
                            continue
                        niy = iy - 1 if face == 2 else iy + 1
                        if 0 <= niy < ny and grid_to_cell[ix, niy, iz] >= 0:
                            continue
                        cells.append(cell)
                        areas.append(
                            (x_verts[ix + 1] - x_verts[ix])
                            * (z_verts[iz + 1] - z_verts[iz])
                        )
    return (
        np.asarray(cells, dtype=np.int64),
        np.asarray(areas, dtype=np.float64),
    )


class AffineParametricModel:
    """Affine-parametric thermal model: shared full-domain plumbing + hooks.

    A model is a domain whose heat sources are exposed as source ports
    (FANTASTIC) and whose boundary is partitioned into one or more groups,
    each controlled by a scalar heat-exchange coefficient ``h`` (BCI).  The
    model exposes h-free full-domain operators plus the per-source and
    per-boundary-group data, and knows how to run a native (unreduced) linear
    reference and solve a reduced model on its own geometry.

    Subclasses must provide a frozen dataclass ``config`` (with ``ambient_K``,
    ``dt_s``, ``duration_s`` and a ``report_dict()``) and implement the
    geometry hooks: ``name``, ``build_geometry``, ``source_ports``,
    ``boundary_groups``, ``boundary_h``, ``group_h_ranges``,
    ``source_power``.  Everything else — full-domain assembly, source-shape
    extraction, boundary affine terms, native reference, reduced solve,
    temperature recovery — is shared here.  ``parameter_points`` has a
    default; override it when a model wants its own parameter-space sampling
    (e.g. a product grid over several boundary groups).
    """

    # ------------------------------------------------------------------ config

    @property
    def name(self) -> str:
        """Registered model name (opaque label, not a concrete class name)."""
        raise NotImplementedError

    config = None  # frozen dataclass with ambient_K / dt_s / duration_s

    @property
    def ambient_K(self) -> float:
        """Single ambient temperature used for boundary conditions."""
        return self.config.ambient_K

    @property
    def dt(self) -> float:
        """Transient output interval (s); the model's natural time scale."""
        return self.config.dt_s

    # ------------------------------------------------- geometry hooks (required)

    def build_geometry(self, study, *, detail, macro):
        """Assemble a metahotspot ``Model`` for the given study/domain split."""
        raise NotImplementedError

    def source_ports(self) -> list[SourcePort]:
        """One :class:`SourcePort` per heat-source port (FANTASTIC ports)."""
        raise NotImplementedError

    def boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        """One :class:`BoundaryGroup` per affine parameter (BCI groups)."""
        raise NotImplementedError

    def boundary_h(self, h_vec) -> dict[str, float]:
        """Map a parameter vector (one h per group, in order) to group names."""
        raise NotImplementedError

    def group_h_ranges(self) -> tuple[tuple[float, float], ...]:
        """Admissible coefficient range per boundary group, in order."""
        raise NotImplementedError

    def source_power(self, t: float) -> np.ndarray:
        """Port power vector ``P(t)`` (one entry per source port, W).

        Default: constant nominal powers.  A model with time-varying sources
        (e.g. chiplet activity traces) overrides this with ``t -> P(t)``.
        """
        return self.nominal_power()

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
        anchors = tuple(float(math.sqrt(lo * hi)) for lo, hi in ranges[1:])
        return [tuple((float(h), *anchors)) for h in axis]

    # --------------------------------------------------- full-domain assembly

    @cached_property
    def _full(self):
        return self.build_geometry(Study.STEADY, detail=True, macro=True).compile()

    @property
    def full_cell_count(self) -> int:
        """Full-domain FVM cell count."""
        return self._full.cell_count

    @cached_property
    def _core(self) -> Operators:
        return normalized_operators(*self._full.assemble())

    def core_operators(self) -> Operators:
        """Full-domain h-free ``Operators(K, C, f)`` (cached).

        ``f`` is the constant heat-source RHS (volumetric power integrated
        over each source cell); ``K`` carries no boundary coefficient (default
        Neumann), the affine Robin terms live in :meth:`boundary_terms`.
        """
        return self._core

    def source_shape(self) -> np.ndarray:
        """Unit-power source-shape matrix ``G_src`` (N, n_src).

        Column k is the unit-power shape of source port k: the constant
        heat-source RHS over the port's cells, normalized by the port's
        nominal power.  ``G_src @ P(t)`` reproduces the native source RHS at
        any power vector (FANTASTIC 2014 eq. 2).
        """
        f = np.asarray(self._core.f, dtype=np.float64)
        ports = self.source_ports()
        G = np.zeros((f.size, len(ports)), dtype=np.float64)
        for k, port in enumerate(ports):
            cells = np.asarray(port.cells, dtype=np.int64)
            scale = max(float(port.power_W), np.finfo(float).tiny)
            G[cells, k] = f[cells] / scale
        return G

    def boundary_terms(self) -> list[sp.diags]:
        """Diagonal affine terms ``H_k = diag(exposed area per cell)``.

        One sparse diagonal matrix per boundary group; ``H_k[cell]`` is the
        exposed area of that cell on group k's surface.  Together with a
        scalar ``h_k`` they form the linear Robin term ``Σ_k h_k H_k``.
        """
        n = self._full.cell_count
        terms = []
        for group in self.boundary_groups():
            diagonal = np.zeros(n)
            diagonal[np.asarray(group.cells, dtype=np.int64)] = group.areas
            terms.append(sp.diags(diagonal))
        return terms

    def boundary_areas(self) -> list[float]:
        """Per-group total exposed area ``A_k`` (m²)."""
        return [float(np.sum(group.areas)) for group in self.boundary_groups()]

    def h_ranges(self) -> np.ndarray:
        """Admissible coefficient ranges as an ``(n_groups, 2)`` array."""
        return np.asarray([g.h_range for g in self.boundary_groups()], dtype=np.float64)

    # --------------------------------------- native reference + recovery

    def full_reference(self, h_vec) -> AffineSolveResult:
        """Native (unreduced) steady+transient reference at ``h_vec``.

        Solves the same full-domain affine-linear system the reduced model
        projects — ``(K + Σ_k h_k H_k) x = G_src P(t)`` in *rise* coordinates
        above ambient — with the full operators, so the reference and the ROM
        differ only by the reduction error (BCI 2015 eqs. 6-7).  Steady uses
        the nominal port powers; transient uses ``source_power(t)``.
        """
        K = self._core.K.tocsc()
        C = self._core.C.tocsc()
        G = self.source_shape()
        terms = self.boundary_terms()

        K_h = K.copy()
        for h, H in zip(h_vec, terms):
            K_h = K_h + float(h) * H
        K_h = (0.5 * (K_h + K_h.T)).tocsc()

        started = time.perf_counter()
        steady_rise = solve_rom_steady(K_h, G, self.nominal_power())
        steady_s = time.perf_counter() - started
        steady_temperature = self.ambient_K + steady_rise

        started = time.perf_counter()
        times, history = solve_rom_transient(
            C,
            K_h,
            G,
            self.source_power,
            dt=self.dt,
            duration=self.config.duration_s,
        )
        transient_s = time.perf_counter() - started
        return AffineSolveResult(
            steady_temperature=steady_temperature,
            times=times,
            history=self.ambient_K + history,
            compile_s=0.0,
            steady_s=steady_s,
            transient_s=transient_s,
            full_order=self._full.cell_count,
        )

    def nominal_power(self) -> np.ndarray:
        """Nominal port power vector (steady-state power input, W)."""
        return np.asarray(
            [port.power_W for port in self.source_ports()], dtype=np.float64
        )

    def junction_temperature(self, field) -> np.ndarray:
        """Per-source-port junction temperature (K) of a full-domain field.

        The port temperature is the *shape-weighted* average of the field
        (FANTASTIC 2014 eq. 3: ``T_j = T0 + Gᵀ x`` with G the port shape
        matrix) — exactly what the reduced model reproduces as
        ``T0 + F̂ᵀ θ``, so the reference and ROM junction temperatures
        coincide by construction on the true field.
        """
        field = np.asarray(field)
        G = self.source_shape()
        if field.ndim == 1:
            return self.ambient_K + G.T @ (field - self.ambient_K)
        return self.ambient_K + (field - self.ambient_K) @ G

    def boundary_temperature(self, field) -> np.ndarray:
        """Per-boundary-group area-averaged temperature (K) of a full-domain field."""
        field = np.asarray(field)
        out = np.empty(len(self.boundary_groups()), dtype=np.float64)
        for k, group in enumerate(self.boundary_groups()):
            cells = np.asarray(group.cells, dtype=np.int64)
            area = float(np.sum(group.areas))
            out[k] = float(np.sum(field[cells] * group.areas) / area) if area else 0.0
        return out

    def recover_temperature(self, theta, basis) -> np.ndarray:
        """Lift reduced interior coordinates back to a full-domain rise field.

        ``theta`` is the reduced interior state (rise above ambient, (n_modes,)
        or (n_times, n_modes)); the recovered field is ``V @ theta`` (K rise).
        """
        theta = np.atleast_2d(theta)
        return theta @ basis.T

    def report_dict(self) -> dict:
        """Opaque scalar configuration, dumped verbatim into result JSON."""
        return self.config.report_dict()

    def solver_options(self, transient: bool) -> SolveOptions:
        """Shared fixed-step BDF1 solve options (reference/integration)."""
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
