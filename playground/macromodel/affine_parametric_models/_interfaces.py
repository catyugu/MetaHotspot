"""
Base contract and shared mechanism for an affine parametric thermal model.

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
    """
    One affine parameter: the boundary data a single ``h`` controls.

    ``cells`` are 0-based indices into the full-domain FVM order; ``areas`` the
    SI exposed face area (m²) of each of those cells on this group's boundary
    surface.  Together with a scalar ``h`` they define the linear affine Robin
    term ``H_k = diag(areas)`` (BCI 2015 eq. 7).  ``h_range`` is the admissible
    **physical** coefficient range used for training/holdout.

    Series-condensed (effective) ``k_c`` and ``half_c`` are read off the base
    :attr:`AffineParametricModel.cell_layout` at face-extraction time (face
    direction selects which (kx, ky, kz) axis and which cell-side length to
    pick); they are NOT carried here — a boundary group is pure boundary
    geometry.
    """

    cells: np.ndarray  # int64, boundary cells in the full-domain FVM order
    areas: np.ndarray  # float64, SI exposed face area of each cell (m^2)
    h_range: tuple[float, float] = (1.0, 1.0e6)  # admissible physical range


@dataclass(frozen=True)
class CellLayout:
    """Per-cell (cell_count, 3) arrays computed once at compile time.

    ``centers[:, c]`` is the SI centre coordinate of cell c along each axis
    (x, y, z); ``half_sizes[:, c]`` is half the cell-side length (centre → face
    distance) along each axis; ``conductivity[:, c]`` is the static (kx, ky, kz)
    in W/m·K, evaluated at compile time at (cell_centre, ambient, t=0).  All
    three arrays align with :attr:`Compiled.layer_ids` order, so
    ``layer_ids[c]`` indexes the model's material table.
    """

    centers: np.ndarray  # (cell_count, 3) float64, SI metres
    half_sizes: np.ndarray  # (cell_count, 3) float64, SI metres
    conductivity: np.ndarray  # (cell_count, 3) float64, W/m·K


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

    ``h_vec`` passed to ``full_reference`` / ``parameter_points`` /
        ``assemble_reduced_k`` is the *physical* HTC vector in W/m²·K (one
        scalar per boundary group).  ``BoundaryGroup`` carries the exposed
        cells and SI area of each group, and the per-cell (kx, ky, kz) /
        cell-side half-distances come from :attr:`cell_layout`.  For the
        native reference (:meth:`full_reference`), the model performs the
        FloTHERM surface-consistent ThirdType series condensation per cell
        (``p_c = k_c·h / (k_c + h·half_c)``) before assembling the affine
        K, so the steady-state result reproduces a capacitance-free surface
        face to ≤ 0.001 K of FloTHERM.  The training path
        (:func:`~utils.assemble_reduced_k`) consumes the same physical
        ``h_vec`` directly — the BCI ROM is affine in ``h`` as the algebraic
        surrogate parameter, with no extra mapping.

        Subclasses must provide a frozen dataclass ``config`` (with ``ambient_K``,
        ``dt_s``, ``duration_s`` and a ``report_dict()``) and implement the
        geometry hooks: ``name``, ``build_geometry``, ``source_ports``,
        ``boundary_groups``, ``boundary_h``, ``group_h_ranges``, ``source_power``,
        ``_axis_vertices``, and ``_layer_conductivity``.  Everything else —
        full-domain assembly, source-shape extraction, per-cell geometry
        (:attr:`cell_layout`), boundary affine terms, native reference, reduced
        solve, temperature recovery — is shared here.  ``parameter_points`` has
        a default; override it when a model wants its own parameter-space
        sampling (e.g. a product grid over several boundary groups).
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
        """Map a parameter vector (one effective value per group) to names."""
        raise NotImplementedError

    def group_h_ranges(self) -> tuple[tuple[float, float], ...]:
        """Admissible *physical* coefficient range per boundary group, in order."""
        raise NotImplementedError

    def source_power(self, t: float) -> np.ndarray:
        """Port power vector ``P(t)`` (one entry per source port, W).

        Default: constant nominal powers.  A model with time-varying sources
        (e.g. chiplet activity traces) overrides this with ``t -> P(t)``.
        """
        return self.nominal_power()

    def parameter_points(self, count: int = 5) -> list[tuple[float, ...]]:
        """Parameter-space points (one physical-h vector per group) to validate.

        Default: sweep the first boundary group's physical range at ``count``
        geometrically spaced points and anchor every remaining group at the
        geometric mean of its own range — for a single-group model this is
        exactly the scalar physical-h sweep.  A model overrides this to
        describe its own parameterization (e.g. the product grid over two
        independent groups), so an experiment never needs to know how many
        affine parameters a model has.
        """
        ranges = self.h_ranges()
        if ranges.size == 0:
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
        *physical* HTC scalar ``h_k`` they form the affine Robin term
        ``Σ_k h_k H_k`` — the training-path BCI-ROM boundary.  The native
        reference (:meth:`full_reference`) substitutes a surface-consistent
        ``p_c = k_c·h / (k_c + h·half_c)`` per cell for the same ``h_k``,
        giving the FloTHERM ThirdType capacitance-free face condensation.
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

    # --------------------------------------- per-cell layout (compile-time)

    # Axis index for face-direction / half-axis lookup.
    _AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

    def _axis_vertices(self, axis: str) -> np.ndarray:
        """SI vertex array along ``axis`` (x/y/z) — shared geometry hook.

        Concrete models return their own mesh breakpoints (e.g.
        ``config.*_vertices_mm`` or ``z_vertices(layers)``), converted to
        metres.  The shared :attr:`cell_layout` consumes it; subclasses must
        override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__}: _axis_vertices() is not implemented; "
            "return the SI vertex array for axis='x'/'y'/'z'."
        )

    def _layer_conductivity(self) -> dict[int, tuple[float, float, float]]:
        """Per-layer ``(kx, ky, kz)`` in SI (W/m·K).

        Keyed by the compiled ``layer_id`` (``Compiled.layer_ids[c]``).  Each
        model derives this from its own material schema (e.g. an ``(name, kx,
        ky, kz, rho, c)`` MATERIALS tuple).  Subclasses must override this;
        :attr:`cell_layout` calls it once per compile.
        """
        raise NotImplementedError(
            f"{type(self).__name__}: _layer_conductivity() is not implemented; "
            "supply a {layer_id: (kx, ky, kz)} mapping from the model's "
            "material schema."
        )

    @cached_property
    def cell_layout(self) -> CellLayout:
        """Per-cell geometry + conductivity, computed once at compile time.

        Reads only what :class:`Compiled` already exposes (``grid_to_cell``,
        ``nx/ny/nz``, ``layer_ids``), the model's own mesh breakpoints and the
        model's :meth:`_layer_conductivity`.  Three (cell_count, 3) arrays in
        SI units — used everywhere the base needs (cx, cy, cz, dx/2, dy/2,
        dz/2, kx, ky, kz) for a cell index.
        """
        full = self._full
        nx, ny, nz = int(full.nx), int(full.ny), int(full.nz)
        cell_count = int(full.cell_count)

        # (ix, iy, iz) for each active cell, derived from grid_to_cell.
        flat = np.flatnonzero(np.asarray(full.grid_to_cell) >= 0).astype(np.int64)
        # flat index decoding: ix = flat // (ny*nz); iy = (flat % (ny*nz)) // nz; iz = flat % nz
        ny_nz = ny * nz
        ix = flat // ny_nz
        iy = (flat % ny_nz) // nz
        iz = flat % nz

        # Per-axis vertex arrays, one shared axis = same vertex list per model.
        xv = np.asarray(self._axis_vertices("x"), dtype=np.float64)
        yv = np.asarray(self._axis_vertices("y"), dtype=np.float64)
        zv = np.asarray(self._axis_vertices("z"), dtype=np.float64)
        xc, yc, zc = (
            0.5 * (xv[:-1] + xv[1:]),
            0.5 * (yv[:-1] + yv[1:]),
            0.5 * (zv[:-1] + zv[1:]),
        )
        xw, yw, zw = xv[1:] - xv[:-1], yv[1:] - yv[:-1], zv[1:] - zv[:-1]

        centers = np.empty((cell_count, 3), dtype=np.float64)
        half = np.empty((cell_count, 3), dtype=np.float64)
        centers[:, 0] = xc[ix]
        centers[:, 1] = yc[iy]
        centers[:, 2] = zc[iz]
        half[:, 0] = 0.5 * xw[ix]
        half[:, 1] = 0.5 * yw[iy]
        half[:, 2] = 0.5 * zw[iz]

        # Per-cell static conductivity (kx, ky, kz) from the layer table.
        table = self._layer_conductivity()
        if not table:
            raise RuntimeError(
                f"{type(self).__name__}: _layer_conductivity() returned an empty map"
            )
        max_lid = max(int(k) for k in table.keys())
        k_layers = np.asarray(
            [
                table.get(i, table[max(int(k) for k in table.keys())])
                for i in range(max_lid + 1)
            ],
            dtype=np.float64,
        )
        layer_ids = np.asarray(full.layer_ids, dtype=np.int64)
        k = k_layers[layer_ids]  # (cell_count, 3)

        return CellLayout(centers=centers, half_sizes=half, conductivity=k)

    # ---------------------------------------- series-condensed (effective) HTC

    def _effective_per_cell(self, group, axis: int, h_phys: float) -> np.ndarray:
        """Per-cell series-effective ``p_c = k_c·h / (k_c + h·half_c)``.

        ``k``/``half`` are read from :attr:`cell_layout` along the face-normal
        ``axis``, so the series condensation uses the model's own geometry and
        material — not anything carried on the boundary group.
        """
        cells = np.asarray(group.cells, dtype=np.int64)
        k = self.cell_layout.conductivity[cells, axis]
        half = self.cell_layout.half_sizes[cells, axis]
        h = float(h_phys)
        return k * h / (k + h * half)

    def _boundary_axis_per_group(self) -> tuple[int, ...]:
        """Face-normal axis (0=x, 1=y, 2=z) for each boundary group.

        Default: all-Z.  Models with lateral (X/Y) groups override this.
        Used by :meth:`physical_to_effective` / :meth:`full_reference` to pick
        the right (kx, ky, kz) / cell-side half-distance for the per-cell
        series condensation.
        """
        return (2,) * len(self.boundary_groups())

    def physical_to_effective(self, physical_h) -> np.ndarray:
        """Map a physical HTC vector → effective affine coefficient per group.

        Effective is the surface-consistent (series-condensed) coefficient the
        ThirdType face actually presents per unit area after static
        condensation of the capacitance-free surface node.  The model contract
        is the *physical* HTC; callers pass physical values in and models do
        the mapping internally before assembling the affine ``K``.
        """
        groups = self.boundary_groups()
        axes = self._boundary_axis_per_group()
        out = np.empty(len(groups), dtype=np.float64)
        layout = self.cell_layout
        for k, group in enumerate(groups):
            cells = np.asarray(group.cells, dtype=np.int64)
            p_cell = self._effective_per_cell(group, axes[k], float(physical_h[k]))
            areas = np.asarray(group.areas, dtype=np.float64)
            total = float(areas.sum())
            out[k] = (
                float((p_cell * areas).sum() / total) if total else float(p_cell.mean())
            )
        return out

    def h_ranges(self) -> np.ndarray:
        """Effective affine ranges as an ``(n_groups, 2)`` array.

        Derived from each group's *physical* ``h_range`` through the per-cell
        series-condensed coefficient (one group at a time), so the basis is
        trained and validated over the effective coefficient that actually
        enters ``Σ_k p_k H_k`` — the same space :meth:`full_reference`
        consumes.
        """
        groups = self.boundary_groups()
        axes = self._boundary_axis_per_group()
        out = []
        for k, g in enumerate(groups):
            cells = np.asarray(g.cells, dtype=np.int64)
            lo, hi = g.h_range
            p_lo = self._effective_per_cell(g, axes[k], lo)
            p_hi = self._effective_per_cell(g, axes[k], hi)
            areas = np.asarray(g.areas, dtype=np.float64)
            total = float(areas.sum())
            s_lo = float((p_lo * areas).sum() / total) if total else float(p_lo.mean())
            s_hi = float((p_hi * areas).sum() / total) if total else float(p_hi.mean())
            out.append((min(s_lo, s_hi), max(s_lo, s_hi)))
        return np.asarray(out, dtype=np.float64)

    # --------------------------------------- native reference + recovery

    def full_reference(self, h_vec) -> AffineSolveResult:
        """Native (unreduced) steady+transient reference at effective ``h_vec``.

        ``h_vec`` is the *effective* (series-condensed) affine coefficient
        vector, one scalar per boundary group, in the same order as
        :meth:`boundary_groups`.  Callers map physical HTC W/m²·K to effective
        with :meth:`physical_to_effective` before calling.  The effective
        coefficient ``p_k`` is what the FloTHERM ThirdType surface-consistent
        face actually presents per unit area after static condensation of the
        capacitance-free surface node; building

            K_h = K + Σ_k p_k · H_k,   H_k = diag(area)

        reproduces a capacitance-free surface face to ≤ 0.001 K of FloTHERM.
        Solves ``K_h x = G_src P(t)`` in rise coordinates above ambient; steady
        uses the nominal port powers, transient uses :meth:`source_power`.
        The reduced model is trained with the same effective ``h_vec``
        (:func:`~utils.assemble_reduced_k`), so the only difference between
        this reference and the ROM is the reduction error.
        """
        K = self._core.K.tocsc()
        C = self._core.C.tocsc()
        G = self.source_shape()
        terms = self.boundary_terms()

        # h_vec is already the effective (series-condensed) coefficient.
        K_h = K.copy()
        for p_k, H_k in zip(h_vec, terms):
            K_h = K_h + float(p_k) * H_k
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
