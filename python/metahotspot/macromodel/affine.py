"""Base contract and shared mechanism for an affine parametric thermal model.

A concrete :class:`AffineParametricModel` is an opaque handle produced by the
factory (:func:`metahotspot.macromodel.affine.create`); experiment scripts
never name a concrete implementation or reach into a config dataclass.

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
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from metahotspot._compiled_data import CellFields, Operators
from metahotspot.enums import Face, Study
from metahotspot.macromodel.geometry import (
    axis_vertices,
    cell_centers,
    cell_sizes,
    exposed_face_mask,
    grid_indices,
)

from metahotspot.macromodel.utils import (
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
    three arrays align with :attr:`Compiled.cells` order, so every column is
    indexed by the same compact Cell ID.
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
    cells: CellFields, face: Face, coord: float, z_range=None
) -> tuple[np.ndarray, np.ndarray]:
    """Exposed-surface cells + SI face areas for one flat face region.

    ``cells`` is the compiled :class:`~metahotspot.compiled.CellFields` view,
    ``face`` a :class:`~metahotspot.enums.Face` and ``coord`` the face's SI
    coordinate.  A cell is on the face if it is truly exposed across it (no
    active neighbour — :func:`exposed_face_mask`) and its face plane sits at
    ``coord``.  ``z_range`` (optional) restricts lateral-face cells to
    those whose z-centre falls inside ``(zmin, zmax)``; it is not applied to the
    Z faces.  Returns ``(cells, areas)``: the full-domain FVM indices of the
    exposed cells (ascending compact order) and their SI face area (m²).
    """
    face = Face(face)
    candidates = np.flatnonzero((exposed_face_mask(cells) >> int(face)) & 1)
    ijk = grid_indices(cells)
    sizes = cell_sizes(cells)
    if face in (Face.ZM, Face.ZP):
        iz = 0 if face == Face.ZM else cells.dz.size - 1
        candidates = candidates[ijk[candidates, 2] == iz]
        z_face = axis_vertices(cells, 2)[0 if face == Face.ZM else cells.dz.size]
        if not (coord - 1.0e-9 <= z_face <= coord + 1.0e-9):
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        areas = sizes[candidates, 0] * sizes[candidates, 1]
    else:
        if face in (Face.XM, Face.XP):
            axis, sign = 0, 1 if face == Face.XP else 0
            tangent = (1, 2)
        else:  # YM / YP
            axis, sign = 1, 1 if face == Face.YP else 0
            tangent = (0, 2)
        verts = axis_vertices(cells, axis)
        plane = verts[ijk[candidates, axis] + sign]
        keep = (coord - 1.0e-9 <= plane) & (plane <= coord + 1.0e-9)
        if z_range is not None:
            z_center = cells.cz[ijk[candidates, 2]]
            keep &= (z_range[0] - 1.0e-9 <= z_center) & (
                z_center <= z_range[1] + 1.0e-9
            )
        candidates = candidates[keep]
        areas = sizes[candidates, tangent[0]] * sizes[candidates, tangent[1]]
    return candidates.astype(np.int64), np.asarray(areas, dtype=np.float64)


class AffineParametricModel:
    """Affine-parametric thermal model: shared full-domain plumbing + hooks.

    A model is a domain whose heat sources are exposed as source ports
    (FANTASTIC) and whose boundary is partitioned into one or more groups,
    each controlled by a scalar heat-exchange coefficient ``h`` (BCI).  The
    model exposes h-free full-domain operators plus the per-source and
    per-boundary-group data, and knows how to run a native (unreduced) linear
    reference and solve a reduced model on its own geometry.

    ``h_vec`` passed to ``full_reference`` / ``parameter_points`` is the
            *physical* HTC vector in W/m²·K (one scalar per boundary group) — the
            public, validation-facing parameter space (FloTHERM calibration).
            ``BoundaryGroup`` carries the exposed cells and SI area of each group,
            and the per-cell (kx, ky, kz) / cell-side half-distances come from
            :attr:`cell_layout`.  Internally the model performs the FloTHERM
            surface-consistent ThirdType series condensation per cell
            (``p_c = k_c·h / (k_c + h·half_c)``), maps the physical ``h`` to the
            effective affine coefficient ``p = area-weighted p_c``
            (:meth:`physical_to_effective`), and assembles ``K_h = K0 + Σ p_k H_k``
            — the steady-state result reproduces a capacitance-free surface face
            to ≤ 0.001 K of FloTHERM (exactly for homogeneous groups).  The BCI
            ROM (``assemble_reduced_k``) is affine in the *same* effective ``p``:
            callers feed it ``model.physical_to_effective(h)``, and
            :meth:`h_ranges` returns the effective (training) coefficient range.
            Callers never map physical→effective themselves for the reference.

        Subclasses must provide a frozen dataclass ``config`` (with ``ambient_K``,
        ``dt_s``, ``duration_s`` and a ``report_dict()``) and implement the
        geometry hooks: ``name``, ``build_geometry``, ``source_ports``,
        ``boundary_groups``, ``boundary_h``, ``group_h_ranges``, ``source_power``,
        model-defined geometry and physical parameters. Everything else —
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
        # Physical validation space (each group's declared physical range),
        # NOT the effective range — parameter_points feeds the physical
        # scenarios that full_reference maps internally.
        ranges = np.asarray(
            [g.h_range for g in self.boundary_groups()], dtype=np.float64
        )
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

    @cached_property
    def cell_layout(self) -> CellLayout:
        """Per-cell geometry and reference material values from native fields."""
        cells = self._full.cells
        values = self._full.eval_materials()
        conductivity = np.column_stack(
            (
                values["conductivity_x"],
                values["conductivity_y"],
                values["conductivity_z"],
            )
        )
        return CellLayout(
            centers=cell_centers(cells),
            half_sizes=cell_sizes(cells) * 0.5,
            conductivity=conductivity,
        )

    @staticmethod
    def _area_weighted(values, areas) -> float:
        """Area-weighted mean; unweighted mean when ``areas`` sums to zero."""
        values = np.asarray(values, dtype=np.float64)
        areas = np.asarray(areas, dtype=np.float64)
        total = float(areas.sum())
        return float((values * areas).sum() / total) if total else float(values.mean())

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
            out[k] = self._area_weighted(p_cell, areas)
        return out

    def h_ranges(self) -> np.ndarray:
        """Effective affine ranges as an ``(n_groups, 2)`` array.

        Derived from each group's *physical* ``h_range`` through the per-cell
        series-condensed coefficient (one group at a time), so the ROM basis is
        trained over the effective coefficient that actually enters
        ``Σ_k p_k H_k`` — the training space that
        :func:`~metahotspot.macromodel.utils.build_parametric_basis` samples
        and that :func:`~metahotspot.macromodel.utils.assemble_reduced_k` (fed
        ``physical_to_effective``) and :meth:`full_reference` consume.
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
            s_lo = self._area_weighted(p_lo, areas)
            s_hi = self._area_weighted(p_hi, areas)
            out.append((min(s_lo, s_hi), max(s_lo, s_hi)))
        return np.asarray(out, dtype=np.float64)

    # --------------------------------------- native reference + recovery

    def full_reference(self, h_vec) -> AffineSolveResult:
        """Native (unreduced) steady+transient reference at physical ``h_vec``.

        ``h_vec`` is the *physical* HTC vector (W/m²·K), one scalar per
        boundary group in :meth:`boundary_groups` order — the public,
        validation-facing space the ROM is calibrated against.  The model maps
        it internally (:meth:`physical_to_effective`) to the surface-consistent
        effective coefficient ``p`` before assembling

            K_h = K + Σ_k p_k · H_k,   H_k = diag(area)

        which reproduces a capacitance-free (ThirdType) surface face to
        ≤ 0.001 K of FloTHERM for homogeneous groups.  Solves
        ``K_h x = G_src P(t)`` in rise coordinates above ambient; steady uses
        the nominal port powers, transient uses :meth:`source_power`.  The
        reduced model is trained with the *same* effective ``p``
        (:func:`~metahotspot.macromodel.utils.assemble_reduced_k` fed
        ``model.physical_to_effective(h)``), so the only difference between
        this reference and the ROM is the reduction error.  Callers pass
        physical HTC directly — no caller-side mapping.
        """
        K = self._core.K.tocsc()
        C = self._core.C.tocsc()
        G = self.source_shape()
        terms = self.boundary_terms()

        # physical h -> effective (series-condensed) affine coefficient.
        p = self.physical_to_effective(h_vec)
        K_h = K.copy()
        for p_k, H_k in zip(p, terms):
            K_h = K_h + float(p_k) * H_k

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


# ---------------------------------------------------------------------------
# factory / registry  (concrete models register here from playground adapters)
# ---------------------------------------------------------------------------


class _Entry:
    """Registered builder plus its quick-mode config overrides.

    ``builder`` is called as ``builder(overrides: dict | None = None, **kw)``
    and must return an :class:`AffineParametricModel`.  ``quick_overrides`` is
    the model's own recipe for a fast smoke experiment; the factory applies it
    when ``create(..., quick=True)`` is used, so the experiment only has to
    say *whether* it is quick, never *what* that means for a given model.
    """

    __slots__ = ("builder", "quick_overrides")

    def __init__(self, builder, quick_overrides=None):
        self.builder = builder
        self.quick_overrides = quick_overrides


_REGISTRY: dict[str, _Entry] = {}


def register(
    name: str,
    builder: Callable[..., AffineParametricModel],
    *,
    quick_overrides: dict | None = None,
) -> None:
    """Register a concrete model under ``name``.

    ``builder`` is called as ``builder(overrides: dict | None = None, **kw)``
    and must return an :class:`AffineParametricModel`.  ``quick_overrides`` is
    the mapping of scalar config fields the model applies in quick mode (its
    own smoke-experiment recipe).  Re-registering a name replaces the previous
    entry.
    """
    if not name or not name.isidentifier():
        raise ValueError(f"invalid model name: {name!r}")
    _REGISTRY[name] = _Entry(builder, quick_overrides)


def create(name: str, *, quick: bool = False, **kwargs: Any) -> AffineParametricModel:
    """Instantiate the registered model ``name``.

    ``quick`` toggles the model's own quick-mode overrides (``True`` applies
    them); additional ``**kwargs`` are forwarded to the registered builder as
    config overrides and take precedence.  Raises ``KeyError`` for unknown
    names.  The returned value is typed as the abstract base; the concrete
    class is private to the registering package.
    """
    try:
        entry = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown affine parametric model {name!r}; "
            f"registered: {sorted(_REGISTRY)}"
        ) from None
    overrides = entry.quick_overrides if quick else None
    return entry.builder(overrides=overrides, **kwargs)


def registered_names() -> list[str]:
    """Sorted names of all registered models."""
    return sorted(_REGISTRY)
