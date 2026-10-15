"""Base contract and shared mechanism for an affine parametric thermal model.

A concrete :class:`AffineParametricModel` is built by a playground adapter
(e.g. ``Case1Model`` in ``model_case1.py``); experiment scripts code against
the base contract, never a concrete implementation or a config dataclass.

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

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp

from metahotspot._compiled_data import Operators
from metahotspot.enums import Study
from metahotspot.macromodel.geometry import BoundarySurface, CellGeometry
from metahotspot.macromodel.utils import (
    normalized_operators,
    solve_rom_steady,
    solve_rom_transient,
)


@dataclass(frozen=True)
class BoundaryGroup:
    """A physical surface controlled by one affine Robin parameter.

    Selected FaceIds supply normals and owner distances; projected areas may
    contain several contributions to the same physical CellId. ``h_range`` is
    the physical HTC range; material conductivity comes from ``cell_layout``.
    """

    surface: BoundarySurface
    h_range: tuple[float, float] = (1.0, 1.0e6)

    @property
    def cells(self):
        return self.surface.cell_ids

    @property
    def areas(self):
        return self.surface.areas

    @property
    def face_ids(self):
        return self.surface.face_ids


@dataclass(frozen=True)
class CellLayout:
    """Geometry and reference conductivity in physical CellId order.

    Rows identify cells; columns identify x/y/z. Conductivity is evaluated at
    the cell center, ambient temperature and t=0, in W/(m K).
    """

    centers: np.ndarray  # (cell_count, 3) float64, SI metres
    half_sizes: np.ndarray  # (cell_count, 3) float64, SI metres
    conductivity: np.ndarray  # (cell_count, 3) float64, W/m·K


@dataclass(frozen=True)
class SourcePort:
    """A power input selecting physical source IDs or physical CellIds.

    ``power_W`` is the signed nominal power. ``activity(t)`` scales it, with
    ``None`` denoting constant power. ``source_ids`` preserves independent
    sources that share cells; ``cells`` selects the assembled source field.
    """

    power_W: float
    cells: np.ndarray | None = None  # explicit physical CellId selection
    source_ids: np.ndarray | None = None  # physical source-table columns
    activity: Callable[[float], float] | None = None

    def __post_init__(self):
        if (self.cells is None) == (self.source_ids is None):
            raise ValueError("SourcePort needs exactly one of cells or source_ids")
        if not np.isfinite(self.power_W) or self.power_W == 0:
            raise ValueError("SourcePort.power_W must be finite and nonzero")


@dataclass(frozen=True)
class AffineSolveResult:
    """Native full-model steady+transient reference at one parameter vector."""

    steady_temperature: np.ndarray  # full-layout temperature field (K)
    times: np.ndarray  # transient output times (s)
    history: np.ndarray  # (n_times, full_cell_count) temperature history (K)
    steady_s: float  # steady solve wall-clock (s)
    transient_s: float  # transient solve wall-clock (s)


class AffineParametricModel:
    """Affine-parametric thermal model: shared full-domain plumbing + hooks.

    A model is a domain whose heat sources are exposed as source ports
    (FANTASTIC) and whose boundary is partitioned into one or more groups,
    each controlled by a scalar heat-exchange coefficient ``h`` (BCI).  The
    model exposes h-free full-domain operators plus the per-source and
    per-boundary-group data, and knows how to run a native (unreduced) linear
    reference and solve a reduced model on its own geometry.

    ``h_vec`` passed to ``full_reference`` is the
    *physical* HTC vector in W/m²·K (one scalar per boundary group) — the
    public, validation-facing parameter space (FloTHERM calibration).  The
    model maps it internally to the surface-consistent *effective* affine
    coefficient ``p`` via :meth:`physical_to_effective`, uses ``p`` to
    assemble ``K_h = K0 + Σ p_k H_k``, and exposes the effective training
    range through :meth:`h_ranges`.  Callers pass physical ``h`` and never do
    the mapping themselves.  The BCI ROM is affine in the same effective ``p``.

    Subclasses must provide a frozen dataclass ``config`` (with ``ambient_K``,
    ``dt_s`` and ``duration_s``) and implement the geometry hooks:
    ``name``, ``build_geometry``, ``source_ports``, ``boundary_groups``,
    ``source_power``; the concrete model owns its geometry and physical
    parameters.  Everything else — full-domain assembly, source-shape
    extraction, per-cell geometry (:attr:`cell_layout`), boundary affine
    terms, native reference, reduced solve, temperature recovery — is shared
    here.
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

    def source_power(self, t: float) -> np.ndarray:
        """Port power vector ``P(t)`` (one entry per source port, W).

        Default: constant nominal powers.  A model with time-varying sources
        (e.g. chiplet activity traces) overrides this with ``t -> P(t)``.
        """
        return self.nominal_power()

    # --------------------------------------------------- full-domain assembly

    @cached_property
    def _full(self):
        return self.build_geometry(Study.STEADY, detail=True, macro=True).compile()

    @cached_property
    def geometry(self) -> CellGeometry:
        return CellGeometry(self._full.cells)

    @property
    def full_cell_count(self) -> int:
        """Full-domain FVM cell count."""
        return self._full.cell_count

    @cached_property
    def core(self) -> Operators:
        """Full-domain h-free ``Operators(K, C, f)``.

        ``f`` is the constant heat-source RHS (volumetric power integrated
        over each source cell); ``K`` carries no boundary coefficient (default
        Neumann), the affine Robin terms live in :meth:`boundary_terms`.
        """
        return normalized_operators(*self._full.assemble())

    @cached_property
    def cell_dofs(self) -> np.ndarray:
        """Explicit projection from physical CellId order to algebraic state order."""
        return self._full.cells.cell_to_dof.astype(np.intp)

    @cached_property
    def source_contributions(self):
        """Native sparse contributions with a separate column per physical source."""
        return self._full.source_operator()

    @cached_property
    def source_shape(self) -> np.ndarray:
        """Unit-power source-shape matrix ``G_src`` (N, n_src).

        Column k is the unit-power shape of source port k: the constant
        heat-source RHS over the port's cells, normalized by the port's
        nonzero nominal power. Signed powers are preserved. ``G_src @ P(t)``
        reproduces the native source RHS at any power vector (FANTASTIC 2014 eq. 2).
        """
        f = np.asarray(self.core.f, dtype=np.float64)
        ports = self.source_ports()
        G = np.zeros((f.size, len(ports)), dtype=np.float64)
        for k, port in enumerate(ports):
            if port.source_ids is not None:
                ids = np.asarray(port.source_ids, dtype=np.intp)
                G[:, k] = (
                    np.asarray(self.source_contributions[:, ids].sum(axis=1)).ravel()
                    / port.power_W
                )
            else:
                cells = np.asarray(port.cells, dtype=np.int64)
                dofs = self.cell_dofs[cells]
                G[dofs, k] = f[dofs] / port.power_W
        return G

    @cached_property
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
        n = self.core.K.shape[0]
        terms = []
        for group in self.boundary_groups():
            diagonal = np.zeros(n)
            np.add.at(
                diagonal,
                self.cell_dofs[np.asarray(group.cells, dtype=np.int64)],
                group.areas,
            )
            terms.append(sp.diags(diagonal))
        return terms

    @cached_property
    def cell_layout(self) -> CellLayout:
        """Per-cell geometry and reference material values from native fields."""
        cells = self.geometry
        values = self._full.eval_materials()
        conductivity = np.column_stack(
            (
                values.conductivity_x,
                values.conductivity_y,
                values.conductivity_z,
            )
        )
        return CellLayout(
            centers=cells.centers,
            half_sizes=cells.half_sizes,
            conductivity=conductivity,
        )

    @staticmethod
    def _area_weighted(values, areas) -> float:
        """Area-weighted mean; unweighted mean when ``areas`` sums to zero."""
        values = np.asarray(values, dtype=np.float64)
        areas = np.asarray(areas, dtype=np.float64)
        total = float(areas.sum())
        return float((values * areas).sum() / total) if total else float(values.mean())

    def _effective_per_patch(self, group, h_phys: float) -> np.ndarray:
        """Series-condensed coefficient on each selected native surface patch."""
        fields = self.geometry.fields
        faces = np.asarray(group.face_ids, dtype=np.intp)
        cells = np.asarray(group.cells, dtype=np.intp)
        axes = fields.face_axis[faces]
        k = self.cell_layout.conductivity[cells, axes]
        half = fields.owner_distance[faces]
        h = float(h_phys)
        return k * h / (k + h * half)

    def physical_to_effective(self, physical_h) -> np.ndarray:
        """Map a physical HTC vector → effective affine coefficient per group.

        Effective is the surface-consistent (series-condensed) coefficient the
        ThirdType face actually presents per unit area after static
        condensation of the capacitance-free surface node.  The model contract
        is the *physical* HTC; callers pass physical values in and models do
        the mapping internally before assembling the affine ``K``.
        """
        groups = self.boundary_groups()
        out = np.empty(len(groups), dtype=np.float64)
        for k, group in enumerate(groups):
            p_cell = self._effective_per_patch(group, float(physical_h[k]))
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
        out = []
        for g in groups:
            lo, hi = g.h_range
            p_lo = self._effective_per_patch(g, lo)
            p_hi = self._effective_per_patch(g, hi)
            areas = np.asarray(g.areas, dtype=np.float64)
            s_lo = self._area_weighted(p_lo, areas)
            s_hi = self._area_weighted(p_hi, areas)
            out.append((min(s_lo, s_hi), max(s_lo, s_hi)))
        return np.asarray(out, dtype=np.float64)

    # --------------------------------------- native reference + recovery

    def full_reference(self, h_vec) -> AffineSolveResult:
        """Native (unreduced) steady+transient reference at physical ``h_vec``.

        ``h_vec`` is one *physical* HTC scalar per boundary group in
        :meth:`boundary_groups` order — the public space the ROM is calibrated
        against.  Builds ``K_h = K + Σ_k p_k H_k`` with ``H_k = diag(area)`` and
        ``p = physical_to_effective(h_vec)``, then solves ``K_h x = G_src P(t)``
        in rise coordinates above ambient (steady: nominal port powers;
        transient: :meth:`source_power`).  Because the ROM
        (:func:`~metahotspot.macromodel.utils.assemble_reduced_k`) is trained in
        the same effective ``p``, the only difference between this reference and
        the ROM is the reduction error.
        """
        K = self.core.K.tocsc()
        C = self.core.C.tocsc()
        G = self.source_shape
        terms = self.boundary_terms

        # physical h -> effective (series-condensed) affine coefficient.
        p = self.physical_to_effective(h_vec)
        K_h = K.copy()
        for p_k, H_k in zip(p, terms):
            K_h = K_h + float(p_k) * H_k

        started = time.perf_counter()
        steady_rise = solve_rom_steady(K_h, G, self.nominal_power())
        steady_s = time.perf_counter() - started
        steady_temperature = self.ambient_K + steady_rise[self.cell_dofs]

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
            history=self.ambient_K + history[:, self.cell_dofs],
            steady_s=steady_s,
            transient_s=transient_s,
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
        G = self.source_shape[self.cell_dofs, :]
        if field.ndim == 1:
            return self.ambient_K + G.T @ (field - self.ambient_K)
        return self.ambient_K + (field - self.ambient_K) @ G
