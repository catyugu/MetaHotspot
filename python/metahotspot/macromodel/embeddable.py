#!/usr/bin/env python3
"""Embeddable FANTASTIC-BCI ROM extraction: extract once, connect everywhere.

A *subdomain* (an arbitrary connected set of cells of the full model) is
reduced a single time.  Its boundary splits in two roles:

* **BCI/ambient faces** — the faces carrying an explicitly declared ambient
  boundary condition.  Their projected terms are assembled directly as
  ``K(p) = K0 + Σ_k p_k Vᵀ H_k V`` at the surface-consistent effective
  coefficient ``p = physical_to_effective(h)``.
* **interface faces** — every *other* boundary face of the subdomain.  These
  become connectable :class:`FacePort` ports.

**Embedded-EROM interface (FloTHERM contract).**  The whole subdomain —
interior *and* boundary cells — is reduced into one modal basis, and the
interface is exposed as a connection surface rather than as ``h``-symbolic
affine terms:

* the interface faces enter the *boundary-parameter sampling during
  extraction* (same affine-BCI treatment as the ambient groups), which is what
  gives the reduced basis its response richness under an arbitrary external
  attachment;
* the exported boundary is the **modal trace** ``Vb`` (the basis restricted to
  the interface cells) plus the per-face **series conduction**
  ``X = (dz/2)/k``, equivalently ``g = A/X = k·A/half``;
* the reduced model is **modalized** so ``M = I`` and ``K = diag(λ)`` with one
  ~0 ambient mode.

Only the *full-FVM* side (:class:`Subdomain`) keeps its interface cells
explicit; the reduced side (:class:`EmbeddableRom`) reproduces its face
temperature through the trace ``T_face = Vb·q``.

The module is model-agnostic (as ``utils.py``): it consumes the ``Operators``
interface, a per-cell geometry view, and the declared ambient groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from metahotspot.enums import Face
from metahotspot.macromodel.utils import (
    build_parametric_basis,
    normalized_operators,
)

# (axis, direction) -> Face enum bit  (XM, XP, YM, YP, ZM, ZP = 0..5)
_FACE_BIT = {
    (0, -1): Face.XM,
    (0, +1): Face.XP,
    (1, -1): Face.YM,
    (1, +1): Face.YP,
    (2, -1): Face.ZM,
    (2, +1): Face.ZP,
}
_AXES = {0: "x", 1: "y", 2: "z"}
_DIR = {-1: "-", +1: "+"}


# ---------------------------------------------------------------------------
# boundary-face ports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FacePort:
    """One planar set of boundary faces sharing a face normal.

    ``cells`` are indices into the *subdomain-local* cell order; ``areas``,
    ``k``, ``half`` are per local cell (``k``/``half`` along the face normal).
    ``rects`` is the (n,4) tangential rectangle ``[t1lo, t1hi, t2lo, t2hi]`` in
    SI, used for area-weight matching of non-conforming interfaces.
    """

    label: str
    axis: int  # 0=x, 1=y, 2=z face normal
    direction: int  # -1 low (Xi/Yi/Zi), +1 high
    cells: np.ndarray  # subdomain-local cell indices
    areas: np.ndarray  # SI face area per cell
    k: np.ndarray  # SI conductivity along `axis` per cell
    half: np.ndarray  # SI centre->face half distance along `axis`
    t1: int  # first tangent global axis
    t2: int  # second tangent global axis
    rects: np.ndarray  # (n,4) tangential rectangles, SI
    ambient: bool = False

    @property
    def g(self) -> np.ndarray:
        """Per-cell series conduction to an interface node: ``k·A/half``."""
        return self.k * self.areas / self.half

    @property
    def normal(self) -> str:
        return f"{_AXES[self.axis]}{_DIR[self.direction]}"


def _tangent_pair(axis: int):
    return [a for a in range(3) if a != axis]


# ---------------------------------------------------------------------------
# subdomain construction + boundary enumeration
# ---------------------------------------------------------------------------


def enumerate_interface_ports(model, cells, *, include_ambient=False) -> list[FacePort]:
    """Every boundary face of ``cells`` not covered by a declared ambient BC.

    A face is a boundary face of the subdomain when the cell across it is either
    outside the grid (exterior) or not part of ``cells`` (an artificial cut). A
    boundary face whose cell belongs to a :meth:`model.boundary_groups` group and
    is genuinely exposed there is *declared* (kept as a BCI ambient group, not a
    port); every other boundary face becomes a connectable :class:`FacePort`.
    """
    cells = np.asarray(cells, dtype=np.int64)
    full = model.geometry
    layout = model.cell_layout
    local = {int(c): i for i, c in enumerate(cells)}
    fields = full.fields
    ambient_faces = {
        int(f) for group in model.boundary_groups() for f in group.surface.facet_ids
    }
    by_key = {}
    invalid = np.iinfo(fields.facet_plus.dtype).max
    for fid, (minus, plus) in enumerate(
        zip(fields.facet_minus, fields.facet_plus, strict=True)
    ):
        for cell, other, sign in (
            (int(minus), int(plus), 1),
            (int(plus), int(minus), -1),
        ):
            if cell == invalid or cell not in local or other in local:
                continue
            ambient = other == invalid and fid in ambient_faces
            if ambient and not include_ambient:
                continue
            key = (int(fields.facet_axis[fid]), sign, ambient)
            by_key.setdefault(key, []).append((local[cell], fid))
    ports = []
    for (axis, direction, ambient), entries in by_key.items():
        rows = np.asarray([r for r, _ in entries], dtype=np.int64)
        facet_ids = np.asarray([f for _, f in entries], dtype=np.int64)
        fc = cells[rows]
        t1, t2 = _tangent_pair(axis)
        rects = fields.facet_bounds[facet_ids]
        distances = full.half_sizes[fc, axis]
        ports.append(
            FacePort(
                label=f"{_AXES[axis]}{_DIR[direction]}",
                axis=axis,
                direction=direction,
                cells=rows,
                areas=(rects[:, 1] - rects[:, 0]) * (rects[:, 3] - rects[:, 2]),
                k=layout.conductivity[fc, axis],
                half=distances,
                t1=t1,
                t2=t2,
                rects=rects,
                ambient=ambient,
            )
        )
    return ports


@dataclass
class Subdomain:
    """A full-FVM side (all cells explicit) plus its connectable boundary ports.

    ``K`` is stored ambient-free (``h=0``) with the phantom cross conductances at
    artificial cuts removed; the declared ambient groups are carried as affine
    ``ambient_terms`` / ``ambient_ranges`` / ``effective_p`` and folded at solve
    time by :meth:`internal_operator`.  ``ports`` are the connectable (non-BC)
    boundary faces and are always full-FVM cells (never reduced).
    """

    name: str
    cells: np.ndarray  # full-domain FVM indices
    K: sp.csc_matrix  # local internal stiffness, ambient-free, cuts adiabatic
    C: sp.csc_matrix
    source: np.ndarray  # (n, n_src) unit-power shapes
    ports: list[FacePort]
    boundary_ports: list[FacePort] = field(default_factory=list)
    ambient_terms: list[sp.csc_matrix] = field(default_factory=list)
    ambient_ranges: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    effective_p: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Accessors for the archived coupling experiments.
    @property
    def dof_order(self) -> int:
        return self.cells.size

    def internal_operator(self) -> sp.csc_matrix:
        """Full-FVM internal stiffness with the ambient folded at effective p."""
        K = self.K.tocsc()
        for pk, Hk in zip(self.effective_p, self.ambient_terms, strict=False):
            K = K + float(pk) * Hk.tocsc()
        return K.tocsc()

    def port(self, label: str) -> FacePort:
        for p in self.ports:
            if p.label == label:
                return p
        raise KeyError(f"no port {label!r} on {self.name}")

    def boundary_trace(self, label: str) -> np.ndarray:
        port = next(p for p in self.boundary_ports if p.label == label)
        return np.asarray(port.cells, dtype=np.int64)

    def boundary_conductance(self, label: str) -> np.ndarray:
        port = next(p for p in self.boundary_ports if p.label == label)
        return np.asarray(port.g, dtype=np.float64)

    @property
    def capacity(self) -> sp.csc_matrix:
        return self.C

    @property
    def source_matrix(self) -> np.ndarray:
        return self.source

    def interface_trace(self, port: FacePort, incidence, xi):
        """Return the common-face trace and conductance for this FVM side."""
        incidence = sp.csr_matrix(incidence)
        h_b = self.boundary_conductance(port.label)
        h_if = np.asarray(xi, dtype=np.float64) * np.asarray(incidence @ h_b).ravel()
        face_cells = self.boundary_trace(port.label)
        V_face = sp.coo_matrix(
            (np.ones(face_cells.size), (np.arange(face_cells.size), face_cells)),
            shape=(face_cells.size, self.dof_order),
        ).tocsr()
        return incidence @ V_face, h_if

    def junction_rise(self, state, offset: int) -> np.ndarray:
        return self.source.T @ np.asarray(state[offset : offset + self.dof_order])


def build_subdomain(
    model,
    cells,
    *,
    name: str,
    physical_h=None,
) -> Subdomain:
    """Assemble a :class:`Subdomain` from a cell set of the full model.

    ``physical_h`` is the physical HTC vector (one scalar per declared ambient
    group); ``None`` leaves the ambient terms with an undefined effective
    coefficient (caller folds them later via ``effective_p``).
    """
    cells = np.asarray(cells, dtype=np.int64)
    core = model.core
    n = cells.size

    K = core.K.tocsc()[cells, :][:, cells].tocsc()
    C = core.C.tocsc()[cells, :][:, cells].tocsc()
    source = np.asarray(model.source_shape[cells, :], dtype=np.float64)

    # Declared ambient groups as local affine terms + effective coefficients.
    ambient_terms, ambient_ranges = [], []
    for term, h_range in zip(model.boundary_terms, model.h_ranges(), strict=True):
        diag = np.asarray(term.diagonal()).ravel()[cells]
        ambient_terms.append(sp.diags(diag))
        ambient_ranges.append(list(h_range))
    if physical_h is None:
        effective_p = np.empty(len(ambient_terms), dtype=np.float64)
    else:
        effective_p = np.asarray(
            model.physical_to_effective(physical_h), dtype=np.float64
        )

    # Remove the phantom cross conductance(s) to now-removed neighbours so the
    # cut faces are genuinely adiabatic inside the subdomain.
    outside = np.setdiff1d(np.arange(model.full_cell_count), cells)
    cross = (core.K.tocsc()[cells, :][:, outside]).tocoo()
    phantom = np.zeros(n)
    for row, value in zip(cross.row, cross.data, strict=True):
        if value < 0.0:
            phantom[row] -= value
    K = (K - sp.diags(phantom)).tocsc()

    all_ports = enumerate_interface_ports(model, cells, include_ambient=True)
    ports = [p for p in all_ports if not p.ambient]
    return Subdomain(
        name=name,
        cells=cells,
        K=K,
        C=C,
        source=source,
        ports=ports,
        boundary_ports=all_ports,
        ambient_terms=ambient_terms,
        ambient_ranges=np.asarray(ambient_ranges, dtype=np.float64),
        effective_p=effective_p,
    )


# ---------------------------------------------------------------------------
# embeddable ROM (whole-subdomain basis + explicit interface nodes)
# ---------------------------------------------------------------------------


@dataclass
class EmbeddableRom:
    """Whole-subdomain BCI ROM with independent physical interface nodes."""

    name: str
    cells: np.ndarray  # full-domain FVM indices (reporting only)
    basis: np.ndarray  # (n_cells, m)
    C_hat: sp.csc_matrix
    K0_hat: sp.csc_matrix
    F_hat: np.ndarray  # Vᵀ·source (m, n_src)
    ambient_hat: list[sp.csc_matrix]  # Vᵀ H_k V
    boundary_traces: dict[str, np.ndarray]  # A_b.T @ V, one trace per boundary
    boundary_conductances: dict[str, np.ndarray]  # h_b per boundary
    ambient_ranges: np.ndarray  # (n_groups, 2) effective ranges
    effective_p: np.ndarray  # effective coefficient of this side's ambient load
    ports: list[FacePort]
    summary: dict = field(default_factory=dict)

    m: int = field(init=False)

    def __post_init__(self):
        self.m = int(self.basis.shape[1])

    # Accessors for the archived coupling experiments.
    @property
    def dof_order(self) -> int:
        return self.m

    def internal_operator(self) -> sp.csc_matrix:
        """Modal effective stiffness: K0_hat + Σ p_k VᵀH_kV, ambient folded."""
        K = self.K0_hat
        for p, H in zip(self.effective_p, self.ambient_hat, strict=False):
            K = K + float(p) * H
        return K.tocsc()

    def port(self, label: str) -> FacePort:
        for p in self.ports:
            if p.label == label:
                return p
        raise KeyError(f"no port {label!r} on {self.name}")

    def boundary_trace(self, label: str) -> np.ndarray:
        return self.boundary_traces[label]

    def boundary_conductance(self, label: str) -> np.ndarray:
        return self.boundary_conductances[label]

    @property
    def capacity(self) -> sp.csc_matrix:
        return self.C_hat

    @property
    def source_matrix(self) -> np.ndarray:
        return self.F_hat

    def interface_trace(self, port: FacePort, incidence, xi):
        """Return the common-face trace and conductance for this ROM side."""
        incidence = sp.csr_matrix(incidence)
        h_b = self.boundary_conductance(port.label)
        h_if = np.asarray(xi, dtype=np.float64) * np.asarray(incidence @ h_b).ravel()
        return (
            np.asarray(incidence @ self.boundary_trace(port.label), dtype=np.float64),
            h_if,
        )

    def junction_rise(self, state, offset: int) -> np.ndarray:
        """Per-source-port temperature rise from a full coupled state.
        ``state`` is the whole coupled state; ``offset`` the position of this
        side's ROM block within it.
        """
        q = np.asarray(state[offset : offset + self.dof_order])
        return self.F_hat.T @ q


def extract_rom(
    subdomain: Subdomain,
    *,
    tolerance=1.0e-3,
    max_order=512,
    probe_rounds=2,
    seed=20260825,
    interface_h_range=(1.0, 1.0e4),
) -> EmbeddableRom:
    """Reduce the whole subdomain and expose the interfaces as physical traces.

    Faithful to the FloTHERM embedded-EROM contract (project skill reference
    ``erom-embedding-contract.md``): the interface faces are treated as *affine
    boundary parameters* during basis enrichment -- the same treatment as the
    declared ambient groups -- so the modal basis captures the subdomain
    response under an arbitrary external attachment.  They are then *exported*
    as a connection surface (the modal trace ``Vb`` restricted to the interface
    cells + the per-face series conduction ``X = (dz/2)/k``, i.e.
    ``g = A/X = k*A/half``) instead of as ``h``-symbolic affine terms.  Declared
    ambient groups keep the affine ``Vᵀ H V`` export.  The reduced model is
    modalized so ``M = I`` and ``K = diag(λ)`` with one ~0 ambient mode.
    """

    cells = np.arange(subdomain.cells.size, dtype=np.int64)
    ops = normalized_operators(subdomain.K, subdomain.C, np.zeros(cells.size))
    G = np.asarray(subdomain.source, dtype=np.float64)

    # Enrichment over the declared ambient groups AND every interface port.
    boundary_terms = list(subdomain.ambient_terms)
    h_ranges = [list(r) for r in subdomain.ambient_ranges]
    interface_areas = np.zeros(cells.size, dtype=np.float64)
    for port in subdomain.ports:
        np.add.at(interface_areas, np.asarray(port.cells, dtype=np.int64), port.areas)
    if subdomain.ports:
        boundary_terms.append(sp.diags(interface_areas, format="csc"))
        h_ranges.append(list(interface_h_range))

    basis, summary = build_parametric_basis(
        ops,
        G,
        boundary_terms,
        np.asarray(h_ranges, dtype=np.float64),
        tolerance=tolerance,
        max_order=max_order,
        probe_rounds=probe_rounds,
        seed=seed,
    )

    return _project_rom(subdomain, basis, summary)


def _project_rom(
    subdomain: Subdomain, basis: np.ndarray, summary: dict
) -> EmbeddableRom:
    """Project and modalize a response basis, then export physical face traces."""
    modal_k, modal_q = scipy.linalg.eigh(
        basis.T @ subdomain.K @ basis,
        basis.T @ subdomain.C @ basis,
        check_finite=False,
    )
    basis = np.ascontiguousarray(basis @ modal_q)
    return EmbeddableRom(
        name=subdomain.name,
        cells=subdomain.cells,
        basis=basis,
        C_hat=sp.eye(modal_k.size, format="csc"),
        K0_hat=sp.diags(modal_k, format="csc"),
        F_hat=basis.T @ subdomain.source,
        ambient_hat=[
            sp.csc_matrix(basis.T @ H @ basis) for H in subdomain.ambient_terms
        ],
        boundary_traces={p.label: basis[p.cells, :] for p in subdomain.boundary_ports},
        boundary_conductances={p.label: p.g for p in subdomain.boundary_ports},
        ambient_ranges=subdomain.ambient_ranges,
        effective_p=subdomain.effective_p,
        ports=subdomain.ports,
        summary=summary,
    )
