#!/usr/bin/env python3
"""Embeddable FANTASTIC–BCI ROM extraction: extract once, connect everywhere.

A *subdomain* (an arbitrary connected set of cells of the full model) is reduced
a single time.  Its boundary splits in two roles:

* **BCI faces** — the faces carrying an explicitly declared ambient boundary
  condition.  Their projected terms are assembled directly as
  ``K(p) = K0 + Σ_k p_k Vᵀ H_k V`` at the surface-consistent effective
  coefficient ``p = physical_to_effective(h)``.
* **interface faces** — every *other* boundary face of the subdomain.  These
  become connectable :class:`FacePort` ports and, following the coupling-method
  gold rule, the interface **cells are kept at full FVM resolution** (never
  reduced through the basis).  Only the interior cells are reduced.  The
  interface-node coupling therefore operates on a full-resolution interface
  differential (no basis-truncation error at the interface).

Reduction and coupling math:

* The per-face series conduction from a boundary cell to an interface node is
  ``g = k·A / half`` along the face normal — **not** a homogenised layer ``k``
  and **not** the raw ``k·A`` (the latter is dimensionally wrong and does not
  reproduce the monolithic solve).
* At an artificial cut the extracted diagonal block still carries the phantom
  cross conductance to the now-removed neighbour; it is subtracted so the cut
  face is genuinely adiabatic, then each interface cell is re-coupled to an
  independent node through ``g``.  Two cells joined through a node give the
  series combination ``g_l·g_r/(g_l+g_r) = k·A/(half_l+half_r)`` — exactly the
  FVM face conductance, so identity coupling reproduces the monolithic
  full-domain solve to machine precision (~1e-9 K).
* Non-conforming meshes are joined by building every common patch of the two
  face sets (area-weight ``E``/fraction ``ξ`` at model-definition level).
* Interface temperatures are never reduced (coupling-method gold rule); likewise
  the interface cells themselves stay full-FVM.

The module is model-agnostic (as ``utils.py``): it consumes the ``Operators``
interface, a per-cell geometry view, and the declared ambient groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from metahotspot.enums import Face

from metahotspot.macromodel.utils import build_parametric_basis, normalized_operators

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
    full = model._full.cells
    layout = model.cell_layout
    n = cells.size
    local = {int(c): i for i, c in enumerate(cells)}
    ijk = full.ijk[cells]  # (n,3)
    nx, ny, nz = full.nx, full.ny, full.nz

    in_group = np.zeros(model.full_cell_count, dtype=bool)
    for g in model.boundary_groups():
        in_group[np.asarray(g.cells, dtype=np.int64)] = True
    exposed = full.exposed_face_mask

    by_key: dict[tuple[int, int], list[int]] = {}
    for r in range(n):
        ci, cj, ck = (int(ijk[r, 0]), int(ijk[r, 1]), int(ijk[r, 2]))
        for axis in range(3):
            for direction in (-1, 1):
                if axis == 0:
                    ni = ci + direction
                    in_grid = 0 <= ni < nx
                    nidx = ni * (ny * nz) + cj * nz + ck
                elif axis == 1:
                    nj = cj + direction
                    in_grid = 0 <= nj < ny
                    nidx = ci * (ny * nz) + nj * nz + ck
                else:
                    nk = ck + direction
                    in_grid = 0 <= nk < nz
                    nidx = ci * (ny * nz) + cj * nz + nk
                if (not in_grid) or (nidx not in local):
                    by_key.setdefault((axis, direction), []).append(r)

    ports = []
    for (axis, direction), rows in by_key.items():
        rows = np.asarray(rows, dtype=np.int64)
        fc = cells[rows]
        bit = int(_FACE_BIT[(axis, direction)])
        declared = in_group[fc] & (((exposed[fc] >> bit) & 1) == 1)
        interface = (
            np.flatnonzero(~declared) if not include_ambient else np.arange(rows.size)
        )
        if interface.size == 0:
            continue
        ir = rows[interface]
        fci = cells[ir]
        t1, t2 = _tangent_pair(axis)
        centers = layout.centers[fci]
        half = layout.half_sizes[fci]
        areas = 2.0 * half[:, t1] * 2.0 * half[:, t2]
        rects = np.column_stack(
            (
                centers[:, t1] - half[:, t1],
                centers[:, t1] + half[:, t1],
                centers[:, t2] - half[:, t2],
                centers[:, t2] + half[:, t2],
            )
        )
        ports.append(
            FacePort(
                label=f"{_AXES[axis]}{_DIR[direction]}",
                axis=axis,
                direction=direction,
                cells=ir,
                areas=areas,
                k=layout.conductivity[fci, axis],
                half=half[:, axis],
                t1=t1,
                t2=t2,
                rects=rects,
                ambient=bool(include_ambient and np.all(declared)),
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

    # ---- uniform "side" interface consumed by connect() -----------------
    @property
    def dof_order(self) -> int:
        return self.cells.size

    @property
    def order(self) -> int:
        """Reduction order = the full FVM side is an identity basis of this size."""
        return self.cells.size

    def internal_operator(self) -> sp.csc_matrix:
        """Full-FVM internal stiffness with the ambient folded at effective p."""
        K = self.K.tocsc()
        for pk, Hk in zip(self.effective_p, self.ambient_terms):
            K = K + float(pk) * Hk.tocsc()
        return K.tocsc()

    def capacitance_op(self) -> sp.csc_matrix:
        return self.C.tocsc()

    def rhs_op(self) -> np.ndarray:
        return np.asarray(self.source, dtype=np.float64)

    def port_dofs(self, port: FacePort) -> np.ndarray:
        """Interface cells of a port are full FVM DOFs (identity numbering)."""
        return np.asarray(port.cells, dtype=np.int64)

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


def build_subdomain(
    model,
    cells,
    *,
    name: str,
    physical_h=None,
    ambient_diag=None,
) -> Subdomain:
    """Assemble a :class:`Subdomain` from a cell set of the full model.

    ``physical_h`` is the physical HTC vector (one scalar per declared ambient
    group); ``ambient_diag`` is an alternative full-domain diagonal conductance
    array folding an arbitrary (e.g. spatially varying) external load instead —
    that load is baked directly into ``K`` and is not a BCI group.
    """
    cells = np.asarray(cells, dtype=np.int64)
    core = model.core_operators()
    n = cells.size

    K = core.K.tocsc()[cells, :][:, cells].tocsc()
    C = core.C.tocsc()[cells, :][:, cells].tocsc()
    source = np.asarray(model.source_shape()[cells, :], dtype=np.float64)

    # Declared ambient groups as local affine terms + effective coefficients.
    ambient_terms, ambient_ranges, effective_p = [], [], []
    if ambient_diag is None:
        for term, h_range in zip(model.boundary_terms(), model.h_ranges()):
            diag = np.asarray(term.diagonal()).ravel()[cells]
            ambient_terms.append(sp.diags(diag))
            ambient_ranges.append(list(h_range))
        if physical_h is None:
            effective_p = np.empty(len(ambient_terms), dtype=np.float64)
        else:
            effective_p = np.asarray(
                model.physical_to_effective(physical_h), dtype=np.float64
            )
    else:
        K = K + sp.diags(np.asarray(ambient_diag, dtype=np.float64)[cells])

    # Remove the phantom cross conductance(s) to now-removed neighbours so the
    # cut faces are genuinely adiabatic inside the subdomain.
    outside = np.setdiff1d(np.arange(model.full_cell_count), cells)
    cross = (core.K.tocsc()[cells, :][:, outside]).tocoo()
    phantom = np.zeros(n)
    for row, value in zip(cross.row, cross.data):
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

    @property
    def order(self) -> int:
        """Reduction order = the number of whole-subdomain ROM modes."""
        return self.m

    # ---- uniform "side" interface consumed by connect() -----------------
    @property
    def dof_order(self) -> int:
        return self.m

    def _q_k(self) -> sp.csc_matrix:
        K = self.K0_hat
        for p, H in zip(self.effective_p, self.ambient_hat):
            K = K + float(p) * H
        return K.tocsc()

    def internal_operator(self) -> sp.csc_matrix:
        return self._q_k()

    def capacitance_op(self) -> sp.csc_matrix:
        return self.C_hat.tocsc()

    def rhs_op(self) -> np.ndarray:
        return np.asarray(self.F_hat, dtype=np.float64)

    def port(self, label: str) -> FacePort:
        for p in self.ports:
            if p.label == label:
                return p
        raise KeyError(f"no port {label!r} on {self.name}")

    def boundary_trace(self, label: str) -> np.ndarray:
        return self.boundary_traces[label]

    def boundary_conductance(self, label: str) -> np.ndarray:
        return self.boundary_conductances[label]

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
) -> EmbeddableRom:
    """Reduce the complete subdomain and expose physical interface traces."""
    cells = np.arange(subdomain.cells.size, dtype=np.int64)
    ops = normalized_operators(subdomain.K, subdomain.C, np.zeros(cells.size))
    G = np.asarray(subdomain.source, dtype=np.float64)
    ambient = list(subdomain.ambient_terms)

    basis, summary = build_parametric_basis(
        ops,
        G,
        ambient,
        subdomain.ambient_ranges,
        tolerance=tolerance,
        max_order=max_order,
        probe_rounds=probe_rounds,
        seed=seed,
    )
    C_hat = sp.csc_matrix(basis.T @ ops.C @ basis)
    K0_hat = sp.csc_matrix(basis.T @ ops.K @ basis)
    F_hat = np.asarray(basis.T @ G, dtype=np.float64)
    ambient_hat = [sp.csc_matrix(basis.T @ H @ basis) for H in ambient]

    # Change only the reduced coordinates to C-orthonormal generalized modes.
    # This makes M_hat the identity and K_hat0 diagonal without adding any
    # interface-response directions to the training set.
    modal_k, modal_q = scipy.linalg.eigh(
        K0_hat.toarray(), C_hat.toarray(), check_finite=False
    )
    modal_q = np.asarray(modal_q, dtype=np.float64)
    basis = np.asarray(basis @ modal_q, dtype=np.float64)
    C_hat = sp.eye(modal_k.size, format="csc")
    K0_hat = sp.diags(modal_k, format="csc")
    F_hat = np.asarray(modal_q.T @ F_hat, dtype=np.float64)
    ambient_hat = [
        sp.csc_matrix(modal_q.T @ H.toarray() @ modal_q) for H in ambient_hat
    ]
    boundary_traces = {
        p.label: np.asarray(basis[np.asarray(p.cells), :], dtype=np.float64)
        for p in subdomain.boundary_ports
    }
    boundary_conductances = {
        p.label: np.asarray(p.g, dtype=np.float64) for p in subdomain.boundary_ports
    }

    return EmbeddableRom(
        name=subdomain.name,
        cells=subdomain.cells,
        basis=np.asarray(basis, dtype=np.float64),
        C_hat=C_hat,
        K0_hat=K0_hat,
        F_hat=np.asarray(F_hat, dtype=np.float64),
        ambient_hat=ambient_hat,
        boundary_traces=boundary_traces,
        boundary_conductances=boundary_conductances,
        ambient_ranges=subdomain.ambient_ranges,
        effective_p=subdomain.effective_p,
        ports=subdomain.ports,
        summary=summary,
    )


def side_junction_rise(state, side, offset: int) -> np.ndarray:
    """Per-source-port temperature rise of any side from a coupled state."""
    if hasattr(side, "junction_rise"):
        return side.junction_rise(state, offset)
    return np.asarray(
        side.source.T @ np.asarray(state[offset : offset + side.dof_order]),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# common-patch area weighting (non-conforming meshes)
# ---------------------------------------------------------------------------


def _contains(rects, xl, xr, yl, yr):
    return np.flatnonzero(
        (rects[:, 0] <= xl + 1.0e-12)
        & (rects[:, 1] >= xr - 1.0e-12)
        & (rects[:, 2] <= yl + 1.0e-12)
        & (rects[:, 3] >= yr - 1.0e-12)
    )


def common_patches(port_l: FacePort, port_r: FacePort):
    """Area-weight intersection of two face ports → the shared interface.

    Following the coupling-method reference (Section 5 of the THERMINIC 2017
    paper), the set of common faces ℱ_if is every non-empty intersection of a
    face of the left port's grid with a face of the right port's grid.  For the
    common face set return:

    * ``areas``  — (M_if,) area of each common face;
    * ``E_l``/``E_r`` — sparse (M_if × M_if,S) binary incidence: one iff common
      face *i* intersects side-*S* face *j* (i.e. which side face owns it);
    * ``xi_l``/``xi_r`` — (M_if,) fraction :math:`ξ_{S,i}` of the area of the
      owning side face that is covered by common face *i*;
    * ``li``/``ri`` — owner face index per common face (row equivalents of E).

    Only the two ports' tangential ``rects`` geometry is needed — pure
    model-definition-level data, never any raw-mesh detail of either side.
    Non-conforming grids thus couple through every common-cell-face
    intersection, with *no* unmatched faces.
    """
    rl = port_l.rects
    rr = port_r.rects
    x_edges = np.unique(np.r_[rl[:, (0, 1)], rr[:, (0, 1)]])
    y_edges = np.unique(np.r_[rl[:, (2, 3)], rr[:, (2, 3)]])

    areas, li, ri = [], [], []
    for xl, xr in zip(x_edges[:-1], x_edges[1:]):
        for yl, yr in zip(y_edges[:-1], y_edges[1:]):
            if xr <= xl or yr <= yl:
                continue
            lm = _contains(rl, xl, xr, yl, yr)
            rm = _contains(rr, xl, xr, yl, yr)
            if lm.size and rm.size:
                areas.append((xr - xl) * (yr - yl))
                li.append(lm[0])
                ri.append(rm[0])
    if not areas:
        raise RuntimeError("interface ports do not overlap")

    areas = np.asarray(areas, dtype=np.float64)
    li = np.asarray(li, dtype=np.int64)
    ri = np.asarray(ri, dtype=np.int64)
    npatch = areas.size

    def incidence(idx, nface):
        return sp.coo_matrix(
            (np.ones(npatch), (np.arange(npatch), idx)), shape=(npatch, nface)
        ).tocsc()

    def fraction(idx, face_areas):
        return areas / np.asarray(face_areas, dtype=np.float64)[idx]

    E_l = incidence(li, port_l.areas.size)
    E_r = incidence(ri, port_r.areas.size)
    return (
        areas,
        E_l,
        E_r,
        fraction(li, port_l.areas),
        fraction(ri, port_r.areas),
        li,
        ri,
    )


# ---------------------------------------------------------------------------
# connection (independent-interface-node coupling, full-resolution interface)
# ---------------------------------------------------------------------------


def _diag_at(diag_vals, rows, size):
    """Diagonal sparse matrix with ``diag_vals`` placed at ``rows`` of ``size``."""
    return sp.coo_matrix((diag_vals, (rows, rows)), shape=(size, size)).tocsc()


def interface_trace(side, port, incidence, xi):
    """Paper Section 4: the interface *trace* ``(V_if, h_if)`` of one side.

    ``incidence`` is the common-patch-to-side-face matrix ``E`` and ``xi`` the
    per-patch area fraction, so ``h_if = xi·(E·g)`` is the per-common-face
    conductance (paper's ``diag(ξ)·diag(E·h)``). ``V_if`` maps each common face
    into the side's DOF
    space — the only structure that differs between a detailed and a reduced
    side:

    * ``Subdomain`` (full-FVM): ``V_b = A_b.T`` for the owning cell DOFs;
    * ``EmbeddableRom``: ``V_b = A_b.T·V`` is stored at extraction time;

    In both cases the common-grid trace is ``V_if = E·V_b``.


    In the coupled system this contributes ``V_ifᵀ H_if V_if`` on the side
    block and ``-V_ifᵀ H_if`` to the shared interface node.
    """
    incidence = sp.csr_matrix(incidence)
    V_b = side.boundary_trace(port.label)
    h_b = side.boundary_conductance(port.label)
    h_if = np.asarray(xi, dtype=np.float64) * np.asarray(incidence @ h_b).ravel()
    if isinstance(side, Subdomain):
        # Subdomain trace is the identity over the owning cell DOFs (A_S).
        V_face = sp.coo_matrix(
            (np.ones(V_b.size), (np.arange(V_b.size), V_b)),
            shape=(V_b.size, side.dof_order),
        ).tocsr()
        V_if = incidence @ V_face
        return V_if, h_if
    if isinstance(side, EmbeddableRom):
        return np.asarray(incidence @ V_b, dtype=np.float64), h_if
    raise TypeError(f"unsupported side type: {type(side).__name__}")


def connect(
    left: Subdomain | EmbeddableRom,
    right: Subdomain | EmbeddableRom,
    left_port: FacePort,
    right_port: FacePort,
    *,
    power=None,
):
    """Couple ``left`` and ``right`` through one interface at shared common patches.

    Interface temperatures stay independent (un-reduced) nodes connecting the two
    sides. Each side exposes its boundary conductance
    ``H_S = diag(h_S)`` (``h_S = k·A/half`` per face); against the common face set
    this is replaced by ``diag(ξ_S)·diag(E_S h_S)`` (non-conforming grids, the
    paper's Section 5).  The shared-node block sees ``gl + gr`` and off-diagonal
    ``-gl``/``-gr``, so the whole system is symmetric PSD and identity coupling
    reproduces the monolithic solve.
    Returns ``(K, C, rhs, left_order, right_order, interface_count)``.
    """
    areas, E_l, E_r, xi_l, xi_r, _li, _ri = common_patches(left_port, right_port)
    Vl, hl = interface_trace(left, left_port, E_l, xi_l)
    Vr, hr = interface_trace(right, right_port, E_r, xi_r)
    Vl_s = sp.csr_matrix(Vl)
    Vr_s = sp.csr_matrix(Vr)

    ldof = left.dof_order
    rdof = right.dof_order
    n_patch = areas.size

    Hl = sp.diags(hl)
    Hr = sp.diags(hr)
    left_k = left.internal_operator() + (Vl_s.T @ Hl @ Vl_s)
    right_k = right.internal_operator() + (Vr_s.T @ Hr @ Vr_s)

    all_n = ldof + n_patch + rdof
    K = sp.lil_matrix((all_n, all_n))
    K[:ldof, :ldof] = left_k
    K[ldof + n_patch :, ldof + n_patch :] = right_k
    node = ldof + np.arange(n_patch)
    K[:ldof, node] = -(Vl_s.T @ Hl)
    K[node, :ldof] = -(Hl @ Vl_s)
    K[ldof + n_patch :, node] = -(Vr_s.T @ Hr)
    K[node, ldof + n_patch :] = -(Hr @ Vr_s)
    for i in range(n_patch):
        K[node[i], node[i]] = hl[i] + hr[i]
    K = K.tocsc()

    C = sp.block_diag(
        (
            left.capacitance_op(),
            sp.csc_matrix((n_patch, n_patch)),
            right.capacitance_op(),
        ),
        format="csc",
    )
    n_src = left.rhs_op().shape[1]
    p = np.asarray(power if power is not None else np.ones(n_src), dtype=np.float64)
    rhs = np.r_[left.rhs_op() @ p, np.zeros(n_patch), right.rhs_op() @ p]
    return K, C, rhs, ldof, rdof, n_patch


def solve_system(K, C, rhs, dt: float, duration: float):
    """Steady + fixed-step BDF1 transient of a coupled system (coordinates)."""
    steady = np.asarray(spla.spsolve(K.tocsc(), rhs)).ravel()
    lhs = (K.tocsc() + C.tocsc() / dt).tocsc()
    solver = spla.splu(lhs)
    state = np.zeros(K.shape[0])
    history = [state.copy()]
    for _ in range(round(duration / dt)):
        state = solver.solve(C @ state / dt + rhs)
        history.append(state.copy())
    return steady, np.asarray(history)
