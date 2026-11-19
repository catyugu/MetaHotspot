"""Historical interface-node experiments kept alongside their research drivers."""

from __future__ import annotations

import itertools

import numpy as np
import pyamg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from legacy_embeddable import EmbeddableRom, FacePort, Subdomain


# ---------------------------------------------------------------------------
# common-patch area weighting (non-conforming meshes)
# ---------------------------------------------------------------------------


_EDGE_TOL = 1.0e-9


def _contains(rects, xl, xr, yl, yr):
    return np.flatnonzero(
        (rects[:, 0] <= xl + _EDGE_TOL)
        & (rects[:, 1] >= xr - _EDGE_TOL)
        & (rects[:, 2] <= yl + _EDGE_TOL)
        & (rects[:, 3] >= yr - _EDGE_TOL)
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
    for xl, xr in itertools.pairwise(x_edges):
        for yl, yr in itertools.pairwise(y_edges):
            if (xr - xl) <= 2.0 * _EDGE_TOL or (yr - yl) <= 2.0 * _EDGE_TOL:
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
    Returns ``(K, C, rhs, left_order, right_order, interfacet_count)``.
    """
    areas, E_l, E_r, xi_l, xi_r, _li, _ri = common_patches(left_port, right_port)
    Vl, hl = left.interface_trace(left_port, E_l, xi_l)
    Vr, hr = right.interface_trace(right_port, E_r, xi_r)
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
            left.capacity.tocsc(),
            sp.csc_matrix((n_patch, n_patch)),
            right.capacity.tocsc(),
        ),
        format="csc",
    )
    left_source = left.source_matrix
    right_source = right.source_matrix
    n_src = left_source.shape[1]
    p = np.asarray(power if power is not None else np.ones(n_src), dtype=np.float64)
    rhs = np.r_[left_source @ p, np.zeros(n_patch), right_source @ p]
    return K, C, rhs, ldof, rdof, n_patch


COUPLED_SOLVE_RTOL = 1.0e-10
COUPLED_SOLVE_MAXITER = 2000


class _AMGPCGSolver:
    """Reusable AMG-preconditioned CG solver for one fixed SPD matrix."""

    def __init__(self, A, *, rtol=COUPLED_SOLVE_RTOL):
        self.A = sp.csr_matrix(A)
        self.rtol = float(rtol)
        hierarchy = pyamg.ruge_stuben_solver(self.A, interpolation="direct")
        self.M = hierarchy.aspreconditioner(cycle="V")

    def solve(self, rhs, *, x0=None, label="solve"):
        x, info = spla.cg(
            self.A,
            np.asarray(rhs, dtype=np.float64).ravel(),
            x0=None if x0 is None else np.asarray(x0, dtype=np.float64).ravel(),
            rtol=self.rtol,
            atol=0.0,
            maxiter=COUPLED_SOLVE_MAXITER,
            M=self.M,
        )
        if info != 0:
            raise RuntimeError(f"AMG-PCG failed for {label}: info={info}")
        return np.asarray(x, dtype=np.float64).ravel()


def solve_system(K, C, rhs, dt: float, duration: float):
    """Steady + BDF1 transient solved by reusable AMG-preconditioned CG."""
    steady_solver = _AMGPCGSolver(K)
    steady = steady_solver.solve(rhs, label="steady solve")

    lhs = (K.tocsc() + C.tocsc() / dt).tocsr()
    transient_solver = _AMGPCGSolver(lhs)
    state = np.zeros(K.shape[0])
    history = [state.copy()]
    for step in range(1, round(duration / dt) + 1):
        transient_rhs = C @ state / dt + rhs
        state = transient_solver.solve(
            transient_rhs,
            x0=state,
            label=f"transient step {step}",
        )
        history.append(state.copy())
    return steady, np.asarray(history)
