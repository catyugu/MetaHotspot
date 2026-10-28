#!/usr/bin/env python3
"""Static-condensation reduction with an exactly preserved physical interface.

Every reduced model here uses the same two-block lifting

    T = V x,   V = [[Psi, W], [I, 0]],   Psi = -K_ii^-1 K_iG,

with ``G`` the interface (port) DOFs, ``I`` the interior, ``Psi`` the exact
static constraint modes and ``W`` an interior enrichment space.  Three facts
follow from the algebra alone and are asserted in ``test_reduction.py``:

1. ``K_r = V^T K V`` is block diagonal between the interface and the
   enrichment coordinates, and its interface block is the static Schur
   complement ``S(0)``.
2. Therefore the *steady* interface relation is exact for **any** interface
   flux, linear or nonlinear: the interface row of the reduced residual is
   ``S(0) T_G + f_G``, independent of ``W``.
3. A port flux produces exactly the static constraint mode
   (``K_ii^-1 f_G = Psi``), so no "interface-load direction" is needed.  What
   is *not* exact is the interior response to an internal **source** load
   ``K_ii^-1 f_source``; that is the Guyan/CB interior error, and it is measured
   rather than assumed.

All full-order solves are AMG-preconditioned CG on a positive real shift, so
every operator handled here is symmetric positive definite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from linalg import Cost, SPDSolver, dense_spd_solve


@dataclass(frozen=True)
class Partition:
    Kii: sp.csc_matrix
    KiG: sp.csc_matrix
    KGG: np.ndarray
    Cii: sp.csc_matrix
    CiG: sp.csc_matrix
    CGG: np.ndarray
    interface: np.ndarray
    interior: np.ndarray

    @property
    def port_count(self) -> int:
        return int(self.interface.size)

    @property
    def interior_count(self) -> int:
        return int(self.interior.size)


def partition(operators, interface: np.ndarray) -> Partition:
    interface = np.asarray(interface, dtype=np.int64)
    interior = np.setdiff1d(np.arange(operators.K.shape[0]), interface)
    K = operators.K.tocsr()
    C = operators.C.tocsr()
    return Partition(
        Kii=K[interior][:, interior].tocsc(),
        KiG=K[interior][:, interface].tocsc(),
        KGG=K[interface][:, interface].toarray(),
        Cii=C[interior][:, interior].tocsc(),
        CiG=C[interior][:, interface].tocsc(),
        CGG=C[interface][:, interface].toarray(),
        interface=interface,
        interior=interior,
    )


# ---------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------


def orthonormal(vectors: np.ndarray, against: np.ndarray | None = None, tol: float = 1.0e-10) -> np.ndarray:
    """Orthonormal basis of the range of ``vectors`` modulo ``against``."""
    block = np.asarray(vectors, dtype=np.float64)
    if block.ndim == 1:
        block = block[:, None]
    if block.size == 0:
        return np.empty((block.shape[0], 0))
    if against is not None and against.shape[1]:
        for _ in range(2):
            block = block - against @ (against.T @ block)
    u, s, _ = np.linalg.svd(block, full_matrices=False)
    return np.ascontiguousarray(u[:, s > tol * s[0]]) if s.size else np.empty((block.shape[0], 0))


def shifted_operator(part: Partition, shift: float):
    """``A_ii(s) = K_ii + s C_ii``, SPD for every ``s >= 0``."""
    return (part.Kii + shift * part.Cii).tocsc()


STATIC_RTOL = 1.0e-12


def static_modes(part: Partition, cost: Cost | None = None) -> np.ndarray:
    """``Psi = -K_ii^-1 K_iG`` by AMG-CG, one hierarchy for all port columns.

    Computed once, so it is solved to a tighter tolerance than the enrichment
    solves.  The structural identities below are therefore exact only up to
    ``STATIC_RTOL``; measured port steady-state deviation is ~1e-8 K, not
    machine zero, and that limit is reported rather than hidden.
    """
    solver = SPDSolver(part.Kii, rtol=STATIC_RTOL, cost=cost)
    return -solver.solve(part.KiG.toarray(), label="static modes")


def response(part: Partition, shift: float, inputs: np.ndarray | None = None, cost: Cost | None = None) -> np.ndarray:
    """Interior response ``-A_ii(s)^-1 b`` at ``s = shift``.

    ``inputs`` defaults to the interface coupling ``A_iG(s)``.  Passing the
    source columns as well makes the same space cover the *source* response,
    which the interface-coupling-only space never does.  At ``shift = 0`` with
    the default inputs this is exactly :func:`static_modes`.
    """
    a_ii = shifted_operator(part, shift)
    if inputs is None:
        rhs = (part.KiG + shift * part.CiG).toarray()
    else:
        rhs = np.asarray(inputs, dtype=np.float64)
    return -SPDSolver(a_ii, cost=cost).solve(rhs, label=f"response s={shift:g}")


def residual_indicator(part: Partition, basis: np.ndarray, shift: float, inputs: np.ndarray | None = None) -> float:
    """Relative Galerkin state residual of the shifted response."""
    a_ii = shifted_operator(part, shift)
    rhs = (part.KiG + shift * part.CiG).toarray() if inputs is None else np.asarray(inputs)
    if basis.shape[1] == 0:
        return 1.0
    coefficients = np.linalg.solve(basis.T @ (a_ii @ basis), basis.T @ rhs)
    return float(np.linalg.norm(rhs + a_ii @ (basis @ coefficients), "fro") / np.linalg.norm(rhs, "fro"))


# ---------------------------------------------------------------------------
# enrichment spaces
# ---------------------------------------------------------------------------


def craig_bampton(part: Partition, order: int) -> np.ndarray:
    """Lowest ``order`` fixed-interface eigenmodes of ``(K_ii, C_ii)``."""
    if order <= 0:
        return np.empty((part.interior_count, 0))
    _, vectors = scipy.linalg.eigh(
        part.Kii.toarray(),
        part.Cii.toarray(),
        subset_by_index=(0, order - 1),
        check_finite=False,
    )
    return np.ascontiguousarray(vectors)


@dataclass
class ResponsePlan:
    """One AMG hierarchy per shift; its response kept as SVD factors.

    A single hierarchy at one shift feeds many greedy steps: successive singular
    directions of the same response block cost no further setup.  ``inputs``
    selects which excitations the plan covers -- the interface coupling only, or
    the interface coupling *and* the internal sources.
    """

    shifts: np.ndarray
    directions: dict[float, np.ndarray]
    values: dict[float, np.ndarray]
    inputs: np.ndarray | None = None

    @classmethod
    def build(cls, part: Partition, shifts, cost: Cost | None = None, inputs: np.ndarray | None = None) -> "ResponsePlan":
        directions, values = {}, {}
        for shift in shifts:
            u, s, _ = np.linalg.svd(response(part, float(shift), inputs, cost), full_matrices=False)
            keep = s > 1.0e-10 * s[0]
            directions[float(shift)] = np.ascontiguousarray(u[:, keep])
            values[float(shift)] = s[keep]
        return cls(
            shifts=np.asarray(shifts, dtype=float),
            directions=directions,
            values=values,
            inputs=None if inputs is None else np.asarray(inputs),
        )

    def rank(self, shift: float) -> int:
        return int(self.directions[shift].shape[1])


def rational_krylov(part: Partition, order: int, plan: ResponsePlan, *, trace: list | None = None) -> np.ndarray:
    """Residual-greedy enrichment of the static space, one direction per step.

    Starts from the static constraint space and repeatedly adds the singular
    direction, from the shift whose Galerkin residual is currently largest,
    that carries the most new information.  Returns the enrichment only; the
    static block is added by :func:`assemble` from the exact ``Psi``.
    """
    basis = orthonormal(static_modes(part))
    m = basis.shape[1]
    used = {float(shift): 0 for shift in plan.shifts}
    target = m + max(0, order)

    while basis.shape[1] < target:
        available = [s for s in used if used[s] < plan.rank(s)]
        if not available:
            break
        indicators = {s: residual_indicator(part, basis, s, plan.inputs) for s in available}
        shift = max(available, key=indicators.get)

        remaining = plan.directions[shift][:, used[shift] :]
        projected = remaining - basis @ (basis.T @ remaining)
        norms = np.linalg.norm(projected, axis=0)
        choice = int(np.argmax(norms))
        used[shift] += choice + 1
        block = orthonormal(remaining[:, choice], against=basis)
        if trace is not None:
            trace.append(
                {
                    "shift": float(shift),
                    "indicator": float(indicators[shift]),
                    "new_component": float(norms[choice]),
                    "dimension": int(basis.shape[1] + block.shape[1]),
                }
            )
        if block.shape[1]:
            basis = np.column_stack((basis, block))
    return np.ascontiguousarray(basis[:, m:])


def source_directions(part: Partition, source: np.ndarray, cost: Cost | None = None) -> np.ndarray:
    """Interior static responses to the source shapes: ``K_ii^-1 f_source``."""
    solver = SPDSolver(part.Kii, cost=cost)
    return solver.solve(np.asarray(source)[part.interior, :], label="source directions")


# ---------------------------------------------------------------------------
# reduced model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reduced:
    K: np.ndarray
    C: np.ndarray
    V: np.ndarray
    interface: np.ndarray
    port_count: int

    @property
    def dimension(self) -> int:
        return int(self.K.shape[0])

    @property
    def interior_dimension(self) -> int:
        return self.dimension - self.port_count


def assemble(operators, part: Partition, enrichment: np.ndarray, psi: np.ndarray) -> Reduced:
    """Galerkin projection on ``V = [[Psi, W], [I, 0]]``."""
    m = part.port_count
    q = enrichment.shape[1]
    V = np.zeros((operators.K.shape[0], m + q))
    V[part.interface, :m] = np.eye(m)
    V[part.interior, :m] = psi
    V[part.interior, m:] = enrichment
    return Reduced(
        K=np.asarray(V.T @ (operators.K @ V)),
        C=np.asarray(V.T @ (operators.C @ V)),
        V=V,
        interface=part.interface,
        port_count=m,
    )


# ---------------------------------------------------------------------------
# interface operators on the positive real axis and their norms
# ---------------------------------------------------------------------------


def exact_interface_operator(part: Partition, extra: np.ndarray, shift: float, cost: Cost | None = None) -> np.ndarray:
    """``S(s) = A_GG + diag(extra) - A_Gi A_ii(s)^-1 A_iG`` at real ``s >= 0``."""
    a_ii = shifted_operator(part, shift)
    a_iG = (part.KiG + shift * part.CiG).toarray()
    solved = SPDSolver(a_ii, cost=cost).solve(a_iG, label=f"exact S(s={shift:g})")
    return part.KGG + np.diag(extra) + shift * part.CGG - a_iG.T @ solved


def reduced_interface_operator(model: Reduced, extra: np.ndarray, shift: float, cost: Cost | None = None) -> np.ndarray:
    """Schur complement of the reduced system with respect to its interface block."""
    m = model.port_count
    a = model.K + shift * model.C
    a = a.copy()
    a[np.arange(m), np.arange(m)] += extra
    if model.dimension == m:
        return a[:m, :m]
    inner = a[m:, m:]
    coupling = a[m:, :m]
    correction = np.column_stack(
        [
            dense_spd_solve(inner, coupling[:, column], cost=cost)
            for column in range(m)
        ]
    )
    return a[:m, :m] - coupling.T @ correction


def scaled_norm(matrix: np.ndarray, scale: np.ndarray) -> float:
    """Spectral norm of ``scale^-1/2 matrix scale^-1/2`` for SPD ``scale``."""
    chol = np.linalg.cholesky(scale)
    left = scipy.linalg.solve_triangular(chol, matrix, lower=True, check_finite=False)
    scaled = scipy.linalg.solve_triangular(chol, left.T, lower=True, check_finite=False).T
    return float(np.linalg.svd(scaled, compute_uv=False)[0])
