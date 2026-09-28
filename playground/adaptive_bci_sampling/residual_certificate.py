"""Output-oriented residual certificates for affine SPD thermal systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sparse_solve import AmgSolver


def select_worst_certificate(relative_scores, absolute_scores) -> int:
    """Select the worst candidate without ordering-dependent infinite ties."""
    relative = np.asarray(relative_scores, dtype=np.float64).ravel()
    absolute = np.asarray(absolute_scores, dtype=np.float64).ravel()
    if relative.shape != absolute.shape or not relative.size:
        raise ValueError("certificate score arrays must have the same nonzero size")
    unresolved = np.flatnonzero(~np.isfinite(relative))
    if unresolved.size:
        return int(unresolved[int(np.argmax(absolute[unresolved]))])
    return int(np.argmax(relative))


@dataclass
class ResidualCertificate:
    """Cheap online primal-dual certificate for one fixed Galerkin basis."""

    projected_base: np.ndarray
    projected_terms: list[np.ndarray]
    projected_source: np.ndarray
    riesz_gram: np.ndarray
    source_count: int
    basis_order: int

    def _solve(self, parameter):
        """Reduced coefficients and the residual coefficients in the Riesz span."""
        parameter = np.asarray(parameter, dtype=np.float64).ravel()
        reduced_operator = self.projected_base.copy()
        for value, term in zip(parameter, self.projected_terms):
            reduced_operator += float(value) * term
        coefficients = scipy.linalg.solve(
            reduced_operator,
            self.projected_source,
            assume_a="pos",
            check_finite=False,
        )
        identity = np.eye(self.source_count)
        residual_coefficients = [identity, -coefficients]
        residual_coefficients.extend(
            -float(value) * coefficients for value in parameter
        )
        return coefficients, np.vstack(residual_coefficients)

    def port_defect_bound(self, parameter) -> tuple[float, float]:
        """Cheap all-input majorant of the relative port defect at one parameter.

        The exact port defect of the fixed basis is

        ``E(p) = R(p)^T A(p)^-1 R(p) = Y(p) - Y_V(p) >= 0``,

        with ``R(p) = G - A(p) V Q(p)``.  Every affine boundary term is positive
        semidefinite, so ``A(p) >= A(h_min)`` and the fixed-anchor Riesz Gram
        gives the matrix majorant

        ``U(p) := R(p)^T A(h_min)^-1 R(p) >= E(p)``.

        Since ``Y_V(p) <= Y(p)``, the same majorant normalized by the *reduced*
        transfer is a cheap all-input score,

        ``lambda_max(E(p), Y(p)) <= lambda_max(U(p), Y_V(p))``.

        Returns ``(bound, absolute)``: the majorant above, and
        ``lambda_max(U(p))`` as the tie-breaker used when ``Y_V`` is not positive
        definite and the relative majorant is infinite.  Both are small dense
        quantities, so scoring a whole candidate grid costs no full-order solve.
        """
        coefficients, residual_coefficients = self._solve(parameter)
        majorant = residual_coefficients.T @ (self.riesz_gram @ residual_coefficients)
        majorant = 0.5 * (majorant + majorant.T)
        absolute = float(np.max(np.linalg.eigvalsh(majorant)))
        transfer = np.ascontiguousarray(self.projected_source.T @ coefficients)
        transfer = 0.5 * (transfer + transfer.T)
        if float(np.min(np.linalg.eigvalsh(transfer))) <= 0.0:
            return float("inf"), absolute
        bound = scipy.linalg.eigvalsh(majorant, transfer, check_finite=False)
        return float(np.max(bound)), absolute


def prepare_residual_certificate(
    base_operator,
    boundary_terms,
    source_shape,
    parameter_ranges,
    basis,
    *,
    minimum_factor=None,
) -> ResidualCertificate:
    """Prepare an ``A(h_min)``-Riesz residual certificate.

    Since every affine boundary term is positive semidefinite,

    ``A(h) >= A(h_min)`` and ``A(h)^-1 <= A(h_min)^-1``.

    Hence the primal and dual residual norms in the fixed minimum-corner Riesz
    map rigorously upper-bound their parameter-dependent energy dual norms.
    All online residuals lie in the span of ``[G, K V, H_1 V, ...]``; the dense
    Gram matrix below makes evaluation independent of the full model size.
    """
    K = sp.csc_matrix(base_operator)
    terms = [sp.csc_matrix(term) for term in boundary_terms]
    source = np.asarray(source_shape, dtype=np.float64)
    ranges = np.asarray(parameter_ranges, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if ranges.shape != (len(terms), 2):
        raise ValueError("parameter_ranges must have shape (n_terms, 2)")
    if basis.ndim != 2 or basis.shape[0] != K.shape[0] or basis.shape[1] == 0:
        raise ValueError("basis must be a nonempty two-dimensional array")

    minimum_operator = K.copy()
    for value, term in zip(ranges[:, 0], terms):
        minimum_operator = minimum_operator + float(value) * term
    factor = minimum_factor or AmgSolver(minimum_operator)

    operator_blocks = [K @ basis]
    operator_blocks.extend(term @ basis for term in terms)
    residual_span = np.ascontiguousarray(
        np.column_stack([source, *operator_blocks])
    )
    riesz_images = factor.solve(residual_span)
    riesz_gram = residual_span.T @ riesz_images
    riesz_gram = np.ascontiguousarray(0.5 * (riesz_gram + riesz_gram.T))

    return ResidualCertificate(
        projected_base=np.ascontiguousarray(basis.T @ (K @ basis)),
        projected_terms=[
            np.ascontiguousarray(basis.T @ (term @ basis)) for term in terms
        ],
        projected_source=np.ascontiguousarray(basis.T @ source),
        riesz_gram=riesz_gram,
        source_count=int(source.shape[1]),
        basis_order=int(basis.shape[1]),
    )
