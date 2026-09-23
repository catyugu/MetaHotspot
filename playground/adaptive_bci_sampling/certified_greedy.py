"""Output-oriented residual certificates for affine SPD thermal systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla


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
        residual_coefficients = np.vstack(residual_coefficients)
        gram_action = self.riesz_gram @ residual_coefficients
        residual_squared = np.einsum(
            "ij,ij->j", residual_coefficients, gram_action
        )
        residual_norms = np.sqrt(np.maximum(residual_squared, 0.0))
        return coefficients, residual_coefficients, residual_norms

    def evaluate(
        self, parameter, power
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ROM junctions and componentwise absolute/relative bounds."""
        power = np.asarray(power, dtype=np.float64).ravel()
        coefficients, residual_coefficients, residual_norms = self._solve(parameter)
        junction = np.asarray(
            self.projected_source.T @ (coefficients @ power)
        ).ravel()

        primal_coefficients = residual_coefficients @ power
        primal_squared = float(
            primal_coefficients @ (self.riesz_gram @ primal_coefficients)
        )
        absolute_bound = np.sqrt(
            residual_norms * residual_norms * max(primal_squared, 0.0)
        )

        denominator = np.abs(junction) - absolute_bound
        relative_bound = np.full_like(absolute_bound, np.inf)
        valid = denominator > 0.0
        relative_bound[valid] = absolute_bound[valid] / denominator[valid]
        return junction, absolute_bound, relative_bound

    def evaluate_entrywise(
        self, parameter
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the full junction transfer and entrywise error bounds."""
        coefficients, _residual_coefficients, residual_norms = self._solve(
            parameter
        )
        transfer = np.ascontiguousarray(self.projected_source.T @ coefficients)
        absolute_bound = np.outer(residual_norms, residual_norms)
        denominator = np.abs(transfer) - absolute_bound
        relative_bound = np.full_like(absolute_bound, np.inf)
        valid = denominator > 0.0
        relative_bound[valid] = absolute_bound[valid] / denominator[valid]
        return transfer, absolute_bound, relative_bound


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
    factor = minimum_factor or spla.splu(minimum_operator.tocsc())

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
